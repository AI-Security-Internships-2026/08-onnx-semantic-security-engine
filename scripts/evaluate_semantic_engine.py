"""
Semantic Engine Integration Evaluation — Local Runner

Runs the full semantic security engine (ConfidenceAnalyzer + DriftDetector +
InputValidator) on real CIC-IDS2018 and ToN-IoT data to produce:
  - experiments/results/semantic_engine_evaluation.json
  - experiments/images/semantic_engine_evaluation_plots.png

Usage:
    python scripts/evaluate_semantic_engine.py
"""

import json
import time
import sys
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.spatial.distance import cosine as cosine_distance
from scipy.special import softmax

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"
DATASETS = BASE_DIR / "datasets"

ONNX_MODEL_PATH     = EXPERIMENTS / "threat_mlp_nf_fp32.onnx"
SCALER_PATH         = EXPERIMENTS / "standard_scaler_nf.joblib"
ENCODER_PATH        = EXPERIMENTS / "label_encoder_nf.joblib"
REF_EMBEDDINGS_PATH = EXPERIMENTS / "reference_embeddings_nf.npz"
FEATURE_STATS_PATH  = EXPERIMENTS / "training_feature_stats_nf.json"

CIC_DIR     = DATASETS / "CIC-IDS2018"
TONIOT_PATH = DATASETS / "ToN-IoT" / "NF-ToN-IoT-V2.parquet"

# ── Evaluation parameters ──
N_IN_DIST_SAMPLES  = 10000
N_OOD_SAMPLES      = 10000
N_NOISE_SAMPLES    = 1000
N_ZERO_SAMPLES     = 500
N_LATENCY_ITERS    = 1000

# ── NF Feature Schema ──
#
# CORRECTED FEATURE MAP (v2) — Re-derived against nProbe's authoritative
# NetFlow field documentation (https://www.ntop.org/guides/nprobe/
# flow_information_elements.html) and CICFlowMeter source definitions.
#
# 8 of the original 21 mappings were semantically mismatched:
#   - SRC_TO_DST_AVG_THROUGHPUT (bps rate) → Fwd Header Length (byte count)
#   - DST_TO_SRC_AVG_THROUGHPUT (bps rate) → Bwd Header Length (byte count)
#   - RETRANSMITTED_IN_PKTS (retransmission count) → Fwd Avg Packets/Bulk
#   - RETRANSMITTED_OUT_PKTS (retransmission count) → Bwd Avg Packets/Bulk
#   - RETRANSMITTED_IN_BYTES (retransmitted bytes) → Fwd Avg Bytes/Bulk
#   - RETRANSMITTED_OUT_BYTES (retransmitted bytes) → Bwd Avg Bytes/Bulk
#   - NUM_PKTS_UP_TO_128_BYTES (size-bucket count) → Subflow Fwd Packets
#   - NUM_PKTS_1024_TO_1514_BYTES (size-bucket count) → Subflow Bwd Packets
#   - TCP_FLAGS (cumulative bitmask) → Fwd PSH Flags (directional count)
#
# These were dropped. 13 genuinely equivalent pairs remain.
# See FEATURE_MAP_NOTES for per-field match quality.
#
# NOTE: The existing ONNX model (threat_mlp_nf_fp32.onnx) was trained on the
# original 21-feature map (LEGACY_FEATURE_MAP below). Retraining with the
# corrected 13-feature subset is required as a follow-up on Kaggle.
#

FEATURE_MAP = {
    # ── Exact matches ──
    "FLOW_DURATION_MILLISECONDS":    "Flow Duration",
    "IN_PKTS":                       "Total Fwd Packets",
    "OUT_PKTS":                      "Total Backward Packets",
    "IN_BYTES":                      "Fwd Packets Length Total",
    "OUT_BYTES":                     "Bwd Packets Length Total",
    "LONGEST_FLOW_PKT":              "Packet Length Max",
    "SHORTEST_FLOW_PKT":             "Packet Length Min",
    # ── Approximate matches (same physical quantity, minor scope difference) ──
    "MAX_IP_PKT_LEN":                "Fwd Packet Length Max",     # nProbe: bidirectional max; CIC: fwd-only max
    "MIN_IP_PKT_LEN":                "Fwd Packet Length Min",     # nProbe: bidirectional min; CIC: fwd-only min
    "SRC_TO_DST_SECOND_BYTES":       "Flow Bytes/s",              # nProbe: src→dst rate; CIC: bidirectional rate
    "TCP_WIN_MAX_IN":                "Init Fwd Win Bytes",        # nProbe: max TCP window; CIC: initial window
    "TCP_WIN_MAX_OUT":               "Init Bwd Win Bytes",        # nProbe: max TCP window; CIC: initial window
}

# Per-field match quality documentation
FEATURE_MAP_NOTES = {
    "FLOW_DURATION_MILLISECONDS":  "exact — both measure flow duration",
    "IN_PKTS":                     "exact — both count forward-direction packets",
    "OUT_PKTS":                    "exact — both count backward-direction packets",
    "IN_BYTES":                    "exact — both count forward-direction bytes",
    "OUT_BYTES":                   "exact — both count backward-direction bytes",
    "LONGEST_FLOW_PKT":            "exact — both measure max packet length (bidirectional)",
    "SHORTEST_FLOW_PKT":           "exact — both measure min packet length (bidirectional)",
    "MAX_IP_PKT_LEN":              "approximate — nProbe is bidirectional max, CIC is fwd-only max",
    "MIN_IP_PKT_LEN":              "approximate — nProbe is bidirectional min, CIC is fwd-only min",
    "SRC_TO_DST_SECOND_BYTES":     "approximate — nProbe is src→dst bytes/s, CIC is bidirectional bytes/s",
    "TCP_WIN_MAX_IN":              "approximate — nProbe is max observed window, CIC is initial window",
    "TCP_WIN_MAX_OUT":             "approximate — nProbe is max observed window, CIC is initial window",
}

# Original 21-feature mapping preserved for reproducibility of prior results.
# This mapping contains 8 semantic mismatches identified during supervisor review.
LEGACY_FEATURE_MAP = {
    "FLOW_DURATION_MILLISECONDS":    "Flow Duration",
    "IN_PKTS":                       "Total Fwd Packets",
    "OUT_PKTS":                      "Total Backward Packets",
    "IN_BYTES":                      "Fwd Packets Length Total",
    "OUT_BYTES":                     "Bwd Packets Length Total",
    "MAX_IP_PKT_LEN":                "Fwd Packet Length Max",
    "MIN_IP_PKT_LEN":                "Fwd Packet Length Min",
    "LONGEST_FLOW_PKT":              "Packet Length Max",
    "SHORTEST_FLOW_PKT":             "Packet Length Min",
    "SRC_TO_DST_SECOND_BYTES":       "Flow Bytes/s",
    "SRC_TO_DST_AVG_THROUGHPUT":     "Fwd Header Length",       # ❌ MISMATCHED: rate (bps) vs byte count
    "DST_TO_SRC_AVG_THROUGHPUT":     "Bwd Header Length",       # ❌ MISMATCHED: rate (bps) vs byte count
    "TCP_FLAGS":                     "Fwd PSH Flags",           # ❌ MISMATCHED: aggregate bitmask vs directional count
    "TCP_WIN_MAX_IN":                "Init Fwd Win Bytes",
    "TCP_WIN_MAX_OUT":               "Init Bwd Win Bytes",
    "RETRANSMITTED_IN_PKTS":         "Fwd Avg Packets/Bulk",    # ❌ MISMATCHED: retransmission vs bulk stat
    "RETRANSMITTED_OUT_PKTS":        "Bwd Avg Packets/Bulk",    # ❌ MISMATCHED: retransmission vs bulk stat
    "RETRANSMITTED_IN_BYTES":        "Fwd Avg Bytes/Bulk",      # ❌ MISMATCHED: retransmission vs bulk stat
    "RETRANSMITTED_OUT_BYTES":       "Bwd Avg Bytes/Bulk",      # ❌ MISMATCHED: retransmission vs bulk stat
    "NUM_PKTS_UP_TO_128_BYTES":      "Subflow Fwd Packets",     # ❌ MISMATCHED: size-bucket vs subflow count
    "NUM_PKTS_1024_TO_1514_BYTES":   "Subflow Bwd Packets",     # ❌ MISMATCHED: size-bucket vs subflow count
}

NF_FEATURES = list(FEATURE_MAP.values())
TONIOT_FEATURES = list(FEATURE_MAP.keys())


def main():
    print("=" * 70)
    print("  SEMANTIC ENGINE INTEGRATION EVALUATION")
    print("=" * 70)

    # ── Verify files ──
    print("\n[Setup] Verifying files...")
    for p in [ONNX_MODEL_PATH, SCALER_PATH, ENCODER_PATH, REF_EMBEDDINGS_PATH,
              FEATURE_STATS_PATH, TONIOT_PATH]:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p.name}")
        if not p.exists():
            print(f"  FATAL: {p} not found. Exiting.")
            sys.exit(1)

    # ── Load artifacts ──
    print("\n[Setup] Loading model artifacts...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]
    print(f"  ONNX outputs: {output_names}")
    assert "embedding" in output_names, "Model must have dual outputs!"

    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)
    print(f"  Classes ({len(class_names)}): {class_names}")

    ref_data = np.load(REF_EMBEDDINGS_PATH, allow_pickle=True)
    global_centroid    = ref_data["global_centroid"]
    class_centroids    = ref_data["class_centroids"]
    covariance_inverse = ref_data["covariance_inverse"]
    cosine_threshold   = float(ref_data["cosine_threshold"])
    mahal_threshold    = float(ref_data["mahal_threshold"])
    ref_class_names    = list(ref_data["class_names"])
    print(f"  Cosine threshold:  {cosine_threshold:.4f}")
    print(f"  Mahalanobis threshold: {mahal_threshold:.4f}")

    with open(FEATURE_STATS_PATH) as f:
        feature_stats_data = json.load(f)
    num_features = feature_stats_data["num_features"]
    feat_means = np.array([f["mean"] for f in feature_stats_data["features"]])
    feat_stds  = np.array([f["std"]  for f in feature_stats_data["features"]])
    feat_names = [f["name"] for f in feature_stats_data["features"]]
    print(f"  Feature stats loaded: {num_features} features")

    # ── Thresholds ──
    CONFIDENCE_THRESHOLD = 0.70
    ZSCORE_THRESHOLD = 5.0
    ZERO_FILL_RATIO = 0.50
    RANGE_TOLERANCE_SIGMAS = 5.0

    # ── Analyzer functions ──
    def analyze_confidence(softmax_probs):
        confidence = float(np.max(softmax_probs))
        flag = "LOW_CONFIDENCE" if confidence < CONFIDENCE_THRESHOLD else "OK"
        return {"confidence_score": round(confidence, 4), "confidence_flag": flag}

    def analyze_drift(embedding):
        cos_dist = cosine_distance(embedding, global_centroid)
        cos_dist = float(np.nan_to_num(cos_dist, nan=0.0))
        diff = embedding - global_centroid
        mahal_dist = float(np.sqrt(max(diff @ covariance_inverse @ diff, 0.0)))
        class_distances = []
        for centroid in class_centroids:
            d = cosine_distance(embedding, centroid)
            class_distances.append(float(np.nan_to_num(d, nan=1.0)))
        nearest_idx = int(np.argmin(class_distances))
        nearest_class = str(ref_class_names[nearest_idx])
        cos_norm = cos_dist / max(cosine_threshold, 1e-8)
        mahal_norm = mahal_dist / max(mahal_threshold, 1e-8)
        drift_score = float(min(max(cos_norm, mahal_norm), 2.0))
        drift_flag = "DRIFT_DETECTED" if (cos_dist > cosine_threshold or mahal_dist > mahal_threshold) else "OK"
        return {
            "cosine_distance": round(cos_dist, 4),
            "mahalanobis_distance": round(mahal_dist, 4),
            "drift_score": round(drift_score, 4),
            "drift_flag": drift_flag,
            "nearest_class": nearest_class,
        }

    def validate_input(raw_features):
        alerts = []
        if len(raw_features) != num_features:
            alerts.append(f"SCHEMA_MISMATCH: Expected {num_features}, got {len(raw_features)}")
            return {"validation_passed": False, "alerts": alerts}
        features = np.asarray(raw_features, dtype=np.float64)
        if np.isnan(features).any():
            alerts.append("NAN_VALUES")
            return {"validation_passed": False, "alerts": alerts}
        if np.isinf(features).any():
            alerts.append("INF_VALUES")
            return {"validation_passed": False, "alerts": alerts}
        zero_count = int(np.sum(features == 0.0))
        zero_ratio = zero_count / num_features
        if zero_ratio > ZERO_FILL_RATIO:
            alerts.append(f"ZERO_FILLED: {zero_count}/{num_features} ({zero_ratio:.0%})")
        range_lower = feat_means - RANGE_TOLERANCE_SIGMAS * feat_stds
        range_upper = feat_means + RANGE_TOLERANCE_SIGMAS * feat_stds
        out_of_range = (features < range_lower) | (features > range_upper)
        if out_of_range.any():
            n_oor = int(out_of_range.sum())
            alerts.append(f"OUT_OF_RANGE: {n_oor}/{num_features} features")
        safe_stds = np.where(feat_stds > 1e-10, feat_stds, 1.0)
        zscores = np.abs((features - feat_means) / safe_stds)
        extreme = zscores > ZSCORE_THRESHOLD
        if extreme.any():
            n_extreme = int(extreme.sum())
            alerts.append(f"EXTREME_OUTLIERS: {n_extreme}/{num_features} features")
        passed = not any(a.startswith(("SCHEMA_", "NAN_", "INF_", "ZERO_FILLED")) for a in alerts)
        return {"validation_passed": passed, "alerts": alerts}

    def compute_verdict(conf, drift, val):
        total_alerts = 0
        if conf["confidence_flag"] != "OK":
            total_alerts += 1
        if drift["drift_flag"] == "DRIFT_DETECTED":
            total_alerts += 1
        total_alerts += len(val["alerts"])
        if not val["validation_passed"]:
            verdict = "REJECTED"
        elif total_alerts >= 2:
            verdict = "HIGH_RISK"
        elif total_alerts >= 1:
            verdict = "SUSPICIOUS"
        else:
            verdict = "CLEAN"
        return verdict, total_alerts

    def run_semantic_engine(raw_batch, scaled_batch, batch_size=512):
        """Run engine on batches to avoid memory issues."""
        results = []
        for start in range(0, len(raw_batch), batch_size):
            end = min(start + batch_size, len(raw_batch))
            batch_scaled = scaled_batch[start:end].astype(np.float32)
            onnx_out = session.run(None, {"input": batch_scaled})
            logits = onnx_out[0]
            embeddings = onnx_out[1]
            for i in range(len(batch_scaled)):
                probs = softmax(logits[i])
                conf = analyze_confidence(probs)
                drift = analyze_drift(embeddings[i])
                val = validate_input(raw_batch[start + i])
                verdict, n_alerts = compute_verdict(conf, drift, val)
                results.append({
                    "confidence": conf,
                    "drift": drift,
                    "validation": val,
                    "verdict": verdict,
                    "total_alerts": n_alerts,
                    "predicted_class": class_names[int(np.argmax(logits[i]))],
                })
            if (start + batch_size) % 1000 == 0 or end == len(raw_batch):
                print(f"    Processed {end}/{len(raw_batch)} samples...")
        return results

    def summarize_results(results, label):
        n = len(results)
        drift_detected = sum(1 for r in results if r["drift"]["drift_flag"] == "DRIFT_DETECTED")
        low_confidence = sum(1 for r in results if r["confidence"]["confidence_flag"] == "LOW_CONFIDENCE")
        val_passed = sum(1 for r in results if r["validation"]["validation_passed"])
        verdicts = {"CLEAN": 0, "SUSPICIOUS": 0, "HIGH_RISK": 0, "REJECTED": 0}
        for r in results:
            verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        drift_scores = [r["drift"]["drift_score"] for r in results]
        cosine_dists = [r["drift"]["cosine_distance"] for r in results]
        mahal_dists  = [r["drift"]["mahalanobis_distance"] for r in results]
        conf_scores  = [r["confidence"]["confidence_score"] for r in results]
        alert_counts = {}
        for r in results:
            for alert in r["validation"]["alerts"]:
                alert_type = alert.split(":")[0]
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1

        summary = {
            "scenario": label,
            "n_samples": n,
            "drift_detected_count": drift_detected,
            "drift_detected_pct": round(drift_detected / n * 100, 2),
            "low_confidence_count": low_confidence,
            "low_confidence_pct": round(low_confidence / n * 100, 2),
            "validation_passed_count": val_passed,
            "validation_passed_pct": round(val_passed / n * 100, 2),
            "drift_score_stats": {
                "mean": round(float(np.mean(drift_scores)), 4),
                "std": round(float(np.std(drift_scores)), 4),
                "min": round(float(np.min(drift_scores)), 4),
                "max": round(float(np.max(drift_scores)), 4),
                "median": round(float(np.median(drift_scores)), 4),
                "p95": round(float(np.percentile(drift_scores, 95)), 4),
            },
            "cosine_distance_stats": {
                "mean": round(float(np.mean(cosine_dists)), 4),
                "std": round(float(np.std(cosine_dists)), 4),
                "median": round(float(np.median(cosine_dists)), 4),
                "p95": round(float(np.percentile(cosine_dists, 95)), 4),
            },
            "mahalanobis_distance_stats": {
                "mean": round(float(np.mean(mahal_dists)), 4),
                "std": round(float(np.std(mahal_dists)), 4),
                "median": round(float(np.median(mahal_dists)), 4),
                "p95": round(float(np.percentile(mahal_dists, 95)), 4),
            },
            "confidence_stats": {
                "mean": round(float(np.mean(conf_scores)), 4),
                "std": round(float(np.std(conf_scores)), 4),
                "min": round(float(np.min(conf_scores)), 4),
                "median": round(float(np.median(conf_scores)), 4),
            },
            "engine_verdicts": verdicts,
            "validation_alert_types": alert_counts,
        }

        print(f"\n{'=' * 70}")
        print(f"  {label}")
        print(f"{'=' * 70}")
        print(f"  Samples:          {n}")
        print(f"  Drift detected:   {drift_detected}/{n} ({drift_detected/n*100:.1f}%)")
        print(f"  Low confidence:   {low_confidence}/{n} ({low_confidence/n*100:.1f}%)")
        print(f"  Validation pass:  {val_passed}/{n} ({val_passed/n*100:.1f}%)")
        print(f"  Avg drift score:  {np.mean(drift_scores):.4f} (+/-{np.std(drift_scores):.4f})")
        print(f"  Avg confidence:   {np.mean(conf_scores):.4f}")
        print(f"  Verdicts: {verdicts}")
        if alert_counts:
            print(f"  Alert types: {alert_counts}")
        return summary

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 1: In-Distribution (CIC-IDS2018)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 1: In-Distribution (CIC-IDS2018)")
    print(f"{'#' * 70}")

    parquet_files = sorted(CIC_DIR.glob("*.parquet"))
    frames = []
    loaded = 0
    for f in parquet_files:
        if loaded >= N_IN_DIST_SAMPLES * 2:
            break
        chunk = pd.read_parquet(f)
        chunk.columns = chunk.columns.str.strip()
        frames.append(chunk)
        loaded += len(chunk)
        print(f"  Loaded {f.name}: {len(chunk)} rows")

    df_cic = pd.concat(frames, ignore_index=True)
    if "Timestamp" in df_cic.columns:
        df_cic = df_cic.drop(columns=["Timestamp"])
    df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cic = df_cic.dropna()

    available_nf = [f for f in NF_FEATURES if f in df_cic.columns]
    print(f"  Available NF features: {len(available_nf)}/{len(NF_FEATURES)}")

    X_cic_raw = df_cic[available_nf].values[:N_IN_DIST_SAMPLES]
    X_cic_scaled = scaler.transform(X_cic_raw).astype(np.float32)
    print(f"  Samples: {len(X_cic_raw)}")

    results_in_dist = run_semantic_engine(X_cic_raw, X_cic_scaled)
    summary_in_dist = summarize_results(results_in_dist, "Scenario 1: In-Distribution (CIC-IDS2018)")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 2: Out-of-Distribution (ToN-IoT)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 2: Out-of-Distribution (ToN-IoT)")
    print(f"{'#' * 70}")

    df_ton = pd.read_parquet(TONIOT_PATH)
    df_ton.columns = df_ton.columns.str.strip()
    print(f"  ToN-IoT shape: {df_ton.shape}")

    X_ton_raw = np.zeros((min(len(df_ton), N_OOD_SAMPLES), len(NF_FEATURES)), dtype=np.float64)
    for i, (ton_col, cic_col) in enumerate(FEATURE_MAP.items()):
        if ton_col in df_ton.columns:
            X_ton_raw[:, i] = df_ton[ton_col].values[:N_OOD_SAMPLES]
        else:
            print(f"  [WARN] Missing column: {ton_col}")

    X_ton_raw = np.nan_to_num(X_ton_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_ton_scaled = scaler.transform(X_ton_raw).astype(np.float32)
    print(f"  Samples: {len(X_ton_raw)}")

    results_ood = run_semantic_engine(X_ton_raw, X_ton_scaled)
    summary_ood = summarize_results(results_ood, "Scenario 2: Out-of-Distribution (ToN-IoT, real data)")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 3: Random Noise
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 3: Random Gaussian Noise")
    print(f"{'#' * 70}")

    np.random.seed(42)
    X_noise_raw = np.random.randn(N_NOISE_SAMPLES, num_features)
    X_noise_scaled = scaler.transform(X_noise_raw).astype(np.float32)
    print(f"  Samples: {len(X_noise_raw)}")

    results_noise = run_semantic_engine(X_noise_raw, X_noise_scaled)
    summary_noise = summarize_results(results_noise, "Scenario 3: Random Gaussian Noise")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 4: Zero-Filled Inputs
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 4: Zero-Filled Inputs (RQ3 failure mode)")
    print(f"{'#' * 70}")

    X_zero_raw = np.zeros((N_ZERO_SAMPLES, num_features))
    X_zero_scaled = scaler.transform(X_zero_raw).astype(np.float32)
    print(f"  Samples: {len(X_zero_raw)}")

    results_zero = run_semantic_engine(X_zero_raw, X_zero_scaled)
    summary_zero = summarize_results(results_zero, "Scenario 4: Zero-Filled Inputs (RQ3 failure mode)")

    # ══════════════════════════════════════════════════════════════
    # LATENCY BENCHMARKING
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# LATENCY BENCHMARKING")
    print(f"{'#' * 70}")

    test_raw = X_cic_raw[0:1]
    test_scaled = X_cic_scaled[0:1]

    # Plain inference
    times_plain = []
    for _ in range(N_LATENCY_ITERS):
        t0 = time.perf_counter()
        outputs = session.run(None, {"input": test_scaled})
        probs = softmax(outputs[0][0])
        pred = int(np.argmax(probs))
        t1 = time.perf_counter()
        times_plain.append((t1 - t0) * 1000)

    # Semantic inference
    times_semantic = []
    for _ in range(N_LATENCY_ITERS):
        t0 = time.perf_counter()
        val = validate_input(test_raw[0])
        outputs = session.run(None, {"input": test_scaled})
        probs = softmax(outputs[0][0])
        emb = outputs[1][0]
        conf = analyze_confidence(probs)
        drift = analyze_drift(emb)
        verdict, _ = compute_verdict(conf, drift, val)
        t1 = time.perf_counter()
        times_semantic.append((t1 - t0) * 1000)

    plain_mean = float(np.mean(times_plain))
    semantic_mean = float(np.mean(times_semantic))
    overhead_pct = ((semantic_mean - plain_mean) / plain_mean) * 100

    latency_results = {
        "iterations": N_LATENCY_ITERS,
        "plain_inference": {
            "mean_ms": round(plain_mean, 4),
            "median_ms": round(float(np.median(times_plain)), 4),
            "p95_ms": round(float(np.percentile(times_plain, 95)), 4),
        },
        "semantic_inference": {
            "mean_ms": round(semantic_mean, 4),
            "median_ms": round(float(np.median(times_semantic)), 4),
            "p95_ms": round(float(np.percentile(times_semantic, 95)), 4),
        },
        "overhead_pct": round(overhead_pct, 2),
        "overhead_ms": round(semantic_mean - plain_mean, 4),
    }

    print(f"  Plain:    {plain_mean:.4f} ms")
    print(f"  Semantic: {semantic_mean:.4f} ms")
    print(f"  Overhead: +{semantic_mean - plain_mean:.4f} ms ({overhead_pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════
    final_results = {
        "experiment": "Semantic Security Engine Integration Evaluation",
        "model": f"ThreatMLP NF-Standardized ({num_features} features, {len(class_names)} classes)",
        "onnx_model": "threat_mlp_nf_fp32.onnx",
        "thresholds": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "cosine_drift_threshold": cosine_threshold,
            "mahalanobis_drift_threshold": mahal_threshold,
            "zscore_threshold": ZSCORE_THRESHOLD,
            "zero_fill_ratio": ZERO_FILL_RATIO,
            "range_tolerance_sigmas": RANGE_TOLERANCE_SIGMAS,
        },
        "scenarios": {
            "in_distribution": summary_in_dist,
            "out_of_distribution": summary_ood,
            "random_noise": summary_noise,
            "zero_filled": summary_zero,
        },
        "latency": latency_results,
    }

    output_path = EXPERIMENTS / "results" / "semantic_engine_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[SAVED] {output_path}")

    # ══════════════════════════════════════════════════════════════
    # GENERATE PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\nGenerating plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Drift Score Distribution
    ax = axes[0]
    in_scores = [r["drift"]["drift_score"] for r in results_in_dist]
    ood_scores = [r["drift"]["drift_score"] for r in results_ood]
    noise_scores = [r["drift"]["drift_score"] for r in results_noise]
    ax.hist(in_scores, bins=50, alpha=0.7, label="In-Dist (CIC)", color="#2196F3", density=True)
    ax.hist(ood_scores, bins=50, alpha=0.7, label="OOD (ToN-IoT)", color="#FF5722", density=True)
    ax.hist(noise_scores, bins=50, alpha=0.5, label="Noise", color="#9E9E9E", density=True)
    ax.axvline(x=1.0, color="red", linestyle="--", linewidth=1.5, label="Drift Threshold")
    ax.set_xlabel("Drift Score")
    ax.set_ylabel("Density")
    ax.set_title("(a) Drift Score Distribution")
    ax.legend(fontsize=8)

    # Plot 2: Confidence Score Distribution
    ax = axes[1]
    in_conf = [r["confidence"]["confidence_score"] for r in results_in_dist]
    ood_conf = [r["confidence"]["confidence_score"] for r in results_ood]
    noise_conf = [r["confidence"]["confidence_score"] for r in results_noise]
    ax.hist(in_conf, bins=50, alpha=0.7, label="In-Dist (CIC)", color="#2196F3", density=True)
    ax.hist(ood_conf, bins=50, alpha=0.7, label="OOD (ToN-IoT)", color="#FF5722", density=True)
    ax.hist(noise_conf, bins=50, alpha=0.5, label="Noise", color="#9E9E9E", density=True)
    ax.axvline(x=CONFIDENCE_THRESHOLD, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({CONFIDENCE_THRESHOLD})")
    ax.set_xlabel("Confidence Score")
    ax.set_ylabel("Density")
    ax.set_title("(b) Confidence Score Distribution")
    ax.legend(fontsize=8)

    # Plot 3: Engine Verdict Distribution
    ax = axes[2]
    verdicts_list = ["CLEAN", "SUSPICIOUS", "HIGH_RISK", "REJECTED"]
    x_pos = np.arange(len(verdicts_list))
    width = 0.2
    in_v = [summary_in_dist["engine_verdicts"].get(v, 0) / summary_in_dist["n_samples"] * 100 for v in verdicts_list]
    ood_v = [summary_ood["engine_verdicts"].get(v, 0) / summary_ood["n_samples"] * 100 for v in verdicts_list]
    noise_v = [summary_noise["engine_verdicts"].get(v, 0) / summary_noise["n_samples"] * 100 for v in verdicts_list]
    zero_v = [summary_zero["engine_verdicts"].get(v, 0) / summary_zero["n_samples"] * 100 for v in verdicts_list]
    ax.bar(x_pos - 1.5*width, in_v, width, label="In-Dist", color="#2196F3")
    ax.bar(x_pos - 0.5*width, ood_v, width, label="OOD (ToN-IoT)", color="#FF5722")
    ax.bar(x_pos + 0.5*width, noise_v, width, label="Noise", color="#9E9E9E")
    ax.bar(x_pos + 1.5*width, zero_v, width, label="Zero-filled", color="#4CAF50")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(verdicts_list, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(c) Engine Verdict Distribution")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = EXPERIMENTS / "images" / "semantic_engine_evaluation_plots.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {plot_path}")

    # ── Final Summary ──
    print(f"\n{'=' * 80}")
    print(f"  SEMANTIC ENGINE EVALUATION — FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Scenario':<40} {'Drift%':>8} {'LowConf%':>10} {'ValPass%':>10}")
    print(f"{'-'*40} {'-'*8} {'-'*10} {'-'*10}")
    for key, s in final_results["scenarios"].items():
        print(f"{s['scenario'][:40]:<40} {s['drift_detected_pct']:>7.1f}% {s['low_confidence_pct']:>9.1f}% {s['validation_passed_pct']:>9.1f}%")

    print(f"\nLatency overhead: +{latency_results['overhead_ms']:.4f} ms ({latency_results['overhead_pct']:.1f}%)")
    print(f"{'=' * 80}")
    print("\nDone! Results saved to:")
    print(f"  {output_path}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
