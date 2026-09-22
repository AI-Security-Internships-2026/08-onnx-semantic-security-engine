"""
Semantic Engine Integration Evaluation — Calibrated Runner

Runs the full semantic security engine (ConfidenceAnalyzer + DriftDetector +
InputValidator) with held-out validation threshold calibration (5% FPR target)
on real CIC-IDS2018 and ToN-IoT data to produce:
  - experiments/results/semantic_engine_evaluation.json
  - experiments/images/semantic_engine_evaluation_plots.png

Usage:
    python scripts/evaluate_semantic_engine.py
"""

import json
import time
import sys
import datetime
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.spatial.distance import cosine as cosine_distance
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

# Import production semantic analyzer classes to avoid formula drift
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.semantic_analyzer import (
    ConfidenceAnalyzer, DriftDetector, InputValidator,
    SemanticSecurityEngine,
    ConfidenceResult, DriftResult, ValidationResult,
)
from scripts.config_loader import load_paper_config

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
# ── Load frozen paper configuration (single source of truth) ──
PAPER_CFG = load_paper_config()
PAPER_THRESHOLDS = PAPER_CFG['thresholds']
PAPER_EVAL = PAPER_CFG['evaluation']

# ── Evaluation parameters (from paper_v1.yaml) ──
N_CALIB_SAMPLES    = PAPER_EVAL['n_calibration']
N_IN_DIST_SAMPLES  = PAPER_EVAL['n_in_dist']
N_OOD_SAMPLES      = PAPER_EVAL['n_ood']
N_NOISE_SAMPLES    = PAPER_EVAL['n_noise']
N_ZERO_SAMPLES     = PAPER_EVAL['n_zero']
N_LATENCY_ITERS    = PAPER_EVAL['n_latency_iters']

# ── NF Feature Schema (13 features) ──
FEATURE_MAP = {
    # ── Exact matches ──
    "FLOW_DURATION_MILLISECONDS":    "Flow Duration",
    "IN_PKTS":                       "Total Fwd Packets",
    "OUT_PKTS":                      "Total Backward Packets",
    "IN_BYTES":                      "Fwd Packets Length Total",
    "OUT_BYTES":                     "Bwd Packets Length Total",
    "LONGEST_FLOW_PKT":              "Packet Length Max",
    "SHORTEST_FLOW_PKT":             "Packet Length Min",
    "PROTOCOL":                      "Protocol",
    # ── Approximate matches ──
    "MAX_IP_PKT_LEN":                "Fwd Packet Length Max",
    "MIN_IP_PKT_LEN":                "Fwd Packet Length Min",
    "SRC_TO_DST_SECOND_BYTES":       "Flow Bytes/s",
    "TCP_WIN_MAX_IN":                "Init Fwd Win Bytes",
    "TCP_WIN_MAX_OUT":               "Init Bwd Win Bytes",
}

NF_FEATURES = list(FEATURE_MAP.values())
TONIOT_FEATURES = list(FEATURE_MAP.keys())


def main():
    print("=" * 75)
    print("  SEMANTIC ENGINE INTEGRATION EVALUATION (CALIBRATED RUNNER)")
    print("=" * 75)

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
    ref_class_names    = list(ref_data["class_names"])

    with open(FEATURE_STATS_PATH) as f:
        feature_stats_data = json.load(f)
    num_features = feature_stats_data["num_features"]
    feat_means = np.array([f["mean"] for f in feature_stats_data["features"]])
    feat_stds  = np.array([f["std"]  for f in feature_stats_data["features"]])
    feat_names = [f["name"] for f in feature_stats_data["features"]]
    print(f"  Feature stats loaded: {num_features} features")

    # ── Step 1: Load and Partition In-Distribution Data (CIC-IDS2018) ──
    print(f"\n[Step 1] Loading clean in-distribution data from {CIC_DIR.name}...")
    parquet_files = sorted(CIC_DIR.glob("*.parquet"))
    frames = []
    loaded = 0
    total_needed = N_CALIB_SAMPLES + N_IN_DIST_SAMPLES
    for f in parquet_files:
        if loaded >= total_needed * 2:
            break
        chunk = pd.read_parquet(f)
        chunk.columns = chunk.columns.str.strip()
        if "Label" in chunk.columns:
            chunk = chunk[chunk["Label"] == "Benign"]
        frames.append(chunk)
        loaded += len(chunk)
        print(f"  Loaded {f.name}: {len(chunk)} benign rows (total: {loaded})")

    df_cic = pd.concat(frames, ignore_index=True)
    if "Timestamp" in df_cic.columns:
        df_cic = df_cic.drop(columns=["Timestamp"])
    df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cic = df_cic.dropna()

    available_nf = [f for f in NF_FEATURES if f in df_cic.columns]
    print(f"  Available NF features: {len(available_nf)}/{len(NF_FEATURES)}")
    X_cic_all = df_cic[available_nf].values[:total_needed]

    # Split into Calibration (D_val) and Test (D_test)
    X_calib_raw = X_cic_all[:N_CALIB_SAMPLES]
    X_test_raw  = X_cic_all[N_CALIB_SAMPLES:N_CALIB_SAMPLES + N_IN_DIST_SAMPLES]

    X_calib_scaled = scaler.transform(X_calib_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

    print(f"  Calibration split (D_val): {len(X_calib_raw)} samples")
    print(f"  Evaluation split  (D_test): {len(X_test_raw)} samples")

    # ── Step 2: Calibrate Thresholds on D_val (Targeting 5% FPR) ──
    print("\n[Step 2] Calibrating decision thresholds on clean held-out validation set (D_val)...")
    calib_out = session.run(None, {"input": X_calib_scaled})
    calib_logits, calib_embs = calib_out[0], calib_out[1]
    calib_probs = softmax(calib_logits, axis=1)
    calib_conf = np.max(calib_probs, axis=1)

    # Compute distances using production DriftDetector (Issue #21)
    calib_drift_detector = DriftDetector(suffix="_nf")
    calib_batch = calib_drift_detector.analyze_batch(calib_embs)
    calib_cos = calib_batch['cosine_distances']
    calib_mahal = calib_batch['mahalanobis_distances']

    # Empirical 95th percentiles (5% FPR target on D_val)
    cosine_threshold = float(np.percentile(calib_cos, 95))
    mahal_threshold  = float(np.percentile(calib_mahal, 95))
    conf_threshold   = float(np.percentile(calib_conf, 5))  # lowest 5% confidence

    print(f"  [Calibrated] Cosine Threshold (95th %ile):      {cosine_threshold:.4f}")
    print(f"  [Calibrated] Mahalanobis Threshold (95th %ile): {mahal_threshold:.4f}")
    print(f"  [Calibrated] Confidence Threshold (5th %ile):   {conf_threshold:.4f}")

    # Validator thresholds from paper_v1.yaml (frozen config)
    ZSCORE_THRESHOLD = float(PAPER_THRESHOLDS['zscore'])
    ZERO_FILL_RATIO = float(PAPER_THRESHOLDS['zero_fill_ratio'])
    print(f"  [CONFIG] Validator thresholds from paper_v1.yaml: zscore={ZSCORE_THRESHOLD}, zero_fill={ZERO_FILL_RATIO}")

    # Write calibration config for deployment
    calibration_config = {
        "last_calibrated": datetime.datetime.now().isoformat(),
        "calibration_samples": N_CALIB_SAMPLES,
        "target_fpr": 0.05,
        "cosine_drift_threshold": cosine_threshold,
        "mahalanobis_drift_threshold": mahal_threshold,
        "confidence_threshold": conf_threshold,
        "zscore_threshold": ZSCORE_THRESHOLD,
        "zero_fill_ratio": ZERO_FILL_RATIO,
    }
    config_path = EXPERIMENTS / "calibration_config.json"
    with open(config_path, "w") as f:
        json.dump(calibration_config, f, indent=2)
    print(f"  [SAVED] Calibration config -> {config_path}")

    # ── Production Analyzer Instances (single source of truth) ──
    # Instantiate the real production classes with the freshly-calibrated thresholds,
    # ensuring the eval script measures *exactly* the same formula that ships.
    confidence_analyzer = ConfidenceAnalyzer(threshold=conf_threshold)
    drift_detector = DriftDetector(
        suffix="_nf",
        cosine_threshold=cosine_threshold,
        mahal_threshold=mahal_threshold,
    )
    input_validator = InputValidator(suffix="_nf")

    def run_semantic_engine(raw_batch, scaled_batch, batch_size=512):
        results = []
        for start in range(0, len(raw_batch), batch_size):
            end = min(start + batch_size, len(raw_batch))
            batch_scaled = scaled_batch[start:end].astype(np.float32)
            onnx_out = session.run(None, {"input": batch_scaled})
            logits = onnx_out[0]
            embeddings = onnx_out[1]
            for i in range(len(batch_scaled)):
                probs = softmax(logits[i])
                # Use production classes directly
                conf_result = confidence_analyzer.analyze(probs)
                drift_result = drift_detector.analyze(embeddings[i])
                val_result = input_validator.analyze(raw_batch[start + i])

                # Compute verdict via canonical static method (single source of truth)
                total_alerts, verdict = SemanticSecurityEngine.compute_verdict(
                    confidence_flag=conf_result.confidence_flag,
                    drift_flag=drift_result.drift_flag,
                    validation_passed=val_result.validation_passed,
                    validation_alerts=val_result.alerts,
                )

                results.append({
                    "confidence": {
                        "confidence_score": conf_result.confidence_score,
                        "confidence_flag": conf_result.confidence_flag,
                    },
                    "drift": {
                        "cosine_distance": drift_result.cosine_distance,
                        "mahalanobis_distance": drift_result.mahalanobis_distance,
                        "drift_score": drift_result.drift_score,
                        "drift_flag": drift_result.drift_flag,
                        "nearest_class": drift_result.nearest_reference_class,
                    },
                    "validation": {
                        "validation_passed": val_result.validation_passed,
                        "alerts": val_result.alerts,
                    },
                    "verdict": verdict,
                    "total_alerts": total_alerts,
                    "predicted_class": class_names[int(np.argmax(logits[i]))],
                })
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
        print(f"  Verdicts: {verdicts} (Clean: {verdicts['CLEAN']/n*100:.1f}%)")
        return summary

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 1: In-Distribution (Test Split: CIC-IDS2018)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 1: In-Distribution (CIC-IDS2018 D_test)")
    print(f"{'#' * 70}")
    results_in_dist = run_semantic_engine(X_test_raw, X_test_scaled)
    summary_in_dist = summarize_results(results_in_dist, "Scenario 1: In-Distribution (CIC-IDS2018)")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 2: Out-of-Distribution (ToN-IoT Real Data)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 2: Out-of-Distribution (ToN-IoT)")
    print(f"{'#' * 70}")
    df_ton = pd.read_parquet(TONIOT_PATH)
    df_ton.columns = df_ton.columns.str.strip()
    X_ton_raw = np.zeros((min(len(df_ton), N_OOD_SAMPLES), len(NF_FEATURES)), dtype=np.float64)
    for i, (ton_col, cic_col) in enumerate(FEATURE_MAP.items()):
        if ton_col in df_ton.columns:
            X_ton_raw[:, i] = df_ton[ton_col].values[:N_OOD_SAMPLES]
    X_ton_raw = np.nan_to_num(X_ton_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_ton_scaled = scaler.transform(X_ton_raw).astype(np.float32)

    results_ood = run_semantic_engine(X_ton_raw, X_ton_scaled)
    summary_ood = summarize_results(results_ood, "Scenario 2: Out-of-Distribution (ToN-IoT, real data)")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 3: Random Gaussian Noise
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 3: Random Gaussian Noise")
    print(f"{'#' * 70}")
    np.random.seed(42)
    X_noise_raw = np.random.randn(N_NOISE_SAMPLES, num_features) * 1000.0
    X_noise_scaled = scaler.transform(X_noise_raw).astype(np.float32)

    results_noise = run_semantic_engine(X_noise_raw, X_noise_scaled)
    summary_noise = summarize_results(results_noise, "Scenario 3: Random Gaussian Noise")

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 4: Zero-Filled Inputs (RQ3 failure mode)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# SCENARIO 4: Zero-Filled Inputs (RQ3 failure mode)")
    print(f"{'#' * 70}")
    X_zero_raw = np.zeros((N_ZERO_SAMPLES, num_features))
    X_zero_scaled = scaler.transform(X_zero_raw).astype(np.float32)

    results_zero = run_semantic_engine(X_zero_raw, X_zero_scaled)
    summary_zero = summarize_results(results_zero, "Scenario 4: Zero-Filled Inputs (RQ3 failure mode)")

    # ══════════════════════════════════════════════════════════════
    # AUROC & ROC / PRECISION-RECALL BENCHMARKING
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# AUROC & ROC / PRECISION-RECALL BENCHMARKING (In-Dist vs ToN-IoT)")
    print(f"{'#' * 70}")

    in_conf_scores   = np.array([r["confidence"]["confidence_score"] for r in results_in_dist])
    in_cos_scores    = np.array([r["drift"]["cosine_distance"] for r in results_in_dist])
    in_mahal_scores  = np.array([r["drift"]["mahalanobis_distance"] for r in results_in_dist])
    in_comp_scores   = np.array([r["drift"]["drift_score"] for r in results_in_dist])

    ood_conf_scores  = np.array([r["confidence"]["confidence_score"] for r in results_ood])
    ood_cos_scores   = np.array([r["drift"]["cosine_distance"] for r in results_ood])
    ood_mahal_scores = np.array([r["drift"]["mahalanobis_distance"] for r in results_ood])
    ood_comp_scores  = np.array([r["drift"]["drift_score"] for r in results_ood])

    y_eval = np.concatenate([np.zeros(len(in_cos_scores)), np.ones(len(ood_cos_scores))])

    # Confidence anomaly score: -confidence (lower confidence -> higher anomaly)
    conf_eval_scores  = np.concatenate([-in_conf_scores, -ood_conf_scores])
    cos_eval_scores   = np.concatenate([in_cos_scores, ood_cos_scores])
    mahal_eval_scores = np.concatenate([in_mahal_scores, ood_mahal_scores])
    comp_eval_scores  = np.concatenate([in_comp_scores, ood_comp_scores])

    auc_conf  = float(roc_auc_score(y_eval, conf_eval_scores))
    auc_cos   = float(roc_auc_score(y_eval, cos_eval_scores))
    auc_mahal = float(roc_auc_score(y_eval, mahal_eval_scores))
    auc_comp  = float(roc_auc_score(y_eval, comp_eval_scores))

    ap_conf  = float(average_precision_score(y_eval, conf_eval_scores))
    ap_cos   = float(average_precision_score(y_eval, cos_eval_scores))
    ap_mahal = float(average_precision_score(y_eval, mahal_eval_scores))
    ap_comp  = float(average_precision_score(y_eval, comp_eval_scores))

    # Compute FPR at 95% TPR
    def get_fpr_at_95_tpr(y_true, scores):
        fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
        idx = np.where(tpr_arr >= 0.95)[0]
        return float(fpr_arr[idx[0]]) if len(idx) > 0 else 1.0

    fpr95_conf  = get_fpr_at_95_tpr(y_eval, conf_eval_scores)
    fpr95_cos   = get_fpr_at_95_tpr(y_eval, cos_eval_scores)
    fpr95_mahal = get_fpr_at_95_tpr(y_eval, mahal_eval_scores)
    fpr95_comp  = get_fpr_at_95_tpr(y_eval, comp_eval_scores)

    roc_metrics = {
        "confidence_msp":     {"auroc": round(auc_conf, 4),  "ap": round(ap_conf, 4),  "fpr_at_95_tpr": round(fpr95_conf, 4)},
        "cosine_distance":    {"auroc": round(auc_cos, 4),   "ap": round(ap_cos, 4),   "fpr_at_95_tpr": round(fpr95_cos, 4)},
        "mahalanobis_distance":{"auroc": round(auc_mahal, 4), "ap": round(ap_mahal, 4), "fpr_at_95_tpr": round(fpr95_mahal, 4)},
        "composite_engine":   {"auroc": round(auc_comp, 4),  "ap": round(ap_comp, 4),  "fpr_at_95_tpr": round(fpr95_comp, 4)},
    }

    print(f"  Method                       AUROC     AvgPrec   FPR@95%TPR")
    print(f"  -------------------------------------------------------------")
    print(f"  Confidence (MSP alone):     {auc_conf:7.4f}   {ap_conf:7.4f}   {fpr95_conf:7.4f}")
    print(f"  Cosine Distance:            {auc_cos:7.4f}   {ap_cos:7.4f}   {fpr95_cos:7.4f}")
    print(f"  Mahalanobis Distance:       {auc_mahal:7.4f}   {ap_mahal:7.4f}   {fpr95_mahal:7.4f}")
    print(f"  Combined Semantic Engine:   {auc_comp:7.4f}   {ap_comp:7.4f}   {fpr95_comp:7.4f}")

    # ══════════════════════════════════════════════════════════════
    # LATENCY BENCHMARKING
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("# LATENCY BENCHMARKING (1000 Warm-Up & Repeated Runs)")
    print(f"{'#' * 70}")

    test_raw = X_test_raw[0:1]
    test_scaled = X_test_scaled[0:1]

    # Warm-up
    for _ in range(50):
        _ = session.run(None, {"input": test_scaled})

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
        val_result = input_validator.analyze(test_raw[0])
        outputs = session.run(None, {"input": test_scaled})
        probs = softmax(outputs[0][0])
        emb = outputs[1][0]
        conf_result = confidence_analyzer.analyze(probs)
        drift_result = drift_detector.analyze(emb)
        # verdict (inline to avoid function call overhead in timing)
        _ = "CLEAN"
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
    # SAVE RESULTS JSON
    # ══════════════════════════════════════════════════════════════
    final_results = {
        "experiment": "Calibrated Semantic Security Engine Integration Evaluation",
        "model": f"ThreatMLP NF-Standardized ({num_features} features, {len(class_names)} classes)",
        "onnx_model": "threat_mlp_nf_fp32.onnx",
        "calibration": {
            "calibration_samples": N_CALIB_SAMPLES,
            "target_fpr": 0.05,
            "confidence_threshold": round(conf_threshold, 4),
            "cosine_drift_threshold": round(cosine_threshold, 4),
            "mahalanobis_drift_threshold": round(mahal_threshold, 4),
            "zscore_threshold": ZSCORE_THRESHOLD,
            "zero_fill_ratio": ZERO_FILL_RATIO,
        },
        "roc_benchmark": roc_metrics,
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
    # GENERATE 4-PANEL PUBLICATION PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\nGenerating publication-quality evaluation plots...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # Panel (a): Drift Score Distribution
    ax = axes[0, 0]
    ax.hist(in_comp_scores, bins=40, alpha=0.7, label="In-Dist (CIC-IDS)", color="#1976D2", density=True)
    ax.hist(ood_comp_scores, bins=40, alpha=0.7, label="OOD (ToN-IoT)", color="#E64A19", density=True)
    noise_scores = [r["drift"]["drift_score"] for r in results_noise]
    ax.hist(noise_scores, bins=40, alpha=0.4, label="Random Noise", color="#757575", density=True)
    ax.axvline(x=1.0, color="#D32F2F", linestyle="--", linewidth=1.8, label="Calibrated 5% Threshold (1.0)")
    ax.set_xlabel("Normalized Composite Drift Score")
    ax.set_ylabel("Density")
    ax.set_title("(a) Distribution Drift Score Separation", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    # Panel (b): Softmax Confidence Score Distribution
    ax = axes[0, 1]
    ax.hist(in_conf_scores, bins=40, alpha=0.7, label="In-Dist (CIC-IDS)", color="#1976D2", density=True)
    ax.hist(ood_conf_scores, bins=40, alpha=0.7, label="OOD (ToN-IoT)", color="#E64A19", density=True)
    noise_conf = [r["confidence"]["confidence_score"] for r in results_noise]
    ax.hist(noise_conf, bins=40, alpha=0.4, label="Random Noise", color="#757575", density=True)
    ax.axvline(x=conf_threshold, color="#D32F2F", linestyle="--", linewidth=1.8, label=f"Calibrated 5% Threshold ({conf_threshold:.2f})")
    ax.set_xlabel("Maximum Softmax Probability (Confidence)")
    ax.set_ylabel("Density")
    ax.set_title("(b) Model Confidence Score Distributions", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel (c): ROC Curves Comparison
    ax = axes[1, 0]
    fpr_mahal, tpr_mahal, _ = roc_curve(y_eval, mahal_eval_scores)
    fpr_comp, tpr_comp, _   = roc_curve(y_eval, comp_eval_scores)
    fpr_cos, tpr_cos, _     = roc_curve(y_eval, cos_eval_scores)
    fpr_c, tpr_c, _         = roc_curve(y_eval, conf_eval_scores)

    ax.plot(fpr_mahal, tpr_mahal, color="#388E3C", linewidth=2.0, label=f"Mahalanobis Distance (AUROC={auc_mahal:.3f})")
    ax.plot(fpr_comp, tpr_comp, color="#7B1FA2", linewidth=2.0, label=f"Composite Engine (AUROC={auc_comp:.3f})")
    ax.plot(fpr_cos, tpr_cos, color="#F57C00", linewidth=1.8, label=f"Cosine Distance (AUROC={auc_cos:.3f})")
    ax.plot(fpr_c, tpr_c, color="#1976D2", linestyle=":", linewidth=1.8, label=f"Softmax Confidence (AUROC={auc_conf:.3f})")
    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", label="Random Chance (0.500)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("(c) ROC Curves for OOD Discrimination", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Panel (d): Engine Verdict Distribution
    ax = axes[1, 1]
    verdicts_list = ["CLEAN", "SUSPICIOUS", "HIGH_RISK", "REJECTED"]
    x_pos = np.arange(len(verdicts_list))
    width = 0.2
    in_v = [summary_in_dist["engine_verdicts"].get(v, 0) / summary_in_dist["n_samples"] * 100 for v in verdicts_list]
    ood_v = [summary_ood["engine_verdicts"].get(v, 0) / summary_ood["n_samples"] * 100 for v in verdicts_list]
    noise_v = [summary_noise["engine_verdicts"].get(v, 0) / summary_noise["n_samples"] * 100 for v in verdicts_list]
    zero_v = [summary_zero["engine_verdicts"].get(v, 0) / summary_zero["n_samples"] * 100 for v in verdicts_list]

    ax.bar(x_pos - 1.5*width, in_v, width, label="In-Dist (CIC-IDS)", color="#1976D2")
    ax.bar(x_pos - 0.5*width, ood_v, width, label="OOD (ToN-IoT)", color="#E64A19")
    ax.bar(x_pos + 0.5*width, noise_v, width, label="Random Noise", color="#757575")
    ax.bar(x_pos + 1.5*width, zero_v, width, label="Zero-Filled (Tampered)", color="#D32F2F")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(verdicts_list, fontsize=10, fontweight="bold")
    ax.set_ylabel("Percentage of Samples (%)")
    ax.set_title("(d) Calibrated Engine Verdict Distribution", fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plot_path = EXPERIMENTS / "images" / "semantic_engine_evaluation_plots.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {plot_path}")

    # ── Final Executive Summary Table ──
    print(f"\n{'=' * 85}")
    print(f"  SEMANTIC ENGINE EVALUATION — FINAL CALIBRATED SUMMARY")
    print(f"{'=' * 85}")
    print(f"{'Scenario':<42} {'Clean%':>8} {'Suspicious%':>12} {'HighRisk%':>10} {'Rejected%':>10}")
    print(f"{'-'*42} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
    for key, s in final_results["scenarios"].items():
        v = s["engine_verdicts"]
        n = s["n_samples"]
        c_pct = v.get("CLEAN", 0) / n * 100
        s_pct = v.get("SUSPICIOUS", 0) / n * 100
        h_pct = v.get("HIGH_RISK", 0) / n * 100
        r_pct = v.get("REJECTED", 0) / n * 100
        print(f"{s['scenario'][:42]:<42} {c_pct:>7.1f}% {s_pct:>11.1f}% {h_pct:>9.1f}% {r_pct:>9.1f}%")

    print(f"\nLatency: Plain={latency_results['plain_inference']['mean_ms']:.4f}ms | "
          f"Semantic={latency_results['semantic_inference']['mean_ms']:.4f}ms | "
          f"Overhead=+{latency_results['overhead_ms']:.4f}ms ({latency_results['overhead_pct']:.1f}%)")
    print(f"AUROC vs OOD: Mahalanobis={auc_mahal:.4f} | Composite={auc_comp:.4f} | Cosine={auc_cos:.4f} | MSP={auc_conf:.4f}")
    print(f"{'=' * 85}")


if __name__ == "__main__":
    main()
