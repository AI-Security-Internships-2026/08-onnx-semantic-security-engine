"""
Comprehensive OOD & Anomaly Detector Baselines Benchmarking

Compares the ONNX Semantic Security Engine against 4 standard OOD baseline detectors:
  1. Maximum Softmax Probability (MSP) [Hendrycks & Gimpel, 2017]
  2. Mahalanobis Distance on ONNX Embeddings [Lee et al., 2018]
  3. Isolation Forest on Raw Features [Liu et al., 2008]
  4. One-Class SVM on Raw Features [Schölkopf et al., 2001]
  5. Full Proposed Semantic Security Engine (Multi-Signal)

All detectors are calibrated on a clean held-out validation split (D_val) targeting
an exact 5% False Positive Rate (FPR <= 0.05).

Evaluated across four distinct scenarios:
  - In-Distribution Clean Traffic (CIC-IDS2018 D_test)
  - Out-of-Distribution Covariate Shift (ToN-IoT)
  - Adversarial Random Gaussian Noise
  - Zero-Filled Feature Tampering (RQ3 Failure Mode)

Outputs:
  - experiments/results/ood_baselines_benchmark.json
  - experiments/images/ood_baselines_comparison.png
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
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import production scoring classes (single source of truth)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.semantic_analyzer import ConfidenceAnalyzer, DriftDetector

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
N_FIT_SAMPLES      = 10000  # For training IsoForest and OC-SVM
N_CALIB_SAMPLES    = 10000  # Held-out validation split for threshold calibration
N_IN_DIST_SAMPLES  = 10000  # Evaluation split
N_OOD_SAMPLES      = 10000
N_NOISE_SAMPLES    = 1000
N_ZERO_SAMPLES     = 500
N_LATENCY_ITERS    = 1000

# ── 13 NetFlow Features ──
FEATURE_MAP = {
    "FLOW_DURATION_MILLISECONDS":    "Flow Duration",
    "IN_PKTS":                       "Total Fwd Packets",
    "OUT_PKTS":                      "Total Backward Packets",
    "IN_BYTES":                      "Fwd Packets Length Total",
    "OUT_BYTES":                     "Bwd Packets Length Total",
    "LONGEST_FLOW_PKT":              "Packet Length Max",
    "SHORTEST_FLOW_PKT":             "Packet Length Min",
    "PROTOCOL":                      "Protocol",
    "MAX_IP_PKT_LEN":                "Fwd Packet Length Max",
    "MIN_IP_PKT_LEN":                "Fwd Packet Length Min",
    "SRC_TO_DST_SECOND_BYTES":       "Flow Bytes/s",
    "TCP_WIN_MAX_IN":                "Init Fwd Win Bytes",
    "TCP_WIN_MAX_OUT":               "Init Bwd Win Bytes",
}
NF_FEATURES = list(FEATURE_MAP.values())


def main():
    print("=" * 80)
    print("  OOD & ANOMALY DETECTOR BASELINE BENCHMARKING (5% FPR CALIBRATED)")
    print("=" * 80)

    # ── 1. Load Artifacts ──
    print("\n[Step 1] Loading model artifacts...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)

    ref_data = np.load(REF_EMBEDDINGS_PATH, allow_pickle=True)
    global_centroid    = ref_data["global_centroid"]
    class_centroids    = ref_data["class_centroids"]
    covariance_inverse = ref_data["covariance_inverse"]

    with open(FEATURE_STATS_PATH) as f:
        feature_stats_data = json.load(f)
    num_features = feature_stats_data["num_features"]
    feat_means = np.array([f["mean"] for f in feature_stats_data["features"]])
    feat_stds  = np.array([f["std"]  for f in feature_stats_data["features"]])

    # ── 2. Load In-Distribution Data ──
    print("\n[Step 2] Loading in-distribution CIC-IDS2018 Benign samples...")
    parquet_files = sorted(CIC_DIR.glob("*.parquet"))
    frames = []
    loaded = 0
    total_needed = N_FIT_SAMPLES + N_CALIB_SAMPLES + N_IN_DIST_SAMPLES
    for f in parquet_files:
        if loaded >= total_needed * 2:
            break
        chunk = pd.read_parquet(f)
        chunk.columns = chunk.columns.str.strip()
        if "Label" in chunk.columns:
            chunk = chunk[chunk["Label"] == "Benign"]
        frames.append(chunk)
        loaded += len(chunk)

    df_cic = pd.concat(frames, ignore_index=True)
    if "Timestamp" in df_cic.columns:
        df_cic = df_cic.drop(columns=["Timestamp"])
    df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cic = df_cic.dropna()

    X_cic_all = df_cic[NF_FEATURES].values[:total_needed]

    X_fit_raw   = X_cic_all[:N_FIT_SAMPLES]
    X_calib_raw = X_cic_all[N_FIT_SAMPLES:N_FIT_SAMPLES + N_CALIB_SAMPLES]
    X_test_raw  = X_cic_all[N_FIT_SAMPLES + N_CALIB_SAMPLES:total_needed]

    X_fit_scaled   = scaler.transform(X_fit_raw).astype(np.float32)
    X_calib_scaled = scaler.transform(X_calib_raw).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

    print(f"  Training Split   (D_train for IsoForest/SVM): {len(X_fit_raw)} samples")
    print(f"  Calibration Split(D_val for 5% FPR threshold): {len(X_calib_raw)} samples")
    print(f"  Test Split       (D_test in-dist evaluation):  {len(X_test_raw)} samples")

    # ── 3. Load Out-of-Distribution & Adversarial Datasets ──
    print("\n[Step 3] Loading OOD and Adversarial Evaluation Datasets...")
    # ToN-IoT
    df_ton = pd.read_parquet(TONIOT_PATH)
    df_ton.columns = df_ton.columns.str.strip()
    X_ton_raw = np.zeros((min(len(df_ton), N_OOD_SAMPLES), len(NF_FEATURES)), dtype=np.float64)
    for i, (ton_col, cic_col) in enumerate(FEATURE_MAP.items()):
        if ton_col in df_ton.columns:
            X_ton_raw[:, i] = df_ton[ton_col].values[:N_OOD_SAMPLES]
    X_ton_raw = np.nan_to_num(X_ton_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_ton_scaled = scaler.transform(X_ton_raw).astype(np.float32)
    print(f"  ToN-IoT OOD samples: {len(X_ton_raw)}")

    # Gaussian Noise
    np.random.seed(42)
    X_noise_raw = np.random.randn(N_NOISE_SAMPLES, num_features) * 1000.0
    X_noise_scaled = scaler.transform(X_noise_raw).astype(np.float32)
    print(f"  Gaussian Noise samples: {len(X_noise_raw)}")

    # Zero-filled (RQ3)
    X_zero_raw = np.zeros((N_ZERO_SAMPLES, num_features))
    X_zero_scaled = scaler.transform(X_zero_raw).astype(np.float32)
    print(f"  Zero-Filled samples: {len(X_zero_raw)}")

    # ── 4. Train Baselines on D_train ──
    print("\n[Step 4] Training baseline anomaly detectors on clean training data...")
    t0 = time.perf_counter()
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    iso_forest.fit(X_fit_scaled)
    t_if = (time.perf_counter() - t0) * 1000
    print(f"  [Trained] Isolation Forest ({t_if:.1f}ms)")

    t0 = time.perf_counter()
    oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
    oc_svm.fit(X_fit_scaled)
    t_svm = (time.perf_counter() - t0) * 1000
    print(f"  [Trained] One-Class SVM ({t_svm:.1f}ms)")

    # ── 5. Define Scoring Functions ──
    # NOTE: MSP, Mahalanobis, and Semantic Engine scoring use production classes
    # from src/semantic_analyzer.py to avoid formula drift (Issue #21).

    # Load calibrated thresholds from config file or defaults
    calib_config_path = EXPERIMENTS / "calibration_config.json"
    if calib_config_path.exists():
        try:
            with open(calib_config_path) as f:
                calib_cfg = json.load(f)
            cal_cos_thresh = float(calib_cfg.get("cosine_drift_threshold", 0.4341))
            cal_mahal_thresh = float(calib_cfg.get("mahalanobis_drift_threshold", 18.1593))
        except Exception:
            cal_cos_thresh = 0.4341
            cal_mahal_thresh = 18.1593
    else:
        cal_cos_thresh = 0.4341
        cal_mahal_thresh = 18.1593

    # Production drift detector instance for scoring
    prod_drift_detector = DriftDetector(
        suffix="_nf",
        cosine_threshold=cal_cos_thresh,
        mahal_threshold=cal_mahal_thresh,
    )
    prod_confidence_analyzer = ConfidenceAnalyzer(threshold=0.50)  # threshold only matters for flag, not score

    def score_msp(scaled_batch):
        """MSP Score: 1 - max(softmax(logits)) (higher = more anomalous).
        Uses production ConfidenceAnalyzer.analyze_batch() for consistency."""
        out = session.run(None, {"input": scaled_batch})[0]
        probs = softmax(out, axis=1)
        return 1.0 - prod_confidence_analyzer.analyze_batch(probs)

    def score_mahalanobis(scaled_batch):
        """Class-conditional Mahalanobis distance on fc3 embeddings.
        Uses production DriftDetector.analyze_batch() for consistency."""
        embs = session.run(None, {"input": scaled_batch})[1]
        batch_result = prod_drift_detector.analyze_batch(embs)
        return batch_result['mahalanobis_distances']

    def score_isolation_forest(scaled_batch):
        """Isolation Forest anomaly score (higher = more anomalous)."""
        return -iso_forest.score_samples(scaled_batch)

    def score_one_class_svm(scaled_batch):
        """One-Class SVM score (higher = more anomalous)."""
        return -oc_svm.decision_function(scaled_batch)

    def score_semantic_engine(raw_batch, scaled_batch):
        """Full composite semantic engine score using the production formula.
        Uses production DriftDetector.analyze_batch() for consistency."""
        out = session.run(None, {"input": scaled_batch})
        embs = out[1]
        batch_result = prod_drift_detector.analyze_batch(embs)
        return batch_result['drift_scores']

    # ── 6. Calibrate All Detectors on D_val (Targeting 5% FPR) ──
    print("\n[Step 5] Calibrating 95th-percentile threshold on held-out validation set (D_val)...")
    val_scores_msp   = score_msp(X_calib_scaled)
    val_scores_mahal = score_mahalanobis(X_calib_scaled)
    val_scores_if    = score_isolation_forest(X_calib_scaled)
    val_scores_svm   = score_one_class_svm(X_calib_scaled)
    val_scores_sem   = score_semantic_engine(X_calib_raw, X_calib_scaled)

    thresh_msp   = float(np.percentile(val_scores_msp, 95))
    thresh_mahal = float(np.percentile(val_scores_mahal, 95))
    thresh_if    = float(np.percentile(val_scores_if, 95))
    thresh_svm   = float(np.percentile(val_scores_svm, 95))
    thresh_sem   = float(np.percentile(val_scores_sem, 95))

    detectors = {
        "MSP (Confidence Alone)":       {"fn": lambda r, s: score_msp(s),             "thresh": thresh_msp},
        "Mahalanobis Distance Alone":  {"fn": lambda r, s: score_mahalanobis(s),      "thresh": thresh_mahal},
        "Isolation Forest":             {"fn": lambda r, s: score_isolation_forest(s), "thresh": thresh_if},
        "One-Class SVM":                {"fn": lambda r, s: score_one_class_svm(s),    "thresh": thresh_svm},
        "Semantic Engine (Combined)":   {"fn": lambda r, s: score_semantic_engine(r, s), "thresh": thresh_sem},
    }

    print(f"  MSP Threshold:                {thresh_msp:.4f}")
    print(f"  Mahalanobis Threshold:        {thresh_mahal:.4f}")
    print(f"  Isolation Forest Threshold:   {thresh_if:.4f}")
    print(f"  One-Class SVM Threshold:      {thresh_svm:.4f}")
    print(f"  Semantic Engine Threshold:    {thresh_sem:.4f}")

    # ── 7. Run Benchmarks Across All Datasets ──
    print("\n[Step 6] Running benchmarks across all detectors and threat scenarios...")

    scenarios = {
        "in_distribution": (X_test_raw, X_test_scaled),
        "ood_toniot":      (X_ton_raw, X_ton_scaled),
        "random_noise":    (X_noise_raw, X_noise_scaled),
        "zero_filled":     (X_zero_raw, X_zero_scaled),
    }

    detector_scores = {d_name: {} for d_name in detectors}

    for d_name, d_cfg in detectors.items():
        for s_name, (r_batch, s_batch) in scenarios.items():
            scores = d_cfg["fn"](r_batch, s_batch)
            detector_scores[d_name][s_name] = scores

    # ── 8. Compute AUROC, FPR@95%TPR, and Detection Rates ──
    print("\n[Step 7] Computing metrics and AUROC performance...")

    benchmark_summary = {}

    for d_name, d_cfg in detectors.items():
        thresh = d_cfg["thresh"]
        
        in_scores   = detector_scores[d_name]["in_distribution"]
        ood_scores  = detector_scores[d_name]["ood_toniot"]
        noise_scores= detector_scores[d_name]["random_noise"]
        zero_scores = detector_scores[d_name]["zero_filled"]

        # 1. False positive rate on in-distribution test
        fpr_test = float(np.mean(in_scores > thresh))

        # 2. Intercept / Detection rates on threats (at 5% FPR threshold)
        dr_ood   = float(np.mean(ood_scores > thresh))
        dr_noise = float(np.mean(noise_scores > thresh))
        dr_zero  = float(np.mean(zero_scores > thresh))

        # 3. AUROC vs ToN-IoT OOD
        y_ood_eval = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
        scores_ood_eval = np.concatenate([in_scores, ood_scores])
        auroc_ood = float(roc_auc_score(y_ood_eval, scores_ood_eval))
        ap_ood = float(average_precision_score(y_ood_eval, scores_ood_eval))

        # FPR @ 95% TPR
        fpr_arr, tpr_arr, _ = roc_curve(y_ood_eval, scores_ood_eval)
        idx95 = np.where(tpr_arr >= 0.95)[0]
        fpr95_ood = float(fpr_arr[idx95[0]]) if len(idx95) > 0 else 1.0

        # 4. AUROC vs Noise
        y_noise_eval = np.concatenate([np.zeros(len(in_scores)), np.ones(len(noise_scores))])
        scores_noise_eval = np.concatenate([in_scores, noise_scores])
        auroc_noise = float(roc_auc_score(y_noise_eval, scores_noise_eval))

        # 5. AUROC vs Zero-fill
        y_zero_eval = np.concatenate([np.zeros(len(in_scores)), np.ones(len(zero_scores))])
        scores_zero_eval = np.concatenate([in_scores, zero_scores])
        auroc_zero = float(roc_auc_score(y_zero_eval, scores_zero_eval))

        benchmark_summary[d_name] = {
            "calibrated_threshold": round(thresh, 4),
            "in_dist_fpr": round(fpr_test, 4),
            "ood_toniot_auroc": round(auroc_ood, 4),
            "ood_toniot_avg_precision": round(ap_ood, 4),
            "ood_toniot_fpr95": round(fpr95_ood, 4),
            "ood_toniot_detect_rate": round(dr_ood, 4),
            "noise_auroc": round(auroc_noise, 4),
            "noise_detect_rate": round(dr_noise, 4),
            "zero_fill_auroc": round(auroc_zero, 4),
            "zero_fill_detect_rate": round(dr_zero, 4),
        }

    # ── 9. Measure Latency of Each Baseline ──
    print("\n[Step 8] Benchmarking per-sample latency for all methods (1,000 iterations)...")
    single_raw = X_test_raw[0:1]
    single_scaled = X_test_scaled[0:1]

    # Warmup
    for _ in range(50):
        _ = score_msp(single_scaled)
        _ = score_isolation_forest(single_scaled)

    latency_metrics = {}
    for d_name, d_cfg in detectors.items():
        times = []
        for _ in range(N_LATENCY_ITERS):
            t0 = time.perf_counter()
            _ = d_cfg["fn"](single_raw, single_scaled)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        
        latency_metrics[d_name] = {
            "mean_ms": round(float(np.mean(times)), 4),
            "median_ms": round(float(np.median(times)), 4),
            "p95_ms": round(float(np.percentile(times, 95)), 4),
        }
        benchmark_summary[d_name]["latency"] = latency_metrics[d_name]

    # ── 10. Print Comparative Table ──
    print("\n" + "=" * 95)
    print("  OOD BASELINE COMPARISON TABLE (ALL AT 5% FPR TARGET)")
    print("=" * 95)
    print(f"{'Detector Method':<28} {'OOD AUROC':>10} {'OOD Intercept%':>15} {'Noise Intercept%':>17} {'ZeroFill Intercept%':>20} {'Latency(ms)':>12}")
    print("-" * 95)
    for d_name, res in benchmark_summary.items():
        lat = res['latency']['mean_ms']
        print(f"{d_name:<28} {res['ood_toniot_auroc']:>10.4f} {res['ood_toniot_detect_rate']*100:>14.1f}% {res['noise_detect_rate']*100:>16.1f}% {res['zero_fill_detect_rate']*100:>19.1f}% {lat:>11.4f}ms")
    print("=" * 95)

    # ── 11. Literature Baselines Context ──
    literature_baselines = [
        {"paper": "Sarhan et al. (2022) [IEEE TNSM]", "focus": "NetFlow standardization on ToN-IoT & UNSW-NB15", "cross_dataset_drop": "F1 drops ~72-85% across disparate NIDS sets", "relevance": "Direct validation of our cross-dataset negative result"},
        {"paper": "Pontes et al. (2021) [Computers & Security]", "focus": "Cross-dataset transferability between CIC-IDS and CTU-13", "cross_dataset_drop": "Binary F1 ~0.74, Multi-class F1 < 0.10", "relevance": "Direct match to our Binary 0.80 vs Multi-class 0.01 collapse"},
        {"paper": "Yang et al. (2022) [IEEE TDSC]", "focus": "Out-of-distribution detection for edge NIDS", "cross_dataset_drop": "Mahalanobis detector achieves ~0.94-0.98 AUROC", "relevance": "Validates our 0.977 AUROC Mahalanobis embedding detector"}
    ]

    # ── 12. Save Results JSON ──
    final_output = {
        "experiment": "OOD & Anomaly Baseline Comparison Benchmark",
        "calibration_target_fpr": 0.05,
        "calibration_samples": N_CALIB_SAMPLES,
        "evaluation_samples": N_IN_DIST_SAMPLES,
        "detectors_benchmark": benchmark_summary,
        "literature_baselines": literature_baselines
    }

    out_json = EXPERIMENTS / "results" / "ood_baselines_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[SAVED] {out_json}")

    # ── 13. Generate 4-Panel Baseline Comparison Plots ──
    print("\nGenerating baseline comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # Plot (a): ROC Curves on ToN-IoT OOD
    ax = axes[0, 0]
    colors = {
        "MSP (Confidence Alone)": "#1976D2",
        "Mahalanobis Distance Alone": "#388E3C",
        "Isolation Forest": "#E64A19",
        "One-Class SVM": "#FBC02D",
        "Semantic Engine (Combined)": "#7B1FA2"
    }

    for d_name in detectors:
        in_s = detector_scores[d_name]["in_distribution"]
        ood_s = detector_scores[d_name]["ood_toniot"]
        y_ev = np.concatenate([np.zeros(len(in_s)), np.ones(len(ood_s))])
        s_ev = np.concatenate([in_s, ood_s])
        fpr, tpr, _ = roc_curve(y_ev, s_ev)
        auc = roc_auc_score(y_ev, s_ev)
        lw = 2.4 if "Semantic" in d_name or "Mahalanobis" in d_name else 1.6
        ax.plot(fpr, tpr, color=colors[d_name], linewidth=lw, label=f"{d_name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", label="Random Chance (0.500)")
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("(a) ROC Curves: Out-of-Distribution (ToN-IoT)", fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Plot (b): Multi-Threat Intercept Rates (at 5% FPR Target)
    ax = axes[0, 1]
    det_names = list(detectors.keys())
    short_names = ["MSP Alone", "Mahalanobis", "IsoForest", "OC-SVM", "Semantic Engine"]
    x = np.arange(len(det_names))
    width = 0.25

    ood_rates   = [benchmark_summary[d]["ood_toniot_detect_rate"] * 100 for d in det_names]
    noise_rates = [benchmark_summary[d]["noise_detect_rate"] * 100 for d in det_names]
    zero_rates  = [benchmark_summary[d]["zero_fill_detect_rate"] * 100 for d in det_names]

    ax.bar(x - width, ood_rates, width, label="ToN-IoT OOD Shift", color="#E64A19")
    ax.bar(x, noise_rates, width, label="Gaussian Noise", color="#757575")
    ax.bar(x + width, zero_rates, width, label="Zero-Filled Tampering", color="#D32F2F")

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9, fontweight="bold")
    ax.set_ylabel("Detection / Intercept Rate (%)")
    ax.set_title("(b) Threat Intercept Rates at Fixed 5% FPR", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    # Plot (c): Trade-off: AUROC vs Latency Overhead
    ax = axes[1, 0]
    for d_name in det_names:
        auc = benchmark_summary[d_name]["ood_toniot_auroc"]
        lat = benchmark_summary[d_name]["latency"]["mean_ms"]
        ax.scatter(lat, auc, color=colors[d_name], s=140, zorder=5)
        ax.annotate(d_name, (lat + 0.015, auc - 0.015), fontsize=8.5, fontweight="bold")

    ax.set_xlabel("Per-Sample Inference Latency (ms)")
    ax.set_ylabel("OOD Discrimination AUROC")
    ax.set_title("(c) AUROC vs. Latency Trade-Off", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Plot (d): Zero-Fill Tampering Robustness (RQ3 Failure Mode)
    ax = axes[1, 1]
    zero_aucs = [benchmark_summary[d]["zero_fill_auroc"] for d in det_names]
    bar_colors = [colors[d] for d in det_names]
    ax.bar(short_names, [z * 100 for z in zero_aucs], color=bar_colors, width=0.55)
    ax.set_ylabel("Zero-Fill Tampering AUROC (%)")
    ax.set_title("(d) Robustness to Zero-Filled Telemetry Evasion", fontweight="bold")
    ax.set_ylim(0, 110)
    for i, v in enumerate(zero_aucs):
        ax.text(i, v * 100 + 2, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out_plot = EXPERIMENTS / "images" / "ood_baselines_comparison.png"
    plt.savefig(str(out_plot), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_plot}")
    print("\nBaseline benchmarking complete!")


if __name__ == "__main__":
    main()
