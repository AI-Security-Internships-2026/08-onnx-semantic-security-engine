"""
Statistical Rigor & Multi-Seed Benchmarking Runner

Executes 5 independent evaluation runs across random seeds (42, 123, 456, 789, 1024)
with disjoint calibration splits, test splits, and noise realizations to compute
mean ± std, 95% confidence intervals, and detailed latency distributions (p50, p95, p99).

Outputs:
  - experiments/results/statistical_rigor_benchmark.json
  - experiments/images/statistical_rigor_plots.png
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
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import production scoring classes (single source of truth — Issue #21)
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.semantic_analyzer import (
    ConfidenceAnalyzer, DriftDetector, InputValidator,
    SemanticSecurityEngine,
)
from scripts.config_loader import load_paper_config, get_provenance_metadata

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
EVAL_SEEDS         = PAPER_EVAL['seeds']
N_CALIB_SAMPLES    = PAPER_EVAL['n_calibration']
N_IN_DIST_SAMPLES  = PAPER_EVAL['n_in_dist']
N_OOD_SAMPLES      = PAPER_EVAL['n_ood']
N_NOISE_SAMPLES    = PAPER_EVAL['n_noise']
N_ZERO_SAMPLES     = PAPER_EVAL['n_zero']
N_LATENCY_ITERS    = PAPER_EVAL['n_latency_iters']

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


def compute_fpr95(y_true, scores):
    """Compute False Positive Rate at 95% True Positive Rate (standard OOD metric)."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr >= 0.95)[0]
    return float(fpr[idx[0]]) * 100.0 if len(idx) > 0 else 100.0


def main():
    parser = argparse.ArgumentParser(description="Statistical rigor and multi-seed evaluation benchmark.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS / "paper_results" / "json"),
        help="Directory to save statistical_rigor_benchmark.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "paper_results" / "figures"),
        help="Directory to save statistical rigor plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  STATISTICAL RIGOR & MULTI-SEED EVALUATION (5 RUNS, MEAN +/- STD, 95% CI)")
    print("=" * 85)

    # ── Load Model Artifacts ──
    print("\n[Step 1] Loading model artifacts...")
    session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    class_names = list(encoder.classes_)
    benign_idx = class_names.index("Benign") if "Benign" in class_names else 0

    ref_data = np.load(REF_EMBEDDINGS_PATH, allow_pickle=True)
    global_centroid    = ref_data["global_centroid"]
    class_centroids    = ref_data["class_centroids"]
    covariance_inverse = ref_data["covariance_inverse"]

    with open(FEATURE_STATS_PATH) as f:
        feature_stats_data = json.load(f)
    num_features = feature_stats_data["num_features"]
    feat_means = np.array([f["mean"] for f in feature_stats_data["features"]])
    feat_stds  = np.array([f["std"]  for f in feature_stats_data["features"]])

    # ── Load Datasets into Memory ──
    print("\n[Step 2] Loading full datasets for multi-seed sampling...")
    parquet_files = sorted(CIC_DIR.glob("*.parquet"))
    cic_frames = []
    loaded = 0
    for f in parquet_files:
        if loaded >= 80000:
            break
        chunk = pd.read_parquet(f)
        chunk.columns = chunk.columns.str.strip()
        if "Label" in chunk.columns:
            chunk = chunk[chunk["Label"] == "Benign"]
        cic_frames.append(chunk)
        loaded += len(chunk)

    df_cic_all = pd.concat(cic_frames, ignore_index=True)
    df_cic_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cic_all = df_cic_all.dropna()
    print(f"  Total clean CIC-IDS2018 pool available: {len(df_cic_all)} samples")

    df_ton_all = pd.read_parquet(TONIOT_PATH)
    df_ton_all.columns = df_ton_all.columns.str.strip()
    print(f"  Total ToN-IoT pool available: {len(df_ton_all)} samples")

    # ── Validation Function: use production InputValidator ──
    prod_validator = InputValidator(suffix="_nf")

    def validate_input(raw_features):
        """Wrapper around production InputValidator for quick pass/fail + alerts."""
        result = prod_validator.analyze(raw_features)
        alert_types = [a.split(":")[0] for a in result.alerts]
        return result.validation_passed, alert_types

    # ── 5-Seed Evaluation Loop ──
    print(f"\n[Step 3] Running 5-seed evaluation ({len(EVAL_SEEDS)} independent runs)...")

    seed_results = []

    for run_idx, seed in enumerate(EVAL_SEEDS, 1):
        print(f"\n  --- Run {run_idx}/5 (Seed = {seed}) ---")
        np.random.seed(seed)

        # 1. Sample Calibration and Test splits from CIC-IDS2018
        df_cic_sampled = df_cic_all.sample(n=N_SAMPLES_PER_SCENARIO * 2, random_state=seed)
        available_nf = [f for f in NF_FEATURES if f in df_cic_sampled.columns]
        if len(available_nf) < len(NF_FEATURES):
            print(f"  [WARN] Missing {len(NF_FEATURES) - len(available_nf)} NF features in CIC data")
        X_calib_raw = df_cic_sampled[available_nf].values[:N_SAMPLES_PER_SCENARIO]
        X_test_raw  = df_cic_sampled[available_nf].values[N_SAMPLES_PER_SCENARIO:]

        X_calib_scaled = scaler.transform(X_calib_raw).astype(np.float32)
        X_test_scaled  = scaler.transform(X_test_raw).astype(np.float32)

        # 2. Calibrate thresholds on D_val targeting 5% FPR
        calib_out = session.run(None, {"input": X_calib_scaled})
        calib_logits, calib_embs = calib_out[0], calib_out[1]
        calib_probs = softmax(calib_logits, axis=1)
        calib_conf = np.max(calib_probs, axis=1)

        # Compute distances using production DriftDetector (Issue #21)
        calib_drift = DriftDetector(suffix="_nf")
        calib_drift_out = calib_drift.analyze_batch(calib_embs)
        calib_cos = calib_drift_out['cosine_distances']
        calib_mahal = calib_drift_out['mahalanobis_distances']

        cos_thresh   = float(np.percentile(calib_cos, 95))
        mahal_thresh = float(np.percentile(calib_mahal, 95))
        conf_thresh  = float(np.percentile(calib_conf, 5))

        # 3. Sample ToN-IoT OOD data
        df_ton_sampled = df_ton_all.sample(n=N_SAMPLES_PER_SCENARIO, random_state=seed)
        X_ton_raw = np.zeros((N_SAMPLES_PER_SCENARIO, len(NF_FEATURES)), dtype=np.float64)
        for i, (ton_col, cic_col) in enumerate(FEATURE_MAP.items()):
            if ton_col in df_ton_sampled.columns:
                X_ton_raw[:, i] = df_ton_sampled[ton_col].values
        X_ton_raw = np.nan_to_num(X_ton_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_ton_scaled = scaler.transform(X_ton_raw).astype(np.float32)

        # 4. Generate Random Gaussian Noise & Zero-fill
        X_noise_raw = np.random.randn(N_NOISE_SAMPLES, num_features) * 1000.0
        X_noise_scaled = scaler.transform(X_noise_raw).astype(np.float32)

        X_zero_raw = np.zeros((N_ZERO_SAMPLES, num_features))
        X_zero_scaled = scaler.transform(X_zero_raw).astype(np.float32)

        # 5. Run Inference and Compute Metrics on In-Dist Test
        test_out = session.run(None, {"input": X_test_scaled})
        test_logits, test_embs = test_out[0], test_out[1]
        test_probs = softmax(test_logits, axis=1)
        test_conf = np.max(test_probs, axis=1)
        
        # Use production DriftDetector for class-conditional scoring
        drift_det = DriftDetector(
            suffix="_nf",
            cosine_threshold=cos_thresh,
            mahal_threshold=mahal_thresh,
        )
        test_drift = drift_det.analyze_batch(test_embs)
        test_cos = test_drift['cosine_distances']
        test_mahal = test_drift['mahalanobis_distances']
        test_comp = test_drift['drift_scores']

        # In-dist verdicts (using production compute_verdict)
        in_clean = 0
        in_high_risk = 0
        for i in range(len(X_test_raw)):
            val_ok, alerts = validate_input(X_test_raw[i])
            is_drift_flag = "DRIFT_DETECTED" if (test_cos[i] > cos_thresh or test_mahal[i] > mahal_thresh) else "OK"
            is_low_conf_flag = "LOW_CONFIDENCE" if test_conf[i] < conf_thresh else "OK"
            n_alerts, verdict = SemanticSecurityEngine.compute_verdict(
                confidence_flag=is_low_conf_flag,
                drift_flag=is_drift_flag,
                validation_passed=val_ok,
                validation_alerts=alerts,
            )
            if verdict == "CLEAN":
                in_clean += 1
            elif verdict in ("HIGH_RISK", "REJECTED"):
                in_high_risk += 1

        in_clean_pct = in_clean / len(X_test_raw) * 100.0
        in_fpr_pct = (len(X_test_raw) - in_clean) / len(X_test_raw) * 100.0

        # 6. Run Inference and Compute Metrics on ToN-IoT OOD
        ton_out = session.run(None, {"input": X_ton_scaled})
        ton_logits, ton_embs = ton_out[0], ton_out[1]
        ton_probs = softmax(ton_logits, axis=1)
        ton_conf = np.max(ton_probs, axis=1)
        
        # Use same production DriftDetector instance
        ton_drift = drift_det.analyze_batch(ton_embs)
        ton_cos = ton_drift['cosine_distances']
        ton_mahal = ton_drift['mahalanobis_distances']
        ton_comp = ton_drift['drift_scores']

        # OOD AUROC and FPR@95TPR metrics
        y_eval = np.concatenate([np.zeros(len(test_cos)), np.ones(len(ton_cos))])
        auroc_mahal = float(roc_auc_score(y_eval, np.concatenate([test_mahal, ton_mahal])))
        auroc_comp  = float(roc_auc_score(y_eval, np.concatenate([test_comp, ton_comp])))
        auroc_cos   = float(roc_auc_score(y_eval, np.concatenate([test_cos, ton_cos])))
        auroc_msp   = float(roc_auc_score(y_eval, np.concatenate([-test_conf, -ton_conf])))
        ap_mahal    = float(average_precision_score(y_eval, np.concatenate([test_mahal, ton_mahal])))

        fpr95_mahal = compute_fpr95(y_eval, np.concatenate([test_mahal, ton_mahal]))
        fpr95_comp  = compute_fpr95(y_eval, np.concatenate([test_comp, ton_comp]))
        fpr95_cos   = compute_fpr95(y_eval, np.concatenate([test_cos, ton_cos]))
        fpr95_msp   = compute_fpr95(y_eval, np.concatenate([-test_conf, -ton_conf]))

        # Intercept rates on threats
        ood_intercept = float(np.mean((ton_cos > cos_thresh) | (ton_mahal > mahal_thresh) | (ton_conf < conf_thresh))) * 100.0

        # 7. Evaluate Noise and Zero-Fill
        noise_out = session.run(None, {"input": X_noise_scaled})
        noise_embs = noise_out[1]
        # Use production DriftDetector for noise Mahalanobis scoring
        noise_drift = drift_det.analyze_batch(noise_embs)
        noise_mahal_scores = noise_drift['mahalanobis_distances']
        noise_intercept = float(np.mean(noise_mahal_scores > mahal_thresh)) * 100.0

        zero_rejected = float(np.mean([not validate_input(X_zero_raw[i])[0] for i in range(len(X_zero_raw))])) * 100.0

        # 8. Cross-dataset Binary F1
        ton_preds_binary = (np.argmax(ton_logits, axis=1) != benign_idx).astype(int)
        # ToN-IoT ground truth: handle Attack string column or Label (int/str)
        if "Attack" in df_ton_sampled.columns:
            ton_true_binary = (df_ton_sampled["Attack"].astype(str).str.lower() != "benign").astype(int)
        elif df_ton_sampled["Label"].dtype.kind in "biu":
            ton_true_binary = (df_ton_sampled["Label"].values != 0).astype(int)
        else:
            ton_true_binary = (df_ton_sampled["Label"].astype(str).str.lower() != "benign").astype(int)
        bin_f1 = float(f1_score(ton_true_binary, ton_preds_binary))

        run_data = {
            "seed": seed,
            "in_clean_pct": in_clean_pct,
            "in_fpr_pct": in_fpr_pct,
            "auroc_mahal": auroc_mahal,
            "auroc_comp": auroc_comp,
            "auroc_cos": auroc_cos,
            "auroc_msp": auroc_msp,
            "fpr95_mahal": fpr95_mahal,
            "fpr95_comp": fpr95_comp,
            "fpr95_cos": fpr95_cos,
            "fpr95_msp": fpr95_msp,
            "ap_mahal": ap_mahal,
            "ood_intercept_pct": ood_intercept,
            "noise_intercept_pct": noise_intercept,
            "zero_rejected_pct": zero_rejected,
            "binary_f1_toniot": bin_f1,
            "cosine_thresh": cos_thresh,
            "mahal_thresh": mahal_thresh,
        }
        seed_results.append(run_data)
        print(f"    In-Dist Clean: {in_clean_pct:.2f}% | Mahal AUROC: {auroc_mahal:.4f} | Binary F1: {bin_f1:.4f}")

    # ── Statistical Aggregation Across 5 Seeds ──
    print("\n[Step 4] Computing mean +/- std and 95% Confidence Intervals across 5 seeds...")

    def get_stats(key):
        vals = [r[key] for r in seed_results]
        mean_val = float(np.mean(vals))
        std_val  = float(np.std(vals, ddof=1))
        # 95% CI with t-distribution for n=5 (t_crit = 2.776 / sqrt(5) = 1.241)
        ci95 = float(2.776 * (std_val / np.sqrt(len(vals))))
        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci95_margin": round(ci95, 4),
            "formatted": f"{mean_val:.4f} +/- {std_val:.4f}",
            "formatted_pct": f"{mean_val:.2f}% +/- {std_val:.2f}%",
        }

    stats_summary = {
        "in_distribution_clean_pct": get_stats("in_clean_pct"),
        "in_distribution_fpr_pct":   get_stats("in_fpr_pct"),
        "mahalanobis_auroc":         get_stats("auroc_mahal"),
        "composite_engine_auroc":    get_stats("auroc_comp"),
        "cosine_distance_auroc":     get_stats("auroc_cos"),
        "msp_confidence_auroc":      get_stats("auroc_msp"),
        "mahalanobis_fpr95":         get_stats("fpr95_mahal"),
        "composite_engine_fpr95":    get_stats("fpr95_comp"),
        "cosine_distance_fpr95":     get_stats("fpr95_cos"),
        "msp_confidence_fpr95":      get_stats("fpr95_msp"),
        "mahalanobis_avg_precision": get_stats("ap_mahal"),
        "ood_toniot_intercept_pct":  get_stats("ood_intercept_pct"),
        "noise_intercept_pct":       get_stats("noise_intercept_pct"),
        "zero_fill_rejected_pct":    get_stats("zero_rejected_pct"),
        "cross_dataset_binary_f1":   get_stats("binary_f1_toniot"),
    }

    # ── Detailed Latency Benchmarking (5 Runs of 1000 Iterations with Warm-up) ──
    print("\n[Step 5] Running 5-run repeated latency benchmarking (with warm-up & percentiles)...")
    single_raw = X_test_raw[0:1]
    single_scaled = X_test_scaled[0:1]

    # Warm-up
    for _ in range(100):
        _ = session.run(None, {"input": single_scaled})

    plain_latencies = []
    semantic_latencies = []

    # Production engine instance for latency benchmarking — loaded from paper_v1.yaml
    prod_engine = SemanticSecurityEngine.from_config("configs/paper_v1.yaml")

    for run_i in range(5):
        run_plain = []
        run_sem = []
        for _ in range(N_LATENCY_ITERS):
            t0 = time.perf_counter()
            _ = session.run(None, {"input": single_scaled})
            t1 = time.perf_counter()
            run_plain.append((t1 - t0) * 1000)

            t0 = time.perf_counter()
            outs = session.run(None, {"input": single_scaled})
            probs = softmax(outs[0][0])
            emb = outs[1][0]
            _ = prod_engine.analyze(raw_features=single_raw[0], softmax_probs=probs, embedding=emb)
            t1 = time.perf_counter()
            run_sem.append((t1 - t0) * 1000)

        plain_latencies.extend(run_plain)
        semantic_latencies.extend(run_sem)

    plain_stats = {
        "mean_ms":   round(float(np.mean(plain_latencies)), 4),
        "std_ms":    round(float(np.std(plain_latencies)), 4),
        "median_p50":round(float(np.median(plain_latencies)), 4),
        "p95_ms":    round(float(np.percentile(plain_latencies, 95)), 4),
        "p99_ms":    round(float(np.percentile(plain_latencies, 99)), 4),
    }

    semantic_stats = {
        "mean_ms":   round(float(np.mean(semantic_latencies)), 4),
        "std_ms":    round(float(np.std(semantic_latencies)), 4),
        "median_p50":round(float(np.median(semantic_latencies)), 4),
        "p95_ms":    round(float(np.percentile(semantic_latencies, 95)), 4),
        "p99_ms":    round(float(np.percentile(semantic_latencies, 99)), 4),
        "overhead_mean_ms": round(float(np.mean(semantic_latencies) - np.mean(plain_latencies)), 4),
        "overhead_pct": round(float((np.mean(semantic_latencies) - np.mean(plain_latencies)) / np.mean(plain_latencies) * 100), 2)
    }

    # ── Print Final Rigorous Table ──
    print("\n" + "=" * 90)
    print("  FINAL 5-SEED STATISTICAL RIGOR BENCHMARK TABLE (MEAN +/- STD, 95% CI)")
    print("=" * 90)
    print(f"{'Evaluation Metric':<35} {'Mean +/- Std (N=5)':>22} {'95% Confidence Interval':>28}")
    print("-" * 90)
    for k, v in stats_summary.items():
        mean_v = v['mean']
        std_v  = v['std']
        margin = v['ci95_margin']
        ci_str = f"[{mean_v - margin:.4f}, {mean_v + margin:.4f}]"
        print(f"{k:<35} {v['formatted']:>22} {ci_str:>28}")
    print("=" * 90)

    print("\n" + "=" * 90)
    print("  LATENCY BENCHMARKING (5,000 RUNS WITH PERCENTILES)")
    print("=" * 90)
    print(f"  Plain ONNX:      Mean={plain_stats['mean_ms']}ms +/- {plain_stats['std_ms']}ms | p50={plain_stats['median_p50']}ms | p95={plain_stats['p95_ms']}ms | p99={plain_stats['p99_ms']}ms")
    print(f"  Semantic Engine: Mean={semantic_stats['mean_ms']}ms +/- {semantic_stats['std_ms']}ms | p50={semantic_stats['median_p50']}ms | p95={semantic_stats['p95_ms']}ms | p99={semantic_stats['p99_ms']}ms")
    print(f"  Net Overhead:    +{semantic_stats['overhead_mean_ms']}ms ({semantic_stats['overhead_pct']}%)")
    print("=" * 90)

    # ── Save Results JSON ──
    output_json_data = {
        "provenance": get_provenance_metadata(),
        "experiment": "Statistical Rigor 5-Seed Evaluation Benchmark",
        "num_seeds": len(EVAL_SEEDS),
        "seeds": EVAL_SEEDS,
        "per_seed_runs": seed_results,
        "aggregated_metrics": stats_summary,
        "latency_percentiles": {
            "plain_inference": plain_stats,
            "semantic_inference": semantic_stats,
        }
    }

    # Save to canonical output directory
    out_file = output_dir / "statistical_rigor_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(output_json_data, f, indent=2)
    print(f"\n[SAVED] {out_file}")

    # Mirror to legacy results directory
    legacy_file = EXPERIMENTS / "results" / "statistical_rigor_benchmark.json"
    if legacy_file.parent.exists() and out_file != legacy_file:
        with open(legacy_file, "w") as f:
            json.dump(output_json_data, f, indent=2)
        print(f"[MIRRORED] {legacy_file}")

    # ── Generate 4-Panel Statistical Rigor Figures ──
    print("\nGenerating 4-panel statistical rigor figures...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.rcParams.update({'font.sans-serif': 'DejaVu Sans', 'font.size': 10})

    # Plot (a): AUROC across 5 seeds (Boxplot + Jitter)
    ax = axes[0, 0]
    methods = ["Mahalanobis", "Composite Engine", "Cosine Distance", "MSP Confidence"]
    auroc_data = [
        [r["auroc_mahal"] for r in seed_results],
        [r["auroc_comp"] for r in seed_results],
        [r["auroc_cos"] for r in seed_results],
        [r["auroc_msp"] for r in seed_results],
    ]
    bp = ax.boxplot(auroc_data, labels=methods, patch_artist=True,
                    boxprops=dict(facecolor="#E3F2FD", color="#1976D2"),
                    medianprops=dict(color="#D32F2F", linewidth=2.0))
    for i, d in enumerate(auroc_data, 1):
        ax.scatter([i]*len(d), d, color="#0D47A1", zorder=5, s=60, alpha=0.8)
    ax.set_ylabel("AUROC Score")
    ax.set_title("(a) OOD AUROC Stability Across 5 Seeds (Boxplot + Runs)", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Plot (b): Clean vs FPR Distribution Across Seeds
    ax = axes[0, 1]
    seed_labels = [f"Seed {s}" for s in EVAL_SEEDS]
    clean_vals = [r["in_clean_pct"] for r in seed_results]
    fpr_vals   = [r["in_fpr_pct"] for r in seed_results]
    x_idx = np.arange(len(EVAL_SEEDS))
    width = 0.35
    ax.bar(x_idx - width/2, clean_vals, width, label="Clean Rate (%)", color="#388E3C")
    ax.bar(x_idx + width/2, fpr_vals, width, label="False Positive Rate (%)", color="#D32F2F")
    ax.axhline(y=95.0, color="#388E3C", linestyle="--", linewidth=1.5, label="Target Clean (95%)")
    ax.axhline(y=5.0, color="#D32F2F", linestyle="--", linewidth=1.5, label="Target FPR (5%)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(seed_labels, fontweight="bold")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(b) In-Distribution Calibration Stability (5 Seeds)", fontweight="bold")
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.25)

    # Plot (c): Threat Intercept Rates (Mean + Std Error Bars)
    ax = axes[1, 0]
    threat_keys = ["ood_toniot_intercept_pct", "noise_intercept_pct", "zero_fill_rejected_pct"]
    threat_names = ["ToN-IoT OOD Shift", "Gaussian Noise", "Zero-Fill Tampering"]
    means = [stats_summary[k]["mean"] for k in threat_keys]
    stds  = [stats_summary[k]["std"] for k in threat_keys]
    ax.bar(threat_names, means, yerr=stds, capsize=8, color=["#E64A19", "#757575", "#7B1FA2"], width=0.55, edgecolor="black")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m / 2, f"{m:.1f}% ± {s:.1f}%", ha="center", color="white", fontweight="bold", fontsize=10)
    ax.set_ylabel("Intercept Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(c) Multi-Threat Interception (Mean ± Std Across 5 Seeds)", fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Plot (d): Latency Distribution (CDF / Percentiles)
    ax = axes[1, 1]
    sorted_plain = np.sort(plain_latencies)
    sorted_sem = np.sort(semantic_latencies)
    p_plain = np.linspace(0, 100, len(sorted_plain))
    p_sem = np.linspace(0, 100, len(sorted_sem))
    ax.plot(sorted_plain, p_plain, color="#1976D2", linewidth=2.2, label=f"Plain ONNX (p95={plain_stats['p95_ms']:.3f}ms)")
    ax.plot(sorted_sem, p_sem, color="#7B1FA2", linewidth=2.2, label=f"Semantic Engine (p95={semantic_stats['p95_ms']:.3f}ms)")
    ax.axvline(x=1.0, color="#D32F2F", linestyle=":", linewidth=1.5, label="1.0 ms Edge Budget")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Percentage (%)")
    ax.set_title("(d) Latency Cumulative Distribution (5,000 Iterations)", fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    canonical_plot = figures_dir / "statistical_rigor_plots.png"
    plt.savefig(str(canonical_plot), dpi=300, bbox_inches="tight")
    print(f"[SAVED] {canonical_plot}")

    legacy_plot = EXPERIMENTS / "images" / "statistical_rigor_plots.png"
    if legacy_plot.parent.exists() and canonical_plot != legacy_plot:
        plt.savefig(str(legacy_plot), dpi=300, bbox_inches="tight")
        print(f"[MIRRORED] {legacy_plot}")
    plt.close()
    print("\nStatistical rigor benchmark complete!")


if __name__ == "__main__":
    main()
