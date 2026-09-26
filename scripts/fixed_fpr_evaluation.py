"""
Issue 3: Fixed-FPR Evaluation, Failure-Mode Coverage & Ablation Study
=====================================================================

Comprehensive evaluation script producing the main experimental
contribution for the revised SEMANTICSHIELD paper.

Evaluates five detectors (MSP, Cosine, Mahalanobis, InputValidator,
Full SEMANTICSHIELD) across seven+ realistic deployment failure modes
at three fixed FPR operating points (0.1%, 1%, 5%).

Outputs:
  JSON  -> experiments/paper_results/json/
    - fixed_fpr_evaluation.json
    - failure_mode_coverage_matrix.json
    - ablation_fixed_fpr.json

  Figures -> experiments/paper_results/figures/
    - roc_curves_ood.png
    - detection_vs_fpr_budget.png
    - corruption_severity.png
    - ablation_impact.png
    - failure_mode_heatmap.png

Usage:
    python scripts/fixed_fpr_evaluation.py
    python scripts/fixed_fpr_evaluation.py --config configs/paper_v1.yaml
"""

import sys
import json
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from scipy.special import softmax
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.semantic_analyzer import (
    ConfidenceAnalyzer, DriftDetector, InputValidator,
    SemanticSecurityEngine,
)
from scripts.config_loader import load_paper_config, get_provenance_metadata

EXPERIMENTS = BASE_DIR / "experiments"
DATASETS = BASE_DIR / "datasets"

# -- NF Feature Schema (13 features) --
FEATURE_MAP = {
    "FLOW_DURATION_MILLISECONDS": "Flow Duration",
    "IN_PKTS":                    "Total Fwd Packets",
    "OUT_PKTS":                   "Total Backward Packets",
    "IN_BYTES":                   "Fwd Packets Length Total",
    "OUT_BYTES":                  "Bwd Packets Length Total",
    "LONGEST_FLOW_PKT":           "Packet Length Max",
    "SHORTEST_FLOW_PKT":          "Packet Length Min",
    "PROTOCOL":                   "Protocol",
    "MAX_IP_PKT_LEN":             "Fwd Packet Length Max",
    "MIN_IP_PKT_LEN":             "Fwd Packet Length Min",
    "SRC_TO_DST_SECOND_BYTES":    "Flow Bytes/s",
    "TCP_WIN_MAX_IN":             "Init Fwd Win Bytes",
    "TCP_WIN_MAX_OUT":            "Init Bwd Win Bytes",
}
NF_FEATURES = list(FEATURE_MAP.values())
TONIOT_FEATURES = list(FEATURE_MAP.keys())


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def safe_auroc(y_true, scores):
    """Compute AUROC, returning NaN if undefined (single class)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true, scores):
    """Compute AUPRC (average precision), returning NaN if undefined."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def fpr_at_tpr(y_true, scores, target_tpr=0.95):
    """Compute FPR at a given TPR level."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    idx = np.where(tpr_arr >= target_tpr)[0]
    return float(fpr_arr[idx[0]]) if len(idx) > 0 else 1.0


def tpr_at_threshold(scores_ood, threshold):
    """Detection rate (TPR) at a given decision threshold."""
    return float(np.mean(scores_ood > threshold))


def bootstrap_metric(y_true, scores, metric_fn, n_boot=1000, seed=42):
    """Bootstrap 95% confidence interval for a metric function."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_values = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        val = metric_fn(y_true[idx], scores[idx])
        if not np.isnan(val):
            boot_values.append(val)
    if len(boot_values) < 10:
        return float("nan"), float("nan"), float("nan")
    boot_values = np.array(boot_values)
    mean_val = float(np.mean(boot_values))
    ci_lo = float(np.percentile(boot_values, 2.5))
    ci_hi = float(np.percentile(boot_values, 97.5))
    return mean_val, ci_lo, ci_hi


def r(val, digits=4):
    """Round helper."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(val, digits)


# ===================================================================
# MAIN
# ===================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Issue 3: Fixed-FPR Evaluation, Failure-Mode Coverage & Ablation"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to paper config YAML (default: configs/paper_v1.yaml)")
    parser.add_argument("--output-dir", type=str,
                        default=str(EXPERIMENTS / "paper_results" / "json"),
                        help="JSON output directory")
    parser.add_argument("--figures-dir", type=str,
                        default=str(EXPERIMENTS / "paper_results" / "figures"),
                        help="Figures output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_paper_config(args.config)
    provenance = get_provenance_metadata(args.config)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -- Configuration --
    thresholds_cfg = cfg["thresholds"]
    eval_cfg = cfg["evaluation"]

    N_CAL = eval_cfg["n_calibration"]          # 10000
    N_TEST = eval_cfg["n_in_dist"]             # 10000
    N_OOD = eval_cfg["n_ood"]                  # 10000
    FPR_BUDGETS = eval_cfg.get("fpr_budgets", [0.001, 0.01, 0.05])
    ZERO_LEVELS = eval_cfg.get("zero_fill_levels", [0.10, 0.25, 0.50, 0.75, 1.00])
    N_BOOT = eval_cfg.get("n_bootstrap", 1000)
    N_FM = eval_cfg.get("n_failure_mode_samples", 5000)

    print("=" * 80)
    print("  ISSUE 3: FIXED-FPR EVALUATION, FAILURE-MODE COVERAGE & ABLATION")
    print("=" * 80)
    print(f"  FPR budgets:     {FPR_BUDGETS}")
    print(f"  Zero-fill levels: {ZERO_LEVELS}")
    print(f"  Bootstrap:       {N_BOOT} resamples")
    print(f"  Failure-mode N:  {N_FM} per dataset")
    print(f"  Calibration N:   {N_CAL}")
    print(f"  Test N:          {N_TEST}")
    print(f"  OOD N:           {N_OOD}")

    # ==============================================================
    # PHASE 1: LOAD ARTIFACTS & DATA
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 1: Loading model artifacts and data")
    print(f"{'#' * 80}")

    onnx_path = BASE_DIR / cfg["model"]["onnx_path"]
    scaler_path = BASE_DIR / cfg["model"]["scaler"]
    encoder_path = BASE_DIR / cfg["model"]["encoder"]
    ref_emb_path = BASE_DIR / cfg["model"]["reference_embeddings"]
    stats_path = BASE_DIR / cfg["model"]["feature_stats"]

    for p in [onnx_path, scaler_path, encoder_path, ref_emb_path, stats_path]:
        if not p.exists():
            print(f"  [FATAL] Missing: {p}")
            sys.exit(1)
        print(f"  [OK] {p.name}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output_names = [o.name for o in session.get_outputs()]
    assert "embedding" in output_names, "ONNX model must have 'embedding' output"

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    class_names = list(encoder.classes_)
    num_features = scaler.n_features_in_

    # Load feature stats for validator and corruption generation
    with open(stats_path) as f:
        feature_stats_data = json.load(f)
    feat_means = np.array([fs["mean"] for fs in feature_stats_data["features"]])
    feat_stds = np.array([fs["std"] for fs in feature_stats_data["features"]])

    # Production drift detector (for batch scoring)
    drift_detector = DriftDetector(
        suffix="_nf",
        cosine_threshold=float(thresholds_cfg["cosine_drift"]),
        mahal_threshold=float(thresholds_cfg["mahalanobis_drift"]),
    )
    input_validator = InputValidator(suffix="_nf")

    # -- Load CIC-IDS2018 benign data --
    print(f"\n  Loading CIC-IDS2018 benign data...")
    cic_dir = DATASETS / "CIC-IDS2018"
    parquet_files = sorted(cic_dir.glob("*.parquet"))
    frames = []
    loaded = 0
    total_needed = N_CAL + N_TEST + 5000  # extra buffer
    for pf in parquet_files:
        if loaded >= total_needed * 2:
            break
        chunk = pd.read_parquet(pf)
        chunk.columns = chunk.columns.str.strip()
        if "Label" in chunk.columns:
            chunk = chunk[chunk["Label"] == "Benign"]
        frames.append(chunk)
        loaded += len(chunk)
        print(f"    {pf.name}: {len(chunk)} benign rows (cumulative: {loaded})")

    df_cic = pd.concat(frames, ignore_index=True)
    if "Timestamp" in df_cic.columns:
        df_cic = df_cic.drop(columns=["Timestamp"])
    df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cic = df_cic.dropna()

    avail_nf = [f for f in NF_FEATURES if f in df_cic.columns]
    X_cic_all = df_cic[avail_nf].values[:N_CAL + N_TEST]

    X_cal_raw = X_cic_all[:N_CAL]
    X_test_raw = X_cic_all[N_CAL:N_CAL + N_TEST]

    X_cal_scaled = scaler.transform(X_cal_raw).astype(np.float32)
    X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    print(f"  D_cal: {len(X_cal_raw)}  |  D_test: {len(X_test_raw)}")

    # -- Load ToN-IoT OOD data --
    print(f"  Loading ToN-IoT OOD data...")
    ton_path = DATASETS / "ToN-IoT" / "NF-ToN-IoT-V2.parquet"
    df_ton = pd.read_parquet(ton_path)
    df_ton.columns = df_ton.columns.str.strip()
    X_ood_raw = np.zeros((min(len(df_ton), N_OOD), len(NF_FEATURES)), dtype=np.float64)
    for i, (ton_col, _) in enumerate(FEATURE_MAP.items()):
        if ton_col in df_ton.columns:
            X_ood_raw[:, i] = df_ton[ton_col].values[:N_OOD]
    X_ood_raw = np.nan_to_num(X_ood_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_ood_scaled = scaler.transform(X_ood_raw).astype(np.float32)
    print(f"  OOD (ToN-IoT): {len(X_ood_raw)}")

    # ==============================================================
    # PHASE 2: RUN ONNX INFERENCE & EXTRACT SIGNALS
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 2: ONNX inference & signal extraction")
    print(f"{'#' * 80}")

    def run_onnx_batch(scaled_batch, batch_size=512):
        """Run ONNX inference and return (softmax_probs, embeddings)."""
        all_probs, all_embs = [], []
        for start in range(0, len(scaled_batch), batch_size):
            end = min(start + batch_size, len(scaled_batch))
            out = session.run(None, {"input": scaled_batch[start:end]})
            all_probs.append(softmax(out[0], axis=1))
            all_embs.append(out[1])
        return np.concatenate(all_probs), np.concatenate(all_embs)

    def extract_scores(scaled_batch):
        """Extract all detector anomaly scores for a batch.
        Returns dict with keys: msp, cosine, mahalanobis, composite."""
        probs, embs = run_onnx_batch(scaled_batch)
        msp_scores = 1.0 - np.max(probs, axis=1)  # higher = more anomalous
        drift_batch = drift_detector.analyze_batch(embs)
        return {
            "msp": msp_scores,
            "cosine": drift_batch["cosine_distances"],
            "mahalanobis": drift_batch["mahalanobis_distances"],
            "composite": drift_batch["drift_scores"],
        }

    def extract_validator_decisions(raw_batch):
        """Run InputValidator on raw features, return binary rejection array."""
        rejections = np.zeros(len(raw_batch), dtype=bool)
        for i, row in enumerate(raw_batch):
            result = input_validator.analyze(row)
            rejections[i] = not result.validation_passed
        return rejections

    # Extract calibration scores
    print("  Scoring D_cal...")
    cal_scores = extract_scores(X_cal_scaled)

    # Extract test (in-dist) scores
    print("  Scoring D_test (in-distribution)...")
    test_scores = extract_scores(X_test_scaled)
    test_val_rejections = extract_validator_decisions(X_test_raw)

    # Extract OOD scores
    print("  Scoring OOD (ToN-IoT)...")
    ood_scores = extract_scores(X_ood_scaled)
    ood_val_rejections = extract_validator_decisions(X_ood_raw)

    # ==============================================================
    # PHASE 3: CALIBRATE THRESHOLDS AT FIXED FPR BUDGETS
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 3: Calibrating thresholds on D_cal at fixed FPR budgets")
    print(f"{'#' * 80}")

    # For each detector, compute threshold at each FPR budget
    # FPR budget X% -> threshold = (100-X)th percentile of in-dist scores
    detector_names = ["msp", "cosine", "mahalanobis", "composite"]
    calibrated_thresholds = {}  # {detector: {fpr_budget: threshold}}

    for det_name in detector_names:
        calibrated_thresholds[det_name] = {}
        for fpr in FPR_BUDGETS:
            percentile = (1.0 - fpr) * 100.0
            thresh = float(np.percentile(cal_scores[det_name], percentile))
            calibrated_thresholds[det_name][str(fpr)] = thresh
        print(f"  {det_name:15s} -> "
              f"0.1%: {calibrated_thresholds[det_name]['0.001']:.4f}  "
              f"1%: {calibrated_thresholds[det_name]['0.01']:.4f}  "
              f"5%: {calibrated_thresholds[det_name]['0.05']:.4f}")

    # Verify observed FPR on D_test (held-out, NOT calibration data)
    print("\n  Observed FPR on D_test (held-out verification):")
    observed_fpr = {}
    for det_name in detector_names:
        observed_fpr[det_name] = {}
        for fpr_str, thresh in calibrated_thresholds[det_name].items():
            obs = float(np.mean(test_scores[det_name] > thresh))
            observed_fpr[det_name][fpr_str] = obs
        print(f"  {det_name:15s} -> "
              f"0.1%: {observed_fpr[det_name]['0.001']*100:.2f}%  "
              f"1%: {observed_fpr[det_name]['0.01']*100:.2f}%  "
              f"5%: {observed_fpr[det_name]['0.05']*100:.2f}%")

    # ==============================================================
    # PHASE 4: GENERATE FAILURE-MODE DATASETS (E1-E7 + Gaussian)
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 4: Generating failure-mode datasets")
    print(f"{'#' * 80}")

    rng = np.random.RandomState(42)
    base_samples = X_test_raw[:N_FM].copy()  # base for corruptions

    failure_modes = {}

    # E1: ID test traffic (already computed as test_scores)
    failure_modes["E1_id_test"] = {
        "label": "E1: ID Test Traffic",
        "raw": X_test_raw[:N_FM],
        "is_anomaly": False,
        "is_structural": False,
    }

    # E2: Real cross-domain OOD (ToN-IoT)
    failure_modes["E2_real_ood"] = {
        "label": "E2: Cross-Domain OOD (ToN-IoT)",
        "raw": X_ood_raw[:N_FM],
        "is_anomaly": True,
        "is_structural": False,
    }

    # E3: NaN/Inf/missing dimensions
    X_nan = base_samples.copy().astype(np.float64)
    n_e3 = min(N_FM, len(X_nan))
    # Inject NaN into ~30% of samples, Inf into ~30%, truncated schema into ~40%
    n_nan = n_e3 // 3
    n_inf = n_e3 // 3
    n_trunc = n_e3 - n_nan - n_inf
    for i in range(n_nan):
        idx = rng.choice(num_features)
        X_nan[i, idx] = np.nan
    for i in range(n_nan, n_nan + n_inf):
        idx = rng.choice(num_features)
        X_nan[i, idx] = np.inf if rng.rand() > 0.5 else -np.inf
    # For truncated schema: set last few features to 0 (simulating missing dimensions)
    for i in range(n_nan + n_inf, n_e3):
        n_missing = rng.randint(1, num_features // 2 + 1)
        X_nan[i, -n_missing:] = 0.0

    failure_modes["E3_nan_inf_missing"] = {
        "label": "E3: NaN/Inf/Missing Dims",
        "raw": X_nan[:n_e3],
        "is_anomaly": True,
        "is_structural": True,
    }

    # E4: Partial zero-fill at various severities
    for zf_level in ZERO_LEVELS:
        X_zf = base_samples.copy()
        n_zf = min(N_FM, len(X_zf))
        n_zero_features = max(1, int(round(num_features * zf_level)))
        for i in range(n_zf):
            zero_indices = rng.choice(num_features, size=n_zero_features, replace=False)
            X_zf[i, zero_indices] = 0.0
        key = f"E4_zero_{int(zf_level*100)}pct"
        failure_modes[key] = {
            "label": f"E4: {int(zf_level*100)}% Zero-Fill",
            "raw": X_zf[:n_zf],
            "is_anomaly": True,
            "is_structural": zf_level >= 0.80,  # only high zero-fill triggers structural
        }

    # E5: Feature permutation (schema/semantic corruption)
    X_perm = base_samples.copy()
    n_e5 = min(N_FM, len(X_perm))
    for i in range(n_e5):
        perm = rng.permutation(num_features)
        X_perm[i] = X_perm[i, perm]
    failure_modes["E5_permutation"] = {
        "label": "E5: Feature Permutation",
        "raw": X_perm[:n_e5],
        "is_anomaly": True,
        "is_structural": False,
    }

    # E6: Unit/scale mismatch (multiply some features by 1000 or 0.001)
    X_scale = base_samples.copy()
    n_e6 = min(N_FM, len(X_scale))
    for i in range(n_e6):
        n_affected = rng.randint(1, num_features // 2 + 1)
        affected_idx = rng.choice(num_features, size=n_affected, replace=False)
        for idx in affected_idx:
            scale_factor = 1000.0 if rng.rand() > 0.5 else 0.001
            X_scale[i, idx] *= scale_factor
    failure_modes["E6_scale_mismatch"] = {
        "label": "E6: Unit/Scale Mismatch",
        "raw": X_scale[:n_e6],
        "is_anomaly": True,
        "is_structural": False,
    }

    # E7: Extreme-but-plausible values (99.99th percentile of training distribution)
    X_extreme = base_samples.copy()
    n_e7 = min(N_FM, len(X_extreme))
    extreme_upper = feat_means + 4.0 * feat_stds  # approximate 99.99th pctile
    extreme_lower = feat_means - 4.0 * feat_stds
    for i in range(n_e7):
        n_extreme = rng.randint(1, num_features // 2 + 1)
        extreme_idx = rng.choice(num_features, size=n_extreme, replace=False)
        for idx in extreme_idx:
            X_extreme[i, idx] = extreme_upper[idx] if rng.rand() > 0.5 else extreme_lower[idx]
    failure_modes["E7_extreme_plausible"] = {
        "label": "E7: Extreme-but-Plausible",
        "raw": X_extreme[:n_e7],
        "is_anomaly": True,
        "is_structural": False,
    }

    # Supplementary: Gaussian noise (not primary OOD evidence)
    X_noise = rng.randn(min(N_FM, 2000), num_features) * 1000.0
    failure_modes["E8_gaussian_noise"] = {
        "label": "E8 (Suppl.): Gaussian Noise",
        "raw": X_noise,
        "is_anomaly": True,
        "is_structural": False,
    }

    for key, fm in failure_modes.items():
        print(f"  {fm['label']:40s} -> {len(fm['raw']):6d} samples  "
              f"(anomaly={fm['is_anomaly']}, structural={fm['is_structural']})")

    # ==============================================================
    # PHASE 5: SCORE ALL FAILURE MODES
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 5: Scoring all failure modes with all detectors")
    print(f"{'#' * 80}")

    fm_scores = {}   # {fm_key: {det_name: scores_array}}
    fm_val = {}      # {fm_key: validator_rejection_array}

    for fm_key, fm_data in failure_modes.items():
        raw = fm_data["raw"]
        # Handle NaN/Inf for ONNX (replace for scaling, but validator uses raw)
        raw_clean = np.nan_to_num(raw, nan=0.0, posinf=1e10, neginf=-1e10)
        scaled = scaler.transform(raw_clean).astype(np.float32)

        scores = extract_scores(scaled)
        fm_scores[fm_key] = scores
        fm_val[fm_key] = extract_validator_decisions(raw)

        det_rate_5pct = tpr_at_threshold(
            scores["composite"], calibrated_thresholds["composite"]["0.05"]
        ) if fm_data["is_anomaly"] else None
        val_rej_rate = float(np.mean(fm_val[fm_key]))

        print(f"  {fm_data['label']:40s} -> "
              f"Validator rej: {val_rej_rate*100:.1f}%"
              + (f"  Composite TPR@5%: {det_rate_5pct*100:.1f}%" if det_rate_5pct is not None else ""))

    # ==============================================================
    # PHASE 6: COMPUTE ALL METRICS
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 6: Computing AUROC, AUPRC, TPR@FPR, FPR@95TPR")
    print(f"{'#' * 80}")

    # Use D_test scores as the in-distribution baseline
    # Limit to N_FM samples to match failure-mode sizes
    id_scores = {det: test_scores[det][:N_FM] for det in detector_names}

    # Main metrics table: per detector x per failure mode
    all_metrics = {}

    for fm_key, fm_data in failure_modes.items():
        if not fm_data["is_anomaly"]:
            continue  # skip E1 (ID test) -- it's the baseline

        fm_label = fm_data["label"]
        all_metrics[fm_key] = {"label": fm_label, "detectors": {}}

        for det_name in detector_names:
            in_s = id_scores[det_name]
            fm_s = fm_scores[fm_key][det_name]

            # Build binary labels: 0 = in-dist, 1 = anomaly
            y = np.concatenate([np.zeros(len(in_s)), np.ones(len(fm_s))])
            s = np.concatenate([in_s, fm_s])

            auroc = safe_auroc(y, s)
            auprc = safe_auprc(y, s)
            fpr95 = fpr_at_tpr(y, s, 0.95)

            # TPR at each FPR budget (using calibrated thresholds)
            tpr_at_fpr = {}
            for fpr_str, thresh in calibrated_thresholds[det_name].items():
                tpr_at_fpr[fpr_str] = tpr_at_threshold(fm_s, thresh)

            all_metrics[fm_key]["detectors"][det_name] = {
                "auroc": r(auroc),
                "auprc": r(auprc),
                "fpr_at_95tpr": r(fpr95),
                "tpr_at_fpr": {k: r(v) for k, v in tpr_at_fpr.items()},
            }

        # Validator (structural) -- report separately
        val_rej = fm_val[fm_key]
        all_metrics[fm_key]["validator_rejection_rate"] = r(float(np.mean(val_rej)))
        all_metrics[fm_key]["validator_id_rejection_rate"] = r(float(np.mean(test_val_rejections[:N_FM])))

    # Print main OOD benchmark table (E2)
    print(f"\n  {'Method':20s} {'AUROC':>8s} {'AUPRC':>8s} {'FPR@95':>8s} "
          f"{'TPR@0.1%':>9s} {'TPR@1%':>8s} {'TPR@5%':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
    e2_metrics = all_metrics.get("E2_real_ood", {}).get("detectors", {})
    for det_name in detector_names:
        m = e2_metrics.get(det_name, {})
        tpr = m.get("tpr_at_fpr", {})
        auroc_v = m.get('auroc', 0) or 0
        auprc_v = m.get('auprc', 0) or 0
        fpr95_v = m.get('fpr_at_95tpr', 0) or 0
        tpr001 = tpr.get('0.001', 0) or 0
        tpr01 = tpr.get('0.01', 0) or 0
        tpr05 = tpr.get('0.05', 0) or 0
        print(f"  {det_name:20s} {auroc_v:>8.4f} {auprc_v:>8.4f} "
              f"{fpr95_v:>8.4f} "
              f"{tpr001:>9.4f} {tpr01:>8.4f} {tpr05:>8.4f}")
    val_rej_e2 = all_metrics.get("E2_real_ood", {}).get("validator_rejection_rate", 0)
    print(f"  {'validator':20s} {'(struct)':>8s} {'(struct)':>8s} {'N/A':>8s} "
          f"{'N/A':>9s} {'N/A':>8s}   rej={val_rej_e2}")

    # ==============================================================
    # PHASE 7: BOOTSTRAP CONFIDENCE INTERVALS (headline metrics on E2)
    # ==============================================================
    print(f"\n{'#' * 80}")
    print(f"# PHASE 7: Bootstrap 95% CIs on E2 (n_boot={N_BOOT})")
    print(f"{'#' * 80}")

    bootstrap_results = {}
    e2_fm_scores = fm_scores["E2_real_ood"]
    for det_name in detector_names:
        in_s = id_scores[det_name]
        ood_s = e2_fm_scores[det_name]
        y = np.concatenate([np.zeros(len(in_s)), np.ones(len(ood_s))])
        s = np.concatenate([in_s, ood_s])

        auroc_mean, auroc_lo, auroc_hi = bootstrap_metric(y, s, safe_auroc, N_BOOT)
        auprc_mean, auprc_lo, auprc_hi = bootstrap_metric(y, s, safe_auprc, N_BOOT)

        bootstrap_results[det_name] = {
            "auroc": {"mean": r(auroc_mean), "ci95_lo": r(auroc_lo), "ci95_hi": r(auroc_hi)},
            "auprc": {"mean": r(auprc_mean), "ci95_lo": r(auprc_lo), "ci95_hi": r(auprc_hi)},
        }
        print(f"  {det_name:15s}  AUROC={auroc_mean:.4f} [{auroc_lo:.4f}, {auroc_hi:.4f}]  "
              f"AUPRC={auprc_mean:.4f} [{auprc_lo:.4f}, {auprc_hi:.4f}]")

    # ==============================================================
    # PHASE 8: ABLATION STUDY
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 8: Component ablation (with AUROC-level metrics)")
    print(f"{'#' * 80}")

    # For ablation, we define composite scoring functions that mask components
    # The composite drift score = min(max(cos/tau_cos, mahal/tau_mahal), 2.0)
    # For ablation, we remove one distance from the max formula

    def compute_ablation_score(cos, mahal, msp_scores, use_msp, use_cos, use_mahal):
        """Compute an ablated composite score.
        
        The full composite score uses max(cos/tau, mahal/tau).
        When a component is removed, it's excluded from the max.
        MSP contributes via a separate 1-confidence signal.
        """
        cos_thresh = calibrated_thresholds["cosine"]["0.05"]
        mahal_thresh = calibrated_thresholds["mahalanobis"]["0.05"]

        components = []
        if use_cos:
            components.append(cos / max(cos_thresh, 1e-8))
        if use_mahal:
            components.append(mahal / max(mahal_thresh, 1e-8))

        if len(components) > 0:
            drift_part = np.maximum.reduce(components)
        else:
            drift_part = np.zeros_like(cos)

        if use_msp:
            # Combine MSP with drift via max (both are anomaly scores)
            msp_thresh = calibrated_thresholds["msp"]["0.05"]
            msp_norm = msp_scores / max(msp_thresh, 1e-8)
            combined = np.maximum(drift_part, msp_norm)
        else:
            combined = drift_part

        return np.minimum(combined, 2.0)

    ablation_configs = {
        "full": {"label": "Full SEMANTICSHIELD", "use_msp": True, "use_cos": True, "use_mahal": True, "use_val": True},
        "no_msp": {"label": "-MSP", "use_msp": False, "use_cos": True, "use_mahal": True, "use_val": True},
        "no_cosine": {"label": "-Cosine", "use_msp": True, "use_cos": False, "use_mahal": True, "use_val": True},
        "no_mahal": {"label": "-Mahalanobis", "use_msp": True, "use_cos": True, "use_mahal": False, "use_val": True},
        "no_validator": {"label": "-Validator", "use_msp": True, "use_cos": True, "use_mahal": True, "use_val": False},
        "msp_only": {"label": "MSP Only", "use_msp": True, "use_cos": False, "use_mahal": False, "use_val": False},
        "cosine_only": {"label": "Cosine Only", "use_msp": False, "use_cos": True, "use_mahal": False, "use_val": False},
        "mahal_only": {"label": "Mahalanobis Only", "use_msp": False, "use_cos": False, "use_mahal": True, "use_val": False},
        "validator_only": {"label": "Validator Only", "use_msp": False, "use_cos": False, "use_mahal": False, "use_val": True},
    }

    # Compute ablation scores for E2 (primary OOD)
    ablation_results = {}

    for abl_key, abl_cfg in ablation_configs.items():
        # In-dist scores
        abl_in_scores = compute_ablation_score(
            id_scores["cosine"], id_scores["mahalanobis"], id_scores["msp"],
            abl_cfg["use_msp"], abl_cfg["use_cos"], abl_cfg["use_mahal"]
        )
        # OOD scores
        abl_ood_scores = compute_ablation_score(
            e2_fm_scores["cosine"], e2_fm_scores["mahalanobis"], e2_fm_scores["msp"],
            abl_cfg["use_msp"], abl_cfg["use_cos"], abl_cfg["use_mahal"]
        )

        y = np.concatenate([np.zeros(len(abl_in_scores)), np.ones(len(abl_ood_scores))])
        s = np.concatenate([abl_in_scores, abl_ood_scores])

        auroc = safe_auroc(y, s)
        auprc = safe_auprc(y, s)
        fpr95 = fpr_at_tpr(y, s, 0.95)

        # Validator contribution (structural rejection rate on E2)
        if abl_cfg["use_val"]:
            val_rej_rate = float(np.mean(fm_val["E2_real_ood"]))
        else:
            val_rej_rate = 0.0

        # Detection rates across all failure modes (at 5% FPR)
        abl_thresh = float(np.percentile(abl_in_scores, 95))
        fm_detection = {}
        for fm_key, fm_data in failure_modes.items():
            if not fm_data["is_anomaly"]:
                continue
            abl_fm_scores = compute_ablation_score(
                fm_scores[fm_key]["cosine"], fm_scores[fm_key]["mahalanobis"],
                fm_scores[fm_key]["msp"],
                abl_cfg["use_msp"], abl_cfg["use_cos"], abl_cfg["use_mahal"]
            )
            stat_det = float(np.mean(abl_fm_scores > abl_thresh))
            # Add validator rejections if enabled
            if abl_cfg["use_val"]:
                combined_det = float(np.mean(
                    (abl_fm_scores > abl_thresh) | fm_val[fm_key]
                ))
            else:
                combined_det = stat_det
            fm_detection[fm_key] = r(combined_det)

        ablation_results[abl_key] = {
            "label": abl_cfg["label"],
            "components": {
                "msp": abl_cfg["use_msp"],
                "cosine": abl_cfg["use_cos"],
                "mahalanobis": abl_cfg["use_mahal"],
                "validator": abl_cfg["use_val"],
            },
            "e2_auroc": r(auroc),
            "e2_auprc": r(auprc),
            "e2_fpr_at_95tpr": r(fpr95),
            "e2_validator_rejection": r(val_rej_rate),
            "detection_rates_at_5pct_fpr": fm_detection,
        }

        print(f"  {abl_cfg['label']:25s}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"FPR@95={fpr95:.4f}  ValRej={val_rej_rate:.3f}")

    # ==============================================================
    # PHASE 9: FAILURE-MODE COVERAGE MATRIX
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 9: Failure-mode coverage matrix")
    print(f"{'#' * 80}")

    # Rows = failure modes, Columns = detectors + validator
    # Cell = detection rate at 1% FPR (or structural rejection rate for validator)
    coverage_matrix = {}

    for fm_key, fm_data in failure_modes.items():
        if not fm_data["is_anomaly"]:
            continue
        row = {"label": fm_data["label"]}
        for det_name in detector_names:
            thresh_1pct = calibrated_thresholds[det_name]["0.01"]
            det_rate = tpr_at_threshold(fm_scores[fm_key][det_name], thresh_1pct)
            row[det_name] = r(det_rate)
        # Validator
        row["validator"] = r(float(np.mean(fm_val[fm_key])))
        # Full system (composite @ 1% FPR OR validator)
        thresh_comp_1pct = calibrated_thresholds["composite"]["0.01"]
        stat_det = fm_scores[fm_key]["composite"] > thresh_comp_1pct
        full_det = float(np.mean(stat_det | fm_val[fm_key]))
        row["full_system"] = r(full_det)
        coverage_matrix[fm_key] = row

    # Print coverage matrix
    print(f"\n  {'Failure Mode':35s} {'MSP':>7s} {'Cosine':>7s} {'Mahal':>7s} {'Valid.':>7s} {'Full':>7s}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for fm_key, row in coverage_matrix.items():
        print(f"  {row['label']:35s} "
              f"{row.get('msp', 0)*100:6.1f}% "
              f"{row.get('cosine', 0)*100:6.1f}% "
              f"{row.get('mahalanobis', 0)*100:6.1f}% "
              f"{row.get('validator', 0)*100:6.1f}% "
              f"{row.get('full_system', 0)*100:6.1f}%")

    # ==============================================================
    # PHASE 10: SAVE JSON OUTPUTS
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 10: Saving JSON outputs")
    print(f"{'#' * 80}")

    # 1. Main evaluation results
    main_output = {
        "provenance": provenance,
        "experiment": "Issue 3: Fixed-FPR Evaluation, Failure-Mode Coverage & Ablation",
        "config": {
            "fpr_budgets": FPR_BUDGETS,
            "zero_fill_levels": ZERO_LEVELS,
            "n_bootstrap": N_BOOT,
            "n_calibration": N_CAL,
            "n_test": N_TEST,
            "n_ood": N_OOD,
            "n_failure_mode": N_FM,
        },
        "calibrated_thresholds": calibrated_thresholds,
        "observed_id_fpr": observed_fpr,
        "metrics_per_failure_mode": all_metrics,
        "bootstrap_ci_e2": bootstrap_results,
        "note": "Structural (validator) rejections are deterministic and reported separately from statistical AUROC. "
                "Gaussian noise is supplementary, not primary OOD evidence.",
    }
    main_path = output_dir / "fixed_fpr_evaluation.json"
    with open(main_path, "w", encoding="utf-8") as f:
        json.dump(main_output, f, indent=2, default=str)
    print(f"  [SAVED] {main_path}")

    # 2. Coverage matrix
    coverage_output = {
        "provenance": provenance,
        "experiment": "Failure-Mode Coverage Matrix",
        "operating_point": "1% FPR (calibrated on D_cal)",
        "matrix": coverage_matrix,
        "note": "Validator column shows structural rejection rate (deterministic). "
                "Other columns show detection rate at 1% FPR threshold. "
                "Full System combines statistical detection OR validator rejection.",
    }
    coverage_path = output_dir / "failure_mode_coverage_matrix.json"
    with open(coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage_output, f, indent=2, default=str)
    print(f"  [SAVED] {coverage_path}")

    # 3. Ablation results
    ablation_output = {
        "provenance": provenance,
        "experiment": "Component Ablation with AUROC-level Metrics",
        "primary_ood_dataset": "E2: Cross-Domain OOD (ToN-IoT)",
        "ablation_results": ablation_results,
        "note": "AUROC/AUPRC computed on E2 (ToN-IoT). Detection rates at 5% FPR across all failure modes.",
    }
    ablation_path = output_dir / "ablation_fixed_fpr.json"
    with open(ablation_path, "w", encoding="utf-8") as f:
        json.dump(ablation_output, f, indent=2, default=str)
    print(f"  [SAVED] {ablation_path}")

    # ==============================================================
    # PHASE 11: GENERATE FIGURES
    # ==============================================================
    print(f"\n{'#' * 80}")
    print("# PHASE 11: Generating publication figures")
    print(f"{'#' * 80}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    COLORS = {
        "msp": "#1976D2",
        "cosine": "#F57C00",
        "mahalanobis": "#388E3C",
        "composite": "#7B1FA2",
        "validator": "#D32F2F",
        "full_system": "#00897B",
    }
    DET_LABELS = {
        "msp": "MSP (1-Confidence)",
        "cosine": "Cosine Distance",
        "mahalanobis": "Mahalanobis Distance",
        "composite": "Composite Drift Score",
    }

    # -- Figure 1: ROC Curves on E2 (primary real OOD) --
    fig, ax = plt.subplots(figsize=(8, 7))
    for det_name in detector_names:
        in_s = id_scores[det_name]
        ood_s = e2_fm_scores[det_name]
        y = np.concatenate([np.zeros(len(in_s)), np.ones(len(ood_s))])
        s = np.concatenate([in_s, ood_s])
        fpr_arr, tpr_arr, _ = roc_curve(y, s)
        auc_val = safe_auroc(y, s)
        ci = bootstrap_results.get(det_name, {}).get("auroc", {})
        ci_str = f" [{ci.get('ci95_lo', '')}, {ci.get('ci95_hi', '')}]" if ci else ""
        lw = 2.5 if det_name in ("mahalanobis", "composite") else 1.8
        ax.plot(fpr_arr, tpr_arr, color=COLORS[det_name], linewidth=lw,
                label=f"{DET_LABELS[det_name]} (AUROC={auc_val:.3f}{ci_str})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--", label="Random (0.500)")
    # Mark FPR budget lines
    for fpr_budget in FPR_BUDGETS:
        ax.axvline(x=fpr_budget, color="#E0E0E0", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("ROC Curves -- Real OOD (ToN-IoT)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    roc_path = figures_dir / "roc_curves_ood.png"
    plt.savefig(str(roc_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {roc_path}")

    # -- Figure 2: Detection vs FPR Budget --
    fig, ax = plt.subplots(figsize=(9, 6))
    fpr_grid = np.logspace(-4, -0.5, 200)
    for det_name in detector_names:
        in_s = id_scores[det_name]
        ood_s = e2_fm_scores[det_name]
        y = np.concatenate([np.zeros(len(in_s)), np.ones(len(ood_s))])
        s = np.concatenate([in_s, ood_s])
        fpr_arr, tpr_arr, _ = roc_curve(y, s)
        # Interpolate TPR at each FPR grid point
        tpr_interp = np.interp(fpr_grid, fpr_arr, tpr_arr)
        ax.plot(fpr_grid * 100, tpr_interp * 100, color=COLORS[det_name],
                linewidth=2.0, label=DET_LABELS[det_name])

    for fpr_budget in FPR_BUDGETS:
        ax.axvline(x=fpr_budget * 100, color="#BDBDBD", linestyle="--", linewidth=1.2)
        ax.annotate(f"{fpr_budget*100:.1f}%", xy=(fpr_budget*100, 5),
                    fontsize=8, color="#757575", ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Allowed ID False-Positive Rate (%)", fontsize=12)
    ax.set_ylabel("OOD Detection Rate (%)", fontsize=12)
    ax.set_title("Detection Rate vs FPR Budget -- ToN-IoT OOD", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2, which="both")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2g}%"))
    plt.tight_layout()
    det_fpr_path = figures_dir / "detection_vs_fpr_budget.png"
    plt.savefig(str(det_fpr_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {det_fpr_path}")

    # -- Figure 3: Corruption Severity (zero-fill) --
    fig, ax = plt.subplots(figsize=(9, 6))
    zero_fm_keys = [f"E4_zero_{int(zl*100)}pct" for zl in ZERO_LEVELS]
    x_levels = [int(zl * 100) for zl in ZERO_LEVELS]

    for det_name in detector_names:
        det_rates = []
        for zfk in zero_fm_keys:
            thresh_5 = calibrated_thresholds[det_name]["0.05"]
            det_rates.append(tpr_at_threshold(fm_scores[zfk][det_name], thresh_5) * 100)
        ax.plot(x_levels, det_rates, marker="o", linewidth=2.0,
                color=COLORS[det_name], label=DET_LABELS[det_name])

    # Validator
    val_rates = [float(np.mean(fm_val[zfk])) * 100 for zfk in zero_fm_keys]
    ax.plot(x_levels, val_rates, marker="s", linewidth=2.0, linestyle="--",
            color=COLORS["validator"], label="Input Validator (structural)")

    ax.set_xlabel("Zero-Fill Corruption Level (%)", fontsize=12)
    ax.set_ylabel("Detection / Rejection Rate (%)", fontsize=12)
    ax.set_title("Detection Rate vs Corruption Severity", fontsize=13, fontweight="bold")
    ax.set_xticks(x_levels)
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    severity_path = figures_dir / "corruption_severity.png"
    plt.savefig(str(severity_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {severity_path}")

    # -- Figure 4: Ablation Impact --
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel (a): AUROC on E2 per ablation
    abl_labels = [ablation_results[k]["label"] for k in ablation_configs]
    abl_aurocs = [ablation_results[k]["e2_auroc"] or 0 for k in ablation_configs]
    full_auroc = abl_aurocs[0]
    abl_deltas = [a - full_auroc for a in abl_aurocs]

    bar_colors = ["#00897B" if k == "full" else "#E64A19" if d < -0.01
                  else "#FFA726" if d < 0 else "#66BB6A"
                  for k, d in zip(ablation_configs, abl_deltas)]

    ax = axes[0]
    x = np.arange(len(abl_labels))
    ax.barh(x, abl_aurocs, color=bar_colors, height=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(abl_labels, fontsize=10)
    ax.set_xlabel("AUROC on E2 (ToN-IoT OOD)", fontsize=11)
    ax.set_title("(a) Ablation: AUROC per Configuration", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for i, v in enumerate(abl_aurocs):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.2)
    ax.invert_yaxis()

    # Panel (b): Detection rate heatmap across failure modes
    ax = axes[1]
    # Pick a subset of ablation configs for readability
    abl_keys_for_heatmap = ["full", "no_msp", "no_cosine", "no_mahal", "no_validator"]
    fm_keys_for_heatmap = [k for k in failure_modes if failure_modes[k]["is_anomaly"]]

    heatmap_data = []
    for abl_key in abl_keys_for_heatmap:
        row = []
        det_rates = ablation_results[abl_key]["detection_rates_at_5pct_fpr"]
        for fm_key in fm_keys_for_heatmap:
            row.append((det_rates.get(fm_key, 0) or 0) * 100)
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(fm_keys_for_heatmap)))
    ax.set_xticklabels(
        [failure_modes[k]["label"].split(":")[0] for k in fm_keys_for_heatmap],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_yticks(range(len(abl_keys_for_heatmap)))
    ax.set_yticklabels(
        [ablation_results[k]["label"] for k in abl_keys_for_heatmap], fontsize=10,
    )
    ax.set_title("(b) Detection Rate at 5% FPR by Ablation x Failure Mode",
                  fontsize=12, fontweight="bold")
    for i in range(len(abl_keys_for_heatmap)):
        for j in range(len(fm_keys_for_heatmap)):
            ax.text(j, i, f"{heatmap_data[i, j]:.0f}%", ha="center", va="center",
                    fontsize=8, color="white" if heatmap_data[i, j] > 60 else "black")
    plt.colorbar(im, ax=ax, label="Detection Rate (%)", shrink=0.8)

    plt.tight_layout()
    ablation_fig_path = figures_dir / "ablation_impact.png"
    plt.savefig(str(ablation_fig_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {ablation_fig_path}")

    # -- Figure 5: Failure-Mode Coverage Heatmap --
    fig, ax = plt.subplots(figsize=(10, 7))
    col_dets = ["msp", "cosine", "mahalanobis", "validator", "full_system"]
    col_labels = ["MSP", "Cosine", "Mahalanobis", "Validator\n(structural)", "Full System"]
    fm_keys_cov = list(coverage_matrix.keys())
    fm_labels_cov = [coverage_matrix[k]["label"] for k in fm_keys_cov]

    cov_data = []
    for fm_key in fm_keys_cov:
        row = [(coverage_matrix[fm_key].get(c, 0) or 0) * 100 for c in col_dets]
        cov_data.append(row)
    cov_data = np.array(cov_data)

    im = ax.imshow(cov_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(fm_labels_cov)))
    ax.set_yticklabels(fm_labels_cov, fontsize=10)
    ax.set_title("Failure-Mode Coverage Matrix (Detection Rate at 1% FPR)",
                  fontsize=13, fontweight="bold", pad=15)

    for i in range(len(fm_keys_cov)):
        for j in range(len(col_dets)):
            val = cov_data[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if val > 60 else "black")

    plt.colorbar(im, ax=ax, label="Detection Rate (%)", shrink=0.8)
    plt.tight_layout()
    heatmap_path = figures_dir / "failure_mode_heatmap.png"
    plt.savefig(str(heatmap_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] {heatmap_path}")

    # ==============================================================
    # FINAL SUMMARY
    # ==============================================================
    print(f"\n{'=' * 80}")
    print("  ISSUE 3 EVALUATION COMPLETE")
    print(f"{'=' * 80}")

    print(f"\n  JSON outputs:")
    print(f"    {main_path}")
    print(f"    {coverage_path}")
    print(f"    {ablation_path}")
    print(f"\n  Figures:")
    print(f"    {roc_path}")
    print(f"    {det_fpr_path}")
    print(f"    {severity_path}")
    print(f"    {ablation_fig_path}")
    print(f"    {heatmap_path}")

    # Summary table
    print(f"\n  Main OOD Benchmark (E2: ToN-IoT):")
    print(f"  {'Method':20s} {'AUROC':>8s} {'AUPRC':>8s} {'FPR@95':>8s} {'TPR@0.1%':>9s} {'TPR@1%':>8s} {'TPR@5%':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
    e2_m = all_metrics.get("E2_real_ood", {}).get("detectors", {})
    for det_name in detector_names:
        m = e2_m.get(det_name, {})
        tpr = m.get("tpr_at_fpr", {})
        ci = bootstrap_results.get(det_name, {}).get("auroc", {})
        auroc_str = f"{m.get('auroc', 0):.4f}"
        if ci.get("ci95_lo") is not None:
            auroc_str += f" [{ci['ci95_lo']:.3f},{ci['ci95_hi']:.3f}]"
        print(f"  {DET_LABELS[det_name]:20s} {auroc_str:>28s} {m.get('auprc', 0):>8.4f} "
              f"{m.get('fpr_at_95tpr', 0):>8.4f} "
              f"{tpr.get('0.001', 0):>9.4f} {tpr.get('0.01', 0):>8.4f} {tpr.get('0.05', 0):>8.4f}")
    e2_val = all_metrics.get("E2_real_ood", {}).get("validator_rejection_rate", 0)
    print(f"  {'Validator (struct.)':20s} {'(deterministic)':>28s} {'N/A':>8s} "
          f"{'N/A':>8s} {'N/A':>9s} {'N/A':>8s}    rej={e2_val:.4f}")

    print(f"\n  Conclusion: The evaluation characterizes complementary coverage across")
    print(f"  deployment failure modes. No claim of universal ensemble superiority.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
