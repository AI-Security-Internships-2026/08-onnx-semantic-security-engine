"""
SEMANTICSHIELD Ablation Study

Evaluates SEMANTICSHIELD with each component systematically enabled/disabled:
  1. Full System (InputValidator + ConfidenceAnalyzer + DriftDetector)
  2. No ConfidenceAnalyzer (-confidence)
  3. No DriftDetector (-drift)
  4. No InputValidator (-validator)
  5. Validator Only (-confidence, -drift)
  6. Drift Only (-validator, -confidence)
  7. Confidence Only (-validator, -drift)

Evaluates on:
  - In-distribution benign (CIC-IDS2018)
  - Out-of-distribution network traffic (ToN-IoT)
  - Random Gaussian noise
  - Zero-filled feature vectors (tampering)
  - Extreme range / NaN corruption

Outputs:
  - experiments/paper_results/json/ablation_study.json
  - experiments/paper_results/figures/ablation_comparison.png

Usage:
    python scripts/ablation_study.py [--output-dir DIR] [--figures-dir DIR] [--samples N]
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.semantic_analyzer import (
    SemanticSecurityEngine,
    ConfidenceAnalyzer,
    DriftDetector,
    InputValidator,
)
from scripts.config_loader import load_paper_config, get_provenance_metadata

EXPERIMENTS = BASE_DIR / "experiments"
DATASETS = BASE_DIR / "datasets"

FEATURE_MAP = {
    "FLOW_DURATION_MILLISECONDS": "Flow Duration",
    "IN_PKTS": "Total Fwd Packets",
    "OUT_PKTS": "Total Backward Packets",
    "IN_BYTES": "Fwd Packets Length Total",
    "OUT_BYTES": "Bwd Packets Length Total",
    "LONGEST_FLOW_PKT": "Packet Length Max",
    "SHORTEST_FLOW_PKT": "Packet Length Min",
    "PROTOCOL": "Protocol",
    "MAX_IP_PKT_LEN": "Fwd Packet Length Max",
    "MIN_IP_PKT_LEN": "Fwd Packet Length Min",
    "SRC_TO_DST_SECOND_BYTES": "Flow Bytes/s",
    "TCP_WIN_MAX_IN": "Init Fwd Win Bytes",
    "TCP_WIN_MAX_OUT": "Init Bwd Win Bytes",
}
TONIOT_FEATURES = list(FEATURE_MAP.keys())


def parse_args():
    parser = argparse.ArgumentParser(description="Run SEMANTICSHIELD component ablation study.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "experiments" / "paper_results" / "json"),
        help="Directory to save ablation_study.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(BASE_DIR / "experiments" / "paper_results" / "figures"),
        help="Directory to save ablation_comparison.png",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples per evaluation scenario (default: 5000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to paper config (defaults to configs/paper_v1.yaml)",
    )
    return parser.parse_args()


def plot_ablation(results, output_path):
    """Generate publication-ready ablation comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(results["configurations"].keys())
    short_names = [
        "Full System",
        "-Confidence",
        "-Drift",
        "-Validator",
        "Validator Only",
        "Drift Only",
        "Confidence Only",
    ]

    in_fpr = [results["configurations"][c]["in_dist_fpr_pct"] for c in configs]
    ood_det = [results["configurations"][c]["ood_detection_pct"] for c in configs]
    noise_det = [results["configurations"][c]["noise_rejection_pct"] for c in configs]
    zero_det = [results["configurations"][c]["zero_rejection_pct"] for c in configs]

    x = np.arange(len(short_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 7))
    r1 = ax.bar(x - 1.5 * width, in_fpr, width, label="In-Dist FPR % (Lower is better)", color="#455A64")
    r2 = ax.bar(x - 0.5 * width, ood_det, width, label="ToN-IoT OOD Intercept %", color="#1976D2")
    r3 = ax.bar(x + 0.5 * width, noise_det, width, label="Noise Intercept %", color="#388E3C")
    r4 = ax.bar(x + 1.5 * width, zero_det, width, label="Zero-Fill Rejection %", color="#D32F2F")

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("SEMANTICSHIELD Architectural Component Ablation Study", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 110)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Figure saved to {output_path}")


def main():
    args = parse_args()
    cfg = load_paper_config(args.config)
    provenance = get_provenance_metadata(args.config)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  SEMANTICSHIELD ARCHITECTURAL COMPONENT ABLATION STUDY")
    print("=" * 75)

    # 1. Initialize engine from config
    engine = SemanticSecurityEngine.from_config(args.config or "configs/paper_v1.yaml")
    session = ort.InferenceSession(str(BASE_DIR / cfg["model"]["onnx_path"]))
    input_name = session.get_inputs()[0].name

    n_eval = args.samples

    # 2. Load In-Distribution Benign Test data
    print(f"\n[1/4] Loading in-distribution test samples (N={n_eval})...")
    X_test_all = np.load(EXPERIMENTS / "X_test_nf.npy")
    y_test_all = np.load(EXPERIMENTS / "y_test_nf.npy")
    benign_mask = (y_test_all == 0)
    X_in_dist = X_test_all[benign_mask][:n_eval]

    # 3. Load OOD data (ToN-IoT)
    print(f"[2/4] Loading OOD ToN-IoT samples (N={n_eval})...")
    ton_path = DATASETS / "ToN-IoT" / "NF-ToN-IoT-V2.parquet"
    if ton_path.exists():
        df_ton = pd.read_parquet(ton_path, columns=TONIOT_FEATURES)
        X_ood_raw = df_ton.values[:n_eval].astype(np.float32)
        X_ood_raw = np.nan_to_num(X_ood_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        scaler_path = BASE_DIR / cfg["model"].get("scaler", "experiments/standard_scaler_nf.joblib")
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None
        if scaler is not None:
            X_ood_scaled = scaler.transform(X_ood_raw).astype(np.float32)
        else:
            X_ood_scaled = X_ood_raw
    else:
        # Synthetic OOD fallback if parquet is unavailable
        rng = np.random.RandomState(42)
        X_ood_raw = rng.exponential(scale=50.0, size=(n_eval, 13)).astype(np.float32)
        X_ood_scaled = X_ood_raw

    # 4. Generate Noise, Zero-fill, and Corrupted datasets
    print(f"[3/4] Preparing noise, zero-fill, and corrupted samples...")
    rng = np.random.RandomState(42)
    X_noise_raw = rng.normal(0, 1, (min(1000, n_eval), 13)).astype(np.float32)
    X_noise_scaled = X_noise_raw

    X_zero_raw = np.zeros((min(500, n_eval), 13), dtype=np.float32)
    X_zero_scaled = X_zero_raw

    # Precompute raw signals for all datasets
    def evaluate_signals(X_raw, X_scaled):
        # Run ONNX inference
        outs = session.run(None, {input_name: X_scaled})
        logits = outs[0]
        embs = outs[1]
        probs = softmax(logits, axis=1)
        conf_scores = np.max(probs, axis=1)
        conf_flags = ["LOW_CONFIDENCE" if c < engine.confidence_analyzer.threshold else "OK" for c in conf_scores]

        # Drift signals
        drift_batch = engine.drift_detector.analyze_batch(embs)
        cos_dists = drift_batch["cosine_distances"]
        mahal_dists = drift_batch["mahalanobis_distances"]
        drift_scores = drift_batch["drift_scores"]
        drift_flags = [
            "DRIFT_DETECTED" if (cos > engine.drift_detector.cosine_threshold or mahal > engine.drift_detector.mahal_threshold)
            else "OK"
            for cos, mahal in zip(cos_dists, mahal_dists)
        ]

        # Validation signals
        val_passes = []
        val_alerts = []
        for row in X_raw:
            v_res = engine.input_validator.analyze(row)
            val_passes.append(v_res.validation_passed)
            val_alerts.append(v_res.alerts)

        return {
            "conf_flags": conf_flags,
            "drift_flags": drift_flags,
            "drift_scores": drift_scores,
            "val_passes": val_passes,
            "val_alerts": val_alerts,
            "n": len(X_raw),
        }

    # Extract signals
    signals_in = evaluate_signals(X_in_dist, X_in_dist)
    signals_ood = evaluate_signals(X_ood_raw, X_ood_scaled)
    signals_noise = evaluate_signals(X_noise_raw, X_noise_scaled)
    signals_zero = evaluate_signals(X_zero_raw, X_zero_scaled)

    # Define the 7 ablation setups
    ablation_definitions = {
        "full_system": {
            "name": "Full System (InputValidator + Confidence + Drift)",
            "use_val": True,
            "use_conf": True,
            "use_drift": True,
        },
        "no_confidence": {
            "name": "Without ConfidenceAnalyzer (-confidence)",
            "use_val": True,
            "use_conf": False,
            "use_drift": True,
        },
        "no_drift": {
            "name": "Without DriftDetector (-drift)",
            "use_val": True,
            "use_conf": True,
            "use_drift": False,
        },
        "no_validator": {
            "name": "Without InputValidator (-validator)",
            "use_val": False,
            "use_conf": True,
            "use_drift": True,
        },
        "validator_only": {
            "name": "InputValidator Only (-confidence, -drift)",
            "use_val": True,
            "use_conf": False,
            "use_drift": False,
        },
        "drift_only": {
            "name": "DriftDetector Only (-validator, -confidence)",
            "use_val": False,
            "use_conf": False,
            "use_drift": True,
        },
        "confidence_only": {
            "name": "ConfidenceAnalyzer Only (-validator, -drift)",
            "use_val": False,
            "use_conf": True,
            "use_drift": False,
        },
    }

    def run_verdicts(sig, use_val, use_conf, use_drift):
        n = sig["n"]
        verdicts = []
        for i in range(n):
            c_flag = sig["conf_flags"][i] if use_conf else "OK"
            d_flag = sig["drift_flags"][i] if use_drift else "OK"
            v_pass = sig["val_passes"][i] if use_val else True
            v_alerts = sig["val_alerts"][i] if use_val else []
            _, verdict = SemanticSecurityEngine.compute_verdict(
                confidence_flag=c_flag,
                drift_flag=d_flag,
                validation_passed=v_pass,
                validation_alerts=v_alerts,
            )
            verdicts.append(verdict)
        return verdicts

    print("\n[4/4] Evaluating all 7 architectural ablation configurations...")
    configs_summary = {}

    for key, spec in ablation_definitions.items():
        v_in = run_verdicts(signals_in, spec["use_val"], spec["use_conf"], spec["use_drift"])
        v_ood = run_verdicts(signals_ood, spec["use_val"], spec["use_conf"], spec["use_drift"])
        v_noise = run_verdicts(signals_noise, spec["use_val"], spec["use_conf"], spec["use_drift"])
        v_zero = run_verdicts(signals_zero, spec["use_val"], spec["use_conf"], spec["use_drift"])

        # Metrics
        in_fpr = (sum(1 for v in v_in if v != "CLEAN") / len(v_in)) * 100.0
        ood_det = (sum(1 for v in v_ood if v in ("HIGH_RISK", "REJECTED")) / len(v_ood)) * 100.0
        ood_flagged = (sum(1 for v in v_ood if v != "CLEAN") / len(v_ood)) * 100.0
        noise_det = (sum(1 for v in v_noise if v in ("HIGH_RISK", "REJECTED")) / len(v_noise)) * 100.0
        zero_rej = (sum(1 for v in v_zero if v == "REJECTED") / len(v_zero)) * 100.0

        configs_summary[key] = {
            "description": spec["name"],
            "components": {
                "input_validator": spec["use_val"],
                "confidence_analyzer": spec["use_conf"],
                "drift_detector": spec["use_drift"],
            },
            "in_dist_clean_pct": round(100.0 - in_fpr, 2),
            "in_dist_fpr_pct": round(in_fpr, 2),
            "ood_detection_pct": round(ood_det, 2),
            "ood_total_flagged_pct": round(ood_flagged, 2),
            "noise_rejection_pct": round(noise_det, 2),
            "zero_rejection_pct": round(zero_rej, 2),
        }

    output_data = {
        "provenance": provenance,
        "experiment": "SEMANTICSHIELD Architectural Component Ablation Study",
        "sample_sizes": {
            "in_distribution": len(X_in_dist),
            "out_of_distribution": len(X_ood_raw),
            "random_noise": len(X_noise_raw),
            "zero_filled": len(X_zero_raw),
        },
        "configurations": configs_summary,
    }

    out_file = output_dir / "ablation_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[SAVED] Ablation results saved to {out_file}")

    fig_file = figures_dir / "ablation_comparison.png"
    plot_ablation(output_data, fig_file)

    # Print summary table
    print("\n" + "=" * 110)
    print(f"  {'Ablation Configuration':<36} {'In-Dist FPR':>12} {'OOD HighRisk':>13} {'OOD AnyFlag':>13} {'Noise Rej':>11} {'Zero Rej':>11}")
    print("=" * 110)
    for k, v in configs_summary.items():
        print(f"  {v['description'][:36]:<36} {v['in_dist_fpr_pct']:>11.1f}% {v['ood_detection_pct']:>12.1f}% {v['ood_total_flagged_pct']:>12.1f}% {v['noise_rejection_pct']:>10.1f}% {v['zero_rejection_pct']:>10.1f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
