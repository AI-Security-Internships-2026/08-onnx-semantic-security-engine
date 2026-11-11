"""
Generate Canonical Classifier Metrics and Confusion Matrix for ThreatMLP NF (13 features)

Evaluates the frozen ONNX model on the standardized test partition (138,069 samples)
and generates:
  - classifier_metrics.json (overall metrics, per-class metrics, confusion matrix array)
  - confusion_matrix_cic.png (publication-quality normalized confusion matrix heatmap)

Usage:
    python scripts/generate_classifier_metrics.py [--output-dir DIR] [--figures-dir DIR]
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import joblib
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.config_loader import load_paper_config, get_provenance_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate classifier metrics and confusion matrix.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BASE_DIR / "experiments" / "paper_results" / "json"),
        help="Directory to save classifier_metrics.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(BASE_DIR / "experiments" / "paper_results" / "figures"),
        help="Directory to save confusion_matrix_cic.png",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to paper configuration file (defaults to configs/paper_v1.yaml)",
    )
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot publication-quality confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Normalize by row (true class)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Recall (Normalized)', rotation=-90, va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    ax.set_title("CSE-CIC-IDS2018 Normalized Confusion Matrix (ThreatMLP NF-13)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

    # Annotate significant values
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j]
            if val >= 0.01:
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=7,
                )

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
    print("  GENERATING CLASSIFIER METRICS (THREATMLP NF-13)")
    print("=" * 75)

    # 1. Resolve paths
    onnx_rel = cfg["model"].get("onnx_path", "experiments/threat_mlp_nf_fp32.onnx")
    onnx_path = BASE_DIR / onnx_rel
    encoder_rel = cfg["model"].get("encoder", "experiments/label_encoder_nf.joblib")
    encoder_path = BASE_DIR / encoder_rel
    x_test_path = BASE_DIR / "experiments" / "X_test_nf.npy"
    y_test_path = BASE_DIR / "experiments" / "y_test_nf.npy"

    for p in [onnx_path, encoder_path, x_test_path, y_test_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact not found: {p}")

    # 2. Load model and data
    print(f"Loading ONNX session: {onnx_path}")
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    
    le = joblib.load(encoder_path)
    class_names = [str(c) for c in le.classes_]
    n_classes = len(class_names)

    print(f"Loading test partitions: X={x_test_path.name}, y={y_test_path.name}")
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    n_samples, n_features = X_test.shape

    print(f"Running inference on {n_samples:,} samples with {n_features} features...")
    preds_logits = session.run(None, {input_name: X_test})[0]
    preds = np.argmax(preds_logits, axis=1)

    # 3. Calculate metrics
    acc = float(accuracy_score(y_test, preds))
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    per_class = {}
    for c in class_names:
        if c in report:
            per_class[c] = {
                "precision": round(float(report[c]["precision"]), 4),
                "recall": round(float(report[c]["recall"]), 4),
                "f1": round(float(report[c]["f1-score"]), 4),
                "support": int(report[c]["support"]),
            }

    overall = {
        "accuracy": round(acc, 4),
        "macro_avg_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_avg_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_avg_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_avg_precision": round(float(report["weighted avg"]["precision"]), 4),
        "weighted_avg_recall": round(float(report["weighted avg"]["recall"]), 4),
        "weighted_avg_f1": round(float(report["weighted avg"]["f1-score"]), 4),
    }

    result = {
        "provenance": provenance,
        "model": f"ThreatMLP NF-Standardized ({n_features} features, {n_classes} classes, all 10 CIC-IDS2018 days)",
        "architecture": f"Input({n_features}) -> 256 -> 128 -> 64 -> {n_classes} (BatchNorm + Dropout 0.3)",
        "dataset": "CSE-CIC-IDS2018",
        "split": "80/20 stratified, random_state=42",
        "evaluation": {
            "test_samples": n_samples,
            "features_count": n_features,
            "classes_count": n_classes,
        },
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist(),
        },
    }

    # 4. Save JSON
    out_file = output_dir / "classifier_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] Classifier metrics saved to {out_file}")

    # 5. Generate and save figure
    fig_file = figures_dir / "confusion_matrix_cic.png"
    plot_confusion_matrix(cm, class_names, fig_file)

    # 6. Print summary
    print("\n" + "=" * 65)
    print(f"  Test Accuracy:     {overall['accuracy']*100:.2f}%")
    print(f"  Macro-Avg F1:      {overall['macro_avg_f1']:.4f}")
    print(f"  Weighted-Avg F1:   {overall['weighted_avg_f1']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
