"""
E2.2-A: Same-Schema Cross-Dataset Shift Evaluation
==================================================

Evaluates domain and distribution shift across datasets with the SAME standardized
NetFlow v2 schema (NF-ToN-IoT-v2 and NF-BoT-IoT-v2), trained on NF-CSE-CIC-IDS2018.

Separates:
  1. In-distribution baseline (held-out CIC-IDS2018 test set)
  2. External dataset #1: NF-ToN-IoT-v2
  3. External dataset #2: NF-BoT-IoT-v2
Across BOTH neural models:
  - ThreatMLP
  - ThreatCNN1D

Outputs:
  JSON: experiments/paper_results/json/cross_dataset_generalization.json
  Tables: experiments/paper_results/tables/table_cross_dataset_generalization.csv
  Figures: experiments/paper_results/figures/cross_dataset_performance_drop.png
           experiments/paper_results/figures/detector_generalization_across_datasets.png
"""

import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score, accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.semantic_analyzer import DriftDetector, InputValidator
from scripts.config_loader import load_paper_config, get_provenance_metadata
from scripts.data_utils import (
    FEATURE_MAP, NF_FEATURES,
    load_standardized_netflow, get_toniot_path, get_botiot_path,
    safe_auroc, safe_auprc, tpr_at_fixed_fpr
)

EXPERIMENTS = BASE_DIR / "experiments"
JSON_DIR = EXPERIMENTS / "paper_results" / "json"
TABLES_DIR = EXPERIMENTS / "paper_results" / "tables"
FIGURES_DIR = EXPERIMENTS / "paper_results" / "figures"


def evaluate_model_on_dataset(
    model_name: str,
    onnx_path: Path,
    scaler_path: Path,
    encoder_path: Path,
    ref_emb_path: Path,
    stats_path: Path,
    X_raw: np.ndarray,
    y_true_binary: np.ndarray,
    dataset_name: str,
    is_id: bool = False,
    X_id_scaled: Optional[np.ndarray] = None,
    cal_scores: Optional[Dict[str, np.ndarray]] = None,
    fpr_budgets: List[float] = [0.001, 0.01, 0.05]
) -> Dict:
    """Run full classifier and assurance evaluation on an evaluation dataset."""
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    # Drift detector
    ref_data = np.load(ref_emb_path)
    cosine_th = float(ref_data["cosine_threshold"])
    mahal_th = float(ref_data["mahal_threshold"])
    
    # Load feature stats for validator
    with open(stats_path) as f:
        fstats = json.load(f)

    # Scale inputs using source scaler
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    # Measure latency
    t0 = time.perf_counter()
    onnx_out = session.run(None, {"input": X_scaled[:1000]})
    sample_latency_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0

    # Full batch inference
    batch_size = 4096
    all_logits = []
    all_embs = []
    for i in range(0, len(X_scaled), batch_size):
        b = X_scaled[i:i + batch_size]
        outs = session.run(None, {"input": b})
        all_logits.append(outs[0])
        all_embs.append(outs[1])
    
    logits = np.vstack(all_logits)
    embeddings = np.vstack(all_embs)
    probs = softmax(logits, axis=1)
    preds = np.argmax(logits, axis=1)

    # Classifier metrics
    # Binary predictions: class 0 is Benign, classes >0 are Attack
    pred_binary = (preds != 0).astype(int)
    bin_acc = float(accuracy_score(y_true_binary, pred_binary))
    bin_f1 = float(f1_score(y_true_binary, pred_binary, zero_division=0))

    # Anomaly/Assurance scores (higher = more anomalous)
    msp_anomaly_scores = 1.0 - np.max(probs, axis=1)
    
    # Compute Cosine distances to class centroids
    class_centroids = ref_data["class_centroids"]
    cov_inv = ref_data["covariance_inverse"]
    
    cosine_dists = np.zeros(len(embeddings), dtype=np.float32)
    mahal_dists = np.zeros(len(embeddings), dtype=np.float32)
    for i, emb in enumerate(embeddings):
        # min cosine to class centroids
        c_dists = [1.0 - (np.dot(emb, c) / (np.linalg.norm(emb) * np.linalg.norm(c) + 1e-9)) for c in class_centroids]
        cosine_dists[i] = min(c_dists)
        
        # min mahalanobis to class centroids
        m_dists = []
        for c in class_centroids:
            diff = emb - c
            md = np.sqrt(max(diff @ cov_inv @ diff, 0.0))
            m_dists.append(md)
        mahal_dists[i] = min(m_dists)

    composite_scores = np.minimum(np.maximum(cosine_dists / cosine_th, mahal_dists / mahal_th), 2.0)

    # If this is ID test data, return calibration/baseline stats
    res = {
        "model": model_name,
        "dataset": dataset_name,
        "is_id": is_id,
        "sample_count": len(X_raw),
        "sample_latency_ms": round(sample_latency_ms, 4),
        "binary_accuracy": round(bin_acc, 4),
        "binary_macro_f1": round(bin_f1, 4),
        "raw_scores": {
            "msp": msp_anomaly_scores,
            "cosine": cosine_dists,
            "mahalanobis": mahal_dists,
            "composite": composite_scores
        }
    }

    if not is_id and cal_scores is not None:
        # Evaluate OOD detection: 0 = In-Dist (from cal_scores), 1 = External OOD
        n_id = len(cal_scores["msp"])
        n_ood = len(msp_anomaly_scores)
        y_eval = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
        
        # Assurance AUROC / AUPRC
        assurance_metrics = {}
        for det_key in ["msp", "cosine", "mahalanobis", "composite"]:
            eval_scores = np.concatenate([cal_scores[det_key], res["raw_scores"][det_key]])
            auroc = safe_auroc(y_eval, eval_scores)
            auprc = safe_auprc(y_eval, eval_scores)
            
            # TPR at fixed FPR budgets
            tpr_points = {}
            for fpr_b in fpr_budgets:
                tpr_val = tpr_at_fixed_fpr(y_eval, eval_scores, target_fpr=fpr_b)
                tpr_points[f"tpr@{fpr_b*100:.1f}%"] = round(tpr_val, 4)
                
            assurance_metrics[det_key] = {
                "auroc": round(auroc, 4),
                "auprc": round(auprc, 4),
                **tpr_points
            }
        
        # Observed ID FPR
        id_fpr_5pct = float(np.mean(cal_scores["composite"] > 1.0))
        res["observed_id_fpr"] = round(id_fpr_5pct, 4)
        res["assurance"] = assurance_metrics
        res["full_assurance_auroc"] = assurance_metrics["composite"]["auroc"]

    return res


def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  E2.2-A: SAME-SCHEMA CROSS-DATASET GENERALIZATION EVALUATION")
    print("=" * 80)

    # 1. Load Datasets
    print("\n[1/4] Loading Datasets...")
    # ID Test data (CIC-IDS2018)
    X_test_id = np.load(EXPERIMENTS / "X_test_nf.npy")
    y_test_id = np.load(EXPERIMENTS / "y_test_nf.npy")
    # Subsample ID test to 20,000 for balanced evaluation
    idx_id = np.random.RandomState(42).choice(len(X_test_id), 20000, replace=False)
    X_id = X_test_id[idx_id]
    y_id_binary = (y_test_id[idx_id] != 0).astype(int)
    print(f"  ID Test (CIC-IDS2018): {len(X_id):,} samples, {y_id_binary.mean()*100:.1f}% attacks")

    # ToN-IoT
    toniot_path = get_toniot_path()
    df_ton, y_ton_bin, _, _ = load_standardized_netflow(toniot_path, n_samples=30000, random_state=42)
    X_ton = df_ton[NF_FEATURES].values.astype(np.float32)
    print(f"  External #1 (ToN-IoT): {len(X_ton):,} samples, {y_ton_bin.mean()*100:.1f}% attacks")

    # BoT-IoT
    botiot_path = get_botiot_path()
    df_bot, y_bot_bin, _, _ = load_standardized_netflow(botiot_path, n_samples=30000, random_state=42)
    X_bot = df_bot[NF_FEATURES].values.astype(np.float32)
    print(f"  External #2 (BoT-IoT): {len(X_bot):,} samples, {y_bot_bin.mean()*100:.1f}% attacks")

    # 2. Evaluate Models
    models_to_eval = [
        {
            "name": "ThreatMLP",
            "onnx": EXPERIMENTS / "threat_mlp_nf_fp32.onnx",
            "scaler": EXPERIMENTS / "standard_scaler_nf.joblib",
            "encoder": EXPERIMENTS / "label_encoder_nf.joblib",
            "ref_emb": EXPERIMENTS / "reference_embeddings_nf.npz",
            "stats": EXPERIMENTS / "training_feature_stats_nf.json",
        },
        {
            "name": "ThreatCNN1D",
            "onnx": EXPERIMENTS / "threat_cnn1d_nf_fp32.onnx",
            "scaler": EXPERIMENTS / "standard_scaler_nf.joblib",
            "encoder": EXPERIMENTS / "label_encoder_nf.joblib",
            "ref_emb": EXPERIMENTS / "reference_embeddings_cnn1d_nf.npz",
            "stats": EXPERIMENTS / "training_feature_stats_cnn1d_nf.json",
        }
    ]

    all_results = []
    table_rows = []

    for m in models_to_eval:
        m_name = m["name"]
        print(f"\n[2/4] Evaluating {m_name}...")
        
        if not m["onnx"].exists():
            print(f"  [SKIP] {m['onnx'].name} does not exist yet.")
            continue

        # In-distribution evaluation
        id_eval = evaluate_model_on_dataset(
            m_name, m["onnx"], m["scaler"], m["encoder"], m["ref_emb"], m["stats"],
            X_id, y_id_binary, "CIC-IDS2018 (In-Dist)", is_id=True
        )
        all_results.append(id_eval)
        table_rows.append({
            "Train Dataset": "NF-CSE-CIC-IDS2018",
            "Test Dataset": "ID (held-out)",
            "Model": m_name,
            "Accuracy": f"{id_eval['binary_accuracy']:.4f}",
            "Macro-F1": f"{id_eval['binary_macro_f1']:.4f}",
            "MSP AUROC": "1.0000 (Ref)",
            "Mahalanobis AUROC": "1.0000 (Ref)",
            "Full Assurance Metric": "1.0000 (Ref)",
        })

        # External ToN-IoT evaluation
        ton_eval = evaluate_model_on_dataset(
            m_name, m["onnx"], m["scaler"], m["encoder"], m["ref_emb"], m["stats"],
            X_ton, y_ton_bin, "NF-ToN-IoT-v2", is_id=False, cal_scores=id_eval["raw_scores"]
        )
        all_results.append(ton_eval)
        table_rows.append({
            "Train Dataset": "NF-CSE-CIC-IDS2018",
            "Test Dataset": "NF-ToN-IoT-v2",
            "Model": m_name,
            "Accuracy": f"{ton_eval['binary_accuracy']:.4f}",
            "Macro-F1": f"{ton_eval['binary_macro_f1']:.4f}",
            "MSP AUROC": f"{ton_eval['assurance']['msp']['auroc']:.4f}",
            "Mahalanobis AUROC": f"{ton_eval['assurance']['mahalanobis']['auroc']:.4f}",
            "Full Assurance Metric": f"{ton_eval['full_assurance_auroc']:.4f}",
        })

        # External BoT-IoT evaluation
        bot_eval = evaluate_model_on_dataset(
            m_name, m["onnx"], m["scaler"], m["encoder"], m["ref_emb"], m["stats"],
            X_bot, y_bot_bin, "NF-BoT-IoT-v2", is_id=False, cal_scores=id_eval["raw_scores"]
        )
        all_results.append(bot_eval)
        table_rows.append({
            "Train Dataset": "NF-CSE-CIC-IDS2018",
            "Test Dataset": "NF-BoT-IoT-v2",
            "Model": m_name,
            "Accuracy": f"{bot_eval['binary_accuracy']:.4f}",
            "Macro-F1": f"{bot_eval['binary_macro_f1']:.4f}",
            "MSP AUROC": f"{bot_eval['assurance']['msp']['auroc']:.4f}",
            "Mahalanobis AUROC": f"{bot_eval['assurance']['mahalanobis']['auroc']:.4f}",
            "Full Assurance Metric": f"{bot_eval['full_assurance_auroc']:.4f}",
        })

    # 3. Save JSON Results (remove raw numpy arrays)
    clean_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items() if k != "raw_scores"}
        clean_results.append(clean_r)

    json_path = JSON_DIR / "cross_dataset_generalization.json"
    with open(json_path, "w") as f:
        json.dump({"cross_dataset_results": clean_results}, f, indent=2)
    print(f"\n[3/4] Saved JSON to: {json_path}")

    # 4. Save Table CSV
    table_df = pd.DataFrame(table_rows)
    csv_path = TABLES_DIR / "table_cross_dataset_generalization.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved Table to: {csv_path}")
    print("\n" + table_df.to_string(index=False))

    # 5. Generate Plots
    print("\n[4/4] Generating Figures...")
    try:
        import matplotlib.pyplot as plt
        
        # Figure 1: ID vs External-Dataset Performance Drop
        fig, ax = plt.subplots(figsize=(9, 5))
        bar_data = []
        for r in clean_results:
            bar_data.append({
                "label": f"{r['model']}\n{r['dataset']}",
                "f1": r["binary_macro_f1"],
                "acc": r["binary_accuracy"]
            })
        
        x = np.arange(len(bar_data))
        width = 0.35
        ax.bar(x - width/2, [b["acc"] for b in bar_data], width, label="Binary Accuracy", color="#3b82f6")
        ax.bar(x + width/2, [b["f1"] for b in bar_data], width, label="Binary Macro-F1", color="#10b981")
        ax.set_ylabel("Score")
        ax.set_title("Classifier Performance Drop: In-Distribution vs External Datasets")
        ax.set_xticks(x)
        ax.set_xticklabels([b["label"] for b in bar_data], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_path1 = FIGURES_DIR / "cross_dataset_performance_drop.png"
        fig.savefig(fig_path1, dpi=300)
        plt.close(fig)
        print(f"  Saved Figure 1: {fig_path1}")

        # Figure 2: Detector Generalization Across Datasets
        fig, ax = plt.subplots(figsize=(9, 5))
        det_data = [r for r in clean_results if not r["is_id"]]
        labels = [f"{r['model']}\n{r['dataset']}" for r in det_data]
        msp_aurocs = [r["assurance"]["msp"]["auroc"] for r in det_data]
        mahal_aurocs = [r["assurance"]["mahalanobis"]["auroc"] for r in det_data]
        comp_aurocs = [r["assurance"]["composite"]["auroc"] for r in det_data]

        x = np.arange(len(det_data))
        w = 0.25
        ax.bar(x - w, msp_aurocs, w, label="MSP AUROC", color="#f59e0b")
        ax.bar(x, mahal_aurocs, w, label="Mahalanobis AUROC", color="#8b5cf6")
        ax.bar(x + w, comp_aurocs, w, label="SEMANTICSHIELD Composite AUROC", color="#06b6d4")
        ax.set_ylabel("AUROC vs In-Distribution")
        ax.set_title("Assurance Detector Generalization Across External NetFlow Datasets")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        fig_path2 = FIGURES_DIR / "detector_generalization_across_datasets.png"
        fig.savefig(fig_path2, dpi=300)
        plt.close(fig)
        print(f"  Saved Figure 2: {fig_path2}")

    except Exception as e:
        print(f"  [WARN] Could not plot figures: {e}")

    print("\nE2.2-A Complete!")


if __name__ == "__main__":
    main()
