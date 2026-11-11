"""
E2.2-B: Semantic / Extractor Mismatch Evaluation
================================================

Separates pure domain shift from feature-extractor / semantic mismatch (Track B).

Evaluates sensitivity across three distinct mapping tiers:
  Tier 1: Exact Mappings Only (8 features) — perfectly aligned physical quantities
  Tier 2: Exact + Approximate (13 features) — canonical SEMANTICSHIELD standardized schema
  Tier 3: With Known Semantic Mismatches (13+ features) — legacy nProbe to CICFlowMeter
          mappings where physical quantities differ (bps vs header bytes, flag bitmasks vs packet counts).

Outputs:
  JSON: experiments/paper_results/json/semantic_mismatch_sensitivity.json
  Tables: experiments/paper_results/tables/table_semantic_mismatch_sensitivity.csv
  Figures: experiments/paper_results/figures/semantic_mapping_sensitivity.png
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.data_utils import (
    FEATURE_MAP, TIER1_EXACT_MAP, TIER2_APPROX_MAP, TIER3_MISMATCHED_MAP,
    NF_FEATURES, load_standardized_netflow, get_toniot_path, get_botiot_path,
    safe_auroc
)

EXPERIMENTS = BASE_DIR / "experiments"
JSON_DIR = EXPERIMENTS / "paper_results" / "json"
TABLES_DIR = EXPERIMENTS / "paper_results" / "tables"
FIGURES_DIR = EXPERIMENTS / "paper_results" / "figures"


def main():
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  E2.2-B: SEMANTIC / EXTRACTOR MISMATCH SENSITIVITY EVALUATION")
    print("=" * 80)

    # Load artifacts for ThreatMLP
    mlp_onnx_path = EXPERIMENTS / "threat_mlp_nf_fp32.onnx"
    cnn_onnx_path = EXPERIMENTS / "threat_cnn1d_nf_fp32.onnx"
    scaler_path = EXPERIMENTS / "standard_scaler_nf.joblib"
    encoder_path = EXPERIMENTS / "label_encoder_nf.joblib"
    ref_path = EXPERIMENTS / "reference_embeddings_nf.npz"

    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    ref_data = np.load(ref_path)
    class_centroids = ref_data["class_centroids"]
    cov_inv = ref_data["covariance_inverse"]

    # Load external datasets
    print("\n[1/3] Loading External Evaluation Datasets...")
    toniot_path = get_toniot_path()
    df_ton_raw = pd.read_parquet(toniot_path, columns=None)
    # Sample 30,000
    df_ton = df_ton_raw.sample(n=min(30000, len(df_ton_raw)), random_state=42).copy()
    y_ton_bin = df_ton["Label"].values.astype(int)

    # In-distribution test baseline
    X_test_id = np.load(EXPERIMENTS / "X_test_nf.npy")
    y_test_id = (np.load(EXPERIMENTS / "y_test_nf.npy") != 0).astype(int)
    idx_id = np.random.RandomState(42).choice(len(X_test_id), 20000, replace=False)
    X_id = X_test_id[idx_id]
    y_id = y_test_id[idx_id]

    models_to_test = [("ThreatMLP", mlp_onnx_path)]
    if cnn_onnx_path.exists():
        models_to_test.append(("ThreatCNN1D", cnn_onnx_path))

    tier_results = []
    table_rows = []

    # Define the 3 experimental tiers
    # Tier 1: Exact Mappings Only (8 features). Missing 5 features are set to training mean (neutral zero in standard scaler)
    # Tier 2: Exact + Approximate (13 features) — Canonical Standardized
    # Tier 3: With Semantic Mismatches: 3 corrupted features mapped to mismatched NetFlow fields
    
    tiers = [
        {"tier": "Tier 1: Exact Only", "desc": "8 genuinely equivalent features (missing 5 set to neutral mean)"},
        {"tier": "Tier 2: Standardized", "desc": "13 features (8 exact + 5 approximate)"},
        {"tier": "Tier 3: With Mismatches", "desc": "13 features with 3 replaced by semantically mismatched fields"},
    ]

    for model_name, onnx_p in models_to_test:
        print(f"\n[2/3] Evaluating {model_name} across Mapping Tiers...")
        session = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        
        # In-dist reference scoring
        id_scaled = scaler.transform(X_id).astype(np.float32)
        id_outs = session.run(None, {"input": id_scaled})
        id_msp_anomaly = 1.0 - np.max(softmax(id_outs[0], axis=1), axis=1)

        for t_info in tiers:
            tier_name = t_info["tier"]
            # Build feature matrix for this tier
            X_eval_raw = np.zeros((len(df_ton), 13), dtype=np.float32)
            
            # Canonical 13 columns in order
            for col_idx, feat_name in enumerate(NF_FEATURES):
                # Find matching NetFlow column
                nf_col = [k for k, v in FEATURE_MAP.items() if v == feat_name][0]
                
                if "Tier 1" in tier_name:
                    if nf_col in TIER1_EXACT_MAP:
                        X_eval_raw[:, col_idx] = df_ton[nf_col].values.astype(np.float32)
                    else:
                        # Neutral value = mean from scaler so scaled value is 0.0
                        X_eval_raw[:, col_idx] = scaler.mean_[col_idx]
                elif "Tier 2" in tier_name:
                    X_eval_raw[:, col_idx] = df_ton[nf_col].values.astype(np.float32)
                elif "Tier 3" in tier_name:
                    # Intentionally corrupt 3 features with mismatched physical units
                    if feat_name == "Flow Duration" and "SRC_TO_DST_AVG_THROUGHPUT" in df_ton.columns:
                        # map throughput (bps) instead of duration
                        X_eval_raw[:, col_idx] = df_ton["SRC_TO_DST_AVG_THROUGHPUT"].values.astype(np.float32)
                    elif feat_name == "Init Fwd Win Bytes" and "TCP_FLAGS" in df_ton.columns:
                        # map flag bitmask instead of window size
                        X_eval_raw[:, col_idx] = df_ton["TCP_FLAGS"].values.astype(np.float32)
                    elif feat_name == "Flow Bytes/s" and "NUM_PKTS_UP_TO_128_BYTES" in df_ton.columns:
                        # map packet count bucket instead of bytes/sec
                        X_eval_raw[:, col_idx] = df_ton["NUM_PKTS_UP_TO_128_BYTES"].values.astype(np.float32)
                    else:
                        X_eval_raw[:, col_idx] = df_ton[nf_col].values.astype(np.float32)

            # Clean NaNs/Infs
            X_eval_raw = np.nan_to_num(X_eval_raw, nan=0.0, posinf=1e6, neginf=-1e6)

            # Scale and infer
            X_eval_scaled = scaler.transform(X_eval_raw).astype(np.float32)
            outs = session.run(None, {"input": X_eval_scaled})
            logits = outs[0]
            embs = outs[1]
            probs = softmax(logits, axis=1)
            preds = np.argmax(logits, axis=1)
            pred_binary = (preds != 0).astype(int)

            acc = float(accuracy_score(y_ton_bin, pred_binary))
            f1 = float(f1_score(y_ton_bin, pred_binary, zero_division=0))

            # MSP AUROC vs In-Dist
            ood_msp_anomaly = 1.0 - np.max(probs, axis=1)
            y_eval_ood = np.concatenate([np.zeros(len(id_msp_anomaly)), np.ones(len(ood_msp_anomaly))])
            scores_eval = np.concatenate([id_msp_anomaly, ood_msp_anomaly])
            msp_auroc = safe_auroc(y_eval_ood, scores_eval)

            # Mahalanobis distance
            mahal_dists = np.zeros(len(embs), dtype=np.float32)
            for i, emb in enumerate(embs):
                c_dists = [np.sqrt(max((emb - c) @ cov_inv @ (emb - c), 0.0)) for c in class_centroids]
                mahal_dists[i] = min(c_dists)

            tier_results.append({
                "model": model_name,
                "tier": tier_name,
                "description": t_info["desc"],
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "msp_auroc": round(msp_auroc, 4),
                "mean_mahalanobis": round(float(np.mean(mahal_dists)), 4)
            })

            table_rows.append({
                "Model": model_name,
                "Mapping Tier": tier_name,
                "Features Active": "8 Exact" if "Tier 1" in tier_name else ("13 Standardized" if "Tier 2" in tier_name else "13 (3 Mismatched)"),
                "Accuracy": f"{acc:.4f}",
                "Macro-F1": f"{f1:.4f}",
                "MSP AUROC": f"{msp_auroc:.4f}",
                "Mean Mahalanobis Dist": f"{np.mean(mahal_dists):.4f}"
            })

    # Save JSON
    json_path = JSON_DIR / "semantic_mismatch_sensitivity.json"
    with open(json_path, "w") as f:
        json.dump({"semantic_mismatch_results": tier_results}, f, indent=2)
    print(f"\n[3/3] Saved JSON to: {json_path}")

    # Save CSV
    table_df = pd.DataFrame(table_rows)
    csv_path = TABLES_DIR / "table_semantic_mismatch_sensitivity.csv"
    table_df.to_csv(csv_path, index=False)
    print(f"  Saved Table to: {csv_path}")
    print("\n" + table_df.to_string(index=False))

    # Generate Figure
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Subplot 1: Macro-F1 across tiers
        tiers_list = [r["tier"] for r in tier_results if r["model"] == "ThreatMLP"]
        mlp_f1s = [r["macro_f1"] for r in tier_results if r["model"] == "ThreatMLP"]
        
        x = np.arange(len(tiers_list))
        w = 0.35
        ax1.bar(x, mlp_f1s, w, label="ThreatMLP F1", color="#3b82f6")
        if any(r["model"] == "ThreatCNN1D" for r in tier_results):
            cnn_f1s = [r["macro_f1"] for r in tier_results if r["model"] == "ThreatCNN1D"]
            ax1.bar(x + w, cnn_f1s, w, label="ThreatCNN1D F1", color="#10b981")
            ax1.set_xticks(x + w/2)
        else:
            ax1.set_xticks(x)
            
        ax1.set_xticklabels(tiers_list, rotation=15, ha="right", fontsize=9)
        ax1.set_ylabel("Macro-F1")
        ax1.set_title("Classifier Performance vs Feature Mapping Tier")
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # Subplot 2: MSP AUROC across tiers
        mlp_aurocs = [r["msp_auroc"] for r in tier_results if r["model"] == "ThreatMLP"]
        ax2.bar(x, mlp_aurocs, w, label="ThreatMLP MSP AUROC", color="#f59e0b")
        if any(r["model"] == "ThreatCNN1D" for r in tier_results):
            cnn_aurocs = [r["msp_auroc"] for r in tier_results if r["model"] == "ThreatCNN1D"]
            ax2.bar(x + w, cnn_aurocs, w, label="ThreatCNN1D MSP AUROC", color="#8b5cf6")
            ax2.set_xticks(x + w/2)
        else:
            ax2.set_xticks(x)
            
        ax2.set_xticklabels(tiers_list, rotation=15, ha="right", fontsize=9)
        ax2.set_ylabel("MSP AUROC")
        ax2.set_title("OOD Detection Ability vs Feature Mapping Tier")
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig_path = FIGURES_DIR / "semantic_mapping_sensitivity.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        print(f"  Saved Figure to: {fig_path}")

    except Exception as e:
        print(f"  [WARN] Could not plot figures: {e}")

    print("\nE2.2-B Complete!")


if __name__ == "__main__":
    main()
