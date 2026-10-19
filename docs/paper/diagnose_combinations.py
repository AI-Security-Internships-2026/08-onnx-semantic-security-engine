"""
Diagnose composite score combination and see what happens when combining
Mahalanobis + MSP + Cosine in different ways.
"""
import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.spatial.distance import cosine as cosine_distance
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).parent.parent.parent
EXP = BASE / "experiments"
DATASETS = BASE / "datasets"

session = ort.InferenceSession(str(EXP / "threat_mlp_nf_fp32.onnx"), providers=["CPUExecutionProvider"])
scaler = joblib.load(EXP / "standard_scaler_nf.joblib")
ref_data = np.load(EXP / "reference_embeddings_nf.npz", allow_pickle=True)
global_centroid = ref_data["global_centroid"]
class_centroids = ref_data["class_centroids"]
covariance_inverse = ref_data["covariance_inverse"]

# Load in-dist
parquet_files = sorted((DATASETS / "CIC-IDS2018").glob("*.parquet"))
df_cic = pd.read_parquet(parquet_files[0])
df_cic.columns = df_cic.columns.str.strip()
if "Label" in df_cic.columns:
    df_cic = df_cic[df_cic["Label"] == "Benign"]
df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cic = df_cic.dropna()

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

X_cic = df_cic[NF_FEATURES].values[:20000]
X_calib = X_cic[:10000]
X_test = X_cic[10000:20000]

# Load OOD ToN-IoT
df_ton = pd.read_parquet(DATASETS / "ToN-IoT" / "NF-ToN-IoT-V2.parquet")
df_ton.columns = df_ton.columns.str.strip()
X_ton_raw = np.zeros((10000, len(NF_FEATURES)), dtype=np.float64)
for i, (ton_col, cic_col) in enumerate(FEATURE_MAP.items()):
    if ton_col in df_ton.columns:
        X_ton_raw[:, i] = df_ton[ton_col].values[:10000]
X_ton_raw = np.nan_to_num(X_ton_raw, nan=0.0, posinf=0.0, neginf=0.0)

# Scaled
X_calib_s = scaler.transform(X_calib).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)
X_ton_s = scaler.transform(X_ton_raw).astype(np.float32)

def extract(X_s):
    out = session.run(None, {"input": X_s})
    logits, embs = out[0], out[1]
    probs = softmax(logits, axis=1)
    confs = np.max(probs, axis=1)
    msp_scores = 1.0 - confs
    
    # Cosine to global centroid
    cos_dists = np.array([float(np.nan_to_num(cosine_distance(emb, global_centroid), nan=0.0)) for emb in embs])
    
    # Class-conditional cosine
    cos_cc = np.zeros(len(embs))
    for i, emb in enumerate(embs):
        dists = [cosine_distance(emb, c) for c in class_centroids]
        cos_cc[i] = min(float(np.nan_to_num(d, nan=1.0)) for d in dists)
        
    # Class-conditional Mahalanobis
    mahal_dists = np.full(len(embs), np.inf)
    for c in class_centroids:
        diffs = embs - c
        d = np.sqrt(np.maximum(np.sum(diffs @ covariance_inverse * diffs, axis=1), 0.0))
        mahal_dists = np.minimum(mahal_dists, d)
        
    return msp_scores, cos_dists, cos_cc, mahal_dists

cal_msp, cal_cos, cal_cos_cc, cal_mahal = extract(X_calib_s)
test_msp, test_cos, test_cos_cc, test_mahal = extract(X_test_s)
ton_msp, ton_cos, ton_cos_cc, ton_mahal = extract(X_ton_s)

y_true = np.concatenate([np.zeros(len(test_msp)), np.ones(len(ton_msp))])

print("INDIVIDUAL DETECTOR AUROCs:")
print(f"  MSP Alone:                     {roc_auc_score(y_true, np.concatenate([test_msp, ton_msp])):.4f}")
print(f"  Cosine (Global):               {roc_auc_score(y_true, np.concatenate([test_cos, ton_cos])):.4f}")
print(f"  Cosine (Class-Conditional):    {roc_auc_score(y_true, np.concatenate([test_cos_cc, ton_cos_cc])):.4f}")
print(f"  Mahalanobis (Class-Conditional): {roc_auc_score(y_true, np.concatenate([test_mahal, ton_mahal])):.4f}")

# Thresholds at 95th percentile of calib
t_msp = np.percentile(cal_msp, 95)
t_cos = np.percentile(cal_cos, 95)
t_cos_cc = np.percentile(cal_cos_cc, 95)
t_mahal = np.percentile(cal_mahal, 95)

# Normalization combinations
norm_test_mahal = test_mahal / t_mahal
norm_ton_mahal = ton_mahal / t_mahal

norm_test_msp = test_msp / t_msp
norm_ton_msp = ton_msp / t_msp

norm_test_cos = test_cos / t_cos
norm_ton_cos = ton_cos / t_cos

norm_test_cos_cc = test_cos_cc / t_cos_cc
norm_ton_cos_cc = ton_cos_cc / t_cos_cc

print("\nCOMBINATIONS AUROC:")
# 1. Current combination (max(cos_global, mahal))
c1_test = np.maximum(norm_test_cos, norm_test_mahal)
c1_ton = np.maximum(norm_ton_cos, norm_ton_mahal)
print(f"  1. Max(Cosine_Global, Mahalanobis):                 {roc_auc_score(y_true, np.concatenate([c1_test, c1_ton])):.4f}")

# 2. Max(Cosine_ClassCond, Mahalanobis)
c2_test = np.maximum(norm_test_cos_cc, norm_test_mahal)
c2_ton = np.maximum(norm_ton_cos_cc, norm_ton_mahal)
print(f"  2. Max(Cosine_ClassCond, Mahalanobis):             {roc_auc_score(y_true, np.concatenate([c2_test, c2_ton])):.4f}")

# 3. Max(MSP, Mahalanobis)
c3_test = np.maximum(norm_test_msp, norm_test_mahal)
c3_ton = np.maximum(norm_ton_msp, norm_ton_mahal)
print(f"  3. Max(MSP, Mahalanobis):                          {roc_auc_score(y_true, np.concatenate([c3_test, c3_ton])):.4f}")

# 4. Soft combination: Mean(MSP, Mahalanobis)
c4_test = (norm_test_msp + norm_test_mahal) / 2.0
c4_ton = (norm_ton_msp + norm_ton_mahal) / 2.0
print(f"  4. Mean(MSP, Mahalanobis):                         {roc_auc_score(y_true, np.concatenate([c4_test, c4_ton])):.4f}")

# 5. Weighted combination: 0.8 * Mahalanobis + 0.2 * MSP
c5_test = 0.8 * norm_test_mahal + 0.2 * norm_test_msp
c5_ton = 0.8 * norm_ton_mahal + 0.2 * norm_ton_msp
print(f"  5. Weighted (0.8*Mahal + 0.2*MSP):                 {roc_auc_score(y_true, np.concatenate([c5_test, c5_ton])):.4f}")

# 6. Weighted with class-cond cosine: 0.7*Mahal + 0.2*MSP + 0.1*Cos_CC
c6_test = 0.7 * norm_test_mahal + 0.2 * norm_test_msp + 0.1 * norm_test_cos_cc
c6_ton = 0.7 * norm_ton_mahal + 0.2 * norm_ton_msp + 0.1 * norm_ton_cos_cc
print(f"  6. Multi-Signal (0.7*Mahal + 0.2*MSP + 0.1*CosCC): {roc_auc_score(y_true, np.concatenate([c6_test, c6_ton])):.4f}")
