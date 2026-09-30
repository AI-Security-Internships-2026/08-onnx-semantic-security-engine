"""
Diagnose WHY Mahalanobis fails for OOD detection.
Compare Mahalanobis distances for in-distribution vs OOD samples.
"""
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
EXP = BASE / "experiments"

# Load model, scaler, reference
session = ort.InferenceSession(str(EXP / "threat_mlp_nf_fp32.onnx"), providers=["CPUExecutionProvider"])
scaler = joblib.load(EXP / "standard_scaler_nf.joblib")
ref = np.load(EXP / "reference_embeddings_nf.npz")
centroid = ref["global_centroid"]
cov_inv = ref["covariance_inverse"]
cos_thresh = float(ref["cosine_threshold"])
mah_thresh = float(ref["mahal_threshold"])

NF_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Fwd Packets Length Total", "Bwd Packets Length Total",
    "Packet Length Max", "Packet Length Min", "Protocol",
    "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Flow Bytes/s", "Init Fwd Win Bytes", "Init Bwd Win Bytes",
]

def get_embeddings(X_raw, n=500):
    """Scale + run through ONNX, return embeddings."""
    X_scaled = scaler.transform(X_raw[:n]).astype(np.float32)
    outputs = session.run(None, {"input": X_scaled})
    return outputs[1]  # embeddings

def compute_distances(embeddings):
    """Compute cosine and Mahalanobis distances."""
    from scipy.spatial.distance import cosine
    cos_dists = np.array([cosine(e, centroid) for e in embeddings])
    diffs = embeddings - centroid
    mah_dists = np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))
    return cos_dists, mah_dists

# ── In-distribution: CIC-IDS2018 ──
print("Loading in-distribution (CIC-IDS2018)...")
cic_files = sorted((BASE / "datasets" / "CIC-IDS2018").glob("*.parquet"))
df_cic = pd.read_parquet(cic_files[0])  # just one file
df_cic.columns = df_cic.columns.str.strip()
df_cic.replace([np.inf, -np.inf], np.nan, inplace=True)
df_cic = df_cic.dropna()
X_cic = df_cic[[f for f in NF_FEATURES if f in df_cic.columns]].values
emb_cic = get_embeddings(X_cic, 500)
cos_cic, mah_cic = compute_distances(emb_cic)

# ── OOD: ToN-IoT ──
print("Loading OOD (ToN-IoT)...")
toniot_dir = BASE / "datasets" / "NF-ToN-IoT-v2"
toniot_files = sorted(toniot_dir.glob("*.parquet")) or sorted(toniot_dir.glob("*.csv"))
if toniot_files:
    if str(toniot_files[0]).endswith(".parquet"):
        df_ton = pd.read_parquet(toniot_files[0])
    else:
        df_ton = pd.read_csv(toniot_files[0], nrows=10000)
    df_ton.columns = df_ton.columns.str.strip()
    df_ton.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ton = df_ton.dropna()
    
    # Map NF-ToN-IoT features to our 13 features
    FEATURE_MAP = {
        "L7_PROTO": None, "L4_DST_PORT": None, "L4_SRC_PORT": None,
        "PROTOCOL": "Protocol", "IN_BYTES": "Fwd Packets Length Total",
        "IN_PKTS": "Total Fwd Packets", "OUT_BYTES": "Bwd Packets Length Total",
        "OUT_PKTS": "Total Backward Packets", "FLOW_DURATION_MILLISECONDS": "Flow Duration",
    }
    # Try direct column names first
    available = [f for f in NF_FEATURES if f in df_ton.columns]
    if available:
        X_ton = df_ton[available].values
    else:
        print(f"  Available columns: {list(df_ton.columns[:20])}")
        # Try NF format columns
        nf_cols = [c for c in df_ton.columns if c in FEATURE_MAP and FEATURE_MAP[c] is not None]
        print(f"  NF columns found: {nf_cols}")
        X_ton = df_ton[nf_cols].values
    
    emb_ton = get_embeddings(X_ton, min(500, len(X_ton)))
    cos_ton, mah_ton = compute_distances(emb_ton)
else:
    print("  ToN-IoT not found, using random noise as OOD proxy")
    rng = np.random.default_rng(42)
    X_ton = rng.normal(0, 1, (500, 13)).astype(np.float32)
    emb_ton = get_embeddings(X_ton, 500)
    cos_ton, mah_ton = compute_distances(emb_ton)

# ── Noise ──
print("Generating noise inputs...")
rng = np.random.default_rng(42)
X_noise = rng.normal(0, 10, (500, 13)).astype(np.float32)
emb_noise = get_embeddings(X_noise, 500)
cos_noise, mah_noise = compute_distances(emb_noise)

# ── Report ──
print("\n" + "="*70)
print("DISTANCE DISTRIBUTIONS (mean ± std)")
print("="*70)
print(f"{'Source':<15} {'Cosine':>20} {'Mahalanobis':>25}")
print("-"*70)
print(f"{'In-dist':<15} {cos_cic.mean():.4f} ± {cos_cic.std():.4f}{'':<5} {mah_cic.mean():.2f} ± {mah_cic.std():.2f}")
print(f"{'OOD':<15} {cos_ton.mean():.4f} ± {cos_ton.std():.4f}{'':<5} {mah_ton.mean():.2f} ± {mah_ton.std():.2f}")
print(f"{'Noise':<15} {cos_noise.mean():.4f} ± {cos_noise.std():.4f}{'':<5} {mah_noise.mean():.2f} ± {mah_noise.std():.2f}")
print("-"*70)
print(f"Thresholds:     cosine={cos_thresh:.4f}   mahal={mah_thresh:.4f}")
print()

# Separation analysis
print("SEPARATION ANALYSIS:")
print(f"  Cosine: OOD mean / In-dist mean = {cos_ton.mean()/cos_cic.mean():.2f}x")
print(f"  Mahal:  OOD mean / In-dist mean = {mah_ton.mean()/mah_cic.mean():.2f}x")
print(f"  Cosine: Noise mean / In-dist mean = {cos_noise.mean()/cos_cic.mean():.2f}x")
print(f"  Mahal:  Noise mean / In-dist mean = {mah_noise.mean()/mah_cic.mean():.2f}x")
print()

# What fraction exceeds threshold?
print("DETECTION RATES (fraction exceeding threshold):")
print(f"  In-dist: cosine={( cos_cic > cos_thresh).mean():.2%}, mahal={(mah_cic > mah_thresh).mean():.2%}")
print(f"  OOD:     cosine={(cos_ton > cos_thresh).mean():.2%}, mahal={(mah_ton > mah_thresh).mean():.2%}")
print(f"  Noise:   cosine={(cos_noise > cos_thresh).mean():.2%}, mahal={(mah_noise > mah_thresh).mean():.2%}")

# The key question: is OOD CLOSER to centroid than in-dist? (would explain AUROC < 0.5)
print()
print("KEY DIAGNOSTIC: Is OOD *closer* than in-dist? (explains AUROC < 0.5)")
print(f"  Mahal median in-dist: {np.median(mah_cic):.2f}")
print(f"  Mahal median OOD:     {np.median(mah_ton):.2f}")
print(f"  Mahal median noise:   {np.median(mah_noise):.2f}")
if np.median(mah_ton) < np.median(mah_cic):
    print("  >>> YES: OOD is CLOSER to centroid than in-dist in Mahalanobis space!")
    print("  >>> This means the covariance captures a direction where OOD looks 'normal'")
else:
    print("  >>> NO: OOD is farther, but distributions overlap heavily")
