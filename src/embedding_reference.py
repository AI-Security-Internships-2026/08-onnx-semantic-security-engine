"""
Reference Embedding & Training Stats Generator

Computes reference embeddings and feature statistics needed by the semantic
security engine's drift detector and input validator.

Since the training dataset may not be available locally, this script derives
training feature statistics from the fitted StandardScaler (which stores the
training data's mean and variance) and generates reference embeddings by
loading a subset of the real training dataset and running it through the
ONNX model to capture the true correlation structure.

Outputs:
    experiments/reference_embeddings.npz
        - global_centroid:   (64,) mean embedding across all training samples
        - class_centroids:   (num_classes, 64) per-class centroids (if labels available)
        - covariance:        (64, 64) covariance matrix of embedding space
        - cosine_threshold:  scalar — 95th percentile of cosine distances (for drift cutoff)
        - mahal_threshold:   scalar — 95th percentile of Mahalanobis distances

    experiments/training_feature_stats.json
        - Per-feature: name, mean, std, min_approx, max_approx
        - Derived from the fitted StandardScaler's parameters

Usage:
    python src/embedding_reference.py          # baseline model
    python src/embedding_reference.py --nf     # NF-standardized model
    python src/embedding_reference.py --nf --num-samples 5000
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.spatial.distance import cosine

# ── CLI Arguments ──
parser = argparse.ArgumentParser(description="Generate reference embeddings and training stats")
parser.add_argument(
    "--nf", action="store_true",
    help="Use NF-standardized model (13 features)"
)
parser.add_argument(
    "--arch", type=str, default="mlp", choices=["mlp", "cnn1d"],
    help="Model architecture: 'mlp' or 'cnn1d' (default: 'mlp')"
)
parser.add_argument(
    "--num-samples", type=int, default=100000,
    help="Number of real samples to load for reference embeddings (default: 100000)"
)
args = parser.parse_args()

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"

suffix = "_nf" if args.nf else ""
arch_tag = f"_{args.arch}" if args.arch != "mlp" else ""
model_label = f"{args.arch.upper()} - {'NF-Standardized (13 features)' if args.nf else 'Baseline (76 features)'}"

# ── NF Feature Names (corrected 13-feature list) ──
# Must match the retrained model's feature order exactly.
# See evaluate_semantic_engine.py FEATURE_MAP for the authoritative mapping.
NF_FEATURE_NAMES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Packet Length Max",
    "Packet Length Min",
    "Protocol",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Flow Bytes/s",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
]

print(f"Reference Embedding Generator — {model_label}")
print(f"  Real training samples to load: {args.num_samples}")

# ── Load StandardScaler ──
scaler_path = EXPERIMENTS / f"standard_scaler{suffix}.joblib"
if not scaler_path.exists():
    print(f"[FAIL] Scaler not found: {scaler_path}")
    exit(1)

scaler = joblib.load(scaler_path)
n_features = scaler.n_features_in_
print(f"  Features: {n_features}")
print(f"  Scaler mean shape: {scaler.mean_.shape}")
print(f"  Scaler scale shape: {scaler.scale_.shape}")

# ── Load Label Encoder ──
encoder_path = EXPERIMENTS / f"label_encoder{suffix}.joblib"
if not encoder_path.exists():
    print(f"[FAIL] Encoder not found: {encoder_path}")
    exit(1)

encoder = joblib.load(encoder_path)
class_names = list(encoder.classes_)
num_classes = len(class_names)
print(f"  Classes ({num_classes}): {class_names}")

# ── Step 1: Generate Training Feature Statistics from StandardScaler ──
print("\n[Step 1/3] Generating training feature statistics from StandardScaler...")

feature_names = NF_FEATURE_NAMES[:n_features] if args.nf else [f"feature_{i}" for i in range(n_features)]

feature_stats = {
    "model_variant": model_label,
    "num_features": n_features,
    "features": []
}

for i in range(n_features):
    mean = float(scaler.mean_[i])
    std = float(scaler.scale_[i])
    # Approximate training min/max as mean ± 4*std (covers ~99.99% of normal data)
    # These are conservative estimates since we don't have the actual training data
    feature_stats["features"].append({
        "index": i,
        "name": feature_names[i],
        "mean": round(mean, 6),
        "std": round(std, 6),
        "min_approx": round(mean - 4 * std, 6),
        "max_approx": round(mean + 4 * std, 6),
    })

stats_path = EXPERIMENTS / f"training_feature_stats{arch_tag}{suffix}.json"
with open(stats_path, "w") as f:
    json.dump(feature_stats, f, indent=2)
print(f"  Saved: {stats_path}")

# ── Step 2: Load ONNX Model and Generate Embeddings ──
print("\n[Step 2/3] Loading ONNX model and generating reference embeddings...")

onnx_path = EXPERIMENTS / f"threat_{args.arch}{suffix}_fp32.onnx"
if not onnx_path.exists():
    print(f"[FAIL] ONNX model not found: {onnx_path}")
    exit(1)

session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

# Check if model has embedding output
output_names = [o.name for o in session.get_outputs()]
print(f"  ONNX outputs: {output_names}")

has_embedding = "embedding" in output_names
if not has_embedding:
    print("[WARN] ONNX model does not have 'embedding' output.")
    print("       Re-export with: python src/export_onnx.py --nf --with-embeddings")
    print("       Falling back to logits-only mode for reference stats.")

# Load a sample of real training data to preserve feature correlations
print(f"  Loading real training data from datasets/CIC-IDS2018 (up to {args.num_samples} samples)...")
dataset_dir = BASE_DIR / "datasets" / "CIC-IDS2018"
parquet_files = sorted(dataset_dir.glob("*.parquet"))
if not parquet_files:
    print(f"[FAIL] No parquet files found in {dataset_dir}")
    exit(1)

frames = []
loaded_samples = 0
for pq_file in parquet_files:
    if loaded_samples >= args.num_samples:
        break
    chunk = pd.read_parquet(pq_file)
    chunk.columns = chunk.columns.str.strip()
    frames.append(chunk)
    loaded_samples += len(chunk)
    print(f"    Loaded {pq_file.name}: {len(chunk)} rows (total: {loaded_samples})")

df = pd.concat(frames, ignore_index=True)
if "Timestamp" in df.columns:
    df = df.drop(columns=["Timestamp"])
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()
df = df.drop_duplicates()

if args.nf:
    available_nf = [f for f in NF_FEATURE_NAMES if f in df.columns]
    X_raw = df[available_nf].values[:args.num_samples]
else:
    X_raw = df.drop(columns=["Label"]).values[:args.num_samples]

y_raw = df["Label"].values[:args.num_samples]
y_encoded = encoder.transform(y_raw)
print(f"  Using {len(X_raw)} samples (requested {args.num_samples})")

# Scale features using the loaded StandardScaler
print("  Scaling features...")
scaled_samples = scaler.transform(X_raw).astype(np.float32)

# Run through ONNX model in batches
batch_size = 256
all_embeddings = []
all_logits = []

for i in range(0, args.num_samples, batch_size):
    batch = scaled_samples[i:i + batch_size]
    outputs = session.run(None, {"input": batch})
    all_logits.append(outputs[0])
    if has_embedding:
        all_embeddings.append(outputs[1])

all_logits = np.concatenate(all_logits, axis=0)

if has_embedding:
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    emb_dim = all_embeddings.shape[1]
    print(f"  Embeddings shape: {all_embeddings.shape} (dim={emb_dim})")
else:
    # Fall back: use logits as a proxy embedding space
    all_embeddings = all_logits
    emb_dim = all_logits.shape[1]
    print(f"  Using logits as proxy embeddings: shape={all_embeddings.shape}")

# ── Step 3: Compute Reference Statistics ──
print("\n[Step 3/3] Computing reference centroids and covariance...")

# Global centroid
global_centroid = np.mean(all_embeddings, axis=0)
print(f"  Global centroid shape: {global_centroid.shape}")

# Covariance matrix (regularized for numerical stability)
covariance = np.cov(all_embeddings.T)
# Stronger regularization (λ=0.01) to shrink condition number
# and stabilize Mahalanobis distances across all input types
covariance += np.eye(emb_dim) * 0.01
print(f"  Covariance matrix shape: {covariance.shape}")
cond_num = np.linalg.cond(covariance)
print(f"  Condition number: {cond_num:.2e}")

# Per-class centroids (based on true ground-truth labels)
class_centroids = np.zeros((num_classes, emb_dim))
for cls_idx in range(num_classes):
    mask = y_encoded == cls_idx
    if mask.sum() > 0:
        class_centroids[cls_idx] = np.mean(all_embeddings[mask], axis=0)
    else:
        class_centroids[cls_idx] = global_centroid  # fallback
print(f"  Per-class centroids shape: {class_centroids.shape}")

# Compute distance thresholds from the training distribution
# Class-conditional cosine distances (min distance to nearest class centroid)
cosine_distances = np.zeros(len(all_embeddings))
for i, emb in enumerate(all_embeddings):
    dists = [cosine(emb, c) for c in class_centroids]
    cosine_distances[i] = min(float(np.nan_to_num(d, nan=1.0)) for d in dists)
cosine_threshold = float(np.percentile(cosine_distances, 95))
print(f"  Class-conditional Cosine distance 95th percentile: {cosine_threshold:.4f}")

# Class-conditional Mahalanobis distances (min distance to nearest class centroid)
# This matches the inference-time class-conditional approach in semantic_analyzer.py
try:
    cov_inv = np.linalg.inv(covariance)
    
    # For each training sample, compute Mahalanobis distance to its nearest class centroid
    mahal_distances = np.zeros(len(all_embeddings))
    for i, emb in enumerate(all_embeddings):
        class_dists = []
        for centroid_c in class_centroids:
            diff = emb - centroid_c
            d = np.sqrt(max(diff @ cov_inv @ diff, 0.0))
            class_dists.append(d)
        mahal_distances[i] = min(class_dists)  # closest class
    
    mahal_threshold = float(np.percentile(mahal_distances, 95))
    print(f"  Class-conditional Mahalanobis 95th percentile: {mahal_threshold:.4f}")
    print(f"  Mahalanobis distance stats: median={np.median(mahal_distances):.4f}, "
          f"mean={np.mean(mahal_distances):.4f}, max={np.max(mahal_distances):.4f}")
except np.linalg.LinAlgError:
    print("[WARN] Covariance matrix is singular, using default Mahalanobis threshold")
    cov_inv = np.linalg.pinv(covariance)
    mahal_threshold = 3.0

# Save reference embeddings
ref_path = EXPERIMENTS / f"reference_embeddings{arch_tag}{suffix}.npz"
np.savez(
    ref_path,
    global_centroid=global_centroid,
    class_centroids=class_centroids,
    covariance=covariance,
    covariance_inverse=cov_inv,
    cosine_threshold=np.array(cosine_threshold),
    mahal_threshold=np.array(mahal_threshold),
    class_names=np.array(class_names),
)
print(f"\n  Saved: {ref_path}")

# Summary
print("\n" + "=" * 60)
print("Reference data generation complete!")
print(f"  Feature stats:        {stats_path}")
print(f"  Reference embeddings: {ref_path}")
print(f"  Embedding dim:        {emb_dim}")
print(f"  Cosine threshold:     {cosine_threshold:.4f}")
print(f"  Mahalanobis threshold: {mahal_threshold:.4f}")
print(f"  Classes:              {num_classes}")
print("=" * 60)
