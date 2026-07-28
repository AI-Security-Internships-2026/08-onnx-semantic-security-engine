"""
Reference Embedding & Training Stats Generator

Computes reference embeddings and feature statistics needed by the semantic
security engine's drift detector and input validator.

Since the training dataset may not be available locally, this script derives
training feature statistics from the fitted StandardScaler (which stores the
training data's mean and variance) and generates reference embeddings by
sampling from the training distribution and running through the ONNX model.

Outputs:
    experiments/reference_embeddings.npz
        - global_centroid:   (64,) mean embedding across all synthetic samples
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
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.spatial.distance import cosine

# ── CLI Arguments ──
parser = argparse.ArgumentParser(description="Generate reference embeddings and training stats")
parser.add_argument(
    "--nf", action="store_true",
    help="Use NF-standardized model (21 features)"
)
parser.add_argument(
    "--num-samples", type=int, default=5000,
    help="Number of synthetic samples to generate for reference embeddings (default: 5000)"
)
args = parser.parse_args()

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"

suffix = "_nf" if args.nf else ""
model_label = "NF-Standardized (21 features)" if args.nf else "Baseline (76 features)"

# ── NF Feature Names ──
NF_FEATURE_NAMES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Fwd Packets Length Total",
    "Bwd Packets Length Total",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Packet Length Max",
    "Packet Length Min",
    "Flow Bytes/s",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd PSH Flags",
    "Init Fwd Win Bytes",
    "Init Bwd Win Bytes",
    "Fwd Avg Packets/Bulk",
    "Bwd Avg Packets/Bulk",
    "Fwd Avg Bytes/Bulk",
    "Bwd Avg Bytes/Bulk",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
]

print(f"Reference Embedding Generator — {model_label}")
print(f"  Synthetic samples: {args.num_samples}")

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

stats_path = EXPERIMENTS / f"training_feature_stats{suffix}.json"
with open(stats_path, "w") as f:
    json.dump(feature_stats, f, indent=2)
print(f"  Saved: {stats_path}")

# ── Step 2: Load ONNX Model and Generate Embeddings ──
print("\n[Step 2/3] Loading ONNX model and generating synthetic embeddings...")

onnx_path = EXPERIMENTS / f"threat_mlp{suffix}_fp32.onnx"
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

# Generate synthetic samples from the training distribution
# Since StandardScaler normalizes to N(0,1), we sample from N(0,1) in scaled space
# and inverse-transform to get realistic raw feature values
np.random.seed(42)
scaled_samples = np.random.randn(args.num_samples, n_features).astype(np.float32)

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
# Add small ridge to ensure invertibility
covariance += np.eye(emb_dim) * 1e-6
print(f"  Covariance matrix shape: {covariance.shape}")

# Per-class centroids (based on predicted labels from synthetic data)
pred_classes = np.argmax(all_logits, axis=1)
class_centroids = np.zeros((num_classes, emb_dim))
for cls_idx in range(num_classes):
    mask = pred_classes == cls_idx
    if mask.sum() > 0:
        class_centroids[cls_idx] = np.mean(all_embeddings[mask], axis=0)
    else:
        class_centroids[cls_idx] = global_centroid  # fallback
print(f"  Per-class centroids shape: {class_centroids.shape}")

# Compute distance thresholds from the training distribution
# Cosine distances to global centroid
cosine_distances = np.array([cosine(emb, global_centroid) for emb in all_embeddings])
cosine_threshold = float(np.percentile(cosine_distances, 95))
print(f"  Cosine distance 95th percentile: {cosine_threshold:.4f}")

# Mahalanobis distances to global centroid
try:
    cov_inv = np.linalg.inv(covariance)
    diffs = all_embeddings - global_centroid
    mahal_distances = np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))
    mahal_threshold = float(np.percentile(mahal_distances, 95))
    print(f"  Mahalanobis distance 95th percentile: {mahal_threshold:.4f}")
except np.linalg.LinAlgError:
    print("[WARN] Covariance matrix is singular, using default Mahalanobis threshold")
    cov_inv = np.linalg.pinv(covariance)
    mahal_threshold = 3.0

# Save reference embeddings
ref_path = EXPERIMENTS / f"reference_embeddings{suffix}.npz"
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
