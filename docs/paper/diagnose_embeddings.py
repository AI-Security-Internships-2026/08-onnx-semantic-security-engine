"""Diagnose current reference embeddings quality."""
import numpy as np

ref = np.load("experiments/reference_embeddings_nf.npz")
print("Keys:", list(ref.keys()))
cov = ref["covariance"]
print(f"Covariance shape: {cov.shape}")
print(f"Covariance diag min/max: {np.diag(cov).min():.6f} / {np.diag(cov).max():.6f}")
cond = np.linalg.cond(cov)
print(f"Condition number: {cond:.2e}")
eigvals = np.linalg.eigvalsh(cov)
print(f"Eigenvalue range: {eigvals.min():.6e} to {eigvals.max():.6e}")
print(f"Ratio max/min: {eigvals.max()/max(eigvals.min(), 1e-30):.2e}")
cos_t = float(ref["cosine_threshold"])
mah_t = float(ref["mahal_threshold"])
print(f"Cosine threshold: {cos_t:.6f}")
print(f"Mahal threshold: {mah_t:.6f}")
centroid = ref["global_centroid"]
print(f"Centroid norm: {np.linalg.norm(centroid):.4f}")
print(f"Centroid stats: min={centroid.min():.4f}, max={centroid.max():.4f}, mean={centroid.mean():.4f}")
class_centroids = ref["class_centroids"]
print(f"Class centroids shape: {class_centroids.shape}")
# Post-ReLU fc3 embeddings should be non-negative
neg_frac = (centroid < 0).mean()
print(f"Centroid negative fraction: {neg_frac:.2%}")
cls_neg = (class_centroids < 0).mean()
print(f"Class centroids negative fraction: {cls_neg:.2%}")

# Try computing Mahalanobis distance on centroid itself (should be ~0)
cov_inv = ref["covariance_inverse"]
diff = centroid - centroid  # zero
mahal_self = np.sqrt(diff @ cov_inv @ diff)
print(f"Mahal distance centroid-to-self: {mahal_self:.6f}")

# Check a random sample
rng = np.random.default_rng(42)
fake_emb = centroid + rng.normal(0, 0.1, size=centroid.shape)
diff2 = fake_emb - centroid
mahal_near = np.sqrt(diff2 @ cov_inv @ diff2)
print(f"Mahal distance centroid+noise(0.1): {mahal_near:.4f}")

fake_far = rng.normal(0, 10, size=centroid.shape).astype(np.float32)
diff3 = fake_far - centroid
mahal_far = np.sqrt(diff3 @ cov_inv @ diff3)
print(f"Mahal distance random far: {mahal_far:.4f}")
