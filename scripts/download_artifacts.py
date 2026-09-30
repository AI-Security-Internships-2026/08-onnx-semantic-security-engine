"""
SEMANTICSHIELD Artifact Verification and Acquisition Manager

Provides verification, integrity validation (SHA-256), and reproduction
instructions for all pre-trained models, embeddings, scalers, and test partitions.

Required artifacts:
  - experiments/threat_mlp_nf_fp32.onnx (+.data)
  - experiments/threat_mlp_nf_fp16.onnx
  - experiments/threat_mlp_nf_int8.onnx
  - experiments/threat_mlp_nf_int4.onnx
  - experiments/threat_cnn1d_nf_fp32.onnx (+.data)
  - experiments/standard_scaler_nf.joblib
  - experiments/label_encoder_nf.joblib
  - experiments/reference_embeddings_nf.npz
  - experiments/reference_embeddings_cnn1d_nf.npz
  - experiments/training_feature_stats_nf.json
  - experiments/training_feature_stats_cnn1d_nf.json
  - experiments/calibration_config.json
  - experiments/X_test_nf.npy
  - experiments/y_test_nf.npy

Usage:
    python scripts/download_artifacts.py --check
    python scripts/download_artifacts.py --verify-checksums
    python scripts/download_artifacts.py --instructions
"""

import sys
import os
import hashlib
import argparse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKSUM_FILE = EXPERIMENTS_DIR / "checksums.sha256"

REQUIRED_ARTIFACTS = [
    "threat_mlp_nf_fp32.onnx",
    "threat_mlp_nf_fp32.onnx.data",
    "threat_mlp_nf_fp16.onnx",
    "threat_mlp_nf_int8.onnx",
    "threat_mlp_nf_int4.onnx",
    "threat_cnn1d_nf_fp32.onnx",
    "threat_cnn1d_nf_fp32.onnx.data",
    "threat_mlp_nf.pth",
    "threat_mlp_nf_best.pth",
    "threat_cnn1d_nf.pth",
    "threat_cnn1d_nf_best.pth",
    "standard_scaler_nf.joblib",
    "label_encoder_nf.joblib",
    "reference_embeddings_nf.npz",
    "reference_embeddings_cnn1d_nf.npz",
    "training_feature_stats_nf.json",
    "training_feature_stats_cnn1d_nf.json",
    "calibration_config.json",
    "X_test_nf.npy",
    "y_test_nf.npy",
]

REPRODUCTION_GUIDE = """
================================================================================
  HOW TO REGENERATE ALL REQUIRED ARTIFACTS LOCALLY FROM RAW DATA
================================================================================

1. Acquire In-Distribution Dataset:
   Download CSE-CIC-IDS2018 into `datasets/CIC-IDS2018/` (see `datasets/README.md`).

2. Train Classifiers:
   # Train ThreatMLP (13 NetFlow features)
   python src/train_classifier.py --nf --arch mlp --epochs 30

   # Train ThreatCNN1D (Cross-model replication)
   python src/train_classifier.py --nf --arch cnn1d --epochs 30

3. Export to ONNX (with dual embedding outputs):
   python src/export_onnx.py --nf --arch mlp --with-embeddings
   python src/export_onnx.py --nf --arch cnn1d --with-embeddings

4. Quantize ONNX Models:
   python src/quantize_model.py --model experiments/threat_mlp_nf_fp32.onnx --test-data experiments/X_test_nf.npy

5. Generate Reference Embeddings & Training Statistics:
   python src/embedding_reference.py --nf --arch mlp
   python src/embedding_reference.py --nf --arch cnn1d

6. Calibrate SEMANTICSHIELD Detection Thresholds:
   python scripts/evaluate_semantic_engine.py

7. Verify Generated Artifacts:
   python scripts/download_artifacts.py --verify-checksums
================================================================================
"""


def check_presence() -> bool:
    """Check whether all required runtime artifacts exist locally."""
    print("=" * 70)
    print("  CHECKING REQUIRED RUNTIME ARTIFACTS")
    print("=" * 70)
    missing = []
    present = []
    for f in REQUIRED_ARTIFACTS:
        p = EXPERIMENTS_DIR / f
        if p.exists():
            size_kb = p.stat().st_size / 1024
            present.append((f, size_kb))
        else:
            missing.append(f)

    for f, sz in present:
        print(f"  [EXISTS]  {f:<42} ({sz:>8.1f} KB)")

    if missing:
        print("\nMissing artifacts:")
        for f in missing:
            print(f"  [MISSING] experiments/{f}")
        print("\nRun `python scripts/download_artifacts.py --instructions` to see reproduction steps.")
        return False

    print(f"\nAll {len(REQUIRED_ARTIFACTS)} required runtime artifacts are present!")
    return True


def verify_checksums() -> bool:
    """Verify SHA-256 checksums of all tracked artifacts against checksums.sha256."""
    print("=" * 70)
    print("  VERIFYING ARTIFACT CRYPTOGRAPHIC INTEGRITY (SHA-256)")
    print("=" * 70)
    if not CHECKSUM_FILE.exists():
        print(f"[FAIL] Checksum file not found: {CHECKSUM_FILE}")
        return False

    all_ok = True
    checked = 0
    passed = 0

    for line in CHECKSUM_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, rel_path = line.split(maxsplit=1)
        target = BASE_DIR / rel_path
        checked += 1

        if not target.exists():
            print(f"  [MISSING]  {rel_path}")
            all_ok = False
            continue

        h = hashlib.sha256(target.read_bytes()).hexdigest()
        if h == expected_hash:
            passed += 1
            print(f"  [VALID]    {rel_path:<46} SHA256: {h[:12]}...")
        else:
            print(f"  [MISMATCH] {rel_path:<46} Expected: {expected_hash[:12]}..., Got: {h[:12]}...")
            all_ok = False

    print("=" * 70)
    if all_ok:
        print(f"SUCCESS: All {passed}/{checked} artifacts verified with 100% cryptographic integrity.")
    else:
        print(f"WARNING: {checked - passed} artifact(s) failed verification.")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="SEMANTICSHIELD Artifact Verification & Acquisition Manager")
    parser.add_argument("--check", action="store_true", help="Check presence of all required artifacts")
    parser.add_argument("--verify-checksums", action="store_true", help="Verify SHA-256 hashes against experiments/checksums.sha256")
    parser.add_argument("--instructions", action="store_true", help="Print instructions to reproduce artifacts from raw data")
    args = parser.parse_args()

    if args.instructions:
        print(REPRODUCTION_GUIDE)
        return

    if args.verify_checksums:
        ok = verify_checksums()
        sys.exit(0 if ok else 1)

    # Default action: check presence and verify checksums if available
    presence_ok = check_presence()
    if presence_ok and CHECKSUM_FILE.exists():
        print()
        checksum_ok = verify_checksums()
        sys.exit(0 if checksum_ok else 1)
    sys.exit(0 if presence_ok else 1)


if __name__ == "__main__":
    main()

