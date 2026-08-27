"""
Download / Verify pre-trained model artifacts.

Required artifacts for the ONNX Semantic Security Engine:
  - experiments/threat_mlp_nf_fp32.onnx (+.data)
  - experiments/threat_mlp_nf_fp16.onnx
  - experiments/threat_mlp_nf_int8.onnx
  - experiments/standard_scaler_nf.joblib
  - experiments/label_encoder_nf.joblib
  - experiments/reference_embeddings_nf.npz
  - experiments/training_feature_stats_nf.json
  - experiments/calibration_config.json

To regenerate from scratch:
  1. Train: Run experiments/notebooks/nf-standardized-training.ipynb on Kaggle
  2. Export: python src/export_onnx.py --nf --with-embeddings
  3. Quantize: python src/quantize_model.py --model experiments/threat_mlp_nf_fp32.onnx --test-data experiments/X_test_nf.npy
  4. Reference: python src/embedding_reference.py --nf
  5. Calibrate: python scripts/evaluate_semantic_engine.py
"""

from pathlib import Path

REQUIRED = [
    "threat_mlp_nf_fp32.onnx",
    "threat_mlp_nf_fp32.onnx.data",
    "threat_mlp_nf_fp16.onnx",
    "threat_mlp_nf_int8.onnx",
    "standard_scaler_nf.joblib",
    "label_encoder_nf.joblib",
    "reference_embeddings_nf.npz",
    "training_feature_stats_nf.json",
    "calibration_config.json",
]


def check():
    base = Path(__file__).parent.parent / "experiments"
    missing = [f for f in REQUIRED if not (base / f).exists()]
    if missing:
        print(f"Missing {len(missing)} artifacts:")
        for f in missing:
            print(f"  [X] experiments/{f}")
        print("\nSee this script's docstring for regeneration instructions.")
        return False
    else:
        print(f"All {len(REQUIRED)} required artifacts present [OK]")
        return True


if __name__ == "__main__":
    check()
