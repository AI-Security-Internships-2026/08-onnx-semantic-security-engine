"""
ONNX Export Pipeline — Supports both Baseline (76-feature) and NF-Standardized (21-feature) models.

Usage:
    # Export baseline model (default)
    python src/export_onnx.py

    # Export NF-standardized model
    python src/export_onnx.py --nf
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
import argparse
from pathlib import Path

from model import ThreatMLP

# ── CLI Arguments ──
parser = argparse.ArgumentParser(description="Export PyTorch ThreatMLP to ONNX format")
parser.add_argument(
    "--nf", action="store_true",
    help="Export the NF-standardized model (21 features) instead of baseline (76 features)"
)
args = parser.parse_args()

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"

if args.nf:
    model_file = "threat_mlp_nf.pth"
    onnx_file = "threat_mlp_nf_fp32.onnx"
    model_label = "NF-Standardized (21 features)"
else:
    model_file = "threat_mlp.pth"
    onnx_file = "threat_mlp_fp32.onnx"
    model_label = "Baseline (76 features)"

model_path = EXPERIMENTS / model_file
onnx_path = EXPERIMENTS / onnx_file

# ── Load model and infer dimensions ──
print(f"Loading trained model: {model_label}")
print(f"  Source: {model_path}")

if not model_path.exists():
    print(f"[FAIL] Model file not found: {model_path}")
    exit(1)

state_dict = torch.load(model_path, map_location="cpu")

# Infer input_dim and num_classes from state_dict weight shapes
input_dim = state_dict["fc1.weight"].shape[1]
num_classes = state_dict["fc4.weight"].shape[0]
print(f"  Detected dimensions: input_dim={input_dim}, num_classes={num_classes}")

# Initialize and load model
model = ThreatMLP(input_dim, num_classes)
model.load_state_dict(state_dict)
model.eval()

# ── Export to ONNX ──
print(f"\nExporting to ONNX format...")
dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    str(onnx_path),
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    verbose=False
)
print(f"  ONNX model exported to: {onnx_path}")

# ── Validate ONNX model structure ──
print("\nValidating ONNX model structure...")
onnx_model = onnx.load(str(onnx_path))
onnx.checker.check_model(onnx_model)
print("[PASS] ONNX model structure is valid")

# ── Critical validation: compare PyTorch vs ONNX Runtime outputs ──
print("\nRunning CRITICAL VALIDATION (PyTorch vs ONNX Runtime)...")
print("=" * 70)

ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

all_match = True
for sample_idx in range(10):
    # Generate random input
    random_input = np.random.randn(1, input_dim).astype(np.float32)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(torch.FloatTensor(random_input))
        pytorch_pred = torch.argmax(pytorch_output, dim=1).item()
    
    # ONNX Runtime inference
    onnx_output = ort_session.run(None, {"input": random_input})[0]
    onnx_pred = np.argmax(onnx_output, axis=1)[0]
    
    # Compare
    match_status = "MATCH" if pytorch_pred == onnx_pred else "MISMATCH"
    if pytorch_pred != onnx_pred:
        all_match = False
    
    print(f"Sample {sample_idx + 1}: PyTorch={pytorch_pred}  ONNX={onnx_pred}  {match_status}")

print("=" * 70)

# Report result
if not all_match:
    print("\n[FAIL] ERROR: PyTorch and ONNX outputs do NOT match!")
    print("Validation FAILED. Exiting with code 1.")
    exit(1)

# Success: print validation passed and file size
print("\n[PASS] VALIDATION PASSED: All 10 samples match between PyTorch and ONNX!")

file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"[PASS] Model file size: {file_size_mb:.2f} MB")
print(f"[PASS] ONNX export complete and validated: {onnx_path}")
