# SEMANTICSHIELD Paper Reproduction Guide

This guide provides complete, step-by-step instructions to reproduce all experimental findings, machine-readable datasets, canonical CSV tables, publication figures, and the compiled PDF manuscript reported in:

> **"SEMANTICSHIELD: Invariant-Aware and Drift-Calibrated Semantic Runtime Assurance for Edge NIDS in ONNX"**  
> *Release Tag:* `paper-v1.0` | *Frozen Config:* `configs/paper_v1.yaml`

---

## Quick Start: One-Command Verification

To verify the integrity and exact numeric agreement of all 89 reported paper numbers, 18 JSON results, 22 figures, 19 tables, and 23 cryptographic artifact hashes:

```bash
# 1. Run complete paper verification suite (JSONs, figures, tables, hashes, claims)
python experiments/reproduce_paper.py --verify

# 2. Run standalone 89-point mathematical assertion suite
python docs/paper/verify_numbers.py

# 3. Verify cryptographic SHA-256 integrity of all model binaries and outputs
python scripts/download_artifacts.py --verify-checksums

# 4. Run PyTest test suite (71 tests)
python -m pytest tests/ -v
```

---

## 1. System Requirements & Environment Setup

### 1.1 Hardware Specifications
- **Host System:** Windows 10/11 or Linux x86_64 (tested on AMD64, 6 physical cores / 12 logical threads, 16 GB RAM).
- **Disk Space:**
  - Fast Reproduction / Pre-packaged test partitions: ~500 MB.
  - Full End-to-End Retraining (raw multi-day captures): ~60 GB.
- **Simulated Edge Environments:**
  - Experiments in Section 5.4 evaluate simulated resource-constrained profiles using ONNX Runtime thread affinity controls (NOT physical edge hardware):
    - **R0 (Reference):** Unconstrained host CPU & RAM (baseline).
    - **R1 (Constrained):** 1 vCPU, 512 MB RAM — strongly constrained edge profile.
    - **R2 (Moderate):** 2 vCPUs, 1024 MB RAM — moderately constrained edge controller.
    - **R3 (Higher):** 4 vCPUs, 2048 MB RAM — less constrained edge gateway.
  - **Note:** These profiles simulate resource constraints via ONNX Runtime `intra_op_num_threads` / `inter_op_num_threads` controls on a commodity x86_64 host. They do NOT reproduce physical ARM microarchitectures, NPU/TPU cores, thermal throttling, or NIC line-rate capture. Physical edge validation remains future work.

### 1.2 Python Environment Setup
We recommend Python 3.10 to 3.14.

```bash
# Clone repository
git clone https://github.com/<org>/08-onnx-semantic-security-engine.git
cd 08-onnx-semantic-security-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install exact frozen dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.3 LaTeX Manuscript Tooling (Optional for PDF Compilation)
To recompile `docs/paper/paper-draft.pdf` directly without installing a 5 GB TeXLive distribution:
- A standalone portable Tectonic binary can be placed in `tools/tectonic.exe` (or installed via `cargo install tectonic` / `brew install tectonic`).
- Compile command: `.\tools\tectonic.exe docs\paper\paper-draft.tex` (or `tectonic docs/paper/paper-draft.tex`).

---

## 2. End-to-End Reproduction Pipeline

The complete pipeline consists of 8 distinct phases, fully automated and config-driven:

```
[Acquire Datasets] ──> [Preprocess & Split] ──> [Train ThreatMLP / CNN1D]
                                                         │
[Empirical Calibration] <── [Export ONNX + Embeddings] <─┘
         │
[Run Benchmarks (15 Stages)] ──> [Export CSV Tables] ──> [Compile PDF]
```

### Phase 1: Dataset Acquisition
Refer to [`datasets/README.md`](datasets/README.md) for full acquisition links.
- **In-Distribution (ID):** Download CSE-CIC-IDS2018 10-day Parquet captures into `datasets/CIC-IDS2018/`.
- **Out-of-Distribution (OOD):** Download `NF-ToN-IoT-V2.parquet` into `datasets/ToN-IoT/` and `NF-BoT-IoT-V2.parquet` into `datasets/NF-BoT-IoT-V2/`.
- *Note:* If you already have pre-packaged `experiments/X_test_nf.npy` and `experiments/y_test_nf.npy`, you can skip Phase 1 and Phase 2.

### Phase 2: Feature Standardization & Preprocessing
Extracts the 13 canonical NetFlow features from raw flow telemetry and standardizes via `StandardScaler`:
```bash
# Verify preprocessed test partitions
python scripts/download_artifacts.py --check
```

### Phase 3: Train Neural Threat Classifiers
Trains the ThreatMLP and ThreatCNN1D models on the 13 standardized NetFlow features:
```bash
# Train primary ThreatMLP (~46K parameters)
python src/train_classifier.py --nf --arch mlp --epochs 30 --batch-size 1024

# Train replication ThreatCNN1D (~34K parameters)
python src/train_classifier.py --nf --arch cnn1d --epochs 30 --batch-size 1024
```
Outputs saved to `experiments/threat_mlp_nf.pth` and `experiments/threat_cnn1d_nf.pth`.

### Phase 4: ONNX Dual-Output Export
Exports PyTorch weights to ONNX format with dual output graph (Logits + Layer 3 Embeddings $\in \mathbb{R}^{64}$):
```bash
# Export ThreatMLP dual-output ONNX
python src/export_onnx.py --nf --arch mlp --with-embeddings

# Export ThreatCNN1D dual-output ONNX
python src/export_onnx.py --nf --arch cnn1d --with-embeddings
```
Outputs: `experiments/threat_mlp_nf_fp32.onnx` and `experiments/threat_cnn1d_nf_fp32.onnx`.

### Phase 5: Quantization Benchmark (FP32 / FP16 / INT8 / INT4)
Performs static calibration and generates reduced-precision ONNX binaries:
```bash
python src/quantize_model.py --model experiments/threat_mlp_nf_fp32.onnx --test-data experiments/X_test_nf.npy
```
Outputs: `threat_mlp_nf_fp16.onnx`, `threat_mlp_nf_int8.onnx`, `threat_mlp_nf_int4.onnx`.

### Phase 6: Reference Embeddings & Feature Statistics Extraction
Extracts training centroid $\mathbf{c} \in \mathbb{R}^{64}$, covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{64 \times 64}$, and per-feature empirical bounds:
```bash
python src/embedding_reference.py --nf --arch mlp
python src/embedding_reference.py --nf --arch cnn1d
```
Outputs: `reference_embeddings_nf.npz`, `training_feature_stats_nf.json`.

### Phase 7: Empirical Percentile Calibration
Calibrates SEMANTICSHIELD thresholds ($\tau_{\text{conf}}, \tau_{\text{cos}}, \tau_{\text{mahal}}$) on 10,000 held-out benign samples at a 5% target FPR budget:
```bash
python scripts/evaluate_semantic_engine.py
```
Output: `experiments/calibration_config.json`.

---

## 3. Running Paper Experiments & Reproducing Tables/Figures

The unified orchestrator [`experiments/reproduce_paper.py`](experiments/reproduce_paper.py) drives all experimental benchmarks:

### 3.1 Execution Modes

```bash
# Dry run: Validate execution pipeline and check results without rerunning
python experiments/reproduce_paper.py --dry-run

# Verification mode: Full integrity and number validation
python experiments/reproduce_paper.py --verify

# Fast mode: Run only quick analytical stages (skip heavy bootstrap / full dataset scans)
python experiments/reproduce_paper.py --skip-slow

# Run a single stage by identifier
python experiments/reproduce_paper.py --only classifier
python experiments/reproduce_paper.py --only fixed_fpr
python experiments/reproduce_paper.py --only resource_simulation
```

### 3.2 Full Stage Registry

| Stage ID | Benchmark Description | Primary Script | Output JSON | Output Figure / Table |
|---|---|---|---|---|
| `classifier` | NIDS Classification & Confusion Matrix | `scripts/generate_classifier_metrics.py` | `classifier_metrics.json` | `confusion_matrix_cic.png` |
| `cross_model` | 76 vs 13 Feature Comparison | `scripts/cross_model_comparison.py` | `cross_model_comparison.json` | `cross_model_comparison.png` |
| `ablation` | Component Ablation Study | `scripts/ablation_study.py` | `ablation_study.json` | `ablation_comparison.png` |
| `quantization` | FP32/FP16/INT8/INT4 Latency & Drop | `scripts/benchmark_quantization.py` | `quantization_benchmark.json` | `quantization_benchmark.png` |
| `semantic_eval` | Calibrated Integration Evaluation | `scripts/evaluate_semantic_engine.py` | `semantic_engine_evaluation.json` | `semantic_engine_evaluation_plots.png` |
| `ood_baselines` | Baseline Detectors (MSP, Mahal, IF, OCSVM) | `scripts/benchmark_ood_baselines.py` | `ood_baselines_benchmark.json` | `ood_baselines_comparison.png` |
| `statistical_rigor` | 5-Seed Repeated Evaluation & 95% CIs | `scripts/statistical_rigor_benchmark.py` | `statistical_rigor_benchmark.json` | `statistical_rigor_plots.png` |
| `training_time` | Training Scalability Across Schemas | `scripts/benchmark_training_time.py` | `training_time_benchmark.json` | `training_time_comparison.png` |
| `int8_investigation` | INT8 Outlier Clipping & Quantization Analysis | `scripts/investigate_int8_quantization.py` | `int8_degradation_investigation.json` | `int8_activation_analysis.png` |
| `resource_simulation` | Simulated Edge Container Constraints (R0-R3) | `scripts/benchmark_resource_simulation.py` | `resource_simulation_benchmark.json` | `resource_simulation_p95_latency.png` |
| `fixed_fpr` | Multi-Threshold Failure Modes (0.1%, 1%, 5% FPR) | `scripts/fixed_fpr_evaluation.py` | `fixed_fpr_evaluation.json` | `detection_vs_fpr_budget.png`, `failure_mode_heatmap.png` |
| `cross_dataset` | Same-Schema Cross-Dataset Shift (ToN-IoT, BoT-IoT) | `scripts/cross_dataset_evaluation.py` | `cross_dataset_generalization.json` | `cross_dataset_performance_drop.png` |
| `semantic_mismatch` | Semantic / Feature-Extractor Mismatch (Tiers 1-3) | `scripts/semantic_mismatch_evaluation.py` | `semantic_mismatch_sensitivity.json` | `semantic_mapping_sensitivity.png` |
| `cross_model_replication` | Architecture Replication (MLP vs CNN1D) | `scripts/cross_model_replication.py` | `cross_model_replication.json` | `table_cross_model_replication.csv` |
| `paper_tables` | Export All 19 Canonical CSV Tables | `scripts/generate_paper_tables.py` | `table1_classification_performance.csv` | `experiments/paper_results/tables/*.csv` |

---

## 4. Re-Generating Canonical CSV Tables & Paper Figures

To re-export all 19 canonical CSV tables directly from machine-readable JSON datasets:
```bash
python scripts/generate_paper_tables.py
```
Outputs are written to `experiments/paper_results/tables/`.

To recompile the LaTeX manuscript:
```bash
.\tools\tectonic.exe docs\paper\paper-draft.tex
```
Output: `docs/paper/paper-draft.pdf`.

---

## 5. Provenance & Artifact Integrity

All release parameters are frozen in:
- Configuration: [`configs/paper_v1.yaml`](configs/paper_v1.yaml)
- Release Metadata: [`experiments/paper_results/RELEASE_METADATA.json`](experiments/paper_results/RELEASE_METADATA.json)
- Checksums Manifest: [`experiments/checksums.sha256`](experiments/checksums.sha256)

To verify that your local environment matches the exact published state:
```bash
python scripts/download_artifacts.py --verify-checksums
```
If any file differs, the script will output a `[MISMATCH]` alert identifying the changed byte hash.
