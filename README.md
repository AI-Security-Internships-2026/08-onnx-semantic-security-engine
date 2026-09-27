# SEMANTICSHIELD: Invariant-Aware and Drift-Calibrated Semantic Runtime Assurance for Edge NIDS in ONNX

[![Release](https://img.shields.io/badge/Release-paper--v1.0-blue.svg)](experiments/paper_results/RELEASE_METADATA.json)
[![Paper PDF](https://img.shields.io/badge/Manuscript-PDF-red.svg)](docs/paper/paper-draft.pdf)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified%20(89%2F89)-brightgreen.svg)](docs/paper/verify_numbers.py)
[![Tests](https://img.shields.io/badge/Tests-71%2F71%20Passed-brightgreen.svg)](tests/)
[![Config](https://img.shields.io/badge/Config-Frozen%20(paper__v1.yaml)-orange.svg)](configs/paper_v1.yaml)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> **CNIT / PNTLab Pisa · TECIP · Scuola Superiore Sant'Anna — AI Security Research**  
> Reference Implementation and Full Reproduction Artifacts for IEEE / ACM Publication.

---

## ⚡ Quick Start: 1-Line Verification & Reproduction

All reported experimental results, figures, tables, and mathematical claims can be verified immediately:

```bash
# 1. Verify all 18 JSON results, 22 figures, 19 tables, 23 checksums, and 89 claims
python experiments/reproduce_paper.py --verify

# 2. Run standalone 89-point numerical claim verification against paper-draft.tex
python docs/paper/verify_numbers.py

# 3. Verify cryptographic SHA-256 integrity of all model binaries and outputs
python scripts/download_artifacts.py --verify-checksums

# 4. Run the complete automated test suite (71 tests passing)
python -m pytest tests/ -v
```

📖 **For full from-scratch retraining, dataset acquisition, and benchmark pipelines, see [`REPRODUCE.md`](REPRODUCE.md).**

---

## 📄 Research Manuscript & Abstract

- **Full Paper PDF:** [**`docs/paper/paper-draft.pdf`**](docs/paper/paper-draft.pdf) *(12 pages, compiled from source)*
- **LaTeX Source:** [**`docs/paper/paper-draft.tex`**](docs/paper/paper-draft.tex)
- **Authoritative Provenance:** [**`experiments/paper_results/RELEASE_METADATA.json`**](experiments/paper_results/RELEASE_METADATA.json)

### Abstract
> Deep learning models for Network Intrusion Detection Systems (NIDS) are vulnerable to out-of-distribution (OOD) shift, adversarial perturbation, and deployment-time data pipeline corruptions. While extensive literature addresses OOD detection in high-performance cloud environments, edge-deployed NIDS operate under strict compute, memory, and false-alarm constraints where traditional heavy anomaly detectors are impractical. 
> 
> We introduce **SEMANTICSHIELD**, a lightweight, invariant-aware, and drift-calibrated runtime assurance architecture integrated directly into the ONNX Runtime execution graph. SEMANTICSHIELD couples deterministic pre-inference invariant validation (enforcing physical networking bounds, IEEE-754 integrity, and zero-fill ratio constraints) with dual-output intermediate representation extraction to evaluate class-conditional Cosine and Mahalanobis manifold drift.
> 
> Evaluated on 10 days of CSE-CIC-IDS2018 (in-distribution), NF-ToN-IoT-v2 (13.1M flows), and NF-BoT-IoT-v2 (30.4M flows), SEMANTICSHIELD achieves **0.9713 AUROC** under real covariate shift and **100% rejection** on malformed inputs at **1.19 ms p95 latency** (1,124 flows/s) under simulated edge container profiles.

---

## 🏛️ System Architecture

SEMANTICSHIELD sits as an inline runtime assurance layer guarding the exported ONNX classification model:

```
Raw Network Flow (13 NetFlow Features)
               │
               ▼
┌──────────────────────────────────────────────┐
│  Stage 1: Pre-Inference InputValidator      │
│  - IEEE-754 NaN / Inf Rejection              │  ──> [INVALID] ──> REJECTED
│  - Truncation / Zero-Fill Ratio Check (≥80%) │
│  - Physical Feature Bounds & Range Checks    │
└──────────────────────┬───────────────────────┘
                       │ [VALID]
                       ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Dual-Output ONNX Runtime Graph     │
│  - Threat Classifier Logits  z ∈ ℝ^K         │
│  - Hidden Representation e ∈ ℝ^64 (Layer 3)  │
└──────────────┬───────────────────────┬───────┘
               │                       │
               ▼                       ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│ Stage 3: Confidence Guard │ │ Stage 4: Manifold Drift   │
│ - MSP Confidence Scoring  │ │ - Cosine Manifold Drift   │
│ - Low Confidence Alert    │ │ - Mahalanobis Drift (Σ^-1)│
└──────────────┬────────────┘ └─────────────┬─────────────┘
               │                            │
               └──────────────┬─────────────┘
                              ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5: Calibrated Verdict Engine (5% Target FPR)     │
│  - REJECTED:    Pre-inference invariant failure         │
│  - HIGH_RISK:   ≥2 independent anomaly signals          │
│  - SUSPICIOUS:  1 anomaly signal                        │
│  - CLEAN:       All validation & drift checks nominal   │
└─────────────────────────────┬───────────────────────────┘
                              ▼
                    Assurance Verdict & Threat Alert
```

---

## 🔬 Mathematical Formulation

All evaluation scripts and runtime modules use the canonical mathematical formulation implemented in [`src/semantic_analyzer.py`](src/semantic_analyzer.py):

$$\text{MSP Confidence} = \max_{k \in \{1,\dots,K\}} \sigma(\mathbf{z})_k$$

$$d_{\cos}(\mathbf{e}) = \min_{c \in C} \left(1 - \frac{\mathbf{e} \cdot \boldsymbol{\mu}_c}{\|\mathbf{e}\|_2 \|\boldsymbol{\mu}_c\|_2}\right)$$

$$d_{\text{mahal}}(\mathbf{e}) = \min_{c \in C} \sqrt{(\mathbf{e} - \boldsymbol{\mu}_c)^T \mathbf{\Sigma}^{-1} (\mathbf{e} - \boldsymbol{\mu}_c)}$$

$$\text{Drift Score} = \min\left(\max\left(\frac{d_{\cos}(\mathbf{e})}{\tau_{\cos}}, \frac{d_{\text{mahal}}(\mathbf{e})}{\tau_{\text{mahal}}}\right), 2.0\right)$$

### Canonical Verdict Decision Logic

$$\text{Verdict} = \begin{cases}
\text{REJECTED} & \text{if } \neg \text{validation\_passed (schema, NaN/Inf, zero-fill)} \\
\text{HIGH\_RISK} & \text{if } \sum \text{alerts} \ge 2 \\
\text{SUSPICIOUS} & \text{if } \sum \text{alerts} = 1 \\
\text{CLEAN} & \text{if } \sum \text{alerts} = 0
\end{cases}$$

---

## 📊 Key Experimental Findings (All Verified)

| Evaluation Aspect | Metric Reported in Manuscript | Verification Check |
|---|---|:---:|
| **NIDS Classification (13 NetFlow Features)** | Accuracy: **87.19%**, Macro-F1: **78.01%**, Weighted-F1: **86.55%** | [OK] Match |
| **Clean Drift Shift (NF-ToN-IoT-v2)** | Mahalanobis AUROC: **0.9713**, AUPRC: **0.9639**, FPR@95TPR: **6.16%** | [OK] Match |
| **Pipeline Truncation (Zero-Fill 100%)** | InputValidator: **1.0000** (Full System: **1.0000**, MSP: **0.0000**) | [OK] Match |
| **Corrupted IEEE-754 Floats (NaN/Inf)** | Full System: **0.7928** (Validator: **0.6910**, MSP: **0.0000**) | [OK] Match |
| **Statistical Rigor (5-Seed Repeated)** | In-Dist Clean: **86.94% ± 0.44%**, OOD Intercept: **61.72%**, Noise Intercept: **100%** | [OK] Match |
| **Simulated Edge Overhead (R0 Profile)** | Plain ONNX p95: **0.386 ms** $\to$ SEMANTICSHIELD p95: **1.192 ms** (1,124 flows/s) | [OK] Match |
| **Quantization Trade-off (FP32 $\to$ INT4)** | FP32 (0.176 MB, 78.01% F1) $\to$ INT4 (0.034 MB, 76.27% F1, **80.7% compression**) | [OK] Match |

---

## 🛠️ Repository Organization

```
├── configs/
│   └── paper_v1.yaml               # Authoritative frozen configuration (single source of truth)
├── datasets/
│   ├── README.md                   # Complete dataset acquisition, schemas & governance guide
│   ├── CSE-CIC-IDS2018.md          # In-distribution training & evaluation data
│   ├── NF-ToN-IoT-v2.md            # Out-of-distribution real covariate shift
│   └── NF-BoT-IoT-v2.md            # Standardized cross-dataset evaluation
├── docs/
│   ├── paper/
│   │   ├── paper-draft.tex         # LaTeX publication manuscript source (IEEE/ACM)
│   │   ├── paper-draft.pdf         # Compiled 12-page PDF publication paper
│   │   ├── verify_numbers.py       # 89-point automated numeric assertion suite
│   │   └── figures/                # Synchronized publication figures
│   └── weekly-progress.md          # Complete project history and milestone progression
├── experiments/
│   ├── checksums.sha256            # Cryptographic SHA-256 hashes of all 23 core artifacts
│   ├── reproduce_paper.py          # Unified 15-stage automated reproduction orchestrator
│   ├── calibration_config.json     # Calibrated empirical decision thresholds
│   ├── standard_scaler_nf.joblib   # Frozen StandardScaler on CIC-IDS2018 training split
│   ├── reference_embeddings_nf.npz # 64-dim training manifold centroids & covariance
│   ├── training_feature_stats_nf.json # Empirical feature means, sigmas & physical ranges
│   ├── X_test_nf.npy, y_test_nf.npy# Preprocessed test partition (138,069 flows)
│   └── paper_results/
│       ├── RELEASE_METADATA.json   # Hardware, commit, seed, and environment provenance
│       ├── json/                   # 18 machine-readable JSON result datasets
│       ├── figures/                # 22 publication PNG figures
│       └── tables/                 # 19 canonical manuscript CSV tables
├── scripts/
│   ├── download_artifacts.py       # Artifact verification & integrity manager
│   ├── generate_paper_tables.py    # Direct CSV table generator from JSONs
│   ├── fixed_fpr_evaluation.py     # Multi-threshold failure-mode evaluation
│   ├── cross_dataset_evaluation.py # Same-schema cross-dataset domain shift
│   ├── benchmark_resource_simulation.py # Simulated edge cgroup profiles (R0-R3)
│   └── ...                         # Dedicated experiment drivers
├── src/
│   ├── semantic_analyzer.py        # Production SEMANTICSHIELD engine implementation
│   ├── inference_engine.py         # FastAPI edge inference server
│   ├── train_classifier.py         # Neural model training (ThreatMLP / ThreatCNN1D)
│   ├── export_onnx.py              # Dual-output ONNX graph export
│   └── quantize_model.py           # FP16 / INT8 / INT4 quantization
├── tests/
│   ├── test_semantic.py            # Unit test suite for assurance modules
│   ├── test_runtime_eval_equivalence.py # Runtime vs eval equivalence guarantees
│   ├── test_resource_simulation.py # Simulated cgroups resource profiling tests
│   └── test_semantic_audit.py      # Feature alignment & schema consistency tests
└── REPRODUCE.md                    # Master step-by-step reproduction guide
```

---

## 💻 Installation & Dependencies

```bash
# 1. Clone repository
git clone https://github.com/AI-Security-Internships-2026/08-onnx-semantic-security-engine.git
cd 08-onnx-semantic-security-engine

# 2. Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify test suite
python -m pytest tests/ -v
```

---

## 🛡️ License & Academic Citation

This project is licensed under the **MIT License**.

If you use SEMANTICSHIELD in your academic research, please cite:

```bibtex
@article{semanticshield2026,
  title={{SEMANTICSHIELD}: Invariant-Aware and Drift-Calibrated Semantic Runtime Assurance for Edge {NIDS} in {ONNX}},
  author={Hussain, Sikandar and Research Team},
  journal={arXiv preprint},
  year={2026},
  note={Artifact Release Tag: paper-v1.0}
}
```
