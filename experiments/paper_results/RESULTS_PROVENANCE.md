# SEMANTICSHIELD Paper Results Provenance & Discrepancy Audit

This document records the exact provenance of all experimental results in the SEMANTICSHIELD manuscript, explains why certain numerical values changed across project milestones, and establishes the canonical baseline for paper reproducibility.

---

## 1. Frozen Baseline Configuration

All canonical results in `experiments/paper_results/` are governed by `configs/paper_v1.yaml`:

- **Config Version:** `paper_v1` (Frozen: `2026-09-11`)
- **Model Architecture:** ThreatMLP NF-Standardized (13 features, 15 classes, BatchNorm, Dropout 0.3)
- **Primary ONNX Artifact:** `experiments/threat_mlp_nf_fp32.onnx`
- **Trained Parameters:** 46,542 (linear layer weights/biases)
- **Calibrated Decision Thresholds (5% Target FPR on CIC-IDS2018 Benign Validation):**
  - Softmax Confidence ($1 - \text{MSP}$ threshold): $\tau_{\text{conf}} = 0.4743$
  - Cosine Distance Threshold: $\tau_{\text{cos}} = 0.4341$
  - Mahalanobis Distance Threshold: $\tau_{\text{mahal}} = 18.1593$
  - Z-Score Range Outlier Threshold: $\tau_z = 15.0$
  - Zero-Fill Tampering Detection Ratio: $\tau_{\text{zero}} = 0.80$
- **Evaluation Protocols:**
  - Multi-seed evaluation seeds: `[42, 123, 456, 789, 1024]`
  - Calibration partition: $N = 10,000$ benign samples
  - Evaluation partition: $N = 10,000$ in-distribution samples, $N = 10,000$ OOD samples
  - Evaluation OOD dataset: CSE-CIC-IDS2018 $\to$ ToN-IoT (`NF-ToN-IoT-V2.parquet`)

---

## 2. Root Cause Analysis: Why Historical Values Changed

During development, multiple intermediate scripts produced conflicting metrics. The root causes and their resolutions are documented below:

| Dimension | Historical Status (Stale / Conflicted) | Canonical Status (Paper Baseline) | Technical Explanation & Root Cause |
|---|---|---|---|
| **Feature Schema Definition** | Legacy scripts referenced "21 NetFlow features" | Frozen **13-feature schema** (`FEATURE_MAP`) | Post-hoc feature audit revealed that 8 of 21 NetFlow features had physical semantic mismatches (e.g. throughput rate mapped to header byte counts). The schema was cleaned to 13 physically comparable features with protocol preservation. |
| **Classification Report Metadata** | `nf_classification_report.json` stated "21 features" in title string | `classifier_metrics.json` reflects 13 features | Test data `X_test_nf.npy` was shape `(138069, 13)` and evaluated on the 13-feature ONNX model; the old report contained a leftover string literal from the 21-feature prototyping era. |
| **Threshold Source of Truth** | Split between `calibration_config.json`, hard-coded fallbacks (`0.4341`, `18.1593`, `0.50`), and `paper_v1.yaml` | All thresholds centralized in `configs/paper_v1.yaml` via `config_loader.py` | Issue 1 eliminated all hard-coded fallbacks and replaced ad-hoc JSON reads with centralized configuration loading. |
| **Composite OOD Discrimination** | Early draft reported 0.966 AUROC; subsequent corrected evaluation reported 0.938 AUROC | Canonical AUROC: **0.938 - 0.977** (Method-specific) | Early prototype calculated AUROC on an uncalibrated composite heuristic. Standardizing on canonical Mahalanobis distance yields 0.959-0.977 AUROC, while the composite decision score with clipping yields 0.938 AUROC. |
| **Zero-Fill Evaluation AUROC** | Early draft claimed 1.000 AUROC for raw model | Canonical: **1.000** for engine; 0.835-0.917 for raw softmax model alone | A raw neural network without input validation fails on zero-filled inputs (AUROC 0.835-0.917) because zero vectors trigger false-confident default predictions. The full SEMANTICSHIELD engine achieves 100% interception via the InputValidator component. |
| **Latency Measurements** | Varied across environments (0.056ms vs 0.068ms for Plain ONNX; 0.346ms vs 0.664ms for Engine) | Documented with hardware specs, mean $\pm$ std, p50, p95, p99 | Inference latency varies with CPU frequency scaling and background OS thread scheduling. The paper reports CPU benchmarks (5,000 runs) with explicit confidence intervals. |
| **Cross-Dataset Generalization** | Varied reporting between binary F1 (0.77 - 0.80) and multi-class macro-F1 (0.01 - 0.05) | Disambiguated in `cross_dataset_alignment.json` | Network intrusion models transfer moderately well for binary threat detection ($\approx 0.77$ F1) but suffer catastrophic collapse under fine-grained multi-class transfer ($\approx 0.01$ - $0.05$ macro-F1) due to covariate shift and label semantic differences. |

---

## 3. Discrepancy Resolution Table

The table below reconciles all historical mismatches previously flagged by `verify_numbers.py`:

| Metric Claimed in Early Draft | Value in Stale JSON | Canonical Value in `paper_results/` | Resolution & Status |
|---|---|---|---|
| Plain ONNX Latency | 0.056 ms | 0.068 ms (mean) / 0.050 ms (ONNX test) | Hardware execution variance across test environments; reported as $0.068 \pm 0.018$ ms (p95: 0.106 ms). |
| Semantic Engine Latency | 0.346 ms | 0.664 ms (mean, 5000 runs) | Comprehensive 5,000-run benchmark under Windows CPU scheduling yields 0.664 ms (under 1.0 ms edge budget). |
| Latency Overhead | 0.289 ms | 0.596 ms | Overhead represents 3-tier validation + dual drift metrics + verdict logic. |
| Isolation Forest Latency | 11.71 ms | 12.29 ms | Tree traversal execution variation; confirmed IF is $>15\times$ slower than SEMANTICSHIELD. |
| Engine ToN-IoT AUROC | 0.966 | 0.938 (composite) / 0.959 (Mahalanobis) | Corrected scoring with canonical drift clipping in `src/semantic_analyzer.py`. |
| Engine Noise AUROC | 1.000 | 0.999 | Corrected numerical precision over 1,000 noise samples. |
| Engine Zero-Fill AUROC | 1.000 | 0.835 (raw model) / 100.0% rejected (engine) | Clarified distinction between raw classifier vulnerability and engine structural validation. |
| Engine Latency in OOD Baseline | 0.380 ms | 0.637 ms | Single-sample measurement reconciled with multi-iteration statistical rigor benchmark. |

---

## 4. Canonical Artifact Registry

| Canonical Result File | Generation Script | Purpose |
|---|---|---|
| `json/classifier_metrics.json` | `scripts/generate_classifier_metrics.py` | In-distribution accuracy, macro/weighted F1, per-class metrics, confusion matrix array |
| `json/cross_model_comparison.json` | `scripts/cross_model_comparison.py` | 76-feature baseline vs 13-feature NF model comparison across complexity, storage, and throughput |
| `json/ablation_study.json` | `scripts/ablation_study.py` | Systematic ablation across all 7 component subsets (InputValidator, Confidence, Drift) |
| `json/quantization_benchmark.json` | `scripts/benchmark_quantization.py` | FP32, FP16, INT8, INT4 quantization metrics, size drops, and throughput |
| `json/ood_baselines_benchmark.json` | `scripts/benchmark_ood_baselines.py` | OOD baseline comparisons (MSP, Mahalanobis, Isolation Forest, OCSVM, SEMANTICSHIELD) |
| `json/semantic_engine_evaluation.json` | `scripts/evaluate_semantic_engine.py` | Full calibrated engine integration evaluation across 4 test scenarios |
| `json/statistical_rigor_benchmark.json` | `scripts/statistical_rigor_benchmark.py` | 5-seed repeated benchmark with mean $\pm$ std, 95% CIs, and FPR@95TPR |
| `json/training_time_benchmark.json` | `scripts/benchmark_training_time.py` | Scalability and training efficiency across 76, 21, and 13 feature schemas |
| `json/latency_benchmark.json` | `scripts/benchmark_realtime_vs_offline.py` | Latency decomposition across offline in-memory and real-time HTTP streaming modes |
| `json/cross_dataset_alignment.json` | `scripts/cross_dataset_alignment_audit.json` | Cross-dataset domain transfer evaluation (CORAL, re-standardization, raw transfer) |
