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
| `json/resource_simulation_benchmark.json` | `scripts/benchmark_resource_simulation.py` | Edge resource constraint simulation (R0-R3 profiles, Plain vs Engine, concurrency, quantization) |
| `json/int8_degradation_investigation.json` | `scripts/investigate_int8_quantization.py` | Layer-wise activation dynamic ranges, outlier clipping, and INT8 degradation empirical analysis |

---

## 5. Edge Resource Constraints Simulation Provenance (Issue 5 / E2.3-A-E)

### 5.1 Controlled Simulation vs Physical Edge Guardrail
To maintain scientific integrity, the manuscript strictly avoids claiming physical edge hardware validation (e.g. Raspberry Pi, Jetson Nano). Instead, the paper frames these experiments as:
> *"We evaluate SEMANTICSHIELD under controlled CPU- and memory-constrained deployment profiles to approximate resource-limited inference conditions. Physical edge hardware validation remains future work."*

### 5.2 Resource Profiles & Execution Parameters
Evaluated via Docker cgroups v2 resource quotas on Linux container runtime (`docker-security-engine:latest`):
- **Profile R0 (Reference):** Unconstrained vCPU, unconstrained RAM. Host baseline.
- **Profile R1 (Low):** 1 vCPU (`--cpus 1.0`), 512 MB RAM (`--memory 512m`). Emulates strongly constrained IoT/edge gateway.
- **Profile R2 (Medium):** 2 vCPU (`--cpus 2.0`), 1024 MB RAM (`--memory 1024m`). Emulates industrial controller.
- **Profile R3 (Higher):** 4 vCPU (`--cpus 4.0`), 2048 MB RAM (`--memory 2048m`). Emulates high-end edge appliance.

### 5.3 Runtime Overhead & Resource Sensitivity Findings
Across 5 repeated measured runs (5,000 flows/run, 500-flow warm-up) per profile:
- **Plain ONNX Latency:** Stable at $0.2705 - 0.2988$ ms (Mean), $0.2533 - 0.2692$ ms (p50), $0.3473 - 0.4280$ ms (p95), yielding $3,354 - 3,680$ flows/s throughput.
- **SEMANTICSHIELD Latency:** Consistently scales to $0.8883 - 0.9309$ ms (Mean), $0.8324 - 0.8528$ ms (p50), $1.2790 - 1.4049$ ms (p95), maintaining $1,073 - 1,124$ flows/s throughput.
- **Absolute Assurance Overhead:** Fixed at $+0.61 - +0.64$ ms across all resource profiles, demonstrating that assurance computational complexity remains constant and bounded even under tight CPU constraints (1 vCPU).
- **Memory Footprint:** Peak RSS remains essentially flat ($122.4 - 124.9$ MB) across both Plain ONNX and SEMANTICSHIELD, operating well within the 512 MB memory boundary of Profile R1.

### 5.4 INT8 Degradation Investigation Findings (E2.3-E)
- **Empirical Observation:** Post-training static INT8 quantization (`threat_mlp_nf_int8.onnx`, 47.9 KB) suffers a severe drop in Macro-F1 ($0.7801 \to 0.4833$, a $38.0\%$ drop), whereas weight-only INT4 quantization (`threat_mlp_nf_int4.onnx`, 34.0 KB) preserves $0.7627$ Macro-F1 ($97.8\%$ retention).
- **Root Cause Evidence:**
  1. *Activation Range Skewness & Clipping:* Intermediate post-ReLU activations exhibit severe positive skewness where 99th percentile activations are $< 12.0$ but extreme burst outliers reach $> 50.0$.
  2. *Calibration Compression:* Static `QUInt8` quantization maps the full range $[0, 255]$ using a linear scale factor ($0.167 - 4.228$). When test flows containing extreme attack traffic are processed, activations saturate and clip at 255, destroying discriminative boundary margins for rare attack classes.
  3. *Weight vs Activation Decoupling:* Because weight-only INT4 retains 0.7627 Macro-F1, the degradation is conclusively proven to stem from activation quantization and outlier clipping, not weight precision reduction.

---

## 6. Semantic Feature Audit & Literature Gap Analysis Provenance (Issue 6)

### 6.1 Semantic Feature Audit & Four-Tier Taxonomy
- **Scope:** Complete per-field physical and mathematical audit across all 21 candidate flow feature pairs between NetFlow v9 / IPFIX (nProbe v9) and CICFlowMeter v4 (CSE-CIC-IDS2018).
- **Four-Tier Classification:**
  - **Equivalent (7 pairs, 33.3%):** Identical physical quantity, unit, directionality, and computation logic. Governed by explicit RFC 7012 Information Elements (e.g. IE 86, IE 87, IE 85, IE 23, IE 4).
  - **Convertible (1 pair, 4.8%):** Identical physical duration differing solely by a known constant linear scaling factor (ms vs $\mu\text{s}$, convertible via $10^{-3}$ scaling).
  - **Approximate (5 pairs, 23.8%):** Compatible physical dimensions with documented directional scope or aggregation window variations.
  - **Incompatible (8 pairs, 38.1%):** Severe dimensional, state-representation, or semantic phenomenon mismatches (e.g. bitrate in bps mapped to header bytes, or cumulative 8-bit TCP control bitmask mapped to discrete forward PSH packet counter).
- **Inter-Rater Reliability:** Dual-review protocol between Reviewer 1 (Network Protocols / RFC specialist) and Reviewer 2 (ML / Data Engineering specialist) yielded $P_o = 1.00$, $P_e = 0.3152$, and Cohen's Kappa $\kappa = 1.00$, grounded in RFC 7012, RFC 793, nProbe manuals, and CICFlowMeter source code (`FlowFeature.java`).
- **Canonical Files:** `docs/semantic_feature_audit.csv`, `docs/semantic_feature_audit.md`, and `experiments/paper_results/tables/table_semantic_feature_audit.csv`.

### 6.2 Prior-Work Comparison Matrix & Eight Thematic Pillars
- **Comparison Scope:** 8 systems (Kitsune, McLaughlin, NetSight, Sarhan, Cantone, Jajal, Yang, and SEMANTICSHIELD) evaluated across 8 operational and methodology columns.
- **Thematic Structuring:** 8 pillars covering OOD foundations, Deep NIDS & base-rate constraints, telemetry standards, cross-dataset collapse, edge ML & quantization, ONNX interoperability, adversarial robustness, and SEMANTICSHIELD positioning.
- **Canonical Files:** `docs/prior_work_comparison.md` and `experiments/paper_results/tables/table_prior_work_comparison.csv`.

### 6.3 Reference Audit & Bibliography Hardening
- **Author Corrections:** Fixed placeholder `{Various Authors}` in `quantedge2023` with verified author team (Hyunho Ahn et al., arXiv:2303.05016). Updated all author placeholders in `docs/literature-review.md`.
- **BibTeX Syntax:** Converted `mitre2024` from incomplete `@inproceedings` to `@misc`.
- **Foundational Additions:** Added Axelsson (ACM CCS 1999, base-rate fallacy), Sommer & Paxson (IEEE S&P 2010, closed world), Hofstede et al. (IEEE Surveys 2014, flow monitoring), RFC 7012, RFC 793, Handigol et al. (NSDI 2014), and Sastry & Oore (ICML 2020).
- **Documentation:** `docs/reference_audit_report.md` and `docs/paper/references.bib`.


