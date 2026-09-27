# SEMANTICSHIELD Manuscript Artifact Manifest

This document provides the authoritative mapping between every table, figure, and empirical claim in the research manuscript (`docs/paper/paper-draft.tex` and `docs/paper/paper-draft.pdf`) and its exact generating script, configuration file, and canonical JSON result artifact in `experiments/paper_results/`.

---

## 1. Main Manuscript Tables Mapping (Tables I – V)

| Manuscript Table | Description | Generating Script | Primary JSON / Data Source | Canonical CSV Table |
|---|---|---|---|---|
| **Table I: Datasets, Models, and Experimental Protocols** | Class distribution, in-distribution test metrics (Accuracy 87.19%, Macro-F1 78.01%, Weighted-F1 86.55%), 13 NetFlow features, and 5-seed baseline protocol | `scripts/generate_classifier_metrics.py` | `experiments/paper_results/json/classifier_metrics.json`<br>`RELEASE_METADATA.json` | `tables/table1_classification_performance.csv` |
| **Table II: Main Fixed-FPR OOD Detection Benchmark** | AUROC, AUPRC, FPR@95TPR, and TPR at 0.1%, 1.0%, 5.0% FPR for MSP, Cosine, Mahalanobis, Composite Drift, and Full Assurance on NF-ToN-IoT-v2 | `scripts/fixed_fpr_evaluation.py` | `experiments/paper_results/json/fixed_fpr_evaluation.json` | `tables/table_main_ood_benchmark.csv` |
| **Table III: Multi-Failure-Mode Coverage Matrix & Architectural Component Ablation** | Detection rates across 8 failure modes (E1 Clean, E2 Real Shift, E3 NaN/Inf, E4 Zero-Fill, E5 Permutation, E6 Unit Scale, E7 Outliers, E8 Gaussian Noise) at 1% FPR + systematic ablation (Full, −MSP, −Cosine, −Mahalanobis, −Validator) | `scripts/fixed_fpr_evaluation.py` | `experiments/paper_results/json/failure_mode_coverage_matrix.json`<br>`experiments/paper_results/json/ablation_fixed_fpr.json` | `tables/table_failure_mode_coverage.csv`<br>`tables/table_ablation_fixed_fpr.csv` |
| **Table IV: Cross-Dataset & Cross-Model Generalization** | Generalization across Track A (same-schema NF-BoT-IoT-v2) and Track B (mismatch NF-ToN-IoT-v2), ThreatMLP vs ThreatCNN1D, and 3 feature tiers | `scripts/cross_dataset_evaluation.py`<br>`scripts/cross_model_replication.py`<br>`scripts/semantic_mismatch_evaluation.py` | `experiments/paper_results/json/cross_dataset_generalization.json`<br>`experiments/paper_results/json/cross_model_replication.json`<br>`experiments/paper_results/json/semantic_mismatch_sensitivity.json` | `tables/table_cross_dataset_generalization.csv`<br>`tables/table_cross_model_replication.csv`<br>`tables/table_semantic_mismatch_sensitivity.csv` |
| **Table V: Containerized Edge Resource Constraints & Quantization Overhead** | Plain ONNX vs SEMANTICSHIELD mean/p50/p95/p99 latency, throughput, peak RSS across R0–R3 profiles + FP32, FP16, INT8, INT4 quantization trade-offs under R1 | `scripts/benchmark_resource_simulation.py`<br>`scripts/investigate_int8_quantization.py` | `experiments/paper_results/json/resource_simulation_benchmark.json`<br>`experiments/paper_results/json/int8_degradation_investigation.json` | `tables/table_resource_profiles.csv`<br>`tables/table_runtime_overhead.csv`<br>`tables/table_quantization_tradeoff.csv` |

---

## 2. Main Manuscript Figures Mapping (Figures 1 – 6)

| Manuscript Figure | Description | Generating Script | Canonical Figure Artifact | Source JSON Data |
|---|---|---|---|---|
| **Figure 1: System Architecture** | Invariant-aware dual-output ONNX runtime assurance workflow | Architecture specification | `docs/paper/figures/confusion_matrix_cic.png` *(and Section III architectural vector)* | `configs/paper_v1.yaml` |
| **Figure 2: Fixed-FPR Behavior & ROC Curves** | 2-panel comparison: Detection rate vs false-alarm budget (0.1%–5% FPR) and OOD ROC curves on NF-ToN-IoT-v2 | `scripts/fixed_fpr_evaluation.py` | `docs/paper/figures/detection_vs_fpr_budget.png`<br>`docs/paper/figures/roc_curves_ood.png` | `fixed_fpr_evaluation.json["roc_data"]` |
| **Figure 3: Failure-Mode Coverage Heatmap & Severity** | Failure-mode coverage matrix heatmap across E2–E8 and partial zero-fill corruption severity curves (10% to 100%) | `scripts/fixed_fpr_evaluation.py` | `docs/paper/figures/failure_mode_heatmap.png`<br>`docs/paper/figures/corruption_severity.png`<br>`docs/paper/figures/ablation_impact.png` | `failure_mode_coverage_matrix.json`<br>`ablation_fixed_fpr.json` |
| **Figure 4: Cross-Dataset & Cross-Model Generalization** | Cross-dataset F1 drop, detector AUROC across datasets, and mapping tier sensitivity (Tier 1 vs Tier 2 vs Tier 3) | `scripts/cross_dataset_evaluation.py`<br>`scripts/semantic_mismatch_evaluation.py` | `docs/paper/figures/cross_dataset_performance_drop.png`<br>`docs/paper/figures/detector_generalization_across_datasets.png`<br>`docs/paper/figures/semantic_mapping_sensitivity.png` | `cross_dataset_generalization.json`<br>`semantic_mismatch_sensitivity.json` |
| **Figure 5: Simulated Edge Container Profiles (R0–R3)** | Single-flow p95 latency and throughput scaling across R0 (Unconstrained), R1 (Moderate), R2 (Constrained), R3 (Extreme Gateway) + concurrency scaling | `scripts/benchmark_resource_simulation.py` | `docs/paper/figures/resource_simulation_p95_latency.png`<br>`docs/paper/figures/resource_simulation_throughput.png`<br>`docs/paper/figures/resource_simulation_concurrency.png` | `resource_simulation_benchmark.json["profiles_benchmark"]` |
| **Figure 6: Quantization Trade-offs & INT8 Analysis** | Footprint vs Macro-F1 vs latency across FP32/FP16/INT8/INT4 and layer-wise activation dynamic range outlier clipping | `scripts/benchmark_resource_simulation.py`<br>`scripts/investigate_int8_quantization.py` | `docs/paper/figures/resource_simulation_quantization.png`<br>`docs/paper/figures/int8_activation_analysis.png` | `resource_simulation_benchmark.json`<br>`int8_degradation_investigation.json` |

---

## 3. Supplementary & Intermediate Benchmark Registry

| Extended Table / Figure | Description | Generating Script | Output Artifact |
|---|---|---|---|
| **Table S1 / Table 1** | Full per-class classification metrics (15 threat classes) | `scripts/generate_classifier_metrics.py` | `tables/table1_classification_performance.csv` |
| **Table S2 / Table 2** | Raw OOD baseline comparison (Isolation Forest, OCSVM) | `scripts/benchmark_ood_baselines.py` | `tables/table2_ood_baselines.csv` |
| **Table S3 / Table 3** | 5-seed statistical rigor distributions (mean $\pm$ std, 95% CIs) | `scripts/statistical_rigor_benchmark.py` | `tables/table3_statistical_rigor.csv` |
| **Table S4 / Table 5** | Training time scalability across 76, 21, and 13 schemas | `scripts/benchmark_training_time.py` | `tables/table5_training_scalability.csv` |
| **Table S5 / Table 7** | 76-feature baseline vs 13-feature standardized model complexity | `scripts/cross_model_comparison.py` | `tables/table7_cross_model_comparison.csv` |
| **Table S6 / Table 8** | Real-time HTTP streaming vs offline in-memory inference | `scripts/benchmark_realtime_vs_offline.py` | `tables/table8_latency_breakdown.csv` |
| **Table S7 / Table 18** | Complete 21-feature NetFlow vs CICFlowMeter semantic audit matrix | `scripts/generate_paper_tables.py` | `tables/table_semantic_feature_audit.csv` |
| **Table S8 / Table 19** | Prior-work comparison against 7 systems across 8 dimensions | `scripts/generate_paper_tables.py` | `tables/table_prior_work_comparison.csv` |
| **Figure S1 / Fig 2** | 3-panel cross-model complexity, F1 generalization, and latency | `scripts/cross_model_comparison.py` | `figures/cross_model_comparison.png` |
| **Figure S2 / Fig 4** | 4-panel OOD baseline ROC, intercept rates, and latency | `scripts/benchmark_ood_baselines.py` | `figures/ood_baselines_comparison.png` |
| **Figure S3 / Fig 5** | 4-panel statistical rigor boxplots, clean/FPR bars, CDF | `scripts/statistical_rigor_benchmark.py` | `figures/statistical_rigor_plots.png` |
| **Figure S4 / Fig 7** | Real-time REST API vs offline latency decomposition | `scripts/benchmark_realtime_vs_offline.py` | `figures/realtime_vs_offline.png` |
| **Figure S5 / Fig 8** | 4-panel calibrated semantic engine integration evaluation | `scripts/evaluate_semantic_engine.py` | `figures/semantic_engine_evaluation_plots.png` |
| **Figure S6 / Fig 9** | Wall-clock training time across schemas | `scripts/benchmark_training_time.py` | `figures/training_time_comparison.png` |

---

## 4. Mandatory Edge Simulation Limitation

> **IMPORTANT PAPER FRAMING GUARDRAIL:**
> *"We evaluate SEMANTICSHIELD under controlled CPU- and memory-constrained deployment profiles to approximate resource-limited inference conditions. These containerized Linux cgroups v2 resource quotas evaluate execution under constrained execution budgets, but do not reproduce physical ARM microarchitectures, dedicated NPU/TPU cores, thermal throttling, memory bus bandwidth saturation, or physical NIC line-rate packet capture. Physical edge hardware validation remains future work."*

---

## 5. Automated Verification & Single Reproduction Entry Point

All artifacts and numbers in this manifest can be verified immediately:

```bash
# Verify presence and integrity of all 18 JSONs, 22 PNGs, 19 CSVs, 23 checksums, and 89 numbers
python experiments/reproduce_paper.py --verify

# Automated numerical assertion test against paper-draft.tex
python docs/paper/verify_numbers.py

# Cryptographic SHA-256 integrity verification
python scripts/download_artifacts.py --verify-checksums
```
