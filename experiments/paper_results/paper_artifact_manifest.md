# SEMANTICSHIELD Manuscript Artifact Manifest

This document maps every table, figure, and empirical claim in the research manuscript to its exact generating script, configuration file, and canonical JSON result artifact.

---

## 1. Tables to Artifact Mapping

| Manuscript Table | Description | Generating Script | Primary JSON Artifact | Key Metric Fields |
|---|---|---|---|---|
| **Table 1: NIDS Classification Performance** | Overall & per-class precision, recall, F1, and support on CSE-CIC-IDS2018 | `scripts/generate_classifier_metrics.py` | `experiments/paper_results/json/classifier_metrics.json` | `overall.accuracy`, `overall.macro_avg_f1`, `per_class.*` |
| **Table 2: OOD & Anomaly Baselines Comparison** | AUROC, latency, and interception across MSP, Mahalanobis, IF, OCSVM, and Engine | `scripts/benchmark_ood_baselines.py` | `experiments/paper_results/json/ood_baselines_benchmark.json` | `detectors_benchmark.*.ood_toniot_auroc`, `detectors_benchmark.*.latency.mean_ms` |
| **Table 3: Multi-Seed Statistical Rigor Benchmark** | Mean $\pm$ std, 95% CI, and FPR@95TPR across 5 random seeds | `scripts/statistical_rigor_benchmark.py` | `experiments/paper_results/json/statistical_rigor_benchmark.json` | `aggregated_metrics.in_distribution_clean_pct`, `aggregated_metrics.mahalanobis_auroc`, `aggregated_metrics.mahalanobis_fpr95` |
| **Table 4: Quantization Impact on Latency & Size** | FP32, FP16, INT8, INT4 size (MB), Macro-F1 drop, latency, and throughput | `scripts/benchmark_quantization.py` | `experiments/paper_results/json/quantization_benchmark.json` | `variants.FP32.*`, `variants.FP16.*`, `variants.INT8.*`, `variants.INT4.*` |
| **Table 5: Feature Schema Training Scalability** | Training duration, epoch time, throughput, and macro-F1 across 76, 21, and 13 schemas | `scripts/benchmark_training_time.py` | `experiments/paper_results/json/training_time_benchmark.json` | `schemas.76-Feature Baseline.*`, `schemas.13-Feature Standardized.*` |
| **Table 6: Architectural Component Ablation** | Security guarantees under individual and combined component ablations | `scripts/ablation_study.py` | `experiments/paper_results/json/ablation_study.json` | `configurations.full_system.*`, `configurations.no_drift.*`, `configurations.validator_only.*` |
| **Table 7: Cross-Model Complexity Comparison** | 76-feature baseline vs 13-feature standardized model metrics | `scripts/cross_model_comparison.py` | `experiments/paper_results/json/cross_model_comparison.json` | `models.baseline_76.*`, `models.standardized_13.*`, `efficiency_gains.*` |
| **Table 8: Real-Time vs Offline Inference Breakdown** | In-memory inference vs HTTP streaming REST endpoint latency decomposition | `scripts/benchmark_realtime_vs_offline.py` | `experiments/paper_results/json/latency_benchmark.json` | `offline_in_memory.*`, `realtime_http_streaming.*` |
| **Table 9: Cross-Dataset Domain Adaptation** | Multi-class and binary transfer under CORAL, z-score re-standardization, and raw transfer | Direct alignment audit script | `experiments/paper_results/json/cross_dataset_alignment.json` | `methods_benchmark.Source Standardization (No Adaptation).*`, `methods_benchmark.CORAL.*` |
| **Table 15: Resource Profiles** | Simulated edge profiles R0, R1, R2, R3 (CPU, memory, runtime, workload) | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/json/resource_simulation_benchmark.json` | `resource_profiles.*` |
| **Table 16: Runtime Overhead** | Plain ONNX vs SEMANTICSHIELD mean/p50/p95/p99 latency, throughput, peak RSS, CPU % across profiles | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/json/resource_simulation_benchmark.json` | `profiles_benchmark.*.plain_onnx.*`, `profiles_benchmark.*.semantic_shield.*`, `profiles_benchmark.*.overhead.*` |
| **Table 17: Quantization Trade-offs** | Precision (FP32/FP16/INT8/INT4), size (MB), accuracy, macro-F1, p95 latency, throughput, peak RSS under R1 | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/json/resource_simulation_benchmark.json` | `profiles_benchmark.R1.quantization_tradeoffs.*` |

---

## 2. Figures to Artifact Mapping

| Manuscript Figure | Description | Generating Script | Canonical Figure Artifact | Source Data |
|---|---|---|---|---|
| **Figure 1: Confusion Matrix Heatmap** | Normalized confusion matrix across 15 threat classes on CSE-CIC-IDS2018 | `scripts/generate_classifier_metrics.py` | `experiments/paper_results/figures/confusion_matrix_cic.png` | `classifier_metrics.json["confusion_matrix"]` |
| **Figure 2: Cross-Model Comparison** | 3-panel comparison: complexity, F1 generalization, and single-flow latency | `scripts/cross_model_comparison.py` | `experiments/paper_results/figures/cross_model_comparison.png` | `cross_model_comparison.json["models"]` |
| **Figure 3: Component Ablation Study** | Detection and false alarm rates across 7 architectural configurations | `scripts/ablation_study.py` | `experiments/paper_results/figures/ablation_comparison.png` | `ablation_study.json["configurations"]` |
| **Figure 4: OOD Baseline Comparison** | 4-panel comparison: ROC curves, intercept rates, latency trade-offs, zero-fill | `scripts/benchmark_ood_baselines.py` | `experiments/paper_results/figures/ood_baselines_comparison.png` | `ood_baselines_benchmark.json["detectors_benchmark"]` |
| **Figure 5: Statistical Rigor & Distributions** | 4-panel analysis: AUROC stability boxplots, clean/FPR bars, latency CDF | `scripts/statistical_rigor_benchmark.py` | `experiments/paper_results/figures/statistical_rigor_plots.png` | `statistical_rigor_benchmark.json["per_seed_runs"]` |
| **Figure 6: Quantization Benchmark** | 4-panel trade-off: Model size, macro-F1, latency percentiles, throughput | `scripts/benchmark_quantization.py` | `experiments/paper_results/figures/quantization_benchmark.png` | `quantization_benchmark.json["variants"]` |
| **Figure 7: Real-Time vs Offline Latency** | 2-panel comparison: Compute vs HTTP network overhead, throughput | `scripts/benchmark_realtime_vs_offline.py` | `experiments/paper_results/figures/realtime_vs_offline.png` | `latency_benchmark.json` |
| **Figure 8: Semantic Engine Evaluation** | 4-panel integration: Drift distributions, threshold curves, ROC, verdicts | `scripts/evaluate_semantic_engine.py` | `experiments/paper_results/figures/semantic_engine_evaluation_plots.png` | `semantic_engine_evaluation.json["scenarios"]` |
| **Figure 9: Training Efficiency Comparison** | 2-panel comparison: Wall-clock training time and throughput across schemas | `scripts/benchmark_training_time.py` | `experiments/paper_results/figures/training_time_comparison.png` | `training_time_benchmark.json["schemas"]` |
| **Figure 10: p95 Latency vs Resource Profile** | Plain ONNX vs SEMANTICSHIELD p95 latency across R0 (Ref), R1 (Low), R2 (Med), R3 (High) | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/figures/resource_simulation_p95_latency.png` | `resource_simulation_benchmark.json["profiles_benchmark"]` |
| **Figure 11: Throughput vs Resource Profile** | Single-flow inference throughput comparison as resources tighten | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/figures/resource_simulation_throughput.png` | `resource_simulation_benchmark.json["profiles_benchmark"]` |
| **Figure 12: Quantization Trade-offs (R1 Profile)** | 4-panel trade-off: Model footprint, Macro-F1, p95 latency, throughput under R1 | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/figures/resource_simulation_quantization.png` | `resource_simulation_benchmark.json["profiles_benchmark"]["R1"]["quantization_tradeoffs"]` |
| **Figure 13: Concurrency Sensitivity** | Per-flow latency and throughput scalability across sequential, moderate, high load | `scripts/benchmark_resource_simulation.py` | `experiments/paper_results/figures/resource_simulation_concurrency.png` | `resource_simulation_benchmark.json["profiles_benchmark"]["*"]["concurrency_sensitivity"]` |
| **Figure 14: INT8 Activation & Degradation Analysis** | 4-panel empirical analysis: per-class drop, calibration representation, activation outliers | `scripts/investigate_int8_quantization.py` | `experiments/paper_results/figures/int8_activation_analysis.png` | `int8_degradation_investigation.json` |

---

## 3. Paper Framing Guardrail

> **IMPORTANT PAPER FRAMING GUARDRAIL:**
> *"We evaluate SEMANTICSHIELD under controlled CPU- and memory-constrained deployment profiles to approximate resource-limited inference conditions."*
> **Do NOT write:** *"We validate SEMANTICSHIELD on edge hardware."*
> Physical ARM/edge-device hardware validation remains future work and an explicit limitation.

---

## 4. Reproduction Command Reference

To reproduce all artifacts end-to-end, execute:

```bash
# 1. Inspect execution plan (no changes made)
python experiments/reproduce_paper.py --dry-run

# 2. Fast reproduction (classifier, cross-model, ablation, quantization, int8_investigation)
python experiments/reproduce_paper.py --skip-slow

# 3. Individual benchmark execution
python experiments/reproduce_paper.py --only classifier
python experiments/reproduce_paper.py --only cross_model
python experiments/reproduce_paper.py --only ablation
python experiments/reproduce_paper.py --only quantization
python experiments/reproduce_paper.py --only int8_investigation
python experiments/reproduce_paper.py --only resource_simulation

# 4. Generate all canonical CSV tables from machine-readable JSON
python scripts/generate_paper_tables.py

# 5. Verify all canonical artifacts
python experiments/reproduce_paper.py --verify
```
