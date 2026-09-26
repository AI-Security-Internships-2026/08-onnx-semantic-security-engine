# Archived Historical Results

This directory contains legacy, superseded, and duplicate experiment results from intermediate development stages of the SEMANTICSHIELD engine.

> **CRITICAL REPRODUCIBILITY NOTICE:**  
> None of the files in this directory should be cited or used in the final manuscript.  
> All canonical, reproducible paper results are located in `experiments/paper_results/`.

---

## Inventory of Archived Files

| Archived File | Original Location | Reason for Archival | Superseded By |
|---|---|---|---|
| `classification_report.json` | `experiments/results/` | Evaluated legacy 76-feature CICFlowMeter model | `experiments/paper_results/json/classifier_metrics.json` |
| `nf_classification_report.json` | `experiments/results/` | Contained stale text metadata ("21 features") despite being evaluated on 13-feature model | `experiments/paper_results/json/classifier_metrics.json` |
| `cross_dataset_comparison.json` | `experiments/results/` | Evaluated legacy 76-feature model across datasets | `experiments/paper_results/json/cross_dataset_alignment.json` |
| `nf_vs_baseline_comparison.json` | `experiments/results/` | Referenced 21-feature NetFlow schema from intermediate prototyping era | `experiments/paper_results/json/cross_model_comparison.json` |
| `quantization_comparison.json` | `experiments/results/` | Fragmentary latency/size data without throughput or statistical bounds | `experiments/paper_results/json/quantization_benchmark.json` |
| `quantization_comparison_full.json` | `experiments/results/` | Partial duplicate of quantization metrics | `experiments/paper_results/json/quantization_benchmark.json` |
| `nf_cross_dataset_comparison.json` | `experiments/` (root) | Stale duplicate of report located in `results/` | `experiments/paper_results/json/cross_dataset_alignment.json` |
| `nf_quantization_comparison.json` | `experiments/` (root) | Stale duplicate of report located in `results/` | `experiments/paper_results/json/quantization_benchmark.json` |
| `nf_comparison_plots.png` | `experiments/` (root) | Orphaned figure in root directory; canonical plots are in `paper_results/figures/` | `experiments/paper_results/figures/` |
| `nf_confusion_matrix_cic.png` | `experiments/` (root) | Orphaned figure in root directory; superseded by regenerated confusion matrix | `experiments/paper_results/figures/confusion_matrix_cic.png` |
| `nf_confusion_matrix_toniot.png` | `experiments/` (root) | Orphaned figure in root directory; superseded by regenerated confusion matrix | `experiments/paper_results/figures/confusion_matrix_toniot.png` |
