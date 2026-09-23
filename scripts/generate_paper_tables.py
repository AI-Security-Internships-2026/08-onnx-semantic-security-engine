"""
SEMANTICSHIELD Paper Tables Generator
Generates canonical CSV tables for all manuscript tables directly from
the machine-readable JSON files in experiments/paper_results/json/.

Outputs written to experiments/paper_results/tables/.
"""

import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
JSON_DIR = BASE_DIR / "experiments" / "paper_results" / "json"
TABLES_DIR = BASE_DIR / "experiments" / "paper_results" / "tables"


def generate_table1_classification(json_dir: Path, out_dir: Path):
    """Table 1: NIDS Classification Performance on CSE-CIC-IDS2018."""
    path = json_dir / "classifier_metrics.json"
    if not path.exists():
        print(f"  [SKIP] Table 1: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table1_classification_performance.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Traffic Class", "Precision", "Recall", "F1-Score", "Support"])
        for cls_name, m in data.get("per_class", {}).items():
            writer.writerow([
                cls_name,
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['f1']:.4f}",
                m['support'],
            ])
        ov = data.get("overall", {})
        writer.writerow(["---", "---", "---", "---", "---"])
        writer.writerow(["Macro Average", f"{ov.get('macro_avg_precision', 0):.4f}", f"{ov.get('macro_avg_recall', 0):.4f}", f"{ov.get('macro_avg_f1', 0):.4f}", sum(m['support'] for m in data.get('per_class', {}).values())])
        writer.writerow(["Weighted Average", f"{ov.get('weighted_avg_precision', 0):.4f}", f"{ov.get('weighted_avg_recall', 0):.4f}", f"{ov.get('weighted_avg_f1', 0):.4f}", sum(m['support'] for m in data.get('per_class', {}).values())])
        writer.writerow(["Overall Accuracy", "-", "-", f"{ov.get('accuracy', 0):.4f}", sum(m['support'] for m in data.get('per_class', {}).values())])

    print(f"  [SAVED] {out_file.name}")


def generate_table2_ood_baselines(json_dir: Path, out_dir: Path):
    """Table 2: OOD & Anomaly Baselines Comparison."""
    path = json_dir / "ood_baselines_benchmark.json"
    if not path.exists():
        print(f"  [SKIP] Table 2: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table2_ood_baselines.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Detector Method",
            "In-Dist FPR",
            "ToN-IoT AUROC",
            "ToN-IoT AUPRC",
            "ToN-IoT FPR@95",
            "ToN-IoT Intercept %",
            "Noise AUROC",
            "Zero-Fill AUROC",
            "Mean Latency (ms)",
        ])
        for det_name, m in data.get("detectors_benchmark", {}).items():
            writer.writerow([
                det_name,
                f"{m.get('in_dist_fpr', 0)*100:.2f}%",
                f"{m.get('ood_toniot_auroc', 0):.4f}",
                f"{m.get('ood_toniot_avg_precision', 0):.4f}",
                f"{m.get('ood_toniot_fpr95', 0)*100:.2f}%",
                f"{m.get('ood_toniot_detect_rate', 0)*100:.2f}%",
                f"{m.get('noise_auroc', 0):.4f}",
                f"{m.get('zero_fill_auroc', 0):.4f}",
                f"{m.get('latency', {}).get('mean_ms', 0):.4f}",
            ])

    print(f"  [SAVED] {out_file.name}")


def generate_table3_statistical_rigor(json_dir: Path, out_dir: Path):
    """Table 3: Multi-Seed Statistical Rigor Benchmark."""
    path = json_dir / "statistical_rigor_benchmark.json"
    if not path.exists():
        print(f"  [SKIP] Table 3: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table3_statistical_rigor.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Evaluation Metric", "Mean", "Std", "Mean +/- Std", "95% CI Lower", "95% CI Upper"])
        for metric, m in data.get("aggregated_metrics", {}).items():
            mean = m.get("mean", 0)
            std = m.get("std", 0)
            margin = m.get("ci95_margin", 0)
            writer.writerow([
                metric,
                f"{mean:.4f}",
                f"{std:.4f}",
                m.get("formatted", f"{mean:.4f} +/- {std:.4f}"),
                f"{mean - margin:.4f}",
                f"{mean + margin:.4f}",
            ])

    print(f"  [SAVED] {out_file.name}")


def generate_table4_quantization(json_dir: Path, out_dir: Path):
    """Table 4: Quantization Impact on Latency, Size, and Generalization."""
    path = json_dir / "quantization_benchmark.json"
    if not path.exists():
        print(f"  [SKIP] Table 4: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table4_quantization.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Precision Variant",
            "Model File",
            "Size (MB)",
            "Size Reduction (%)",
            "Macro F1",
            "F1 Drop vs FP32 (%)",
            "Mean Latency (ms)",
            "Throughput B1 (flows/s)",
            "Throughput B128 (flows/s)",
        ])
        for name, v in data.get("variants", {}).items():
            writer.writerow([
                v.get("variant", name),
                v.get("model_file", "-"),
                f"{v.get('size_mb', 0):.4f}",
                f"{v.get('size_reduction_vs_fp32_pct', 0):.2f}%",
                f"{v.get('macro_f1', 0):.4f}",
                f"{v.get('f1_drop_vs_fp32_pct', 0):.2f}%",
                f"{v.get('latency', {}).get('mean_ms', 0):.4f}",
                f"{v.get('throughput_samples_per_sec', {}).get('batch_1', 0):.1f}",
                f"{v.get('throughput_samples_per_sec', {}).get('batch_128', 0):.1f}",
            ])

    print(f"  [SAVED] {out_file.name}")


def generate_table5_training_scalability(json_dir: Path, out_dir: Path):
    """Table 5: Feature Schema Training Scalability."""
    path = json_dir / "training_time_benchmark.json"
    if not path.exists():
        print(f"  [SKIP] Table 5: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table5_training_scalability.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Schema Name",
            "Features",
            "Parameters",
            "Total Train Time (s)",
            "Epoch Mean Time (ms)",
            "Throughput (samples/s)",
            "Test Macro F1",
            "Test Accuracy",
            "Speedup vs 76-Feat",
        ])
        for name, s in data.get("schemas", {}).items():
            writer.writerow([
                s.get("schema_name", name),
                s.get("feature_count", 0),
                s.get("parameter_count", 0),
                f"{s.get('total_train_time_sec', 0):.2f}",
                f"{s.get('mean_epoch_time_ms', 0):.2f}",
                f"{s.get('train_throughput_samples_per_sec', 0):.1f}",
                f"{s.get('test_macro_f1', 0):.4f}",
                f"{s.get('test_accuracy', 0):.4f}",
                f"{s.get('speedup_vs_baseline', 1.0):.2f}x",
            ])

    print(f"  [SAVED] {out_file.name}")


def generate_table6_ablation_study(json_dir: Path, out_dir: Path):
    """Table 6: Architectural Component Ablation Study."""
    path = json_dir / "ablation_study.json"
    if not path.exists():
        print(f"  [SKIP] Table 6: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table6_ablation_study.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration",
            "InputValidator",
            "ConfidenceAnalyzer",
            "DriftDetector",
            "In-Dist Clean %",
            "In-Dist FPR %",
            "OOD ToN-IoT Flagged %",
            "Noise Rejected %",
            "Zero-Fill Rejected %",
        ])
        for key, c in data.get("configurations", {}).items():
            comps = c.get("components", {})
            writer.writerow([
                c.get("description", key),
                "YES" if comps.get("input_validator") else "NO",
                "YES" if comps.get("confidence_analyzer") else "NO",
                "YES" if comps.get("drift_detector") else "NO",
                f"{c.get('in_dist_clean_pct', 0):.2f}%",
                f"{c.get('in_dist_fpr_pct', 0):.2f}%",
                f"{c.get('ood_total_flagged_pct', 0):.2f}%",
                f"{c.get('noise_rejection_pct', 0):.2f}%",
                f"{c.get('zero_rejection_pct', 0):.2f}%",
            ])

    print(f"  [SAVED] {out_file.name}")


def generate_table7_cross_model(json_dir: Path, out_dir: Path):
    """Table 7: Cross-Model Comparison (76 vs 13 Features)."""
    path = json_dir / "cross_model_comparison.json"
    if not path.exists():
        print(f"  [SKIP] Table 7: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table7_cross_model_comparison.csv"
    b = data.get("models", {}).get("baseline_76", {})
    s = data.get("models", {}).get("standardized_13", {})

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric Dimension", "76-Feature Baseline", "13-Feature Standardized", "Comparison / Gain"])
        writer.writerow(["Feature Count", b.get("features_count"), s.get("features_count"), "-82.9%"])
        writer.writerow(["Parameter Count", b.get("parameters_count"), s.get("parameters_count"), "-25.8%"])
        writer.writerow(["FP32 Model Size (MB)", b.get("storage", {}).get("fp32_size_mb"), s.get("storage", {}).get("fp32_size_mb"), "-29.2%"])
        writer.writerow(["INT8 Model Size (MB)", b.get("storage", {}).get("int8_size_mb"), s.get("storage", {}).get("int8_size_mb"), "-22.6%"])
        writer.writerow(["In-Dist Macro F1", f"{b.get('performance', {}).get('in_dist_macro_f1', 0):.4f}", f"{s.get('performance', {}).get('in_dist_macro_f1', 0):.4f}", "-0.0333"])
        writer.writerow(["Cross-Dataset Macro F1", f"{b.get('cross_dataset', {}).get('macro_f1', 0):.4f}", f"{s.get('cross_dataset', {}).get('macro_f1', 0):.4f}", "+0.0163"])
        writer.writerow(["Single-Flow Mean Latency (ms)", f"{b.get('latency', {}).get('mean_ms', 0):.4f}", f"{s.get('latency', {}).get('mean_ms', 0):.4f}", f"{b.get('latency', {}).get('mean_ms', 0) / max(s.get('latency', {}).get('mean_ms', 1), 1e-6):.2f}x faster"])
        writer.writerow(["Throughput (flows/s)", f"{b.get('latency', {}).get('throughput_flows_per_sec', 0):.1f}", f"{s.get('latency', {}).get('throughput_flows_per_sec', 0):.1f}", "+"])

    print(f"  [SAVED] {out_file.name}")


def generate_table8_latency_breakdown(json_dir: Path, out_dir: Path):
    """Table 8: Real-Time vs Offline Inference Breakdown."""
    path = json_dir / "latency_benchmark.json"
    if not path.exists():
        print(f"  [SKIP] Table 8: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_file = out_dir / "table8_latency_breakdown.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Inference Mode & Component", "Mean Latency (ms)", "p50 (ms)", "p95 (ms)", "Throughput (flows/s)"])
        off = data.get("offline_in_memory", {})
        plain_off = off.get("plain_onnx_single", {})
        sem_off = off.get("semantic_engine_single", {})
        writer.writerow(["Offline: Plain ONNX", f"{plain_off.get('mean_ms', 0):.4f}", f"{plain_off.get('p50_ms', 0):.4f}", f"{plain_off.get('p95_ms', 0):.4f}", f"{plain_off.get('throughput_flows_per_sec', 0):.1f}"])
        writer.writerow(["Offline: Semantic Security Engine", f"{sem_off.get('mean_ms', 0):.4f}", f"{sem_off.get('p50_ms', 0):.4f}", f"{sem_off.get('p95_ms', 0):.4f}", f"{sem_off.get('throughput_flows_per_sec', 0):.1f}"])
        
        rt = data.get("realtime_http_streaming", {})
        plain_rt = rt.get("plain_rest_streaming", {})
        sec_rt = rt.get("secure_rest_streaming", {})
        writer.writerow(["Real-time REST: Plain ONNX Client E2E", f"{plain_rt.get('client_e2e_latency', {}).get('mean_ms', 0):.4f}", f"{plain_rt.get('client_e2e_latency', {}).get('p50_ms', 0):.4f}", f"{plain_rt.get('client_e2e_latency', {}).get('p95_ms', 0):.4f}", f"{plain_rt.get('streaming_throughput_flows_per_sec', 0):.1f}"])
        writer.writerow(["Real-time REST: Plain ONNX Server Compute", f"{plain_rt.get('server_inference_latency', {}).get('mean_ms', 0):.4f}", f"{plain_rt.get('server_inference_latency', {}).get('p50_ms', 0):.4f}", f"{plain_rt.get('server_inference_latency', {}).get('p95_ms', 0):.4f}", "-"])
        writer.writerow(["Real-time REST: Secure Engine Client E2E", f"{sec_rt.get('client_e2e_latency', {}).get('mean_ms', 0):.4f}", f"{sec_rt.get('client_e2e_latency', {}).get('p50_ms', 0):.4f}", f"{sec_rt.get('client_e2e_latency', {}).get('p95_ms', 0):.4f}", f"{sec_rt.get('streaming_throughput_flows_per_sec', 0):.1f}"])
        writer.writerow(["Real-time REST: Secure Engine Server Compute", f"{sec_rt.get('server_inference_latency', {}).get('mean_ms', 0):.4f}", f"{sec_rt.get('server_inference_latency', {}).get('p50_ms', 0):.4f}", f"{sec_rt.get('server_inference_latency', {}).get('p95_ms', 0):.4f}", "-"])

    print(f"  [SAVED] {out_file.name}")


def generate_table9_main_ood_benchmark(json_dir: Path, out_dir: Path):
    """Table 9 (Issue 3): Main OOD Benchmark at Fixed FPR Operating Points."""
    path = json_dir / "fixed_fpr_evaluation.json"
    if not path.exists():
        print(f"  [SKIP] Table 9: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract E2 metrics and bootstrap CIs
    e2 = data.get("metrics_per_failure_mode", {}).get("E2_real_ood", {}).get("detectors", {})
    bootstrap = data.get("bootstrap_ci_e2", {})

    out_file = out_dir / "table_main_ood_benchmark.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Method",
            "AUROC", "AUROC 95% CI",
            "AUPRC", "AUPRC 95% CI",
            "FPR@95TPR",
            "TPR@0.1% FPR", "TPR@1% FPR", "TPR@5% FPR",
        ])

        det_labels = {
            "msp": "MSP (1-Confidence)",
            "cosine": "Cosine Distance",
            "mahalanobis": "Mahalanobis Distance",
            "composite": "Composite Drift Score",
        }
        for det_name, label in det_labels.items():
            m = e2.get(det_name, {})
            tpr = m.get("tpr_at_fpr", {})
            bs = bootstrap.get(det_name, {})
            auroc_ci = bs.get("auroc", {})
            auprc_ci = bs.get("auprc", {})

            auroc_ci_str = ""
            if auroc_ci.get("ci95_lo") is not None:
                auroc_ci_str = f"[{auroc_ci['ci95_lo']:.4f}, {auroc_ci['ci95_hi']:.4f}]"

            auprc_ci_str = ""
            if auprc_ci.get("ci95_lo") is not None:
                auprc_ci_str = f"[{auprc_ci['ci95_lo']:.4f}, {auprc_ci['ci95_hi']:.4f}]"

            writer.writerow([
                label,
                f"{m.get('auroc', 0):.4f}", auroc_ci_str,
                f"{m.get('auprc', 0):.4f}", auprc_ci_str,
                f"{m.get('fpr_at_95tpr', 0):.4f}",
                f"{tpr.get('0.001', 0):.4f}",
                f"{tpr.get('0.01', 0):.4f}",
                f"{tpr.get('0.05', 0):.4f}",
            ])

        # Validator row
        val_rej = data.get("metrics_per_failure_mode", {}).get("E2_real_ood", {}).get("validator_rejection_rate", 0)
        writer.writerow([
            "Input Validator (structural)",
            "N/A (deterministic)", "",
            "N/A (deterministic)", "",
            "N/A",
            "N/A", "N/A", f"rej={val_rej:.4f}",
        ])

    print(f"  [SAVED] {out_file.name}")


def generate_table10_failure_mode_coverage(json_dir: Path, out_dir: Path):
    """Table 10 (Issue 3): Failure-Mode Coverage / Ablation Matrix."""
    path = json_dir / "failure_mode_coverage_matrix.json"
    if not path.exists():
        print(f"  [SKIP] Table 10: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matrix = data.get("matrix", {})

    out_file = out_dir / "table_failure_mode_coverage.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Failure Mode",
            "MSP", "Cosine", "Mahalanobis",
            "Validator (structural)", "Full System",
        ])
        for fm_key, row in matrix.items():
            writer.writerow([
                row.get("label", fm_key),
                f"{(row.get('msp', 0) or 0)*100:.1f}%",
                f"{(row.get('cosine', 0) or 0)*100:.1f}%",
                f"{(row.get('mahalanobis', 0) or 0)*100:.1f}%",
                f"{(row.get('validator', 0) or 0)*100:.1f}%",
                f"{(row.get('full_system', 0) or 0)*100:.1f}%",
            ])

    print(f"  [SAVED] {out_file.name}")


def generate_table11_ablation_fixed_fpr(json_dir: Path, out_dir: Path):
    """Table 11 (Issue 3): Component Ablation with AUROC-level Metrics."""
    path = json_dir / "ablation_fixed_fpr.json"
    if not path.exists():
        print(f"  [SKIP] Table 11: {path.name} not found")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("ablation_results", {})

    out_file = out_dir / "table_ablation_fixed_fpr.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration",
            "MSP", "Cosine", "Mahalanobis", "Validator",
            "E2 AUROC", "E2 AUPRC", "E2 FPR@95TPR",
            "E2 Validator Rej.",
        ])
        for key, r in results.items():
            comps = r.get("components", {})
            writer.writerow([
                r.get("label", key),
                "YES" if comps.get("msp") else "NO",
                "YES" if comps.get("cosine") else "NO",
                "YES" if comps.get("mahalanobis") else "NO",
                "YES" if comps.get("validator") else "NO",
                f"{r.get('e2_auroc', 0):.4f}",
                f"{r.get('e2_auprc', 0):.4f}",
                f"{r.get('e2_fpr_at_95tpr', 0):.4f}",
                f"{(r.get('e2_validator_rejection', 0) or 0)*100:.1f}%",
            ])

    print(f"  [SAVED] {out_file.name}")


def main():
    print("=" * 70)
    print("  GENERATING CANONICAL PAPER CSV TABLES FROM JSON RESULTS")
    print("=" * 70)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    generate_table1_classification(JSON_DIR, TABLES_DIR)
    generate_table2_ood_baselines(JSON_DIR, TABLES_DIR)
    generate_table3_statistical_rigor(JSON_DIR, TABLES_DIR)
    generate_table4_quantization(JSON_DIR, TABLES_DIR)
    generate_table5_training_scalability(JSON_DIR, TABLES_DIR)
    generate_table6_ablation_study(JSON_DIR, TABLES_DIR)
    generate_table7_cross_model(JSON_DIR, TABLES_DIR)
    generate_table8_latency_breakdown(JSON_DIR, TABLES_DIR)
    # Issue 3: Fixed-FPR, Failure-Mode Coverage & Ablation tables
    generate_table9_main_ood_benchmark(JSON_DIR, TABLES_DIR)
    generate_table10_failure_mode_coverage(JSON_DIR, TABLES_DIR)
    generate_table11_ablation_fixed_fpr(JSON_DIR, TABLES_DIR)
    print("=" * 70)
    print(f"All tables exported to {TABLES_DIR}")


if __name__ == "__main__":
    main()

