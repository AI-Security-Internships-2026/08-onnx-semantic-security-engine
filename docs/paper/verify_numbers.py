"""
Verify all numeric claims in paper-draft.tex against source JSON files.
Run: python docs/paper/verify_numbers.py
"""
import json
from pathlib import Path

RESULTS = Path(__file__).parent.parent.parent / "experiments" / "results"
EXPERIMENTS = Path(__file__).parent.parent.parent / "experiments"

errors = []
checks = 0

def check(description, actual, expected, tolerance=0.01):
    global checks, errors
    checks += 1
    if abs(actual - expected) > tolerance:
        errors.append(f"  MISMATCH: {description}: paper says {expected}, JSON says {actual}")
    else:
        print(f"  OK: {description} = {actual}")

print("=" * 70)
print("PAPER NUMBER VERIFICATION")
print("=" * 70)

# --- statistical_rigor_benchmark.json ---
print("\n[1] Statistical Rigor Benchmark")
with open(RESULTS / "statistical_rigor_benchmark.json") as f:
    sr = json.load(f)
agg = sr["aggregated_metrics"]
check("In-dist clean %", agg["in_distribution_clean_pct"]["mean"], 86.94, 0.01)
check("In-dist clean std", agg["in_distribution_clean_pct"]["std"], 0.44, 0.01)
check("In-dist FPR %", agg["in_distribution_fpr_pct"]["mean"], 13.06, 0.01)
check("Binary F1 mean", agg["cross_dataset_binary_f1"]["mean"], 0.7734, 0.001)
check("Binary F1 std", agg["cross_dataset_binary_f1"]["std"], 0.0049, 0.001)
check("MSP AUROC mean", agg["msp_confidence_auroc"]["mean"], 0.8096, 0.001)
check("Cosine AUROC mean", agg["cosine_distance_auroc"]["mean"], 0.7799, 0.001)
check("Mahalanobis AUROC mean", agg["mahalanobis_auroc"]["mean"], 0.7744, 0.001)
check("OOD intercept %", agg["ood_toniot_intercept_pct"]["mean"], 61.72, 0.01)
check("Noise intercept %", agg["noise_intercept_pct"]["mean"], 100.0, 0.01)
check("Zero-fill rejected %", agg["zero_fill_rejected_pct"]["mean"], 100.0, 0.001)

lat = sr["latency_percentiles"]
check("Plain ONNX latency ms", lat["plain_inference"]["mean_ms"], 0.056, 0.001)
check("Semantic latency ms", lat["semantic_inference"]["mean_ms"], 0.346, 0.001)
check("Overhead ms", lat["semantic_inference"]["overhead_mean_ms"], 0.2893, 0.001)
check("Plain P95 ms", lat["plain_inference"]["p95_ms"], 0.066, 0.001)
check("Semantic P95 ms", lat["semantic_inference"]["p95_ms"], 0.405, 0.001)

# --- ood_baselines_benchmark.json ---
print("\n[2] OOD Baselines Benchmark")
with open(RESULTS / "ood_baselines_benchmark.json") as f:
    ood = json.load(f)
det = ood["detectors_benchmark"]
check("MSP ToN-IoT AUROC", det["MSP (Confidence Alone)"]["ood_toniot_auroc"], 0.850, 0.001)
check("MSP noise AUROC", det["MSP (Confidence Alone)"]["noise_auroc"], 0.031, 0.001)
check("MSP zero AUROC", det["MSP (Confidence Alone)"]["zero_fill_auroc"], 0.917, 0.001)
check("MSP latency ms", det["MSP (Confidence Alone)"]["latency"]["mean_ms"], 0.088, 0.001)
check("Mahal ToN-IoT AUROC", det["Mahalanobis Distance Alone"]["ood_toniot_auroc"], 0.959, 0.001)
check("Mahal noise AUROC", det["Mahalanobis Distance Alone"]["noise_auroc"], 1.000, 0.001)
check("IF ToN-IoT AUROC", det["Isolation Forest"]["ood_toniot_auroc"], 0.638, 0.001)
check("IF latency ms", det["Isolation Forest"]["latency"]["mean_ms"], 11.71, 0.01)
check("OCSVM ToN-IoT AUROC", det["One-Class SVM"]["ood_toniot_auroc"], 0.213, 0.001)
check("OCSVM noise AUROC", det["One-Class SVM"]["noise_auroc"], 1.000, 0.001)
check("Engine ToN-IoT AUROC", det["Semantic Engine (Combined)"]["ood_toniot_auroc"], 0.966, 0.001)
check("Engine noise AUROC", det["Semantic Engine (Combined)"]["noise_auroc"], 1.000, 0.001)
check("Engine zero AUROC", det["Semantic Engine (Combined)"]["zero_fill_auroc"], 1.000, 0.001)
check("Engine latency ms", det["Semantic Engine (Combined)"]["latency"]["mean_ms"], 0.380, 0.001)

# --- quantization_benchmark.json ---
print("\n[3] Quantization Benchmark")
with open(RESULTS / "quantization_benchmark.json") as f:
    quant = json.load(f)
v = quant["variants"]
check("FP32 size MB", v["FP32"]["size_mb"], 0.177, 0.001)
check("FP32 Macro-F1", v["FP32"]["macro_f1"], 0.7801, 0.001)
check("FP16 size MB", v["FP16"]["size_mb"], 0.089, 0.001)
check("FP16 Macro-F1", v["FP16"]["macro_f1"], 0.7802, 0.001)
check("FP16 size reduction %", v["FP16"]["size_reduction_vs_fp32_pct"], 49.6, 0.1)
check("FP16 F1 drop %", v["FP16"]["f1_drop_vs_fp32_pct"], -0.01, 0.01)
check("INT8 size MB", v["INT8"]["size_mb"], 0.048, 0.001)
check("INT8 Macro-F1", v["INT8"]["macro_f1"], 0.4833, 0.001)
check("INT8 F1 drop %", v["INT8"]["f1_drop_vs_fp32_pct"], 38.0, 0.1)
check("INT4 size MB", v["INT4"]["size_mb"], 0.034, 0.001)
check("INT4 Macro-F1", v["INT4"]["macro_f1"], 0.7627, 0.001)
check("INT4 size reduction %", v["INT4"]["size_reduction_vs_fp32_pct"], 80.7, 0.1)
check("INT4 F1 drop %", v["INT4"]["f1_drop_vs_fp32_pct"], 2.24, 0.01)

# --- cross_dataset_alignment_audit.json ---
print("\n[4] Cross-Dataset Alignment Audit")
with open(RESULTS / "cross_dataset_alignment_audit.json") as f:
    cda = json.load(f)
m = cda["methods_benchmark"]
check("Source Binary F1", m["Source Standardization (No Adaptation)"]["binary_f1"], 0.800, 0.001)
check("Source Multi-class F1", m["Source Standardization (No Adaptation)"]["multiclass_macro_f1"], 0.012, 0.001)
check("Re-std Binary F1", m["Target Re-Standardization (Z-score)"]["binary_f1"], 0.771, 0.001)
check("Re-std Multi-class F1", m["Target Re-Standardization (Z-score)"]["multiclass_macro_f1"], 0.019, 0.001)
check("CORAL Binary F1", m["CORAL (Correlation Alignment)"]["binary_f1"], 0.783, 0.001)
check("CORAL Multi-class F1", m["CORAL (Correlation Alignment)"]["multiclass_macro_f1"], 0.005, 0.001)

# --- nf_cross_dataset_comparison.json ---
print("\n[5] NF Cross-Dataset Comparison")
with open(RESULTS / "nf_cross_dataset_comparison.json") as f:
    nfc = json.load(f)
check("NF CIC F1", nfc["nf_cic_ids2018_f1"], 0.766, 0.001)
check("NF ToN-IoT F1", nfc["nf_toniot_f1"], 0.057, 0.001)
fmq = nfc["feature_mapping_quality_caveat"]
check("Exact matches", fmq["exact_matches"], 7, 0)
check("Approximate matches", fmq["approximate_matches"], 5, 0)
check("Mismatched pairs", fmq["mismatched_pairs"], 8, 0)

# --- realtime_vs_offline_benchmark.json ---
print("\n[6] Realtime vs Offline Benchmark")
with open(RESULTS / "realtime_vs_offline_benchmark.json") as f:
    rto = json.load(f)
check("Plain ONNX single ms", rto["offline_in_memory"]["plain_onnx_single"]["mean_ms"], 0.064, 0.001)
check("Semantic engine single ms", rto["offline_in_memory"]["semantic_engine_single"]["mean_ms"], 1.066, 0.001)
check("Plain REST mean ms", rto["realtime_http_streaming"]["plain_rest_streaming"]["client_e2e_latency"]["mean_ms"], 4.26, 0.01)
check("Secure REST mean ms", rto["realtime_http_streaming"]["secure_rest_streaming"]["client_e2e_latency"]["mean_ms"], 5.05, 0.01)
check("Plain REST throughput", rto["realtime_http_streaming"]["plain_rest_streaming"]["streaming_throughput_flows_per_sec"], 235, 1)
check("Secure REST throughput", rto["realtime_http_streaming"]["secure_rest_streaming"]["streaming_throughput_flows_per_sec"], 198, 1)

# --- semantic_engine_evaluation.json ---
print("\n[7] Semantic Engine Evaluation")
with open(RESULTS / "semantic_engine_evaluation.json") as f:
    see = json.load(f)
sc = see["scenarios"]
check("In-dist drift %", sc["in_distribution"]["drift_detected_pct"], 6.82, 0.1)
check("OOD drift %", sc["out_of_distribution"]["drift_detected_pct"], 74.31, 0.1)
check("Noise drift %", sc["random_noise"]["drift_detected_pct"], 100.0, 0.1)
check("Zero-fill REJECTED", sc["zero_filled"]["engine_verdicts"]["REJECTED"], 500, 0)

# --- Summary ---
print("\n" + "=" * 70)
print(f"TOTAL CHECKS: {checks}")
if errors:
    print(f"MISMATCHES FOUND: {len(errors)}")
    for e in errors:
        print(e)
else:
    print("ALL NUMBERS MATCH — paper is numerically truthful")
print("=" * 70)
