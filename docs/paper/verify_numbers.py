"""
Verify all numeric claims in paper-draft.tex against canonical JSON files in experiments/paper_results/json/.
Run: python docs/paper/verify_numbers.py
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
CANONICAL_JSON = BASE_DIR / "experiments" / "paper_results" / "json"

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
print("CANONICAL PAPER NUMBER VERIFICATION (Issue 7)")
print("=" * 70)

# --- 1. classifier_metrics.json (Table I Baseline) ---
print("\n[1] Classifier Metrics on CSE-CIC-IDS2018")
with open(CANONICAL_JSON / "classifier_metrics.json") as f:
    clf = json.load(f)
ov = clf["overall"]
check("Baseline Accuracy", ov["accuracy"], 0.8719, 0.001)
check("Baseline Macro-F1", ov["macro_avg_f1"], 0.7801, 0.001)
check("Baseline Weighted-F1", ov["weighted_avg_f1"], 0.8655, 0.001)
check("Baseline Macro Precision", ov["macro_avg_precision"], 0.7597, 0.001)
check("Baseline Macro Recall", ov["macro_avg_recall"], 0.8299, 0.001)

# --- 2. statistical_rigor_benchmark.json ---
print("\n[2] Statistical Rigor Benchmark (5 Seeds)")
with open(CANONICAL_JSON / "statistical_rigor_benchmark.json") as f:
    sr = json.load(f)
agg = sr["aggregated_metrics"]
check("In-dist clean %", agg["in_distribution_clean_pct"]["mean"], 86.94, 0.01)
check("In-dist clean std", agg["in_distribution_clean_pct"]["std"], 0.44, 0.01)
check("In-dist FPR %", agg["in_distribution_fpr_pct"]["mean"], 13.06, 0.01)
check("In-dist FPR std", agg["in_distribution_fpr_pct"]["std"], 0.44, 0.01)
check("OOD intercept %", agg["ood_toniot_intercept_pct"]["mean"], 61.72, 0.01)
check("Noise intercept %", agg["noise_intercept_pct"]["mean"], 100.0, 0.01)
check("Zero-fill rejected %", agg["zero_fill_rejected_pct"]["mean"], 100.0, 0.001)

# --- 3. fixed_fpr_evaluation.json (Table II: Main OOD Benchmark) ---
print("\n[3] Main Fixed-FPR OOD Benchmark (NF-ToN-IoT-v2)")
with open(CANONICAL_JSON / "fixed_fpr_evaluation.json") as f:
    ffe = json.load(f)
e2_det = ffe["metrics_per_failure_mode"]["E2_real_ood"]["detectors"]

# MSP
check("MSP AUROC", e2_det["msp"]["auroc"], 0.8501, 0.001)
check("MSP AUPRC", e2_det["msp"]["auprc"], 0.9073, 0.001)
check("MSP FPR@95TPR", e2_det["msp"]["fpr_at_95tpr"], 0.7212, 0.001)
check("MSP TPR@1% FPR", e2_det["msp"]["tpr_at_fpr"]["0.01"], 0.7922, 0.001)

# Cosine
check("Cosine AUROC", e2_det["cosine"]["auroc"], 0.9455, 0.001)
check("Cosine AUPRC", e2_det["cosine"]["auprc"], 0.9447, 0.001)
check("Cosine FPR@95TPR", e2_det["cosine"]["fpr_at_95tpr"], 0.1452, 0.001)
check("Cosine TPR@5% FPR", e2_det["cosine"]["tpr_at_fpr"]["0.05"], 0.7604, 0.001)

# Mahalanobis
check("Mahalanobis AUROC", e2_det["mahalanobis"]["auroc"], 0.9713, 0.001)
check("Mahalanobis AUPRC", e2_det["mahalanobis"]["auprc"], 0.9639, 0.001)
check("Mahalanobis FPR@95TPR", e2_det["mahalanobis"]["fpr_at_95tpr"], 0.0616, 0.001)
check("Mahalanobis TPR@0.1% FPR", e2_det["mahalanobis"]["tpr_at_fpr"]["0.001"], 0.0034, 0.001)
check("Mahalanobis TPR@1% FPR", e2_det["mahalanobis"]["tpr_at_fpr"]["0.01"], 0.0118, 0.001)
check("Mahalanobis TPR@5% FPR", e2_det["mahalanobis"]["tpr_at_fpr"]["0.05"], 0.7962, 0.001)

# Composite Drift Score
check("Composite Drift AUROC", e2_det["composite"]["auroc"], 0.9486, 0.001)
check("Composite Drift AUPRC", e2_det["composite"]["auprc"], 0.9376, 0.001)
check("Composite Drift FPR@95TPR", e2_det["composite"]["fpr_at_95tpr"], 0.1186, 0.001)
check("Composite TPR@1% FPR", e2_det["composite"]["tpr_at_fpr"]["0.01"], 0.0114, 0.001)
check("Composite TPR@5% FPR", e2_det["composite"]["tpr_at_fpr"]["0.05"], 0.7852, 0.001)

# --- 4. failure_mode_coverage_matrix.json (Table III) ---
print("\n[4] Failure-Mode Coverage Matrix (at 1% FPR)")
with open(CANONICAL_JSON / "failure_mode_coverage_matrix.json") as f:
    fmc = json.load(f)["matrix"]

check("E2 ToN-IoT MSP", fmc["E2_real_ood"]["msp"], 0.7922, 0.002)
check("E3 NaN/Inf Full System", fmc["E3_nan_inf_missing"]["full_system"], 0.7928, 0.002)
check("E3 NaN/Inf Validator", fmc["E3_nan_inf_missing"]["validator"], 0.691, 0.002)
check("E4 100% Zero-fill Validator", fmc["E4_zero_100pct"]["validator"], 1.000, 0.001)
check("E4 100% Zero-fill Full System", fmc["E4_zero_100pct"]["full_system"], 1.000, 0.001)
check("E4 100% Zero-fill MSP", fmc["E4_zero_100pct"]["msp"], 0.000, 0.001)
check("E4 100% Zero-fill Mahalanobis", fmc["E4_zero_100pct"]["mahalanobis"], 0.000, 0.001)
check("E5 Permutation Full System", fmc["E5_permutation"]["full_system"], 0.965, 0.002)
check("E5 Permutation Mahalanobis", fmc["E5_permutation"]["mahalanobis"], 0.964, 0.002)
check("E6 Unit Scale Mahalanobis", fmc["E6_scale_mismatch"]["mahalanobis"], 0.6874, 0.002)
check("E6 Unit Scale Full System", fmc["E6_scale_mismatch"]["full_system"], 0.682, 0.002)
check("E7 Outliers Mahalanobis", fmc["E7_extreme_plausible"]["mahalanobis"], 0.9046, 0.002)
check("E7 Outliers Full System", fmc["E7_extreme_plausible"]["full_system"], 0.9032, 0.002)
check("E8 Gaussian Noise Full System", fmc["E8_gaussian_noise"]["full_system"], 1.000, 0.001)

# --- 5. ablation_fixed_fpr.json (Table III Ablation) ---
print("\n[5] Component Ablation on Clean Shift")
with open(CANONICAL_JSON / "ablation_fixed_fpr.json") as f:
    abl = json.load(f)["ablation_results"]

check("Full System AUROC", abl["full"]["e2_auroc"], 0.8313, 0.001)
check("Full System AUPRC", abl["full"]["e2_auprc"], 0.8807, 0.001)
check("Full System FPR@95", abl["full"]["e2_fpr_at_95tpr"], 0.8036, 0.001)
check("Ablation -MSP AUROC", abl["no_msp"]["e2_auroc"], 0.9486, 0.001)
check("Ablation -MSP AUPRC", abl["no_msp"]["e2_auprc"], 0.9370, 0.001)
check("Ablation -MSP FPR@95", abl["no_msp"]["e2_fpr_at_95tpr"], 0.1186, 0.001)
check("Ablation -Cosine AUROC", abl["no_cosine"]["e2_auroc"], 0.8544, 0.001)
check("Ablation -Mahal AUROC", abl["no_mahal"]["e2_auroc"], 0.8229, 0.001)

# --- 6. cross_dataset_generalization.json & cross_model_replication.json (Table IV) ---
print("\n[6] Cross-Dataset & Cross-Model Generalization")
with open(CANONICAL_JSON / "cross_dataset_generalization.json") as f:
    cdg = json.load(f)["cross_dataset_results"]

ton_mlp = next(r for r in cdg if r["model"] == "ThreatMLP" and r["dataset"] == "NF-ToN-IoT-v2")
bot_mlp = next(r for r in cdg if r["model"] == "ThreatMLP" and r["dataset"] == "NF-BoT-IoT-v2")
ton_cnn = next(r for r in cdg if r["model"] == "ThreatCNN1D" and r["dataset"] == "NF-ToN-IoT-v2")
bot_cnn = next(r for r in cdg if r["model"] == "ThreatCNN1D" and r["dataset"] == "NF-BoT-IoT-v2")

check("ThreatMLP ToN-IoT Binary F1", ton_mlp["binary_macro_f1"], 0.7686, 0.001)
check("ThreatMLP ToN-IoT Mahal AUROC", ton_mlp["assurance"]["mahalanobis"]["auroc"], 0.8159, 0.001)
check("ThreatMLP ToN-IoT Assurance AUROC", ton_mlp["full_assurance_auroc"], 0.8184, 0.001)

check("ThreatMLP BoT-IoT Binary F1", bot_mlp["binary_macro_f1"], 0.6977, 0.001)
check("ThreatMLP BoT-IoT Mahal AUROC", bot_mlp["assurance"]["mahalanobis"]["auroc"], 0.9980, 0.001)
check("ThreatMLP BoT-IoT Assurance AUROC", bot_mlp["full_assurance_auroc"], 0.9983, 0.001)

check("ThreatCNN1D ToN-IoT Binary F1", ton_cnn["binary_macro_f1"], 0.5896, 0.001)
check("ThreatCNN1D ToN-IoT Mahal AUROC", ton_cnn["assurance"]["mahalanobis"]["auroc"], 0.3495, 0.001)
check("ThreatCNN1D ToN-IoT Assurance AUROC", ton_cnn["full_assurance_auroc"], 0.3549, 0.001)

check("ThreatCNN1D BoT-IoT Binary F1", bot_cnn["binary_macro_f1"], 0.2194, 0.001)
check("ThreatCNN1D BoT-IoT Mahal AUROC", bot_cnn["assurance"]["mahalanobis"]["auroc"], 0.4952, 0.001)
check("ThreatCNN1D BoT-IoT Assurance AUROC", bot_cnn["full_assurance_auroc"], 0.4952, 0.001)

# --- 7. resource_simulation_benchmark.json (Table V) ---
print("\n[7] Simulated Edge Resource Constraints & Quantization")
with open(CANONICAL_JSON / "resource_simulation_benchmark.json") as f:
    rsb = json.load(f)["profiles_benchmark"]

# Profile R0
r0_plain = rsb["R0"]["plain_onnx"]["aggregated"]
r0_sec = rsb["R0"]["semantic_shield"]["aggregated"]
check("R0 Plain Mean ms", r0_plain["mean_ms"], 0.2781, 0.001)
check("R0 Plain p95 ms", r0_plain["p95_ms"], 0.3863, 0.001)
check("R0 Plain Throughput", r0_plain["throughput_flows_sec"], 3581.6, 1.0)
check("R0 Plain RSS MB", r0_plain["peak_rss_mb"], 123.7, 0.1)
check("R0 Shield Mean ms", r0_sec["mean_ms"], 0.8883, 0.001)
check("R0 Shield p95 ms", r0_sec["p95_ms"], 1.1920, 0.001)
check("R0 Shield Throughput", r0_sec["throughput_flows_sec"], 1124.1, 1.0)
check("R0 Shield RSS MB", r0_sec["peak_rss_mb"], 124.9, 0.1)

# Profile R1
r1_plain = rsb["R1"]["plain_onnx"]["aggregated"]
r1_sec = rsb["R1"]["semantic_shield"]["aggregated"]
check("R1 Plain Mean ms", r1_plain["mean_ms"], 0.2705, 0.001)
check("R1 Plain p95 ms", r1_plain["p95_ms"], 0.3473, 0.001)
check("R1 Plain Throughput", r1_plain["throughput_flows_sec"], 3679.8, 1.0)
check("R1 Plain RSS MB", r1_plain["peak_rss_mb"], 122.4, 0.1)
check("R1 Shield Mean ms", r1_sec["mean_ms"], 0.9111, 0.001)
check("R1 Shield p95 ms", r1_sec["p95_ms"], 1.3404, 0.001)
check("R1 Shield Throughput", r1_sec["throughput_flows_sec"], 1100.7, 1.0)
check("R1 Shield RSS MB", r1_sec["peak_rss_mb"], 124.5, 0.1)

# Profile R1 Quantization
r1_q = rsb["R1"]["quantization_tradeoffs"]
check("FP32 Size MB", r1_q["FP32"]["size_mb"], 0.1765, 0.001)
check("FP32 Macro-F1", r1_q["FP32"]["macro_f1"], 0.7801, 0.001)
check("FP16 Size MB", r1_q["FP16"]["size_mb"], 0.0889, 0.001)
check("FP16 Macro-F1", r1_q["FP16"]["macro_f1"], 0.7802, 0.001)
check("INT8 Size MB", r1_q["INT8"]["size_mb"], 0.0479, 0.001)
check("INT8 Macro-F1", r1_q["INT8"]["macro_f1"], 0.4833, 0.001)
check("INT4 Size MB", r1_q["INT4"]["size_mb"], 0.0340, 0.001)
check("INT4 Macro-F1", r1_q["INT4"]["macro_f1"], 0.7627, 0.001)

# --- Summary ---
print("\n" + "=" * 70)
print(f"TOTAL CHECKS: {checks}")
if errors:
    print(f"MISMATCHES FOUND: {len(errors)}")
    for e in errors:
        print(e)
else:
    print("ALL CANONICAL NUMBERS MATCH PERFECTLY — 100% Truthful")
print("=" * 70)
