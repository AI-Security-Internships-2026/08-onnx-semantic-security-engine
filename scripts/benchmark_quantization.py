"""
Full Precision / Quantization Comparison Benchmark

Benchmarks FP32, FP16, and INT8 ONNX model variants on the same test set.
Reports for each variant:
  - Model size (MB)
  - Macro-F1 (accuracy cost of quantizing)
  - Mean +/- std inference latency with p50 / p95 / p99
  - Throughput (samples/sec)
  - 5,000+ repeated single-sample runs for variance

Outputs:
  - experiments/results/quantization_benchmark.json
  - experiments/images/quantization_benchmark.png
"""

import json
import time
import sys
import numpy as np
import joblib
import onnxruntime as ort
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS / "results"
IMAGES_DIR = EXPERIMENTS / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──
N_WARMUP = 500
N_LATENCY_RUNS = 5000
BATCH_SIZES = [1, 32, 128]  # For throughput measurement

# ── Model variants to benchmark ──
VARIANTS = {
    "FP32": EXPERIMENTS / "threat_mlp_nf_fp32.onnx",
    "FP16": EXPERIMENTS / "threat_mlp_nf_fp16.onnx",
    "INT8": EXPERIMENTS / "threat_mlp_nf_int8.onnx",
}

def get_model_size_mb(path: Path) -> float:
    """Get total model size including external data files."""
    total = path.stat().st_size
    # Check for external .data file (ONNX sometimes splits weights)
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    return total / (1024 * 1024)

def benchmark_variant(name: str, model_path: Path, X_test: np.ndarray, y_test: np.ndarray, encoder) -> dict:
    """Full benchmark for a single model variant."""
    print(f"\n{'='*70}")
    print(f"  Benchmarking: {name} ({model_path.name})")
    print(f"{'='*70}")
    
    if not model_path.exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None
    
    # ── Load model ──
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_tensor = session.get_inputs()[0]
    input_name = input_tensor.name
    output_names = [o.name for o in session.get_outputs()]
    
    # Cast input array to match model expected data type (e.g. float16 vs float32)
    if "float16" in input_tensor.type:
        X_eval = X_test.astype(np.float16)
    else:
        X_eval = X_test.astype(np.float32)
    
    # ── Model size ──
    size_mb = get_model_size_mb(model_path)
    print(f"  Model size: {size_mb:.4f} MB")
    
    # ── Accuracy (Macro-F1) on full test set ──
    print(f"  Computing Macro-F1 on {len(X_eval)} test samples...")
    all_preds = []
    batch_size = 512
    for i in range(0, len(X_eval), batch_size):
        batch = X_eval[i:i+batch_size]
        outputs = session.run(output_names, {input_name: batch})
        logits = outputs[0]
        preds = np.argmax(logits, axis=1)
        all_preds.extend(preds.tolist())
    
    all_preds = np.array(all_preds)
    macro_f1 = f1_score(y_test, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, all_preds)
    print(f"  Macro-F1: {macro_f1:.6f}")
    print(f"  Accuracy: {accuracy:.6f}")
    
    # ── Single-sample latency benchmark (N_LATENCY_RUNS iterations) ──
    print(f"  Latency benchmark: {N_WARMUP} warm-up + {N_LATENCY_RUNS} timed runs...")
    single_sample = X_eval[:1].copy()
    
    # Warm-up
    for _ in range(N_WARMUP):
        session.run(output_names, {input_name: single_sample})
    
    # Timed runs
    latencies = []
    for _ in range(N_LATENCY_RUNS):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: single_sample})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms
    
    latencies = np.array(latencies)
    lat_mean = float(np.mean(latencies))
    lat_std = float(np.std(latencies))
    lat_p50 = float(np.percentile(latencies, 50))
    lat_p95 = float(np.percentile(latencies, 95))
    lat_p99 = float(np.percentile(latencies, 99))
    lat_min = float(np.min(latencies))
    lat_max = float(np.max(latencies))
    
    print(f"  Latency: {lat_mean:.4f} +/- {lat_std:.4f} ms")
    print(f"  Percentiles: p50={lat_p50:.4f}, p95={lat_p95:.4f}, p99={lat_p99:.4f} ms")
    
    # ── Throughput (samples/sec) at different batch sizes ──
    throughput_results = {}
    for bs in BATCH_SIZES:
        batch = X_eval[:bs].copy()
        # Warm-up
        for _ in range(100):
            session.run(output_names, {input_name: batch})
        # Timed
        n_iters = 1000
        t0 = time.perf_counter()
        for _ in range(n_iters):
            session.run(output_names, {input_name: batch})
        t1 = time.perf_counter()
        elapsed = t1 - t0
        samples_per_sec = (bs * n_iters) / elapsed
        throughput_results[f"batch_{bs}"] = round(samples_per_sec, 1)
        print(f"  Throughput (batch={bs}): {samples_per_sec:,.1f} samples/sec")
    
    return {
        "variant": name,
        "model_file": model_path.name,
        "size_mb": round(size_mb, 4),
        "macro_f1": round(macro_f1, 6),
        "accuracy": round(accuracy, 6),
        "latency": {
            "n_warmup": N_WARMUP,
            "n_runs": N_LATENCY_RUNS,
            "mean_ms": round(lat_mean, 4),
            "std_ms": round(lat_std, 4),
            "min_ms": round(lat_min, 4),
            "max_ms": round(lat_max, 4),
            "p50_ms": round(lat_p50, 4),
            "p95_ms": round(lat_p95, 4),
            "p99_ms": round(lat_p99, 4),
        },
        "throughput_samples_per_sec": throughput_results,
    }


def generate_plots(results: dict):
    """Generate publication-quality comparison plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    variants = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ONNX Quantization Benchmark: FP32 vs FP16 vs INT8", fontsize=14, fontweight='bold')
    
    colors = {'FP32': '#2196F3', 'FP16': '#4CAF50', 'INT8': '#FF9800'}
    
    # ── Panel 1: Model Size ──
    ax = axes[0, 0]
    sizes = [results[v]["size_mb"] for v in variants]
    bars = ax.bar(variants, sizes, color=[colors[v] for v in variants], edgecolor='black', linewidth=0.5)
    for bar, s in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, f"{s:.3f} MB", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Size Comparison")
    ax.set_ylim(0, max(sizes) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 2: Macro-F1 ──
    ax = axes[0, 1]
    f1s = [results[v]["macro_f1"] for v in variants]
    bars = ax.bar(variants, f1s, color=[colors[v] for v in variants], edgecolor='black', linewidth=0.5)
    for bar, f in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{f:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Classification Accuracy (Macro-F1)")
    ax.set_ylim(min(f1s) * 0.9, max(f1s) * 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 3: Latency Distribution ──
    ax = axes[1, 0]
    positions = range(len(variants))
    for i, v in enumerate(variants):
        lat = results[v]["latency"]
        mean = lat["mean_ms"]
        std = lat["std_ms"]
        p50 = lat["p50_ms"]
        p95 = lat["p95_ms"]
        p99 = lat["p99_ms"]
        
        ax.bar(i, mean, color=colors[v], edgecolor='black', linewidth=0.5, alpha=0.7, label=v)
        ax.errorbar(i, mean, yerr=std, fmt='o', color='black', capsize=5, markersize=4)
        
        text = f"mean={mean:.3f}\np50={p50:.3f}\np95={p95:.3f}\np99={p99:.3f}"
        ax.text(i, mean + std + 0.003, text, ha='center', va='bottom', fontsize=7, family='monospace')
    
    ax.set_xticks(list(positions))
    ax.set_xticklabels(variants)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Inference Latency (mean +/- std, {N_LATENCY_RUNS} runs)")
    ax.grid(axis='y', alpha=0.3)
    
    # ── Panel 4: Throughput ──
    ax = axes[1, 1]
    x = np.arange(len(BATCH_SIZES))
    width = 0.25
    for i, v in enumerate(variants):
        tp = results[v]["throughput_samples_per_sec"]
        vals = [tp[f"batch_{bs}"] for bs in BATCH_SIZES]
        bars = ax.bar(x + i * width, vals, width, label=v, color=colors[v], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val/1000:.1f}k", ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Batch={bs}" for bs in BATCH_SIZES])
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title("Inference Throughput")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = IMAGES_DIR / "quantization_benchmark.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] {plot_path}")


def main():
    print("=" * 70)
    print("  FULL PRECISION / QUANTIZATION COMPARISON BENCHMARK")
    print("=" * 70)
    
    # ── Load test data ──
    X_test_path = EXPERIMENTS / "X_test_nf.npy"
    y_test_path = EXPERIMENTS / "y_test_nf.npy"
    encoder_path = EXPERIMENTS / "label_encoder_nf.joblib"
    
    for p in [X_test_path, y_test_path, encoder_path]:
        if not p.exists():
            print(f"[ERROR] Missing: {p}")
            sys.exit(1)
    
    X_test = np.load(X_test_path).astype(np.float32)
    y_test = np.load(y_test_path)
    encoder = joblib.load(encoder_path)
    
    print(f"Test set: {len(X_test)} samples, {X_test.shape[1]} features")
    print(f"Classes: {len(encoder.classes_)}")
    
    # ── Check which variants exist ──
    available = {}
    for name, path in VARIANTS.items():
        if path.exists():
            available[name] = path
            print(f"  [OK] {name}: {path.name}")
        else:
            print(f"  [MISSING] {name}: {path.name}")
    
    if not available:
        print("[ERROR] No ONNX model variants found!")
        sys.exit(1)
    
    # ── Benchmark each variant ──
    results = {}
    for name, path in available.items():
        result = benchmark_variant(name, path, X_test, y_test, encoder)
        if result:
            results[name] = result
    
    # ── Compute relative metrics ──
    if "FP32" in results:
        fp32_f1 = results["FP32"]["macro_f1"]
        fp32_size = results["FP32"]["size_mb"]
        fp32_lat = results["FP32"]["latency"]["mean_ms"]
        for name, r in results.items():
            r["f1_drop_vs_fp32_pct"] = round((1 - r["macro_f1"] / fp32_f1) * 100, 2) if fp32_f1 > 0 else 0
            r["size_reduction_vs_fp32_pct"] = round((1 - r["size_mb"] / fp32_size) * 100, 2) if fp32_size > 0 else 0
            r["latency_change_vs_fp32_pct"] = round((r["latency"]["mean_ms"] / fp32_lat - 1) * 100, 2) if fp32_lat > 0 else 0
    
    # ── Save JSON ──
    output = {
        "experiment": "Full Precision/Quantization Comparison Benchmark",
        "test_samples": len(X_test),
        "n_warmup": N_WARMUP,
        "n_latency_runs": N_LATENCY_RUNS,
        "batch_sizes_throughput": BATCH_SIZES,
        "variants": results,
    }
    
    json_path = RESULTS_DIR / "quantization_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {json_path}")
    
    # ── Generate plots ──
    print("\nGenerating publication-quality benchmark plots...")
    generate_plots(results)
    
    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"  QUANTIZATION BENCHMARK — SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"{'Variant':<8} {'Size(MB)':>10} {'Macro-F1':>10} {'F1 Drop%':>10} {'Latency(ms)':>14} {'p95(ms)':>10} {'p99(ms)':>10} {'Throughput':>12}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        lat = r["latency"]
        tp = r["throughput_samples_per_sec"].get("batch_1", 0)
        drop = r.get("f1_drop_vs_fp32_pct", 0)
        print(f"{name:<8} {r['size_mb']:>10.4f} {r['macro_f1']:>10.6f} {drop:>9.2f}% {lat['mean_ms']:>8.4f}+/-{lat['std_ms']:.4f} {lat['p95_ms']:>10.4f} {lat['p99_ms']:>10.4f} {tp:>10.1f}/s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
