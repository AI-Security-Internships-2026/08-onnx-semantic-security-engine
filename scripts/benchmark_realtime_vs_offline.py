"""
Real-Time vs. Offline Inference Comparison Benchmark

Compares:
  1. Offline In-Memory Inference:
     - Plain ONNX (single sample & batch=128)
     - Semantic Security Engine (single sample & batch=128)
  2. Real-Time Streaming Service (HTTP / FastAPI endpoint):
     - Plain REST endpoint (/predict)
     - Secure REST endpoint (/predict/secure) with full semantic analysis
     - Measures End-to-End Client Latency (Network + Serialization + Processing) vs Server Internal Latency

Outputs:
  - experiments/results/realtime_vs_offline_benchmark.json
  - experiments/images/realtime_vs_offline.png
"""

import json
import time
import sys
import os
import argparse
import requests
import numpy as np
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.special import softmax

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.config_loader import load_paper_config, get_provenance_metadata

EXPERIMENTS = BASE_DIR / "experiments"
RESULTS_DIR = EXPERIMENTS / "results"
IMAGES_DIR = EXPERIMENTS / "images"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──
N_OFFLINE_WARMUP = 200
N_OFFLINE_RUNS = 2000
N_REALTIME_RUNS = 500
SERVER_URL = os.environ.get("ENGINE_URL", "http://localhost:8000")

def run_offline_benchmarks(X_test: np.ndarray, model_path: Path):
    """Run offline in-memory benchmarks for plain ONNX and Semantic Engine."""
    print(f"\n{'='*70}")
    print(f"  [1/2] RUNNING OFFLINE (IN-MEMORY) BENCHMARKS")
    print(f"{'='*70}")
    
    from src.semantic_analyzer import SemanticSecurityEngine
    
    # ── Load ONNX Model ──
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    scaler = joblib.load(EXPERIMENTS / "standard_scaler_nf.joblib")
    
    # ── Initialize Semantic Engine from frozen paper config ──
    engine = SemanticSecurityEngine.from_config("configs/paper_v1.yaml")
    
    single_raw = X_test[0].tolist()
    single_scaled = scaler.transform([single_raw]).astype(np.float32)
    
    # --- 1A. Plain ONNX (Single Flow, In-Memory) ---
    print(f"  Benchmarking Plain ONNX (single sample, {N_OFFLINE_RUNS} runs)...")
    for _ in range(N_OFFLINE_WARMUP):
        session.run(output_names, {input_name: single_scaled})
    
    latencies_plain_single = []
    for _ in range(N_OFFLINE_RUNS):
        t0 = time.perf_counter()
        session.run(output_names, {input_name: single_scaled})
        t1 = time.perf_counter()
        latencies_plain_single.append((t1 - t0) * 1000)
    
    # --- 1B. Full Semantic Engine (Single Flow, In-Memory) ---
    print(f"  Benchmarking Semantic Security Engine (single sample, {N_OFFLINE_RUNS} runs)...")
    for _ in range(N_OFFLINE_WARMUP):
        outputs = session.run(output_names, {input_name: single_scaled})
        logits = outputs[0][0]
        embs = outputs[1][0] if len(outputs) > 1 else None
        probs = softmax(logits)
        engine.analyze(raw_features=np.array(single_raw), softmax_probs=probs, embedding=embs)
    
    latencies_semantic_single = []
    for _ in range(N_OFFLINE_RUNS):
        t0 = time.perf_counter()
        scaled = scaler.transform([single_raw]).astype(np.float32)
        outputs = session.run(output_names, {input_name: scaled})
        logits = outputs[0][0]
        embs = outputs[1][0] if len(outputs) > 1 else None
        probs = softmax(logits)
        engine.analyze(raw_features=np.array(single_raw), softmax_probs=probs, embedding=embs)
        t1 = time.perf_counter()
        latencies_semantic_single.append((t1 - t0) * 1000)
    
    # --- 1C. Plain ONNX Batch=128 Throughput ---
    batch_raw = X_test[:128]
    batch_scaled = scaler.transform(batch_raw).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(500):
        session.run(output_names, {input_name: batch_scaled})
    t1 = time.perf_counter()
    tp_plain_batch128 = (128 * 500) / (t1 - t0)
    
    def compute_stats(lats):
        a = np.array(lats)
        return {
            "mean_ms": round(float(np.mean(a)), 4),
            "std_ms": round(float(np.std(a)), 4),
            "p50_ms": round(float(np.percentile(a, 50)), 4),
            "p95_ms": round(float(np.percentile(a, 95)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "throughput_flows_per_sec": round(1000.0 / float(np.mean(a)), 1)
        }
    
    res_plain = compute_stats(latencies_plain_single)
    res_semantic = compute_stats(latencies_semantic_single)
    
    print(f"  -> Plain ONNX Offline:     {res_plain['mean_ms']:.4f} +/- {res_plain['std_ms']:.4f} ms ({res_plain['throughput_flows_per_sec']:,.1f} flows/s)")
    print(f"  -> Semantic Engine Offline: {res_semantic['mean_ms']:.4f} +/- {res_semantic['std_ms']:.4f} ms ({res_semantic['throughput_flows_per_sec']:,.1f} flows/s)")
    print(f"  -> Plain Batch-128 Throughput: {tp_plain_batch128:,.1f} flows/s")
    
    return {
        "plain_onnx_single": res_plain,
        "semantic_engine_single": res_semantic,
        "plain_onnx_batch128_throughput": round(tp_plain_batch128, 1)
    }

def run_realtime_benchmarks(X_test: np.ndarray, server_url: str):
    """Run real-time streaming benchmarks against live FastAPI server."""
    print(f"\n{'='*70}")
    print(f"  [2/2] RUNNING REAL-TIME STREAMING (HTTP REST API) BENCHMARKS")
    print(f"  Target Server: {server_url}")
    print(f"{'='*70}")
    
    # Check server availability
    try:
        r = requests.get(f"{server_url}/health", timeout=3)
        if r.status_code != 200:
            print(f"  [ERROR] Server returned status code {r.status_code}")
            return None
        print(f"  [OK] Connected to Inference Server: {r.json()}")
    except Exception as e:
        print(f"  [ERROR] Cannot connect to {server_url}/health: {e}")
        return None
    
    http_session = requests.Session()
    single_raw = X_test[0].tolist()
    payload = {"features": single_raw}
    
    # --- 2A. Plain REST Endpoint (/predict) ---
    print(f"  Streaming {N_REALTIME_RUNS} individual requests to /predict...")
    client_lats_plain = []
    server_lats_plain = []
    
    # Warm-up
    for _ in range(50):
        http_session.post(f"{server_url}/predict", json=payload)
    
    for _ in range(N_REALTIME_RUNS):
        t0 = time.perf_counter()
        resp = http_session.post(f"{server_url}/predict", json=payload)
        t1 = time.perf_counter()
        client_lats_plain.append((t1 - t0) * 1000)
        if resp.status_code == 200:
            server_lats_plain.append(resp.json().get("latency_ms", 0.0))
    
    # --- 2B. Secure REST Endpoint (/predict/secure) ---
    print(f"  Streaming {N_REALTIME_RUNS} individual requests to /predict/secure...")
    client_lats_secure = []
    server_lats_secure = []
    
    # Warm-up
    for _ in range(50):
        http_session.post(f"{server_url}/predict/secure", json=payload)
    
    for _ in range(N_REALTIME_RUNS):
        t0 = time.perf_counter()
        resp = http_session.post(f"{server_url}/predict/secure", json=payload)
        t1 = time.perf_counter()
        client_lats_secure.append((t1 - t0) * 1000)
        if resp.status_code == 200:
            server_lats_secure.append(resp.json().get("latency_ms", 0.0))
    
    def compute_rt_stats(client_lats, server_lats):
        c = np.array(client_lats)
        s = np.array(server_lats) if server_lats else np.array([0.0])
        net_overhead = float(np.mean(c) - np.mean(s))
        return {
            "client_e2e_latency": {
                "mean_ms": round(float(np.mean(c)), 4),
                "std_ms": round(float(np.std(c)), 4),
                "p50_ms": round(float(np.percentile(c, 50)), 4),
                "p95_ms": round(float(np.percentile(c, 95)), 4),
                "p99_ms": round(float(np.percentile(c, 99)), 4),
            },
            "server_inference_latency": {
                "mean_ms": round(float(np.mean(s)), 4),
                "std_ms": round(float(np.std(s)), 4),
                "p50_ms": round(float(np.percentile(s, 50)), 4),
                "p95_ms": round(float(np.percentile(s, 95)), 4),
                "p99_ms": round(float(np.percentile(s, 99)), 4),
            },
            "network_and_serialization_overhead_ms": round(net_overhead, 4),
            "streaming_throughput_flows_per_sec": round(1000.0 / float(np.mean(c)), 1)
        }
    
    res_plain_rt = compute_rt_stats(client_lats_plain, server_lats_plain)
    res_secure_rt = compute_rt_stats(client_lats_secure, server_lats_secure)
    
    print(f"  -> Plain REST Client E2E:   {res_plain_rt['client_e2e_latency']['mean_ms']:.4f} +/- {res_plain_rt['client_e2e_latency']['std_ms']:.4f} ms (Server: {res_plain_rt['server_inference_latency']['mean_ms']:.4f} ms, Network Overhead: {res_plain_rt['network_and_serialization_overhead_ms']:.4f} ms)")
    print(f"  -> Secure REST Client E2E:  {res_secure_rt['client_e2e_latency']['mean_ms']:.4f} +/- {res_secure_rt['client_e2e_latency']['std_ms']:.4f} ms (Server: {res_secure_rt['server_inference_latency']['mean_ms']:.4f} ms, Network Overhead: {res_secure_rt['network_and_serialization_overhead_ms']:.4f} ms)")
    print(f"  -> Real-Time Streaming Rate: {res_secure_rt['streaming_throughput_flows_per_sec']:,.1f} flows/s")
    
    return {
        "plain_rest_streaming": res_plain_rt,
        "secure_rest_streaming": res_secure_rt
    }

def generate_comparison_plots(offline_res: dict, realtime_res: dict, figures_dir=None):
    """Generate side-by-side visualization comparing Offline vs Real-Time Streaming."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Real-Time Streaming Service vs. Offline In-Memory Inference", fontsize=14, fontweight='bold')
    
    # ── Panel 1: Latency Decomposition ──
    ax = axes[0]
    categories = ['Offline Plain', 'Offline Secure', 'Real-Time Plain (E2E)', 'Real-Time Secure (E2E)']
    
    off_plain_lat = offline_res["plain_onnx_single"]["mean_ms"]
    off_sec_lat = offline_res["semantic_engine_single"]["mean_ms"]
    
    if realtime_res:
        rt_plain_server = realtime_res["plain_rest_streaming"]["server_inference_latency"]["mean_ms"]
        rt_plain_net = realtime_res["plain_rest_streaming"]["network_and_serialization_overhead_ms"]
        rt_sec_server = realtime_res["secure_rest_streaming"]["server_inference_latency"]["mean_ms"]
        rt_sec_net = realtime_res["secure_rest_streaming"]["network_and_serialization_overhead_ms"]
        
        server_lats = [off_plain_lat, off_sec_lat, rt_plain_server, rt_sec_server]
        net_lats = [0.0, 0.0, rt_plain_net, rt_sec_net]
        
        b1 = ax.bar(categories, server_lats, label='Model / Engine Inference', color='#2196F3', edgecolor='black', linewidth=0.5)
        b2 = ax.bar(categories, net_lats, bottom=server_lats, label='HTTP Network + JSON Serialization', color='#FF9800', edgecolor='black', linewidth=0.5)
        
        for i, total in enumerate([off_plain_lat, off_sec_lat, rt_plain_server + rt_plain_net, rt_sec_server + rt_sec_net]):
            ax.text(i, total + 0.1, f"{total:.2f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax.bar(categories[:2], [off_plain_lat, off_sec_lat], color='#2196F3', edgecolor='black')
    
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Breakdown: Compute vs Network Overhead")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    # ── Panel 2: Throughput Comparison ──
    ax2 = axes[1]
    tp_cats = ['Offline Batch-128', 'Offline Single Stream', 'Offline Secure Stream']
    tp_vals = [
        offline_res["plain_onnx_batch128_throughput"],
        offline_res["plain_onnx_single"]["throughput_flows_per_sec"],
        offline_res["semantic_engine_single"]["throughput_flows_per_sec"]
    ]
    colors = ['#4CAF50', '#2196F3', '#9C27B0']
    
    if realtime_res:
        tp_cats.append('Real-Time Streaming Service')
        tp_vals.append(realtime_res["secure_rest_streaming"]["streaming_throughput_flows_per_sec"])
        colors.append('#FF5722')
    
    bars = ax2.bar(tp_cats, tp_vals, color=colors, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, tp_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f"{val:,.0f}/s", ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel("Throughput (flows/second)")
    ax2.set_title("Processing Throughput Across Operational Modes")
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
    
    plt.tight_layout()
    if figures_dir is not None:
        canonical_plot = Path(figures_dir) / "realtime_vs_offline.png"
        plt.savefig(canonical_plot, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {canonical_plot}")

    plot_path = IMAGES_DIR / "realtime_vs_offline.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Real-time vs offline benchmark.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(EXPERIMENTS / "paper_results" / "json"),
        help="Directory to save realtime_vs_offline_benchmark.json",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(EXPERIMENTS / "paper_results" / "figures"),
        help="Directory to save realtime vs offline plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  REAL-TIME VS OFFLINE INFERENCE COMPARISON BENCHMARK")
    print("=" * 70)
    
    # ── Load test data ──
    X_test_path = EXPERIMENTS / "X_test_nf.npy"
    if not X_test_path.exists():
        print(f"[ERROR] Missing: {X_test_path}")
        sys.exit(1)
    
    X_test = np.load(X_test_path)
    model_path = EXPERIMENTS / "threat_mlp_nf_fp32.onnx"
    
    # ── Run Offline Benchmarks ──
    offline_results = run_offline_benchmarks(X_test, model_path)
    
    # ── Run Real-Time Benchmarks ──
    realtime_results = run_realtime_benchmarks(X_test, SERVER_URL)
    
    if realtime_results is None:
        print("\n[NOTE] Live server was not detected. To include live REST results, start the Docker engine or uvicorn server:")
        print("       docker compose -f docker/docker-compose.yml up -d security-engine")
        print("       OR: uvicorn src.inference_engine:app --port 8000")
        sys.exit(2)
    
    # ── Save Consolidated Results ──
    consolidated = {
        "provenance": get_provenance_metadata(),
        "experiment": "Real-Time Streaming vs. Offline Batch Inference Comparison",
        "model": "ThreatMLP NF (13 features, FP32)",
        "offline_in_memory": offline_results,
        "realtime_http_streaming": realtime_results,
    }
    
    canonical_json = output_dir / "realtime_vs_offline_benchmark.json"
    with open(canonical_json, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"\n[SAVED] {canonical_json}")

    latency_json = output_dir / "latency_benchmark.json"
    with open(latency_json, "w") as f:
        json.dump(consolidated, f, indent=2)
    print(f"[SAVED] {latency_json}")

    legacy_json = RESULTS_DIR / "realtime_vs_offline_benchmark.json"
    if legacy_json.parent.exists() and canonical_json != legacy_json:
        with open(legacy_json, "w") as f:
            json.dump(consolidated, f, indent=2)
        print(f"[MIRRORED] {legacy_json}")
    
    # ── Generate Plots ──
    generate_comparison_plots(offline_results, realtime_results, figures_dir=figures_dir)
    
    # ── Print Summary Table ──
    print(f"\n{'='*95}")
    print(f"  REAL-TIME VS OFFLINE COMPARISON — SUMMARY TABLE")
    print(f"{'='*95}")
    print(f"{'Operational Mode':<32} {'Latency (Mean ± Std)':>22} {'p95':>10} {'p99':>10} {'Throughput':>15}")
    print("-" * 95)
    
    off_p = offline_results["plain_onnx_single"]
    off_s = offline_results["semantic_engine_single"]
    print(f"{'Offline Plain (In-Memory)':<32} {off_p['mean_ms']:>12.4f}±{off_p['std_ms']:.4f} ms {off_p['p95_ms']:>8.4f} ms {off_p['p99_ms']:>8.4f} ms {off_p['throughput_flows_per_sec']:>13.1f}/s")
    print(f"{'Offline Secure (In-Memory)':<32} {off_s['mean_ms']:>12.4f}±{off_s['std_ms']:.4f} ms {off_s['p95_ms']:>8.4f} ms {off_s['p99_ms']:>8.4f} ms {off_s['throughput_flows_per_sec']:>13.1f}/s")
    
    if realtime_res := realtime_results:
        rt_p = realtime_res["plain_rest_streaming"]["client_e2e_latency"]
        rt_s = realtime_res["secure_rest_streaming"]["client_e2e_latency"]
        rt_p_tp = realtime_res["plain_rest_streaming"]["streaming_throughput_flows_per_sec"]
        rt_s_tp = realtime_res["secure_rest_streaming"]["streaming_throughput_flows_per_sec"]
        print(f"{'Real-Time Plain REST (E2E)':<32} {rt_p['mean_ms']:>12.4f}±{rt_p['std_ms']:.4f} ms {rt_p['p95_ms']:>8.4f} ms {rt_p['p99_ms']:>8.4f} ms {rt_p_tp:>13.1f}/s")
        print(f"{'Real-Time Secure REST (E2E)':<32} {rt_s['mean_ms']:>12.4f}±{rt_s['std_ms']:.4f} ms {rt_s['p95_ms']:>8.4f} ms {rt_s['p99_ms']:>8.4f} ms {rt_s_tp:>13.1f}/s")
    print(f"{'='*95}")

if __name__ == "__main__":
    main()
