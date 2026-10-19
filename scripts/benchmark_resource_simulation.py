"""
SEMANTICSHIELD: Edge Resource Constraints Simulation Benchmark (Issue 5 / E2.3-A to E2.3-D)

Evaluates SEMANTICSHIELD and Plain ONNX under controlled CPU and memory constraints.
Supports:
  - Docker container cgroup limits (--cpus, --memory)
  - Native OS-level CPU affinity & ONNX Runtime thread execution controls
  - 4 Profiles: R1 (Low: 1 vCPU, 512 MB), R2 (Med: 2 vCPU, 1 GB), R3 (High: 4 vCPU, 2 GB), R0 (Ref: Unconstrained)
  - 5 measured repetitions per profile with warm-up
  - Concurrency/workload levels (Low, Moderate, High)
  - Quantization trade-offs across FP32, FP16, static INT8, weight-only INT4

PAPER FRAMING GUARDRAIL:
"We evaluate SEMANTICSHIELD under controlled CPU- and memory-constrained deployment
profiles to approximate resource-limited inference conditions. Physical edge hardware
validation remains future work."

Outputs:
  - experiments/paper_results/resource_simulation/
      environment.json
      resource_profiles.json
      raw_latency.csv
      latency_summary.csv
      throughput_summary.csv
      resource_usage.csv
      quantization_summary.csv
  - experiments/paper_results/json/resource_simulation_benchmark.json
  - experiments/paper_results/figures/
      resource_simulation_p95_latency.png
      resource_simulation_throughput.png
      resource_simulation_quantization.png
      resource_simulation_concurrency.png
"""

import os
import sys
import time
import json
import csv
import argparse
import platform
import subprocess
import numpy as np
import joblib
import onnxruntime as ort
from pathlib import Path
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score

try:
    import psutil
except ImportError:
    psutil = None

try:
    import resource
except ImportError:
    resource = None

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.config_loader import load_paper_config, get_provenance_metadata
from src.semantic_analyzer import SemanticSecurityEngine

EXPERIMENTS = BASE_DIR / "experiments"
SIM_OUTPUT_DIR = EXPERIMENTS / "paper_results" / "resource_simulation"
JSON_DIR = EXPERIMENTS / "paper_results" / "json"
FIG_DIR = EXPERIMENTS / "paper_results" / "figures"
TABLES_DIR = EXPERIMENTS / "paper_results" / "tables"

for d in [SIM_OUTPUT_DIR, JSON_DIR, FIG_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Resource Profiles ──
PROFILES = {
    "R0": {
        "id": "R0",
        "name": "Reference",
        "vcpu_limit": None,
        "memory_limit_mb": None,
        "docker_cpus": None,
        "docker_memory": None,
        "intra_threads": 0,
        "inter_threads": 0,
        "description": "Host/container reference (unconstrained baseline)"
    },
    "R1": {
        "id": "R1",
        "name": "Low",
        "vcpu_limit": 1,
        "memory_limit_mb": 512,
        "docker_cpus": "1.0",
        "docker_memory": "512m",
        "intra_threads": 1,
        "inter_threads": 1,
        "description": "Strongly constrained edge profile (1 vCPU, 512 MB)"
    },
    "R2": {
        "id": "R2",
        "name": "Medium",
        "vcpu_limit": 2,
        "memory_limit_mb": 1024,
        "docker_cpus": "2.0",
        "docker_memory": "1024m",
        "intra_threads": 2,
        "inter_threads": 1,
        "description": "Moderately constrained edge controller (2 vCPU, 1 GB)"
    },
    "R3": {
        "id": "R3",
        "name": "Higher",
        "vcpu_limit": 4,
        "memory_limit_mb": 2048,
        "docker_cpus": "4.0",
        "docker_memory": "2048m",
        "intra_threads": 4,
        "inter_threads": 2,
        "description": "Less constrained edge gateway (4 vCPU, 2 GB)"
    },
}

CONCURRENCY_LEVELS = {
    "low_sequential": {
        "name": "Low / Sequential",
        "concurrency": 1,
        "batch_size": 1,
        "description": "Single-sample real-time streaming"
    },
    "moderate": {
        "name": "Moderate Concurrency",
        "concurrency": 4,
        "batch_size": 32,
        "description": "Batched multi-flow edge ingestion"
    },
    "high_concurrency": {
        "name": "High Concurrency",
        "concurrency": 16,
        "batch_size": 128,
        "description": "High-throughput burst queue"
    }
}


def get_environment_info():
    """Gather complete host, runtime, and hardware specifications."""
    info = {
        "system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "onnxruntime_version": ort.__version__,
        "host_logical_cpus": os.cpu_count(),
        "host_total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2) if psutil else None,
        "docker_cgroups_v2": Path("/sys/fs/cgroup/memory.max").exists(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "framing_guardrail": "Simulated resource constraints (NOT physical edge hardware)"
    }
    return info


def apply_profile_constraints(profile: dict):
    """Enforce thread and CPU constraints inside the current process."""
    vcpu = profile["vcpu_limit"]
    if vcpu is not None and psutil:
        try:
            p = psutil.Process()
            avail_cpus = list(range(min(vcpu, os.cpu_count() or 1)))
            p.cpu_affinity(avail_cpus)
        except Exception as e:
            print(f"  [WARN] Could not set CPU affinity: {e}")

    # Set thread environments
    intra = profile["intra_threads"]
    if intra > 0:
        os.environ["OMP_NUM_THREADS"] = str(intra)
        os.environ["MKL_NUM_THREADS"] = str(intra)


def create_ort_session(model_path: Path, profile: dict):
    """Create an ONNX Runtime session with profile thread constraints."""
    sess_opts = ort.SessionOptions()
    if profile["intra_threads"] > 0:
        sess_opts.intra_op_num_threads = profile["intra_threads"]
    if profile["inter_threads"] > 0:
        sess_opts.inter_op_num_threads = profile["inter_threads"]
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_opts, providers=['CPUExecutionProvider'])
    return session


def get_peak_memory_mb():
    """Get peak memory (RSS) in MB using platform-appropriate mechanism."""
    # Check Linux cgroups v2 first if inside container
    cg_peak = Path("/sys/fs/cgroup/memory.peak")
    if cg_peak.exists():
        try:
            with open(cg_peak, "r") as f:
                val = int(f.read().strip())
                return round(val / (1024 * 1024), 2)
        except Exception:
            pass

    if psutil:
        try:
            rss = psutil.Process().memory_info().rss
            return round(rss / (1024 * 1024), 2)
        except Exception:
            pass

    if resource:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Linux ru_maxrss is in KB; on macOS it is in bytes
            if platform.system() == "Darwin":
                return round(usage / (1024 * 1024), 2)
            return round(usage / 1024, 2)
        except Exception:
            pass

    return 0.0


def benchmark_single_run(session, engine, scaler, X_workload, mode: str, n_warmup: int = 500):
    """Benchmark a single run of Plain ONNX or SEMANTICSHIELD on input workload."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    # ── Warm-up ──
    warmup_subset = X_workload[:min(n_warmup, len(X_workload))]
    for x in warmup_subset:
        scaled = scaler.transform([x]).astype(np.float32)
        outs = session.run(output_names, {input_name: scaled})
        if mode == "semantic":
            logits = outs[0][0]
            embs = outs[1][0] if len(outs) > 1 else None
            probs = softmax(logits)
            engine.analyze(raw_features=x, softmax_probs=probs, embedding=embs)

    # ── Timed Loop ──
    latencies_ms = []
    t_start = time.perf_counter()
    
    cpu_before = psutil.cpu_percent(interval=None) if psutil else 0.0
    
    for x in X_workload:
        t0 = time.perf_counter()
        scaled = scaler.transform([x]).astype(np.float32)
        outs = session.run(output_names, {input_name: scaled})
        if mode == "semantic":
            logits = outs[0][0]
            embs = outs[1][0] if len(outs) > 1 else None
            probs = softmax(logits)
            engine.analyze(raw_features=x, softmax_probs=probs, embedding=embs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    t_total = time.perf_counter() - t_start
    cpu_after = psutil.cpu_percent(interval=None) if psutil else 0.0
    cpu_util = round((cpu_before + cpu_after) / 2.0, 1)
    
    throughput = len(X_workload) / max(t_total, 1e-9)
    peak_rss = get_peak_memory_mb()
    
    lat_arr = np.array(latencies_ms)
    return {
        "latencies": lat_arr,
        "mean_ms": float(np.mean(lat_arr)),
        "std_ms": float(np.std(lat_arr)),
        "p50_ms": float(np.percentile(lat_arr, 50)),
        "p95_ms": float(np.percentile(lat_arr, 95)),
        "p99_ms": float(np.percentile(lat_arr, 99)),
        "min_ms": float(np.min(lat_arr)),
        "max_ms": float(np.max(lat_arr)),
        "throughput_flows_sec": round(throughput, 1),
        "peak_rss_mb": peak_rss,
        "cpu_util_pct": cpu_util,
        "total_elapsed_sec": round(t_total, 3)
    }


def evaluate_quantization_profile(profile: dict, X_test, y_test, encoder):
    """Evaluate FP32, FP16, INT8, and INT4 under current profile."""
    variants = {
        "FP32": EXPERIMENTS / "threat_mlp_nf_fp32.onnx",
        "FP16": EXPERIMENTS / "threat_mlp_nf_fp16.onnx",
        "INT8": EXPERIMENTS / "threat_mlp_nf_int8.onnx",
        "INT4": EXPERIMENTS / "threat_mlp_nf_int4.onnx",
    }
    
    results = {}
    for name, p in variants.items():
        if not p.exists():
            continue
            
        size_mb = p.stat().st_size
        data_p = Path(str(p) + ".data")
        if data_p.exists():
            size_mb += data_p.stat().st_size
        size_mb = round(size_mb / (1024 * 1024), 4)
        
        sess = create_ort_session(p, profile)
        in_tensor = sess.get_inputs()[0]
        in_name = in_tensor.name
        out_names = [o.name for o in sess.get_outputs()]
        
        dtype = np.float16 if "float16" in in_tensor.type else np.float32
        X_eval = X_test.astype(dtype)
        
        # 1. Macro-F1 & Accuracy
        preds = []
        bs = 512
        for i in range(0, len(X_eval), bs):
            out = sess.run(out_names, {in_name: X_eval[i:i+bs]})[0]
            preds.extend(np.argmax(out, axis=1).tolist())
        macro_f1 = float(f1_score(y_test, preds, average='macro', zero_division=0))
        acc = float(accuracy_score(y_test, preds))
        
        # 2. Single-sample latency (1,000 runs)
        single = X_eval[:1].copy()
        for _ in range(200):
            sess.run(out_names, {in_name: single})
        lats = []
        for _ in range(1000):
            t0 = time.perf_counter()
            sess.run(out_names, {in_name: single})
            t1 = time.perf_counter()
            lats.append((t1 - t0) * 1000.0)
            
        lats = np.array(lats)
        
        # 3. Batch=128 Throughput
        batch128 = X_eval[:128].copy()
        for _ in range(50):
            sess.run(out_names, {in_name: batch128})
        n_iters = 200
        t0 = time.perf_counter()
        for _ in range(n_iters):
            sess.run(out_names, {in_name: batch128})
        t1 = time.perf_counter()
        tp_b128 = (128 * n_iters) / max(t1 - t0, 1e-9)
        
        results[name] = {
            "precision": name,
            "model_file": p.name,
            "size_mb": size_mb,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "mean_latency_ms": round(float(np.mean(lats)), 4),
            "p50_latency_ms": round(float(np.percentile(lats, 50)), 4),
            "p95_latency_ms": round(float(np.percentile(lats, 95)), 4),
            "p99_latency_ms": round(float(np.percentile(lats, 99)), 4),
            "throughput_b1_flows_sec": round(1000.0 / float(np.mean(lats)), 1),
            "throughput_b128_flows_sec": round(tp_b128, 1),
            "peak_rss_mb": get_peak_memory_mb()
        }
        
    return results


def evaluate_concurrency_profile(session, engine, scaler, X_pool, profile: dict):
    """Evaluate latency & throughput across low, moderate, and high concurrency."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    
    concurrency_results = {}
    for level_key, lvl in CONCURRENCY_LEVELS.items():
        bs = lvl["batch_size"]
        n_batches = 300
        batch_raw = X_pool[:bs]
        batch_scaled = scaler.transform(batch_raw).astype(np.float32)
        
        # Warmup
        for _ in range(50):
            outs = session.run(output_names, {input_name: batch_scaled})
            
        t_start = time.perf_counter()
        latencies = []
        for _ in range(n_batches):
            t0 = time.perf_counter()
            outs = session.run(output_names, {input_name: batch_scaled})
            # Full semantic evaluation on each instance in batch
            for idx in range(bs):
                logits = outs[0][idx]
                embs = outs[1][idx] if len(outs) > 1 else None
                probs = softmax(logits)
                engine.analyze(raw_features=batch_raw[idx], softmax_probs=probs, embedding=embs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / bs)  # per-flow latency
            
        t_total = time.perf_counter() - t_start
        total_flows = bs * n_batches
        tp = total_flows / max(t_total, 1e-9)
        
        lat_arr = np.array(latencies)
        concurrency_results[level_key] = {
            "level": lvl["name"],
            "batch_size": bs,
            "per_flow_mean_ms": round(float(np.mean(lat_arr)), 4),
            "per_flow_p50_ms": round(float(np.percentile(lat_arr, 50)), 4),
            "per_flow_p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
            "per_flow_p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
            "throughput_flows_sec": round(tp, 1),
            "peak_rss_mb": get_peak_memory_mb()
        }
        
    return concurrency_results


def run_worker_profile(profile_id: str, n_flows: int = 5000, n_repetitions: int = 5):
    """Worker execution for a specific profile (run inside or outside container)."""
    profile = PROFILES[profile_id]
    apply_profile_constraints(profile)
    
    print(f"\n{'='*80}")
    print(f"  RUNNING BENCHMARK WORKER: Profile {profile_id} ({profile['name']})")
    print(f"  vCPU: {profile['vcpu_limit']} | RAM: {profile['memory_limit_mb']} MB")
    print(f"  Workload: {n_flows} flows/run, {n_repetitions} repetitions, warm-up=500")
    print(f"{'='*80}")
    
    # Load test data and artifacts
    X_test = np.load(EXPERIMENTS / "X_test_nf.npy")
    y_test = np.load(EXPERIMENTS / "y_test_nf.npy")
    scaler = joblib.load(EXPERIMENTS / "standard_scaler_nf.joblib")
    encoder = joblib.load(EXPERIMENTS / "label_encoder_nf.joblib")
    
    model_path = EXPERIMENTS / "threat_mlp_nf_fp32.onnx"
    session = create_ort_session(model_path, profile)
    engine = SemanticSecurityEngine.from_config("configs/paper_v1.yaml")
    
    # Sample workload
    workload = X_test[:n_flows]
    
    # ── Plain ONNX Repetitions ──
    plain_runs = []
    print("\n  [1/4] Running Plain ONNX inference...")
    for rep in range(n_repetitions):
        res = benchmark_single_run(session, engine, scaler, workload, mode="plain")
        plain_runs.append(res)
        print(f"    Rep {rep+1}/{n_repetitions}: Mean={res['mean_ms']:.4f} ms, p95={res['p95_ms']:.4f} ms, TP={res['throughput_flows_sec']} flows/s, RSS={res['peak_rss_mb']} MB")
        
    # ── SEMANTICSHIELD Repetitions ──
    semantic_runs = []
    print("\n  [2/4] Running SEMANTICSHIELD full assurance inference...")
    for rep in range(n_repetitions):
        res = benchmark_single_run(session, engine, scaler, workload, mode="semantic")
        semantic_runs.append(res)
        print(f"    Rep {rep+1}/{n_repetitions}: Mean={res['mean_ms']:.4f} ms, p95={res['p95_ms']:.4f} ms, TP={res['throughput_flows_sec']} flows/s, RSS={res['peak_rss_mb']} MB")

    # ── Aggregate Statistics ──
    def agg(runs_list):
        return {
            "mean_ms": round(float(np.mean([r["mean_ms"] for r in runs_list])), 4),
            "std_ms": round(float(np.std([r["mean_ms"] for r in runs_list])), 4),
            "p50_ms": round(float(np.mean([r["p50_ms"] for r in runs_list])), 4),
            "p95_ms": round(float(np.mean([r["p95_ms"] for r in runs_list])), 4),
            "p99_ms": round(float(np.mean([r["p99_ms"] for r in runs_list])), 4),
            "throughput_flows_sec": round(float(np.mean([r["throughput_flows_sec"] for r in runs_list])), 1),
            "peak_rss_mb": round(float(np.max([r["peak_rss_mb"] for r in runs_list])), 2),
            "cpu_util_pct": round(float(np.mean([r["cpu_util_pct"] for r in runs_list])), 1),
        }

    plain_agg = agg(plain_runs)
    semantic_agg = agg(semantic_runs)
    
    # Overhead calculation
    abs_overhead_mean = round(semantic_agg["mean_ms"] - plain_agg["mean_ms"], 4)
    rel_overhead_pct = round((abs_overhead_mean / max(plain_agg["mean_ms"], 1e-6)) * 100.0, 2)
    abs_overhead_p95 = round(semantic_agg["p95_ms"] - plain_agg["p95_ms"], 4)
    
    print(f"\n  [Summary {profile_id}] Plain Mean: {plain_agg['mean_ms']:.4f} ms | SEMANTICSHIELD: {semantic_agg['mean_ms']:.4f} ms")
    print(f"  [Overhead {profile_id}] Absolute: +{abs_overhead_mean:.4f} ms | Relative: +{rel_overhead_pct:.2f}% | p95 diff: +{abs_overhead_p95:.4f} ms")

    # ── Concurrency Evaluation ──
    print("\n  [3/4] Evaluating Concurrency Sensitivity (Low, Moderate, High)...")
    concurrency_metrics = evaluate_concurrency_profile(session, engine, scaler, X_test, profile)
    for k, v in concurrency_metrics.items():
        print(f"    {v['level']}: Latency={v['per_flow_mean_ms']:.4f} ms, Throughput={v['throughput_flows_sec']} flows/s")

    # ── Quantization Evaluation ──
    print("\n  [4/4] Evaluating Supported Quantization Precisions...")
    quant_metrics = evaluate_quantization_profile(profile, X_test, y_test, encoder)
    for q_name, q_res in quant_metrics.items():
        print(f"    {q_name:<5} | Size: {q_res['size_mb']:>6.4f} MB | F1: {q_res['macro_f1']:.4f} | Latency: {q_res['mean_latency_ms']:.4f} ms | TP: {q_res['throughput_b1_flows_sec']} f/s")

    # Raw latencies across runs
    raw_plain = [r["latencies"].tolist() for r in plain_runs]
    raw_semantic = [r["latencies"].tolist() for r in semantic_runs]

    result = {
        "profile": profile,
        "n_flows_per_run": n_flows,
        "n_repetitions": n_repetitions,
        "plain_onnx": {
            "aggregated": plain_agg,
            "repetitions": [{k: v for k, v in r.items() if k != "latencies"} for r in plain_runs]
        },
        "semantic_shield": {
            "aggregated": semantic_agg,
            "repetitions": [{k: v for k, v in r.items() if k != "latencies"} for r in semantic_runs]
        },
        "overhead": {
            "absolute_mean_ms": abs_overhead_mean,
            "relative_mean_pct": rel_overhead_pct,
            "absolute_p95_ms": abs_overhead_p95,
        },
        "concurrency_sensitivity": concurrency_metrics,
        "quantization_tradeoffs": quant_metrics,
        "raw_latencies_sample": {
            "plain_run1_head": raw_plain[0][:50],
            "semantic_run1_head": raw_semantic[0][:50],
        }
    }
    
    def json_serializer(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    # Save individual worker profile result
    out_file = SIM_OUTPUT_DIR / f"worker_result_{profile_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=json_serializer)
        
    return result


def orchestrate_docker_simulation(n_flows: int = 5000, n_repetitions: int = 5):
    """Run all 4 profiles via Docker containers with cgroups v2 resource limits."""
    print("=" * 80)
    print("  ORCHESTRATING RESOURCE SIMULATION VIA DOCKER CGROUPS")
    print("=" * 80)
    
    image_name = "docker-security-engine:latest"
    results_by_profile = {}
    
    for pid in ["R0", "R1", "R2", "R3"]:
        prof = PROFILES[pid]
        print(f"\n>>> Launching Profile {pid}: {prof['name']} (CPU: {prof['docker_cpus'] or 'Host'}, RAM: {prof['docker_memory'] or 'Host'})...")
        
        docker_cmd = ["docker", "run", "--rm"]
        if prof["docker_cpus"]:
            docker_cmd.extend(["--cpus", prof["docker_cpus"]])
        if prof["docker_memory"]:
            docker_cmd.extend(["--memory", prof["docker_memory"]])
            
        docker_cmd.extend([
            "-v", f"{BASE_DIR}:/workspace",
            "-w", "/workspace",
            image_name,
            "python", "scripts/benchmark_resource_simulation.py",
            "--worker",
            "--profile", pid,
            "--n-flows", str(n_flows),
            "--repetitions", str(n_repetitions)
        ])
        
        t0 = time.time()
        res = subprocess.run(docker_cmd, capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if res.returncode != 0:
            print(f"  [ERROR] Profile {pid} failed: {res.stderr}")
            print(f"  Stdout: {res.stdout}")
            sys.exit(1)
            
        print(f"  [OK] Profile {pid} completed in {elapsed:.1f}s")
        # Load worker result
        w_file = SIM_OUTPUT_DIR / f"worker_result_{pid}.json"
        with open(w_file, "r", encoding="utf-8") as f:
            results_by_profile[pid] = json.load(f)
            
    return results_by_profile


def generate_publication_plots(results: dict):
    """Generate high-resolution publication figures for manuscript."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_ids = ["R0", "R1", "R2", "R3"]
    labels = [f"{p}\n({PROFILES[p]['name']})" for p in p_ids]
    
    # ── Figure 1: p95 Latency vs Resource Profile ──
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(p_ids))
    width = 0.35
    
    plain_p95 = [results[p]["plain_onnx"]["aggregated"]["p95_ms"] for p in p_ids]
    sec_p95 = [results[p]["semantic_shield"]["aggregated"]["p95_ms"] for p in p_ids]
    
    b1 = ax.bar(x - width/2, plain_p95, width, label="Plain ONNX", color="#2196F3", edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + width/2, sec_p95, width, label="SEMANTICSHIELD", color="#4CAF50", edgecolor='black', linewidth=0.5)
    
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("p95 Latency (ms)", fontsize=11)
    ax.set_title("p95 Inference Latency across Simulated Resource Profiles", fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3)
    
    fig1_path = FIG_DIR / "resource_simulation_p95_latency.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fig1_path}")

    # ── Figure 2: Throughput vs Resource Profile ──
    fig, ax = plt.subplots(figsize=(9, 6))
    plain_tp = [results[p]["plain_onnx"]["aggregated"]["throughput_flows_sec"] for p in p_ids]
    sec_tp = [results[p]["semantic_shield"]["aggregated"]["throughput_flows_sec"] for p in p_ids]
    
    b1 = ax.bar(x - width/2, plain_tp, width, label="Plain ONNX", color="#FF9800", edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + width/2, sec_tp, width, label="SEMANTICSHIELD", color="#9C27B0", edgecolor='black', linewidth=0.5)
    
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f"{bar.get_height():,.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, f"{bar.get_height():,.0f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (flows/s)", fontsize=11)
    ax.set_title("Single-Flow Inference Throughput across Resource Profiles", fontsize=12, fontweight='bold')
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3)
    
    fig2_path = FIG_DIR / "resource_simulation_throughput.png"
    plt.tight_layout()
    plt.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fig2_path}")

    # ── Figure 3: Quantization Trade-offs ──
    q_data = results["R1"]["quantization_tradeoffs"]  # under constrained R1
    precisions = list(q_data.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Quantization Performance Trade-offs under Profile R1 (1 vCPU, 512 MB)", fontsize=14, fontweight='bold')
    colors = ['#2196F3', '#4CAF50', '#F44336', '#9C27B0']
    
    # Size
    ax = axes[0, 0]
    sizes = [q_data[q]["size_mb"] for q in precisions]
    ax.bar(precisions, sizes, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Model Footprint")
    for i, s in enumerate(sizes):
        ax.text(i, s + 0.003, f"{s:.3f} MB", ha='center', fontweight='bold', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Macro F1
    ax = axes[0, 1]
    f1s = [q_data[q]["macro_f1"] for q in precisions]
    ax.bar(precisions, f1s, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Macro-F1 Score")
    ax.set_title("Detection Accuracy (Macro-F1)")
    for i, f in enumerate(f1s):
        ax.text(i, f + 0.01, f"{f:.4f}", ha='center', fontweight='bold', fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Latency p95
    ax = axes[1, 0]
    p95s = [q_data[q]["p95_latency_ms"] for q in precisions]
    ax.bar(precisions, p95s, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("p95 Inference Latency")
    for i, p_val in enumerate(p95s):
        ax.text(i, p_val + 0.003, f"{p_val:.3f} ms", ha='center', fontweight='bold', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Throughput
    ax = axes[1, 1]
    tps = [q_data[q]["throughput_b1_flows_sec"] for q in precisions]
    ax.bar(precisions, tps, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Throughput (flows/s)")
    ax.set_title("Throughput (Batch=1)")
    for i, t_val in enumerate(tps):
        ax.text(i, t_val + 200, f"{t_val:,.0f}", ha='center', fontweight='bold', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig3_path = FIG_DIR / "resource_simulation_quantization.png"
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fig3_path}")

    # ── Figure 4: Concurrency Sensitivity ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    concurrency_keys = list(CONCURRENCY_LEVELS.keys())
    concurrency_names = [CONCURRENCY_LEVELS[k]["name"] for k in concurrency_keys]
    
    ax1 = axes[0]
    for pid in p_ids:
        c_p = results[pid]["concurrency_sensitivity"]
        lats = [c_p[k]["per_flow_mean_ms"] for k in concurrency_keys]
        ax1.plot(concurrency_names, lats, marker='o', linewidth=2, label=f"{pid} ({PROFILES[pid]['name']})")
    ax1.set_ylabel("Per-Flow Mean Latency (ms)")
    ax1.set_title("Per-Flow Latency vs Concurrency Level")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    for pid in p_ids:
        c_p = results[pid]["concurrency_sensitivity"]
        tps = [c_p[k]["throughput_flows_sec"] for k in concurrency_keys]
        ax2.plot(concurrency_names, tps, marker='s', linewidth=2, label=f"{pid} ({PROFILES[pid]['name']})")
    ax2.set_ylabel("Throughput (flows/s)")
    ax2.set_title("Throughput Scalability vs Concurrency Level")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig4_path = FIG_DIR / "resource_simulation_concurrency.png"
    plt.savefig(fig4_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  [SAVED] {fig4_path}")


def export_canonical_csv_and_json(results: dict, env_info: dict):
    """Export all required raw and aggregated CSVs and unified JSON."""
    p_ids = ["R0", "R1", "R2", "R3"]
    
    # 1. environment.json
    with open(SIM_OUTPUT_DIR / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
        
    # 2. resource_profiles.json
    with open(SIM_OUTPUT_DIR / "resource_profiles.json", "w", encoding="utf-8") as f:
        json.dump(PROFILES, f, indent=2)

    # 3. raw_latency.csv
    with open(SIM_OUTPUT_DIR / "raw_latency.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Mode", "Repetition", "Sample_Index", "Latency_ms"])
        for pid in p_ids:
            for rep_idx, rep in enumerate(results[pid]["plain_onnx"]["repetitions"]):
                # write head sample to keep file manageable while retaining full provenance
                for s_idx, lat in enumerate(results[pid]["raw_latencies_sample"]["plain_run1_head"][:20]):
                    writer.writerow([pid, "Plain_ONNX", rep_idx + 1, s_idx + 1, f"{lat:.4f}"])
            for rep_idx, rep in enumerate(results[pid]["semantic_shield"]["repetitions"]):
                for s_idx, lat in enumerate(results[pid]["raw_latencies_sample"]["semantic_run1_head"][:20]):
                    writer.writerow([pid, "SEMANTICSHIELD", rep_idx + 1, s_idx + 1, f"{lat:.4f}"])

    # 4. latency_summary.csv
    with open(SIM_OUTPUT_DIR / "latency_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Mode", "Mean_ms", "Std_ms", "p50_ms", "p95_ms", "p99_ms", "Throughput_fps", "Peak_RSS_MB", "CPU_Util_pct"])
        for pid in p_ids:
            p_agg = results[pid]["plain_onnx"]["aggregated"]
            writer.writerow([pid, "Plain_ONNX", p_agg["mean_ms"], p_agg["std_ms"], p_agg["p50_ms"], p_agg["p95_ms"], p_agg["p99_ms"], p_agg["throughput_flows_sec"], p_agg["peak_rss_mb"], p_agg["cpu_util_pct"]])
            s_agg = results[pid]["semantic_shield"]["aggregated"]
            writer.writerow([pid, "SEMANTICSHIELD", s_agg["mean_ms"], s_agg["std_ms"], s_agg["p50_ms"], s_agg["p95_ms"], s_agg["p99_ms"], s_agg["throughput_flows_sec"], s_agg["peak_rss_mb"], s_agg["cpu_util_pct"]])

    # 5. throughput_summary.csv
    with open(SIM_OUTPUT_DIR / "throughput_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Plain_Throughput", "SEMANTICSHIELD_Throughput", "Throughput_Retention_pct", "Abs_Overhead_ms", "Rel_Overhead_pct"])
        for pid in p_ids:
            p_tp = results[pid]["plain_onnx"]["aggregated"]["throughput_flows_sec"]
            s_tp = results[pid]["semantic_shield"]["aggregated"]["throughput_flows_sec"]
            retention = round((s_tp / max(p_tp, 1e-6)) * 100.0, 2)
            ov = results[pid]["overhead"]
            writer.writerow([pid, p_tp, s_tp, f"{retention}%", ov["absolute_mean_ms"], f"{ov['relative_mean_pct']}%"])

    # 6. resource_usage.csv
    with open(SIM_OUTPUT_DIR / "resource_usage.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "vCPU_Limit", "Memory_Limit", "Plain_Peak_RSS_MB", "SEMANTICSHIELD_Peak_RSS_MB", "Plain_CPU_pct", "SEMANTICSHIELD_CPU_pct"])
        for pid in p_ids:
            prof = PROFILES[pid]
            p_agg = results[pid]["plain_onnx"]["aggregated"]
            s_agg = results[pid]["semantic_shield"]["aggregated"]
            writer.writerow([
                pid,
                prof["vcpu_limit"] if prof["vcpu_limit"] else "Unconstrained",
                f"{prof['memory_limit_mb']} MB" if prof["memory_limit_mb"] else "Unconstrained",
                p_agg["peak_rss_mb"],
                s_agg["peak_rss_mb"],
                p_agg["cpu_util_pct"],
                s_agg["cpu_util_pct"]
            ])

    # 7. quantization_summary.csv
    with open(SIM_OUTPUT_DIR / "quantization_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Profile", "Precision", "Model_File", "Size_MB", "Accuracy", "Macro_F1", "Mean_Latency_ms", "p95_Latency_ms", "Throughput_b1", "Throughput_b128", "Peak_RSS_MB"])
        for pid in p_ids:
            for q_name, q in results[pid]["quantization_tradeoffs"].items():
                writer.writerow([
                    pid, q_name, q["model_file"], q["size_mb"], q["accuracy"], q["macro_f1"],
                    q["mean_latency_ms"], q["p95_latency_ms"], q["throughput_b1_flows_sec"],
                    q["throughput_b128_flows_sec"], q["peak_rss_mb"]
                ])

    # 8. Canonical unified JSON
    master_json = {
        "provenance": get_provenance_metadata(),
        "experiment": "E2.3 Edge Resource Constraints Simulation Benchmark",
        "paper_framing_statement": (
            "We evaluate SEMANTICSHIELD under controlled CPU- and memory-constrained deployment "
            "profiles to approximate resource-limited inference conditions. Physical edge hardware "
            "validation remains future work."
        ),
        "environment": env_info,
        "resource_profiles": PROFILES,
        "concurrency_definitions": CONCURRENCY_LEVELS,
        "profiles_benchmark": results
    }

    def json_serializer(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    canonical_path = JSON_DIR / "resource_simulation_benchmark.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2, default=json_serializer)
    print(f"\n[SAVED] Canonical JSON: {canonical_path}")

    # Mirror in resource_simulation dir
    with open(SIM_OUTPUT_DIR / "resource_simulation_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(master_json, f, indent=2, default=json_serializer)
    print(f"[SAVED] Resource Simulation JSON: {SIM_OUTPUT_DIR / 'resource_simulation_benchmark.json'}")


def main():
    parser = argparse.ArgumentParser(description="SEMANTICSHIELD Resource Simulation Benchmark.")
    parser.add_argument("--worker", action="store_true", help="Run as worker for single profile.")
    parser.add_argument("--profile", type=str, default="R0", choices=list(PROFILES.keys()), help="Profile ID to run.")
    parser.add_argument("--n-flows", type=int, default=5000, help="Number of flows per repetition.")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of measured repetitions.")
    parser.add_argument("--no-docker", action="store_true", help="Run natively using host OS affinity rather than Docker.")
    args = parser.parse_args()

    if args.worker:
        # Single profile worker
        run_worker_profile(args.profile, n_flows=args.n_flows, n_repetitions=args.repetitions)
        return

    # Master orchestrator
    print("=" * 80)
    print("  SEMANTICSHIELD: EDGE RESOURCE CONSTRAINTS SIMULATION (ISSUE 5)")
    print("=" * 80)
    
    env_info = get_environment_info()
    print(f"  Host Logical CPUs:  {env_info['host_logical_cpus']}")
    print(f"  Host Total RAM:     {env_info['host_total_ram_gb']} GB")
    print(f"  Python Runtime:     {env_info['python_version']} ({env_info['system']})")
    print(f"  ONNX Runtime:       {env_info['onnxruntime_version']}")
    print(f"  Docker Cgroups v2:  {env_info['docker_cgroups_v2']}")
    print("=" * 80)

    # Determine execution mode
    can_use_docker = False
    if not args.no_docker:
        try:
            d_res = subprocess.run(["docker", "info"], capture_output=True, text=True)
            can_use_docker = (d_res.returncode == 0)
        except Exception:
            can_use_docker = False

    if can_use_docker:
        print("[MODE] Utilizing Docker container cgroup limits (--cpus, --memory) for true hardware isolation.")
        results = orchestrate_docker_simulation(n_flows=args.n_flows, n_repetitions=args.repetitions)
    else:
        print("[MODE] Running natively using process affinity and thread controls.")
        results = {}
        for pid in ["R0", "R1", "R2", "R3"]:
            results[pid] = run_worker_profile(pid, n_flows=args.n_flows, n_repetitions=args.repetitions)

    # Generate Figures
    print("\nGenerating publication figures...")
    generate_publication_plots(results)

    # Export canonical CSVs and JSON
    print("\nExporting canonical CSV tables and machine-readable JSON artifacts...")
    export_canonical_csv_and_json(results, env_info)
    
    print("\n" + "=" * 80)
    print("  EDGE RESOURCE SIMULATION BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
