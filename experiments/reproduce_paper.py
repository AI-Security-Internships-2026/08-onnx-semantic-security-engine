"""
SEMANTICSHIELD Paper Results Reproduction Orchestrator

Single authoritative entry point to reproduce all experimental findings,
metrics tables, and publication figures reported in the manuscript.

All experiments are driven by configs/paper_v1.yaml and write to:
  - experiments/paper_results/json/
  - experiments/paper_results/figures/

Usage:
    python experiments/reproduce_paper.py --dry-run
    python experiments/reproduce_paper.py --skip-slow
    python experiments/reproduce_paper.py --only classifier
    python experiments/reproduce_paper.py --verify
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.config_loader import load_paper_config, get_provenance_metadata

PAPER_RESULTS_DIR = BASE_DIR / "experiments" / "paper_results"
JSON_DIR = PAPER_RESULTS_DIR / "json"
FIG_DIR = PAPER_RESULTS_DIR / "figures"

STAGES = [
    {
        "id": "classifier",
        "name": "Threat Classification & Confusion Matrix",
        "script": "scripts/generate_classifier_metrics.py",
        "args": [],
        "output_json": "classifier_metrics.json",
        "output_fig": "confusion_matrix_cic.png",
        "slow": False,
        "description": "Evaluates 13-feature ThreatMLP NF on 138,069 test samples",
    },
    {
        "id": "cross_model",
        "name": "Cross-Model Comparison (76 vs 13 Features)",
        "script": "scripts/cross_model_comparison.py",
        "args": ["--n-latency-runs", "500"],
        "output_json": "cross_model_comparison.json",
        "output_fig": "cross_model_comparison.png",
        "slow": False,
        "description": "Compares 76-feat baseline vs 13-feat NF model across params, size, F1, and latency",
    },
    {
        "id": "ablation",
        "name": "Architectural Component Ablation Study",
        "script": "scripts/ablation_study.py",
        "args": ["--samples", "5000"],
        "output_json": "ablation_study.json",
        "output_fig": "ablation_comparison.png",
        "slow": False,
        "description": "Evaluates 7 component combinations (InputValidator, Confidence, Drift)",
    },
    {
        "id": "quantization",
        "name": "Quantization Benchmark (FP32/FP16/INT8/INT4)",
        "script": "scripts/benchmark_quantization.py",
        "args": [],
        "output_json": "quantization_benchmark.json",
        "output_fig": "quantization_benchmark.png",
        "slow": False,
        "description": "Benchmarks size, Macro-F1 drop, latency, and throughput across 4 precisions",
    },
    {
        "id": "semantic_eval",
        "name": "Semantic Engine Calibrated Integration Evaluation",
        "script": "scripts/evaluate_semantic_engine.py",
        "args": [],
        "output_json": "semantic_engine_evaluation.json",
        "output_fig": "semantic_engine_evaluation_plots.png",
        "slow": True,
        "description": "Calibrated held-out validation evaluation on CIC-IDS2018 and ToN-IoT datasets",
    },
    {
        "id": "ood_baselines",
        "name": "OOD & Anomaly Detector Baselines Comparison",
        "script": "scripts/benchmark_ood_baselines.py",
        "args": [],
        "output_json": "ood_baselines_benchmark.json",
        "output_fig": "ood_baselines_comparison.png",
        "slow": True,
        "description": "Compares MSP, Mahalanobis, IF, OCSVM, and Semantic Engine on ToN-IoT & corruptions",
    },
    {
        "id": "statistical_rigor",
        "name": "Statistical Rigor 5-Seed Benchmark",
        "script": "scripts/statistical_rigor_benchmark.py",
        "args": [],
        "output_json": "statistical_rigor_benchmark.json",
        "output_fig": "statistical_rigor_plots.png",
        "slow": True,
        "description": "5-seed repeated evaluation reporting mean, std, 95% CI, and FPR@95TPR",
    },
    {
        "id": "training_time",
        "name": "Training Time per Feature-Schema Stage",
        "script": "scripts/benchmark_training_time.py",
        "args": [],
        "output_json": "training_time_benchmark.json",
        "output_fig": "training_time_comparison.png",
        "slow": True,
        "description": "Trains models from scratch across 76, 21, and 13-feature schemas",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce all SEMANTICSHIELD paper results.")
    parser.add_argument("--dry-run", action="store_true", help="Print execution plan without running scripts.")
    parser.add_argument("--skip-slow", action="store_true", help="Skip dataset-heavy / slow training benchmarks.")
    parser.add_argument("--only", type=str, default=None, choices=[s["id"] for s in STAGES], help="Run only a specific stage.")
    parser.add_argument("--verify", action="store_true", help="Verify generated numbers against paper claims.")
    parser.add_argument("--output-dir", type=str, default=str(JSON_DIR), help="Output directory for JSON results.")
    parser.add_argument("--figures-dir", type=str, default=str(FIG_DIR), help="Output directory for figures.")
    parser.add_argument("--config", type=str, default=None, help="Path to paper config (defaults to configs/paper_v1.yaml).")
    return parser.parse_args()


def print_banner(cfg, prov):
    print("=" * 80)
    print("  SEMANTICSHIELD: AUTOMATED PAPER REPRODUCIBILITY ORCHESTRATOR")
    print("=" * 80)
    print(f"  Config Version:       {prov['config_version']} (frozen: {prov['config_frozen_date']})")
    print(f"  Git Commit:           {prov['git_commit'][:10]} [{prov['git_branch']}]")
    print(f"  Python Runtime:       {prov['python_version']} ({prov['platform']})")
    print(f"  Canonical Output:     {PAPER_RESULTS_DIR}")
    print(f"  Execution Timestamp:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)


def verify_results(json_dir: Path):
    """Verify presence and validity of all canonical results."""
    print("\n" + "=" * 80)
    print("  VERIFYING CANONICAL PAPER ARTIFACTS")
    print("=" * 80)

    required_files = [
        "classifier_metrics.json",
        "cross_model_comparison.json",
        "ablation_study.json",
        "quantization_benchmark.json",
        "ood_baselines_benchmark.json",
        "semantic_engine_evaluation.json",
        "statistical_rigor_benchmark.json",
        "training_time_benchmark.json",
        "latency_benchmark.json",
    ]

    all_ok = True
    for fname in required_files:
        p = json_dir / fname
        if not p.exists():
            print(f"  [MISSING] {fname}")
            all_ok = False
        else:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                size_kb = p.stat().st_size / 1024
                keys = list(data.keys())[:3]
                print(f"  [OK]      {fname:<32} ({size_kb:>6.1f} KB, keys: {keys})")
            except Exception as e:
                print(f"  [CORRUPT] {fname}: {e}")
                all_ok = False

    return all_ok


def main():
    args = parse_args()
    cfg = load_paper_config(args.config)
    prov = get_provenance_metadata(args.config)

    print_banner(cfg, prov)

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Filter stages
    stages_to_run = STAGES
    if args.only:
        stages_to_run = [s for s in STAGES if s["id"] == args.only]
    elif args.skip_slow:
        stages_to_run = [s for s in STAGES if not s["slow"]]

    print(f"\n[Plan] {len(stages_to_run)} benchmark stages selected for execution:")
    for i, s in enumerate(stages_to_run, 1):
        slow_tag = " [SLOW]" if s["slow"] else ""
        print(f"  {i}. [{s['id']}] {s['name']}{slow_tag}")
        print(f"     Script: {s['script']}")
        print(f"     Output: {s['output_json']} | Figure: {s['output_fig']}")

    if args.dry_run:
        print("\n[DRY RUN] Pipeline validated successfully. No commands executed.")
        verify_results(output_dir)
        return

    # Execute stages
    results_summary = []
    print("\n" + "=" * 80)
    print("  EXECUTING BENCHMARK STAGES")
    print("=" * 80)

    for i, s in enumerate(stages_to_run, 1):
        print(f"\n>>> [{i}/{len(stages_to_run)}] Running {s['name']} ({s['script']})...")
        t0 = time.time()
        
        cmd = [
            sys.executable,
            str(BASE_DIR / s["script"]),
            "--output-dir", str(output_dir),
            "--figures-dir", str(figures_dir),
        ] + s["args"]

        try:
            res = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True)
            elapsed = time.time() - t0
            if res.returncode == 0:
                print(f"    Status: SUCCESS ({elapsed:.2f}s)")
                # Print last 5 lines of stdout for context
                lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
                for l in lines[-4:]:
                    print(f"    | {l}")
                results_summary.append({
                    "stage": s["id"],
                    "name": s["name"],
                    "status": "SUCCESS",
                    "duration_sec": round(elapsed, 2),
                    "json": s["output_json"],
                    "fig": s["output_fig"],
                })
            else:
                print(f"    Status: FAILED (code {res.returncode}, {elapsed:.2f}s)")
                print(f"    Stderr: {res.stderr[:300]}")
                results_summary.append({
                    "stage": s["id"],
                    "name": s["name"],
                    "status": f"FAILED ({res.returncode})",
                    "duration_sec": round(elapsed, 2),
                    "json": s["output_json"],
                    "fig": s["output_fig"],
                })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"    Status: ERROR ({e})")
            results_summary.append({
                "stage": s["id"],
                "name": s["name"],
                "status": f"ERROR: {e}",
                "duration_sec": round(elapsed, 2),
                "json": s["output_json"],
                "fig": s["output_fig"],
            })

    # Summary table
    print("\n" + "=" * 80)
    print("  REPRODUCTION EXECUTION SUMMARY")
    print("=" * 80)
    print(f"  {'Stage':<20} {'Status':<12} {'Time (s)':<10} {'Output JSON':<32}")
    print("  " + "-" * 76)
    for r in results_summary:
        print(f"  {r['stage']:<20} {r['status']:<12} {r['duration_sec']:<10.2f} {r['json']:<32}")
    print("=" * 80)

    # Verification
    verify_results(output_dir)


if __name__ == "__main__":
    main()
