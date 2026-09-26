"""
Tests for Issue 5: Simulated Edge Resource Constraints Evaluation.

Validates that:
  1. Profiles R0, R1, R2, R3 are defined with proper limits.
  2. All canonical CSV tables, figures, and machine-readable JSON artifacts exist.
  3. Plain ONNX vs SEMANTICSHIELD metrics include mean, p50, p95, p99, throughput, peak RSS, and CPU %.
  4. At least 5 repetitions were performed.
  5. The paper framing guardrail strictly disclaims physical edge validation.
  6. E2.3-E INT8 degradation investigation artifact exists and documents activation clipping evidence.
"""

import json
import csv
import pytest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PAPER_RESULTS = BASE_DIR / "experiments" / "paper_results"
JSON_DIR = PAPER_RESULTS / "json"
TABLES_DIR = PAPER_RESULTS / "tables"
FIGURES_DIR = PAPER_RESULTS / "figures"
SIM_DIR = PAPER_RESULTS / "resource_simulation"


class TestResourceSimulationArtifacts:
    """Verifies artifact existence and schema integrity for Issue 5."""

    def test_json_artifacts_exist(self):
        master_json = JSON_DIR / "resource_simulation_benchmark.json"
        int8_json = JSON_DIR / "int8_degradation_investigation.json"
        assert master_json.exists(), "resource_simulation_benchmark.json missing"
        assert int8_json.exists(), "int8_degradation_investigation.json missing"

    def test_paper_framing_guardrail(self):
        master_json = JSON_DIR / "resource_simulation_benchmark.json"
        with open(master_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        statement = data.get("paper_framing_statement", "")
        assert "controlled CPU- and memory-constrained deployment" in statement
        assert "Physical edge hardware validation remains future work" in statement

    def test_profiles_structure_and_repetitions(self):
        master_json = JSON_DIR / "resource_simulation_benchmark.json"
        with open(master_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        bench = data.get("profiles_benchmark", {})
        assert set(bench.keys()) == {"R0", "R1", "R2", "R3"}

        for pid in ["R0", "R1", "R2", "R3"]:
            p = bench[pid]
            assert p["n_repetitions"] >= 5
            assert p["n_flows_per_run"] >= 5000

            # Plain ONNX
            plain = p["plain_onnx"]["aggregated"]
            for m in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_flows_sec", "peak_rss_mb", "cpu_util_pct"]:
                assert m in plain
                assert plain[m] > 0

            # SEMANTICSHIELD
            sec = p["semantic_shield"]["aggregated"]
            for m in ["mean_ms", "p50_ms", "p95_ms", "p99_ms", "throughput_flows_sec", "peak_rss_mb", "cpu_util_pct"]:
                assert m in sec
                assert sec[m] > 0

            # Overhead
            ov = p["overhead"]
            assert "absolute_mean_ms" in ov
            assert "relative_mean_pct" in ov

    def test_concurrency_evaluation(self):
        master_json = JSON_DIR / "resource_simulation_benchmark.json"
        with open(master_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for pid in ["R0", "R1", "R2", "R3"]:
            c_sens = data["profiles_benchmark"][pid]["concurrency_sensitivity"]
            assert "low_sequential" in c_sens
            assert "moderate" in c_sens
            assert "high_concurrency" in c_sens
            for lvl in c_sens.values():
                assert lvl["throughput_flows_sec"] > 0
                assert lvl["per_flow_mean_ms"] > 0

    def test_quantization_tradeoffs(self):
        master_json = JSON_DIR / "resource_simulation_benchmark.json"
        with open(master_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        r1_q = data["profiles_benchmark"]["R1"]["quantization_tradeoffs"]
        for prec in ["FP32", "FP16", "INT8", "INT4"]:
            assert prec in r1_q
            assert r1_q[prec]["size_mb"] > 0
            assert r1_q[prec]["macro_f1"] > 0
            assert r1_q[prec]["p95_latency_ms"] > 0
            assert r1_q[prec]["throughput_b1_flows_sec"] > 0

    def test_int8_investigation_evidence(self):
        int8_json = JSON_DIR / "int8_degradation_investigation.json"
        with open(int8_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "calibration_inspection" in data
        assert "layer_activations_summary" in data
        assert "evidence_summary" in data
        assert data["evidence_summary"]["is_activation_clipping_confirmed"] is True

    def test_canonical_tables_exist_and_valid(self):
        tables = [
            "table_resource_profiles.csv",
            "table_runtime_overhead.csv",
            "table_quantization_tradeoff.csv",
        ]
        for t in tables:
            path = TABLES_DIR / t
            assert path.exists(), f"Table {t} missing"
            with open(path, "r", encoding="utf-8") as f:
                reader = list(csv.reader(f))
                assert len(reader) >= 4, f"Table {t} has too few rows"

    def test_figures_exist_and_non_empty(self):
        figures = [
            "resource_simulation_p95_latency.png",
            "resource_simulation_throughput.png",
            "resource_simulation_quantization.png",
            "resource_simulation_concurrency.png",
            "int8_activation_analysis.png",
        ]
        for fig in figures:
            p = FIGURES_DIR / fig
            assert p.exists(), f"Figure {fig} missing"
            assert p.stat().st_size > 10000, f"Figure {fig} is empty or suspiciously small"
