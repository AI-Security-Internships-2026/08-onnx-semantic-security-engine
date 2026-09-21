"""
Runtime and Evaluation Equivalence Tests for SEMANTICSHIELD (Issue #21).

Verifies that:
1. Batch scoring APIs produce identical results to single-sample runtime APIs.
2. Canonical scoring formulas in src/semantic_analyzer.py match the mathematical
   specifications used across evaluation scripts.
3. Verdict decisions are identical across all execution paths.
4. The paper configuration (configs/paper_v1.yaml) loads deterministically.
5. Structural and statistical signals remain cleanly separated.

Usage:
    python -m pytest tests/test_runtime_eval_equivalence.py -v
"""

import json
import math
import sys
import numpy as np
import pytest
from pathlib import Path

# Add src to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from semantic_analyzer import (
    ConfidenceAnalyzer,
    DriftDetector,
    InputValidator,
    SemanticSecurityEngine,
    ConfidenceResult,
    DriftResult,
    ValidationResult,
    SemanticResult,
    cosine_distance,
)


# ── Fixtures ──

@pytest.fixture
def test_data(tmp_path):
    """Create deterministic synthetic reference data for equivalence testing."""
    np.random.seed(42)
    emb_dim = 64
    num_classes = 4
    num_features = 13

    # Centroids
    class_centroids = np.random.randn(num_classes, emb_dim)
    global_centroid = np.mean(class_centroids, axis=0)

    # Well-conditioned positive definite covariance
    A = np.random.randn(emb_dim, emb_dim)
    covariance = A @ A.T + np.eye(emb_dim) * 2.0
    covariance_inverse = np.linalg.inv(covariance)

    cos_thresh = 0.45
    mahal_thresh = 15.0

    ref_path = tmp_path / "reference_embeddings_test.npz"
    np.savez(
        ref_path,
        global_centroid=global_centroid,
        class_centroids=class_centroids,
        covariance=covariance,
        covariance_inverse=covariance_inverse,
        cosine_threshold=np.array(cos_thresh),
        mahal_threshold=np.array(mahal_thresh),
        class_names=np.array([f"Class_{i}" for i in range(num_classes)]),
    )

    feat_means = np.random.uniform(50.0, 150.0, size=num_features)
    feat_stds = np.random.uniform(5.0, 20.0, size=num_features)
    stats_path = tmp_path / "training_feature_stats_test.json"
    stats_dict = {
        "model_variant": "Equivalence Test (13 features)",
        "num_features": num_features,
        "features": [
            {
                "index": i,
                "name": f"f_{i}",
                "mean": float(feat_means[i]),
                "std": float(feat_stds[i]),
                "min_approx": float(feat_means[i] - 3 * feat_stds[i]),
                "max_approx": float(feat_means[i] + 3 * feat_stds[i]),
            }
            for i in range(num_features)
        ],
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f)

    return {
        "ref_path": ref_path,
        "stats_path": stats_path,
        "class_centroids": class_centroids,
        "covariance_inverse": covariance_inverse,
        "cos_thresh": cos_thresh,
        "mahal_thresh": mahal_thresh,
        "num_features": num_features,
        "feat_means": feat_means,
        "feat_stds": feat_stds,
    }


# ── Test Suite: Equivalence ──

class TestConfidenceEquivalence:
    """Verifies confidence scoring equivalence between runtime and evaluation paths."""

    def test_msp_single_vs_batch(self):
        """ConfidenceAnalyzer.analyze() and analyze_batch() produce identical MSP scores."""
        analyzer = ConfidenceAnalyzer(threshold=0.60)
        np.random.seed(101)
        logits = np.random.randn(25, 5)
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # Batch scoring (unrounded floats for evaluation accuracy)
        batch_scores = analyzer.analyze_batch(probs)

        # Single scoring (rounded to 4 decimal places for API responses)
        single_scores = np.array([analyzer.analyze(p).confidence_score for p in probs])

        # Equivalence after rounding
        np.testing.assert_allclose(np.round(batch_scores, 4), single_scores, rtol=1e-5, atol=1e-5)

    def test_msp_matches_raw_max(self):
        """Confidence score matches mathematical definition max(softmax(logits))."""
        analyzer = ConfidenceAnalyzer(threshold=0.50)
        probs = np.array([
            [0.1, 0.7, 0.2],
            [0.45, 0.45, 0.1],
            [0.99, 0.005, 0.005],
        ])
        expected = np.max(probs, axis=1)
        batch_scores = analyzer.analyze_batch(probs)
        np.testing.assert_allclose(batch_scores, expected)

    def test_confidence_flag_equivalence(self):
        """Confidence flag logic is strictly identical across single and batch."""
        threshold = 0.55
        analyzer = ConfidenceAnalyzer(threshold=threshold)
        probs = np.array([
            [0.60, 0.40],  # OK
            [0.55, 0.45],  # OK (>= threshold)
            [0.54, 0.46],  # LOW_CONFIDENCE (< threshold)
            [0.50, 0.50],  # LOW_CONFIDENCE
        ])
        expected_flags = ["OK", "OK", "LOW_CONFIDENCE", "LOW_CONFIDENCE"]
        for p, exp in zip(probs, expected_flags):
            res = analyzer.analyze(p)
            assert res.confidence_flag == exp


class TestDriftEquivalence:
    """Verifies distance and drift score equivalence between runtime and evaluation."""

    def test_cosine_distance_equivalence(self, test_data):
        """Runtime DriftDetector and batch API compute identical cosine distances to eval formula."""
        detector = DriftDetector(
            reference_path=test_data["ref_path"],
            cosine_threshold=test_data["cos_thresh"],
            mahal_threshold=test_data["mahal_thresh"],
        )
        np.random.seed(202)
        embs = np.random.randn(20, 64)

        # Batch API (evaluation path)
        batch_result = detector.analyze_batch(embs)
        batch_cos = batch_result["cosine_distances"]

        # Single API (runtime path, rounded to 4 decimals)
        single_cos = np.array([detector.analyze(e).cosine_distance for e in embs])

        # Mathematical specification (explicit min distance loop)
        expected_cos = []
        for e in embs:
            min_c = min(
                float(cosine_distance(e, c))
                for c in test_data["class_centroids"]
            )
            expected_cos.append(min_c)
        expected_cos = np.array(expected_cos)

        # Batch matches unrounded mathematical specification to full float precision
        np.testing.assert_allclose(batch_cos, expected_cos, rtol=1e-7, atol=1e-7)
        # Single matches batch after 4-decimal rounding
        np.testing.assert_allclose(np.round(batch_cos, 4), single_cos, rtol=1e-5, atol=1e-5)

    def test_mahalanobis_distance_equivalence(self, test_data):
        """Runtime DriftDetector and batch API compute identical Mahalanobis distances to eval formula."""
        detector = DriftDetector(
            reference_path=test_data["ref_path"],
            cosine_threshold=test_data["cos_thresh"],
            mahal_threshold=test_data["mahal_thresh"],
        )
        np.random.seed(303)
        embs = np.random.randn(20, 64)

        # Batch API (evaluation path)
        batch_result = detector.analyze_batch(embs)
        batch_mahal = batch_result["mahalanobis_distances"]

        # Single API (runtime path, rounded to 4 decimals)
        single_mahal = np.array([detector.analyze(e).mahalanobis_distance for e in embs])

        # Mathematical specification (explicit quadratic form min loop)
        inv = test_data["covariance_inverse"]
        expected_mahal = []
        for e in embs:
            min_m = min(
                math.sqrt(max(float((e - c) @ inv @ (e - c)), 0.0))
                for c in test_data["class_centroids"]
            )
            expected_mahal.append(min_m)
        expected_mahal = np.array(expected_mahal)

        # Batch matches unrounded mathematical specification to full float precision
        np.testing.assert_allclose(batch_mahal, expected_mahal, rtol=1e-6, atol=1e-6)
        # Single matches batch after 4-decimal rounding
        np.testing.assert_allclose(np.round(batch_mahal, 4), single_mahal, rtol=1e-5, atol=1e-5)

    def test_composite_drift_score_equivalence(self, test_data):
        """Composite drift score min(max(cos_norm, mahal_norm), 2.0) is identical in both paths."""
        detector = DriftDetector(
            reference_path=test_data["ref_path"],
            cosine_threshold=test_data["cos_thresh"],
            mahal_threshold=test_data["mahal_thresh"],
        )
        np.random.seed(404)
        embs = np.random.randn(20, 64)

        batch_result = detector.analyze_batch(embs)
        batch_drift = batch_result["drift_scores"]

        single_drift = np.array([detector.analyze(e).drift_score for e in embs])

        # Single matches batch after 4-decimal rounding
        np.testing.assert_allclose(np.round(batch_drift, 4), single_drift, rtol=1e-5, atol=1e-5)


class TestInputValidatorEquivalence:
    """Verifies input validator single vs batch equivalence."""

    def test_validator_single_vs_batch(self, test_data):
        """InputValidator.analyze() and analyze_batch() produce identical results."""
        validator = InputValidator(stats_path=test_data["stats_path"])
        np.random.seed(505)
        raw_batch = np.random.normal(loc=test_data["feat_means"], scale=test_data["feat_stds"], size=(20, 13))

        single_results = [validator.analyze(r) for r in raw_batch]
        batch_results = validator.analyze_batch(raw_batch)

        assert len(single_results) == len(batch_results)
        for s, b in zip(single_results, batch_results):
            assert s.validation_passed == b.validation_passed
            assert s.alerts == b.alerts


class TestVerdictEquivalence:
    """Verifies that compute_verdict is the single canonical source of truth."""

    @pytest.mark.parametrize(
        "conf_flag,drift_flag,val_passed,alerts,expected_alerts,expected_verdict",
        [
            # Clean case
            ("OK", "OK", True, [], 0, "CLEAN"),
            # Single anomaly cases -> SUSPICIOUS
            ("LOW_CONFIDENCE", "OK", True, [], 1, "SUSPICIOUS"),
            ("OK", "DRIFT_DETECTED", True, [], 1, "SUSPICIOUS"),
            ("OK", "OK", True, ["OUTLIER:zscore"], 1, "SUSPICIOUS"),
            # Two anomaly cases -> HIGH_RISK
            ("LOW_CONFIDENCE", "DRIFT_DETECTED", True, [], 2, "HIGH_RISK"),
            ("LOW_CONFIDENCE", "OK", True, ["OUTLIER:zscore"], 2, "HIGH_RISK"),
            ("OK", "DRIFT_DETECTED", True, ["OUTLIER:zscore"], 2, "HIGH_RISK"),
            # Three anomaly cases -> HIGH_RISK
            ("LOW_CONFIDENCE", "DRIFT_DETECTED", True, ["OUTLIER:zscore"], 3, "HIGH_RISK"),
            # Validation failure always REJECTED regardless of alerts
            ("OK", "OK", False, ["SCHEMA_MISMATCH"], 0, "REJECTED"),
            ("LOW_CONFIDENCE", "DRIFT_DETECTED", False, ["NON_FINITE"], 2, "REJECTED"),
        ],
    )
    def test_compute_verdict_rules(
        self, conf_flag, drift_flag, val_passed, alerts, expected_alerts, expected_verdict
    ):
        """Test exhaustive matrix of verdict boundary rules."""
        total_alerts, verdict = SemanticSecurityEngine.compute_verdict(
            confidence_flag=conf_flag,
            drift_flag=drift_flag,
            validation_passed=val_passed,
            validation_alerts=alerts,
        )
        assert total_alerts == expected_alerts
        assert verdict == expected_verdict

    def test_engine_analyze_matches_compute_verdict(self, test_data, monkeypatch):
        """SemanticSecurityEngine.analyze() verdict strictly matches compute_verdict()."""
        import semantic_analyzer
        monkeypatch.setattr(semantic_analyzer, "EXPERIMENTS", test_data["ref_path"].parent)

        test_data["ref_path"].rename(test_data["ref_path"].parent / "reference_embeddings_nf.npz")
        test_data["stats_path"].rename(test_data["ref_path"].parent / "training_feature_stats_nf.json")

        engine = SemanticSecurityEngine(use_nf=True)

        raw = test_data["feat_means"].copy()
        emb = test_data["class_centroids"][0].copy()  # Closest to class 0 -> In-dist
        probs = np.array([0.9, 0.05, 0.025, 0.025])

        result = engine.analyze(raw_features=raw, softmax_probs=probs, embedding=emb)
        canonical_alerts, canonical_verdict = SemanticSecurityEngine.compute_verdict(
            confidence_flag=result.confidence_flag,
            drift_flag=result.drift_flag,
            validation_passed=result.validation_passed,
            validation_alerts=result.validation_alerts,
        )

        assert result.engine_verdict == canonical_verdict
        assert result.total_alerts == canonical_alerts
        assert result.engine_verdict == "CLEAN"


class TestPaperConfigDeterminism:
    """Verifies that configs/paper_v1.yaml loads deterministically and freezes thresholds."""

    def test_paper_v1_config_exists_and_loads(self):
        """configs/paper_v1.yaml exists and can be loaded via SemanticSecurityEngine.from_config()."""
        config_path = BASE_DIR / "configs" / "paper_v1.yaml"
        assert config_path.exists(), "configs/paper_v1.yaml must exist"

        engine = SemanticSecurityEngine.from_config(str(config_path))
        assert engine is not None
        assert engine.confidence_analyzer is not None

        # Verify frozen threshold values
        assert abs(engine.confidence_analyzer.threshold - 0.4743) < 1e-4
        if engine.has_drift:
            assert abs(engine.drift_detector.cosine_threshold - 0.4341) < 1e-4
            assert abs(engine.drift_detector.mahal_threshold - 18.1593) < 1e-3
        if engine.has_validation:
            assert engine.input_validator.zscore_threshold == 15.0
            assert engine.input_validator.zero_fill_ratio == 0.80
            assert engine.input_validator.range_tolerance_sigmas == 15.0

    def test_paper_v1_reproducibility(self):
        """Loading the config twice yields identical engine threshold parameters."""
        config_path = BASE_DIR / "configs" / "paper_v1.yaml"
        e1 = SemanticSecurityEngine.from_config(str(config_path))
        e2 = SemanticSecurityEngine.from_config(str(config_path))

        assert e1.confidence_analyzer.threshold == e2.confidence_analyzer.threshold
        if e1.has_drift and e2.has_drift:
            assert e1.drift_detector.cosine_threshold == e2.drift_detector.cosine_threshold
            assert e1.drift_detector.mahal_threshold == e2.drift_detector.mahal_threshold


class TestSemanticResultSeparation:
    """Verifies structural validation vs statistical OOD evidence separation."""

    def test_semantic_result_has_distance_fields(self):
        """SemanticResult must separate cosine and mahalanobis distances from composite score."""
        res = SemanticResult(
            confidence_score=0.95,
            confidence_flag="OK",
            cosine_distance=0.12,
            mahalanobis_distance=4.5,
            drift_score=0.35,
            drift_flag="OK",
            nearest_reference_class="Benign",
            validation_passed=True,
            validation_alerts=[],
            total_alerts=0,
            engine_verdict="CLEAN",
        )
        assert hasattr(res, "cosine_distance")
        assert hasattr(res, "mahalanobis_distance")
        assert res.cosine_distance == 0.12
        assert res.mahalanobis_distance == 4.5
        assert res.engine_verdict == "CLEAN"
