"""
Unit tests for the Semantic Security Engine.

Tests cover all three analyzers (confidence, drift, input validation)
and the orchestrator, using synthetic data that doesn't require
the actual ONNX model or training dataset.

Usage:
    python -m pytest tests/test_semantic.py -v
"""

import json
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_analyzer import (
    ConfidenceAnalyzer,
    DriftDetector,
    InputValidator,
    SemanticSecurityEngine,
    ConfidenceResult,
    DriftResult,
    ValidationResult,
    SemanticResult,
)


# ── Fixtures ──

@pytest.fixture
def confidence_analyzer():
    """ConfidenceAnalyzer with default 0.70 threshold."""
    return ConfidenceAnalyzer(threshold=0.70)


@pytest.fixture
def tmp_reference_dir(tmp_path):
    """Create temporary reference embeddings and feature stats files."""
    emb_dim = 64
    num_classes = 3
    class_names = ["Benign", "DoS", "BruteForce"]

    # Global centroid: all zeros (center of distribution)
    global_centroid = np.zeros(emb_dim)

    # Per-class centroids: slightly offset from origin
    class_centroids = np.zeros((num_classes, emb_dim))
    class_centroids[0, 0] = 1.0   # Benign offset on dim 0
    class_centroids[1, 1] = 1.0   # DoS offset on dim 1
    class_centroids[2, 2] = 1.0   # BruteForce offset on dim 2

    # Identity covariance (no correlations)
    covariance = np.eye(emb_dim) * 1.0
    covariance_inverse = np.eye(emb_dim)

    np.savez(
        tmp_path / "reference_embeddings_test.npz",
        global_centroid=global_centroid,
        class_centroids=class_centroids,
        covariance=covariance,
        covariance_inverse=covariance_inverse,
        cosine_threshold=np.array(0.5),
        mahal_threshold=np.array(10.0),
        class_names=np.array(class_names),
    )

    # Training feature stats (5 features for testing)
    feature_stats = {
        "model_variant": "Test (5 features)",
        "num_features": 5,
        "features": [
            {"index": i, "name": f"feature_{i}", "mean": 100.0, "std": 10.0,
             "min_approx": 60.0, "max_approx": 140.0}
            for i in range(5)
        ],
    }
    with open(tmp_path / "training_feature_stats_test.json", "w") as f:
        json.dump(feature_stats, f)

    return tmp_path


@pytest.fixture
def drift_detector(tmp_reference_dir):
    """DriftDetector loaded from temporary reference data."""
    return DriftDetector(reference_path=tmp_reference_dir / "reference_embeddings_test.npz")


@pytest.fixture
def input_validator(tmp_reference_dir):
    """InputValidator loaded from temporary feature stats."""
    return InputValidator(stats_path=tmp_reference_dir / "training_feature_stats_test.json")


# ═══════════════════════════════════════════════
# Feature A: ConfidenceAnalyzer Tests
# ═══════════════════════════════════════════════

class TestConfidenceAnalyzer:

    def test_high_confidence_ok(self, confidence_analyzer):
        """High-confidence predictions should be flagged OK."""
        probs = np.array([0.05, 0.05, 0.85, 0.05])  # max = 0.85
        result = confidence_analyzer.analyze(probs)

        assert result.confidence_flag == "OK"
        assert result.confidence_score == 0.85

    def test_low_confidence_flagged(self, confidence_analyzer):
        """Low-confidence predictions should be flagged LOW_CONFIDENCE."""
        probs = np.array([0.30, 0.25, 0.25, 0.20])  # max = 0.30
        result = confidence_analyzer.analyze(probs)

        assert result.confidence_flag == "LOW_CONFIDENCE"
        assert result.confidence_score == 0.30

    def test_exact_threshold_ok(self, confidence_analyzer):
        """Confidence exactly at threshold should be OK."""
        probs = np.array([0.10, 0.10, 0.70, 0.10])  # max = 0.70
        result = confidence_analyzer.analyze(probs)

        assert result.confidence_flag == "OK"
        assert result.confidence_score == 0.70

    def test_just_below_threshold(self, confidence_analyzer):
        """Confidence just below threshold should be flagged."""
        probs = np.array([0.10, 0.10, 0.699, 0.101])
        result = confidence_analyzer.analyze(probs)

        assert result.confidence_flag == "LOW_CONFIDENCE"

    def test_perfect_confidence(self, confidence_analyzer):
        """Perfect 1.0 confidence should be OK."""
        probs = np.array([0.0, 0.0, 1.0, 0.0])
        result = confidence_analyzer.analyze(probs)

        assert result.confidence_flag == "OK"
        assert result.confidence_score == 1.0

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        analyzer = ConfidenceAnalyzer(threshold=0.90)
        probs = np.array([0.10, 0.05, 0.80, 0.05])  # max = 0.80

        result = analyzer.analyze(probs)
        assert result.confidence_flag == "LOW_CONFIDENCE"

    def test_returns_correct_type(self, confidence_analyzer):
        """Should return ConfidenceResult dataclass."""
        probs = np.array([0.5, 0.5])
        result = confidence_analyzer.analyze(probs)

        assert isinstance(result, ConfidenceResult)


# ═══════════════════════════════════════════════
# Feature B: DriftDetector Tests
# ═══════════════════════════════════════════════

class TestDriftDetector:

    def test_in_distribution_no_drift(self, drift_detector):
        """Embedding near the global centroid should not trigger drift."""
        # Small perturbation from centroid (which is all zeros)
        embedding = np.random.randn(64) * 0.1
        result = drift_detector.analyze(embedding)

        assert result.drift_flag == "OK"
        assert result.drift_score < 1.0

    def test_out_of_distribution_drift(self, drift_detector):
        """Extreme embedding should trigger drift detection."""
        # Very far from centroid
        embedding = np.ones(64) * 100.0
        result = drift_detector.analyze(embedding)

        assert result.drift_flag == "DRIFT_DETECTED"
        assert result.drift_score > 1.0

    def test_nearest_class_identification(self, drift_detector):
        """Should identify the nearest reference class correctly."""
        # Embedding close to class 0 (Benign) centroid, which has offset on dim 0
        embedding = np.zeros(64)
        embedding[0] = 0.9  # close to Benign centroid
        result = drift_detector.analyze(embedding)

        assert result.nearest_reference_class == "Benign"

    def test_nearest_class_dos(self, drift_detector):
        """Should identify DoS when embedding is near DoS centroid."""
        embedding = np.zeros(64)
        embedding[1] = 0.9  # close to DoS centroid (offset on dim 1)
        result = drift_detector.analyze(embedding)

        assert result.nearest_reference_class == "DoS"

    def test_returns_correct_type(self, drift_detector):
        """Should return DriftResult dataclass."""
        embedding = np.zeros(64)
        result = drift_detector.analyze(embedding)

        assert isinstance(result, DriftResult)
        assert hasattr(result, "cosine_distance")
        assert hasattr(result, "mahalanobis_distance")
        assert hasattr(result, "drift_score")

    def test_missing_reference_file(self, tmp_path):
        """Should raise FileNotFoundError if reference file is missing."""
        with pytest.raises(FileNotFoundError):
            DriftDetector(reference_path=tmp_path / "nonexistent.npz")


# ═══════════════════════════════════════════════
# Feature C: InputValidator Tests
# ═══════════════════════════════════════════════

class TestInputValidator:

    def test_valid_input_passes(self, input_validator):
        """Normal input within expected ranges should pass all checks."""
        features = np.array([100.0, 95.0, 105.0, 110.0, 90.0])
        result = input_validator.analyze(features)

        assert result.validation_passed is True
        assert len(result.alerts) == 0

    def test_schema_mismatch_fails(self, input_validator):
        """Wrong number of features should fail with SCHEMA_MISMATCH."""
        features = np.array([100.0, 95.0, 105.0])  # 3 instead of 5
        result = input_validator.analyze(features)

        assert result.validation_passed is False
        assert any("SCHEMA_MISMATCH" in a for a in result.alerts)

    def test_nan_values_rejected(self, input_validator):
        """NaN values should be rejected."""
        features = np.array([100.0, np.nan, 105.0, 110.0, 90.0])
        result = input_validator.analyze(features)

        assert result.validation_passed is False
        assert any("NAN_VALUES" in a for a in result.alerts)

    def test_inf_values_rejected(self, input_validator):
        """Infinity values should be rejected."""
        features = np.array([100.0, np.inf, 105.0, 110.0, 90.0])
        result = input_validator.analyze(features)

        assert result.validation_passed is False
        assert any("INF_VALUES" in a for a in result.alerts)

    def test_zero_filled_detection(self, input_validator):
        """Majority-zero input should trigger ZERO_FILLED alert (RQ3 failure)."""
        features = np.array([0.0, 0.0, 0.0, 0.0, 100.0])  # 4/5 = 80% zero
        result = input_validator.analyze(features)

        # Still passes (soft warning) but has alerts
        assert any("ZERO_FILLED" in a for a in result.alerts)
        assert "RQ3" in str(result.alerts)  # mentions the RQ3 failure mode

    def test_zero_filled_below_threshold(self, input_validator):
        """Less than 50% zeros should not trigger zero-fill alert."""
        features = np.array([0.0, 0.0, 100.0, 100.0, 100.0])  # 2/5 = 40%
        result = input_validator.analyze(features)

        assert not any("ZERO_FILLED" in a for a in result.alerts)

    def test_extreme_outliers(self, input_validator):
        """Values far from training mean should trigger EXTREME_OUTLIERS."""
        features = np.array([100.0, 100.0, 100.0, 100.0, 999999.0])  # last one is extreme
        result = input_validator.analyze(features)

        assert any("EXTREME_OUTLIERS" in a for a in result.alerts)

    def test_out_of_range(self, input_validator):
        """Values outside training range should trigger OUT_OF_RANGE."""
        features = np.array([100.0, 100.0, 100.0, 100.0, -500.0])  # far below min
        result = input_validator.analyze(features)

        assert any("OUT_OF_RANGE" in a for a in result.alerts)

    def test_returns_correct_type(self, input_validator):
        """Should return ValidationResult dataclass."""
        features = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        result = input_validator.analyze(features)

        assert isinstance(result, ValidationResult)

    def test_missing_stats_file(self, tmp_path):
        """Should raise FileNotFoundError if stats file is missing."""
        with pytest.raises(FileNotFoundError):
            InputValidator(stats_path=tmp_path / "nonexistent.json")


# ═══════════════════════════════════════════════
# SemanticSecurityEngine Orchestrator Tests
# ═══════════════════════════════════════════════

class TestSemanticSecurityEngine:

    @pytest.fixture
    def mock_engine(self, tmp_reference_dir, monkeypatch):
        """Create engine with test reference data."""
        import semantic_analyzer
        monkeypatch.setattr(semantic_analyzer, "EXPERIMENTS", tmp_reference_dir)

        # Rename test files to match expected naming convention
        (tmp_reference_dir / "reference_embeddings_test.npz").rename(
            tmp_reference_dir / "reference_embeddings_nf.npz"
        )
        (tmp_reference_dir / "training_feature_stats_test.json").rename(
            tmp_reference_dir / "training_feature_stats_nf.json"
        )

        return SemanticSecurityEngine(use_nf=True)

    def test_clean_verdict(self, mock_engine):
        """All OK inputs should produce CLEAN verdict."""
        raw_features = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        softmax_probs = np.array([0.05, 0.90, 0.05])  # high confidence
        embedding = np.zeros(64) + 0.01  # near centroid

        result = mock_engine.analyze(raw_features, softmax_probs, embedding)

        assert result.engine_verdict == "CLEAN"
        assert result.confidence_flag == "OK"

    def test_suspicious_low_confidence(self, mock_engine):
        """Low confidence should produce SUSPICIOUS verdict."""
        raw_features = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        softmax_probs = np.array([0.35, 0.35, 0.30])  # low confidence
        embedding = np.zeros(64) + 0.01

        result = mock_engine.analyze(raw_features, softmax_probs, embedding)

        assert result.confidence_flag == "LOW_CONFIDENCE"
        assert result.engine_verdict == "SUSPICIOUS"

    def test_high_risk_multiple_alerts(self, mock_engine):
        """Multiple alerts (low confidence + drift) should produce HIGH_RISK verdict."""
        raw_features = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        softmax_probs = np.array([0.35, 0.35, 0.30])  # low confidence
        embedding = np.ones(64) * 100.0  # extreme drift
        
        result = mock_engine.analyze(raw_features, softmax_probs, embedding)
        
        assert result.confidence_flag == "LOW_CONFIDENCE"
        assert result.drift_flag == "DRIFT_DETECTED"
        assert result.engine_verdict == "HIGH_RISK"

    def test_rejected_invalid_input(self, mock_engine):
        """Invalid input (NaN) should produce REJECTED verdict."""
        raw_features = np.array([np.nan, 100.0, 100.0, 100.0, 100.0])
        softmax_probs = np.array([0.05, 0.90, 0.05])
        embedding = np.zeros(64)

        result = mock_engine.analyze(raw_features, softmax_probs, embedding)

        assert result.engine_verdict == "REJECTED"
        assert result.validation_passed is False

    def test_returns_correct_type(self, mock_engine):
        """Should return SemanticResult dataclass."""
        raw_features = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        softmax_probs = np.array([0.05, 0.90, 0.05])
        embedding = np.zeros(64)

        result = mock_engine.analyze(raw_features, softmax_probs, embedding)

        assert isinstance(result, SemanticResult)
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "drift_score")
        assert hasattr(result, "validation_passed")
        assert hasattr(result, "engine_verdict")

    def test_without_embedding(self, mock_engine):
        """Should work even without embedding (drift unavailable)."""
        raw_features = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        softmax_probs = np.array([0.05, 0.90, 0.05])

        # Pass None for embedding
        result = mock_engine.analyze(raw_features, softmax_probs, embedding=None)

        assert result.drift_flag == "UNAVAILABLE"
        assert isinstance(result, SemanticResult)
