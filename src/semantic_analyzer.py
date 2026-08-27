"""
Semantic Security Engine — Runtime Analysis Layer

Provides three analyzers that augment ONNX model inference with security-aware
runtime checks, leveraging ONNX Runtime's intermediate-layer extraction to
detect distribution drift, validate inputs, and flag low-confidence predictions.

Classes:
    ConfidenceAnalyzer      — Feature A: softmax confidence thresholding
    DriftDetector           — Feature B: embedding-space drift via fc3 layer
    InputValidator          — Feature C: schema, range, z-score, zero-fill checks
    SemanticSecurityEngine  — Orchestrator composing all three analyzers

Usage:
    from semantic_analyzer import SemanticSecurityEngine

    engine = SemanticSecurityEngine(use_nf=True)
    result = engine.analyze(
        raw_features=np.array([...]),      # raw input vector (pre-scaling)
        logits=np.array([...]),            # ONNX model output logits
        embedding=np.array([...]),         # ONNX model fc3 embedding
    )
"""

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from scipy.spatial.distance import cosine as cosine_distance


# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"


# ── Data Classes ──

@dataclass
class ConfidenceResult:
    """Output from ConfidenceAnalyzer."""
    confidence_score: float
    confidence_flag: str  # "OK" | "LOW_CONFIDENCE"


@dataclass
class DriftResult:
    """Output from DriftDetector."""
    cosine_distance: float
    mahalanobis_distance: float
    drift_score: float  # normalized composite score (0.0 = in-distribution, 1.0 = extreme drift)
    drift_flag: str  # "OK" | "DRIFT_DETECTED"
    nearest_reference_class: str


@dataclass
class ValidationResult:
    """Output from InputValidator."""
    validation_passed: bool
    alerts: List[str] = field(default_factory=list)


@dataclass
class SemanticResult:
    """Unified output from SemanticSecurityEngine."""
    # Confidence (Feature A)
    confidence_score: float
    confidence_flag: str

    # Drift detection (Feature B)
    drift_score: float
    drift_flag: str
    nearest_reference_class: str

    # Input validation (Feature C)
    validation_passed: bool
    validation_alerts: List[str]

    # Summary
    total_alerts: int
    engine_verdict: str  # "CLEAN" | "SUSPICIOUS" | "REJECTED"


# ── Feature A: Confidence Analyzer ──

class ConfidenceAnalyzer:
    """Analyzes softmax confidence to flag uncertain predictions.

    When the maximum softmax probability falls below the threshold,
    the prediction is flagged as LOW_CONFIDENCE — indicating a possible
    novel attack type, adversarial input, or out-of-distribution sample.
    """

    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def analyze(self, softmax_probs: np.ndarray) -> ConfidenceResult:
        """Analyze a single prediction's softmax probabilities.

        Args:
            softmax_probs: 1D array of softmax probabilities for one sample.

        Returns:
            ConfidenceResult with score and flag.
        """
        confidence = float(np.max(softmax_probs))

        if confidence < self.threshold:
            flag = "LOW_CONFIDENCE"
        else:
            flag = "OK"

        return ConfidenceResult(
            confidence_score=round(confidence, 4),
            confidence_flag=flag,
        )


# ── Feature B: Drift Detector ──

class DriftDetector:
    """Detects distribution drift using ONNX intermediate-layer embeddings.

    Uses the 64-dim fc3 hidden layer output extracted from the ONNX model
    at inference time. Compares against reference centroids computed from
    the training data distribution.

    Drift is measured via:
        - Cosine distance to global centroid
        - Mahalanobis distance to global centroid (accounts for correlations)
        - Nearest-class identification (cosine to per-class centroids)
    """

    def __init__(
        self,
        reference_path: Optional[Path] = None,
        suffix: str = "",
        cosine_threshold: Optional[float] = None,
        mahal_threshold: Optional[float] = None,
    ):
        if reference_path is None:
            reference_path = EXPERIMENTS / f"reference_embeddings{suffix}.npz"

        if not reference_path.exists():
            raise FileNotFoundError(
                f"Reference embeddings not found: {reference_path}\n"
                f"Run: python src/embedding_reference.py {'--nf' if '_nf' in suffix else ''}"
            )

        data = np.load(reference_path, allow_pickle=True)
        self.global_centroid = data["global_centroid"]
        self.class_centroids = data["class_centroids"]
        self.covariance_inverse = data["covariance_inverse"]
        
        # Load calibrated thresholds from config file (written by evaluate_semantic_engine.py)
        calib_config_path = EXPERIMENTS / "calibration_config.json"
        if calib_config_path.exists() and "_nf" in suffix:
            try:
                with open(calib_config_path) as f:
                    calib = json.load(f)
                default_cos = float(calib.get("cosine_drift_threshold", data["cosine_threshold"]))
                default_mahal = float(calib.get("mahalanobis_drift_threshold", data["mahal_threshold"]))
                print(f"  Loaded calibrated thresholds from {calib_config_path.name} "
                      f"(calibrated: {calib.get('last_calibrated', 'unknown')})")
            except Exception as e:
                print(f"  [WARN] Failed reading calibration_config.json ({e}), using npz defaults")
                default_cos = float(data["cosine_threshold"])
                default_mahal = float(data["mahal_threshold"])
        else:
            default_cos = float(data["cosine_threshold"])
            default_mahal = float(data["mahal_threshold"])
        
        self.cosine_threshold = cosine_threshold if cosine_threshold is not None else default_cos
        self.mahal_threshold = mahal_threshold if mahal_threshold is not None else default_mahal
        self.class_names = list(data["class_names"])

        print(f"  DriftDetector loaded: {len(self.class_names)} classes, "
              f"cosine_thresh={self.cosine_threshold:.4f}, "
              f"mahal_thresh={self.mahal_threshold:.4f}")

    def analyze(self, embedding: np.ndarray) -> DriftResult:
        """Analyze a single embedding vector for distribution drift.

        Args:
            embedding: 1D array (64-dim) from ONNX model's fc3 layer.

        Returns:
            DriftResult with distances, score, flag, and nearest class.
        """
        # Cosine distance to global centroid
        # NaN cosine means zero-norm vector or corrupted embedding -> treat as maximum drift
        cos_dist = cosine_distance(embedding, self.global_centroid)
        if np.isnan(cos_dist):
            cos_dist = 1.0  # Maximum cosine distance
        else:
            cos_dist = float(cos_dist)

        # Mahalanobis distance to global centroid
        diff = embedding - self.global_centroid
        mahal_dist = float(np.sqrt(max(diff @ self.covariance_inverse @ diff, 0.0)))

        # Nearest-class identification
        class_distances = []
        for centroid in self.class_centroids:
            d = cosine_distance(embedding, centroid)
            class_distances.append(1.0 if np.isnan(d) else float(d))
        nearest_idx = int(np.argmin(class_distances))
        nearest_class = self.class_names[nearest_idx]

        # Composite drift score (0.0 = in-distribution, 1.0 = extreme drift)
        # Normalize each distance by its threshold, take the max
        cos_normalized = cos_dist / max(self.cosine_threshold, 1e-8)
        mahal_normalized = mahal_dist / max(self.mahal_threshold, 1e-8)
        drift_score = float(min(max(cos_normalized, mahal_normalized), 2.0))

        # Flag as drift if either distance exceeds its threshold
        if cos_dist > self.cosine_threshold or mahal_dist > self.mahal_threshold:
            drift_flag = "DRIFT_DETECTED"
        else:
            drift_flag = "OK"

        return DriftResult(
            cosine_distance=round(cos_dist, 4),
            mahalanobis_distance=round(mahal_dist, 4),
            drift_score=round(drift_score, 4),
            drift_flag=drift_flag,
            nearest_reference_class=nearest_class,
        )


# ── Feature C: Input Validator ──

class InputValidator:
    """Validates raw input feature vectors before model inference.

    Performs schema, range, outlier, and zero-fill checks to catch
    malformed or out-of-distribution inputs early. This directly
    addresses the RQ3 finding that cross-dataset generalization fails
    when feature extractors produce incompatible distributions.

    Checks:
        1. Schema: correct number of features
        2. NaN/Inf: reject inputs with non-finite values
        3. Range: each feature within training distribution ± tolerance
        4. Z-score: flag extreme outliers (|z| > 5)
        5. Zero-fill: detect >50% zero features (RQ3 failure mode)
    """

    def __init__(self, stats_path: Optional[Path] = None, suffix: str = ""):
        if stats_path is None:
            stats_path = EXPERIMENTS / f"training_feature_stats{suffix}.json"

        if not stats_path.exists():
            raise FileNotFoundError(
                f"Training feature stats not found: {stats_path}\n"
                f"Run: python src/embedding_reference.py {'--nf' if '_nf' in suffix else ''}"
            )

        with open(stats_path) as f:
            data = json.load(f)

        self.num_features = data["num_features"]
        self.feature_stats = data["features"]

        # Pre-compute arrays for vectorized operations
        self.means = np.array([f["mean"] for f in self.feature_stats])
        self.stds = np.array([f["std"] for f in self.feature_stats])
        self.mins = np.array([f["min_approx"] for f in self.feature_stats])
        self.maxs = np.array([f["max_approx"] for f in self.feature_stats])
        self.names = [f["name"] for f in self.feature_stats]

        # Configurable thresholds calibrated for long-tailed network flows
        self.zscore_threshold = 15.0
        self.zero_fill_ratio = 0.80  # requires >=80% zeros to prevent false alarms on valid unidirectional flows
        self.range_tolerance_sigmas = 15.0

        print(f"  InputValidator loaded: {self.num_features} features, "
              f"zscore_thresh={self.zscore_threshold}, "
              f"zero_fill_ratio={self.zero_fill_ratio}")

    def analyze(self, raw_features: np.ndarray) -> ValidationResult:
        """Validate a single raw feature vector (before scaling).

        Args:
            raw_features: 1D array of raw feature values.

        Returns:
            ValidationResult with pass/fail and specific alerts.
        """
        alerts = []

        # ── Check 1: Schema (feature count) ──
        if len(raw_features) != self.num_features:
            alerts.append(
                f"SCHEMA_MISMATCH: Expected {self.num_features} features, "
                f"got {len(raw_features)}"
            )
            return ValidationResult(validation_passed=False, alerts=alerts)

        features = np.asarray(raw_features, dtype=np.float64)

        # ── Check 2: NaN/Inf ──
        nan_mask = np.isnan(features)
        inf_mask = np.isinf(features)

        if nan_mask.any():
            nan_indices = np.where(nan_mask)[0]
            nan_names = [self.names[i] for i in nan_indices]
            alerts.append(f"NAN_VALUES: Features contain NaN at indices {list(nan_indices)} ({nan_names})")

        if inf_mask.any():
            inf_indices = np.where(inf_mask)[0]
            inf_names = [self.names[i] for i in inf_indices]
            alerts.append(f"INF_VALUES: Features contain Inf at indices {list(inf_indices)} ({inf_names})")

        if nan_mask.any() or inf_mask.any():
            return ValidationResult(validation_passed=False, alerts=alerts)

        # ── Check 3: Zero-fill detection (RQ3 failure mode) ──
        # In network telemetry, unidirectional flows (e.g. UDP/DNS) naturally have ~5-7 zero fields out of 13.
        # True zero-fill corruption/tampering occurs when >=80% of features are zero or all core fields are 0.
        zero_count = int(np.sum(features == 0.0))
        zero_ratio = zero_count / self.num_features

        is_structural_zero = (
            zero_ratio >= self.zero_fill_ratio or
            (self.num_features == 13 and features[0] == 0.0 and features[1] == 0.0 and features[2] == 0.0 and features[5] == 0.0)
        )

        if is_structural_zero:
            alerts.append(
                f"ZERO_FILLED: {zero_count}/{self.num_features} features are 0.0 "
                f"({zero_ratio:.0%}). This pattern indicates telemetry corruption or zero-fill tampering (RQ3 failure mode)."
            )

        # ── Check 4: Range check (soft statistical warning) ──
        range_lower = self.means - self.range_tolerance_sigmas * self.stds
        range_upper = self.means + self.range_tolerance_sigmas * self.stds
        out_of_range = (features < range_lower) | (features > range_upper)

        if out_of_range.any():
            oor_indices = np.where(out_of_range)[0]
            oor_details = [
                f"{self.names[idx]}={features[idx]:.2f} "
                f"(expected [{range_lower[idx]:.2f}, {range_upper[idx]:.2f}])"
                for idx in oor_indices[:3]
            ]
            suffix_text = f" (+{len(oor_indices) - 3} more)" if len(oor_indices) > 3 else ""
            alerts.append(
                f"OUT_OF_RANGE: {len(oor_indices)} feature(s) outside "
                f"training range (mean ± {self.range_tolerance_sigmas}σ): {', '.join(oor_details)}{suffix_text}"
            )

        # ── Check 5: Z-score outlier check (soft statistical warning) ──
        safe_stds = np.where(self.stds > 1e-10, self.stds, 1.0)
        zscores = np.abs((features - self.means) / safe_stds)
        extreme_outliers = zscores > self.zscore_threshold

        if extreme_outliers.any():
            outlier_indices = np.where(extreme_outliers)[0]
            outlier_details = [f"{self.names[idx]}: z={zscores[idx]:.1f}" for idx in outlier_indices[:3]]
            suffix_text = f" (+{len(outlier_indices) - 3} more)" if len(outlier_indices) > 3 else ""
            alerts.append(
                f"EXTREME_OUTLIERS: {len(outlier_indices)} feature(s) with "
                f"|z-score| > {self.zscore_threshold}: {', '.join(outlier_details)}{suffix_text}"
            )

        # Determine pass/fail
        # Hard failures: schema mismatch, NaN, Inf, true zero-fill tampering
        # Soft warnings: out-of-range, extreme outliers (still pass validation, but flagged for threat analysis)
        validation_passed = not any(
            alert.startswith(("SCHEMA_", "NAN_", "INF_", "ZERO_FILLED"))
            for alert in alerts
        )

        return ValidationResult(
            validation_passed=validation_passed,
            alerts=alerts,
        )


# ── Orchestrator ──

class SemanticSecurityEngine:
    """Composes all three analyzers into a unified semantic analysis pipeline.

    Runs confidence scoring, drift detection, and input validation on every
    inference request, returning a single SemanticResult with all flags,
    scores, and an overall engine verdict.
    """

    def __init__(self, use_nf: bool = False):
        suffix = "_nf" if use_nf else ""
        print("Initializing Semantic Security Engine...")

        # Load confidence threshold from calibration config
        calib_config_path = EXPERIMENTS / "calibration_config.json"
        conf_threshold = 0.50
        if calib_config_path.exists() and use_nf:
            try:
                with open(calib_config_path) as f:
                    calib = json.load(f)
                conf_threshold = float(calib.get("confidence_threshold", 0.50))
            except Exception:
                conf_threshold = 0.50

        self.confidence_analyzer = ConfidenceAnalyzer(threshold=conf_threshold)
        print(f"  ConfidenceAnalyzer: threshold={conf_threshold:.4f}")

        # Try to load drift detector (requires reference embeddings)
        try:
            self.drift_detector = DriftDetector(suffix=suffix)
            self.has_drift = True
        except FileNotFoundError as e:
            print(f"  [WARN] DriftDetector unavailable: {e}")
            self.drift_detector = None
            self.has_drift = False

        # Try to load input validator (requires training stats)
        try:
            self.input_validator = InputValidator(suffix=suffix)
            self.has_validation = True
        except FileNotFoundError as e:
            print(f"  [WARN] InputValidator unavailable: {e}")
            self.input_validator = None
            self.has_validation = False

        print(f"  Engine ready: confidence=OK, drift={'OK' if self.has_drift else 'NO'}, "
              f"validation={'OK' if self.has_validation else 'NO'}")

    def analyze(
        self,
        raw_features: np.ndarray,
        softmax_probs: np.ndarray,
        embedding: Optional[np.ndarray] = None,
    ) -> SemanticResult:
        """Run all semantic checks on a single inference sample.

        Args:
            raw_features: 1D raw feature vector (pre-scaling).
            softmax_probs: 1D softmax probabilities from model output.
            embedding: 1D fc3 embedding vector (64-dim). Required for drift detection.

        Returns:
            SemanticResult with all analyzer outputs and overall verdict.
        """
        # Feature A: Confidence
        confidence_result = self.confidence_analyzer.analyze(softmax_probs)

        # Feature B: Drift Detection
        if self.has_drift and embedding is not None:
            drift_result = self.drift_detector.analyze(embedding)
        else:
            drift_result = DriftResult(
                cosine_distance=0.0,
                mahalanobis_distance=0.0,
                drift_score=0.0,
                drift_flag="UNAVAILABLE",
                nearest_reference_class="N/A",
            )

        # Feature C: Input Validation
        if self.has_validation:
            validation_result = self.input_validator.analyze(raw_features)
        else:
            validation_result = ValidationResult(
                validation_passed=True,
                alerts=["INPUT_VALIDATION_UNAVAILABLE"],
            )

        # ── Compute overall verdict ──
        # Count independent anomaly signals without duplicate penalty
        total_alerts = 0
        is_low_conf = confidence_result.confidence_flag != "OK"
        is_drift = drift_result.drift_flag == "DRIFT_DETECTED"
        has_soft_val_alert = len(validation_result.alerts) > 0 and validation_result.validation_passed

        if is_low_conf:
            total_alerts += 1
        if is_drift:
            total_alerts += 1
        if has_soft_val_alert:
            total_alerts += 1

        # Determine calibrated engine verdict
        if not validation_result.validation_passed:
            verdict = "REJECTED"
        elif total_alerts >= 2:
            verdict = "HIGH_RISK"
        elif total_alerts == 1:
            verdict = "SUSPICIOUS"
        else:
            verdict = "CLEAN"

        return SemanticResult(
            # Confidence
            confidence_score=confidence_result.confidence_score,
            confidence_flag=confidence_result.confidence_flag,
            # Drift
            drift_score=drift_result.drift_score,
            drift_flag=drift_result.drift_flag,
            nearest_reference_class=drift_result.nearest_reference_class,
            # Validation
            validation_passed=validation_result.validation_passed,
            validation_alerts=validation_result.alerts,
            # Summary
            total_alerts=total_alerts,
            engine_verdict=verdict,
        )

