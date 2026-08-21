"""
ONNX Semantic Security Engine — FastAPI Inference Server

Supports both Baseline (76-feature) and NF-Standardized (13-feature) models.
Provides standard prediction endpoints and a /predict/secure endpoint with
full semantic analysis (confidence, drift detection, input validation).

Usage:
    # Run with NF model (default)
    uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000

    # Run with INT8 quantized model
    USE_QUANTIZED=true uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000
    
Then visit: http://localhost:8000/docs for Swagger UI
"""

import os, time, numpy as np, joblib
import onnxruntime as ort
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from scipy.special import softmax

# ── Import MITRE mapping ──
from src.mitre_mapping import get_mitre_label, get_all_mappings

# ── Import Semantic Security Engine ──
from src.semantic_analyzer import SemanticSecurityEngine

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"

# ── Pydantic Models ──
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Raw feature vector (dimensions depend on model: 76 for baseline, 13 for NF)")

class BatchPredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="List of feature vectors")

class PredictionResult(BaseModel):
    label: str
    confidence: float
    confidence_flag: str
    mitre_technique_id: Optional[str]
    mitre_technique_name: str
    mitre_tactic: str

class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    model_type: str
    model_variant: str
    latency_ms: float


# ── Secure prediction response models ──

class SecurePredictionResult(BaseModel):
    """Extended prediction result with semantic analysis fields."""
    label: str
    confidence: float
    confidence_flag: str
    drift_score: float
    drift_flag: str
    nearest_reference_class: str
    validation_passed: bool
    validation_alerts: List[str]
    mitre_technique_id: Optional[str]
    mitre_technique_name: str
    mitre_tactic: str

class SemanticSummary(BaseModel):
    total_alerts: int
    engine_verdict: str

class SecurePredictResponse(BaseModel):
    predictions: List[SecurePredictionResult]
    semantic_summary: SemanticSummary
    model_type: str
    model_variant: str
    latency_ms: float


# ── Confidence threshold for anomaly flagging (calibrated on clean validation data) ──
CONFIDENCE_THRESHOLD = 0.50


# ── Engine Class ──
class OnnxSecurityEngine:
    def __init__(self, use_nf: bool = True, quantized: bool = False):
        # Determine model file name based on variant and precision
        prefix = "threat_mlp_nf" if use_nf else "threat_mlp"
        if quantized:
            model_name = f"{prefix}_int8.onnx"
        else:
            model_name = f"{prefix}_fp32.onnx"
        
        model_path = EXPERIMENTS / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_type = "INT8" if quantized else "FP32"
        self.model_variant = "NF-Standardized (13 features)" if use_nf else "Baseline (76 features)"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        
        # Detect available outputs
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.has_embedding = "embedding" in self.output_names
        
        # Load matching scaler and encoder
        scaler_suffix = "_nf" if use_nf else ""
        encoder_suffix = "_nf" if use_nf else ""
        
        scaler_path = EXPERIMENTS / f"standard_scaler{scaler_suffix}.joblib"
        encoder_path = EXPERIMENTS / f"label_encoder{encoder_suffix}.joblib"
        
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {encoder_path}")
        
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        self.input_dim = self.session.get_inputs()[0].shape[1]
        self.start_time = time.time()
        
        print(f"[OK] Engine loaded: {model_name} | Variant: {self.model_variant}")
        print(f"  Classes: {list(self.encoder.classes_)}")
        print(f"  Input features: {self.input_dim}")
        print(f"  ONNX outputs: {self.output_names}")
        print(f"  Embedding output: {'available (64-dim)' if self.has_embedding else 'not available'}")
    
    def predict(self, features: np.ndarray) -> list[dict]:
        """Run inference on a batch of feature vectors."""
        # Scale
        scaled = self.scaler.transform(features)
        
        # ONNX inference
        logits = self.session.run(None, {"input": scaled.astype(np.float32)})[0]
        
        # Post-process
        probs = softmax(logits, axis=1)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        labels = self.encoder.inverse_transform(pred_indices)
        
        results = []
        for label, conf in zip(labels, confidences):
            mitre = get_mitre_label(label)
            
            # Confidence-based anomaly flagging (Feature A)
            if conf < CONFIDENCE_THRESHOLD:
                confidence_flag = "LOW_CONFIDENCE — possible novel attack or adversarial input"
            else:
                confidence_flag = "OK"
            
            results.append({
                "label": label,
                "confidence": round(float(conf), 4),
                "confidence_flag": confidence_flag,
                "mitre_technique_id": mitre["technique_id"],
                "mitre_technique_name": mitre["technique"],
                "mitre_tactic": mitre["tactic"],
            })
        return results

    def predict_secure(self, raw_features: np.ndarray, semantic_engine: SemanticSecurityEngine) -> list[dict]:
        """Run inference with full semantic analysis on a batch.

        Returns extended results including drift scores, validation alerts,
        and overall engine verdict.
        """
        # Scale features
        scaled = self.scaler.transform(raw_features)

        # ONNX inference — get both logits and embeddings
        onnx_outputs = self.session.run(None, {"input": scaled.astype(np.float32)})
        logits = onnx_outputs[0]
        embeddings = onnx_outputs[1] if self.has_embedding else None

        # Softmax probabilities
        probs = softmax(logits, axis=1)
        pred_indices = np.argmax(probs, axis=1)
        labels = self.encoder.inverse_transform(pred_indices)

        results = []
        max_alerts = 0

        for i, label in enumerate(labels):
            mitre = get_mitre_label(label)

            # Run semantic analysis
            embedding_i = embeddings[i] if embeddings is not None else None
            semantic_result = semantic_engine.analyze(
                raw_features=raw_features[i],
                softmax_probs=probs[i],
                embedding=embedding_i,
            )

            max_alerts = max(max_alerts, semantic_result.total_alerts)

            results.append({
                "label": label,
                "confidence": semantic_result.confidence_score,
                "confidence_flag": semantic_result.confidence_flag,
                "drift_score": semantic_result.drift_score,
                "drift_flag": semantic_result.drift_flag,
                "nearest_reference_class": semantic_result.nearest_reference_class,
                "validation_passed": semantic_result.validation_passed,
                "validation_alerts": semantic_result.validation_alerts,
                "mitre_technique_id": mitre["technique_id"],
                "mitre_technique_name": mitre["technique"],
                "mitre_tactic": mitre["tactic"],
                "_total_alerts": semantic_result.total_alerts,
                "_engine_verdict": semantic_result.engine_verdict,
            })

        return results


# ── Determine model configuration from environment variables ──
USE_NF = os.environ.get("USE_NF", "true").lower() == "true"
USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "false").lower() == "true"
engine = OnnxSecurityEngine(use_nf=USE_NF, quantized=USE_QUANTIZED)

# ── Initialize Semantic Security Engine ──
semantic_engine = SemanticSecurityEngine(use_nf=USE_NF)

# ── FastAPI App ──
app = FastAPI(
    title="ONNX Semantic Security Engine",
    description=(
        "Edge-ready threat classifier with MITRE ATT&CK labeling, "
        "confidence-based anomaly flagging, semantic drift detection, "
        "and input validation."
    ),
    version="3.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_type": engine.model_type,
        "model_variant": engine.model_variant,
        "uptime_seconds": round(time.time() - engine.start_time, 1),
        "classes": list(engine.encoder.classes_),
        "input_features": engine.input_dim,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "semantic_features": {
            "confidence": True,
            "drift_detection": engine.has_embedding and semantic_engine.has_drift,
            "input_validation": semantic_engine.has_validation,
        },
    }


@app.get("/model/info")
def model_info():
    prefix = "threat_mlp_nf" if USE_NF else "threat_mlp"
    model_name = f"{prefix}_{'int8' if engine.model_type == 'INT8' else 'fp32'}.onnx"
    model_path = EXPERIMENTS / model_name
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return {
        "model_name": model_name,
        "model_type": engine.model_type,
        "model_variant": engine.model_variant,
        "size_mb": round(size_mb, 3),
        "num_classes": len(engine.encoder.classes_),
        "classes": list(engine.encoder.classes_),
        "input_dim": engine.input_dim,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "onnx_outputs": engine.output_names,
        "has_embedding_output": engine.has_embedding,
    }


@app.get("/mitre/mappings")
def mitre_mappings():
    return get_all_mappings()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if len(request.features) != engine.input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {engine.input_dim} features, got {len(request.features)}"
        )
    
    start = time.perf_counter()
    features = np.array([request.features])
    results = engine.predict(features)
    latency = (time.perf_counter() - start) * 1000
    
    return PredictResponse(
        predictions=[PredictionResult(**r) for r in results],
        model_type=engine.model_type,
        model_variant=engine.model_variant,
        latency_ms=round(latency, 3),
    )


@app.post("/predict/batch", response_model=PredictResponse)
def predict_batch(request: BatchPredictRequest):
    for inst in request.instances:
        if len(inst) != engine.input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {engine.input_dim} features per instance, got {len(inst)}"
            )
    
    start = time.perf_counter()
    features = np.array(request.instances)
    results = engine.predict(features)
    latency = (time.perf_counter() - start) * 1000
    
    return PredictResponse(
        predictions=[PredictionResult(**r) for r in results],
        model_type=engine.model_type,
        model_variant=engine.model_variant,
        latency_ms=round(latency, 3),
    )


@app.post("/predict/secure", response_model=SecurePredictResponse)
def predict_secure(request: PredictRequest):
    """Secure prediction with full semantic analysis.

    Runs all three semantic checks on the input:
    - Feature A: Confidence scoring (softmax threshold)
    - Feature B: Drift detection (ONNX fc3 embedding distance)
    - Feature C: Input validation (schema, range, zero-fill checks)

    Returns extended results with drift_score, validation_alerts, and
    an overall engine_verdict (CLEAN / SUSPICIOUS / REJECTED).
    """
    if len(request.features) != engine.input_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {engine.input_dim} features, got {len(request.features)}"
        )

    start = time.perf_counter()
    features = np.array([request.features])
    results = engine.predict_secure(features, semantic_engine)
    latency = (time.perf_counter() - start) * 1000

    # Extract semantic summary from results
    total_alerts = max(r["_total_alerts"] for r in results)
    engine_verdict = results[0]["_engine_verdict"]

    # Clean internal fields before response
    clean_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(r_copy)

    return SecurePredictResponse(
        predictions=[SecurePredictionResult(**r) for r in clean_results],
        semantic_summary=SemanticSummary(
            total_alerts=total_alerts,
            engine_verdict=engine_verdict,
        ),
        model_type=engine.model_type,
        model_variant=engine.model_variant,
        latency_ms=round(latency, 3),
    )


@app.post("/predict/secure/batch", response_model=SecurePredictResponse)
def predict_secure_batch(request: BatchPredictRequest):
    """Batch secure prediction with full semantic analysis."""
    for inst in request.instances:
        if len(inst) != engine.input_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {engine.input_dim} features per instance, got {len(inst)}"
            )

    start = time.perf_counter()
    features = np.array(request.instances)
    results = engine.predict_secure(features, semantic_engine)
    latency = (time.perf_counter() - start) * 1000

    total_alerts = max(r["_total_alerts"] for r in results)
    # Use worst verdict across batch
    verdicts = [r["_engine_verdict"] for r in results]
    if "REJECTED" in verdicts:
        engine_verdict = "REJECTED"
    elif "SUSPICIOUS" in verdicts:
        engine_verdict = "SUSPICIOUS"
    else:
        engine_verdict = "CLEAN"

    clean_results = []
    for r in results:
        r_copy = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results.append(r_copy)

    return SecurePredictResponse(
        predictions=[SecurePredictionResult(**r) for r in clean_results],
        semantic_summary=SemanticSummary(
            total_alerts=total_alerts,
            engine_verdict=engine_verdict,
        ),
        model_type=engine.model_type,
        model_variant=engine.model_variant,
        latency_ms=round(latency, 3),
    )
