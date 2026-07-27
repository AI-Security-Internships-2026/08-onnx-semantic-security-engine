"""
ONNX Semantic Security Engine — FastAPI Inference Server

Supports both Baseline (76-feature) and NF-Standardized (21-feature) models.

Usage:
    # Run with baseline model (default)
    uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000

    # Run with NF-standardized model
    USE_NF=true uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000

    # Run with INT8 quantized model
    USE_QUANTIZED=true uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000

    # Run with NF + INT8
    USE_NF=true USE_QUANTIZED=true uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000
    
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

# ── Paths ──
BASE_DIR = Path(__file__).parent.parent
EXPERIMENTS = BASE_DIR / "experiments"

# ── Pydantic Models ──
class PredictRequest(BaseModel):
    features: List[float] = Field(..., description="Raw feature vector (dimensions depend on model: 76 for baseline, 21 for NF)")

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


# ── Confidence threshold for anomaly flagging ──
CONFIDENCE_THRESHOLD = 0.70


# ── Engine Class ──
class OnnxSecurityEngine:
    def __init__(self, use_nf: bool = False, quantized: bool = False):
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
        self.model_variant = "NF-Standardized (21 features)" if use_nf else "Baseline (76 features)"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        
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
        
        print(f"✓ Engine loaded: {model_name} | Variant: {self.model_variant}")
        print(f"  Classes: {list(self.encoder.classes_)}")
        print(f"  Input features: {self.input_dim}")
    
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


# ── Determine model configuration from environment variables ──
USE_NF = os.environ.get("USE_NF", "false").lower() == "true"
USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "false").lower() == "true"
engine = OnnxSecurityEngine(use_nf=USE_NF, quantized=USE_QUANTIZED)

# ── FastAPI App ──
app = FastAPI(
    title="ONNX Semantic Security Engine",
    description="Edge-ready threat classifier with MITRE ATT&CK labeling and confidence-based anomaly flagging",
    version="2.0.0",
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
