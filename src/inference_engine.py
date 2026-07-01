"""
ONNX Semantic Security Engine — FastAPI Inference Server

Usage:
    uvicorn src.inference_engine:app --host 0.0.0.0 --port 8000
    
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
    features: List[float] = Field(..., description="Raw feature vector (79 floats for CIC-IDS2018)")

class BatchPredictRequest(BaseModel):
    instances: List[List[float]] = Field(..., description="List of feature vectors")

class PredictionResult(BaseModel):
    label: str
    confidence: float
    mitre_technique_id: Optional[str]
    mitre_technique_name: str
    mitre_tactic: str

class PredictResponse(BaseModel):
    predictions: List[PredictionResult]
    model_type: str
    latency_ms: float


# ── Engine Class ──
class OnnxSecurityEngine:
    def __init__(self, quantized: bool = False):
        model_name = "threat_mlp_int8.onnx" if quantized else "threat_mlp_fp32.onnx"
        model_path = EXPERIMENTS / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model_type = "INT8" if quantized else "FP32"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.scaler = joblib.load(EXPERIMENTS / "standard_scaler.joblib")
        self.encoder = joblib.load(EXPERIMENTS / "label_encoder.joblib")
        self.input_dim = self.session.get_inputs()[0].shape[1]
        self.start_time = time.time()
        
        print(f"✓ Engine loaded: {model_name} | Classes: {list(self.encoder.classes_)}")
    
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
            results.append({
                "label": label,
                "confidence": round(float(conf), 4),
                "mitre_technique_id": mitre["technique_id"],
                "mitre_technique_name": mitre["technique"],
                "mitre_tactic": mitre["tactic"],
            })
        return results


# ── Determine model type from environment variable ──
USE_QUANTIZED = os.environ.get("USE_QUANTIZED", "false").lower() == "true"
engine = OnnxSecurityEngine(quantized=USE_QUANTIZED)

# ── FastAPI App ──
app = FastAPI(
    title="ONNX Semantic Security Engine",
    description="Edge-ready threat classifier with MITRE ATT&CK labeling",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_type": engine.model_type,
        "uptime_seconds": round(time.time() - engine.start_time, 1),
        "classes": list(engine.encoder.classes_),
        "input_features": engine.input_dim,
    }


@app.get("/model/info")
def model_info():
    model_name = f"threat_mlp_{'int8' if engine.model_type == 'INT8' else 'fp32'}.onnx"
    model_path = EXPERIMENTS / model_name
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    return {
        "model_name": model_name,
        "model_type": engine.model_type,
        "size_mb": round(size_mb, 3),
        "num_classes": len(engine.encoder.classes_),
        "classes": list(engine.encoder.classes_),
        "input_dim": engine.input_dim,
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
        latency_ms=round(latency, 3),
    )
