"""
Quantize ONNX Model — Supports FP16 and INT8 (Static) Quantization

Usage:
    python src/quantize_model.py --model experiments/threat_mlp_nf_fp32.onnx --test-data experiments/X_test_nf.npy
"""

import argparse
import numpy as np
import onnx
import onnxruntime as ort
import onnxruntime.quantization as q
from onnxruntime.quantization import shape_inference
import os
from pathlib import Path

class DataReader(q.CalibrationDataReader):
    def __init__(self, data, input_name, batch_size=100):
        self.data = data
        self.input_name = input_name
        self.batch_size = batch_size
        self.current_index = 0

    def get_next(self):
        if self.current_index < len(self.data):
            batch = self.data[self.current_index : self.current_index + self.batch_size]
            self.current_index += self.batch_size
            return {self.input_name: batch}
        else:
            return None

def quantize_fp16(model_path, output_path):
    print(f"Quantizing to FP16: {output_path}")
    import onnxconverter_common
    model = onnx.load(model_path)
    model_fp16 = onnxconverter_common.float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_path)
    print("FP16 Quantization complete.")

def quantize_int8_static(model_path, output_path, data_path):
    print("Running Shape Inference...")
    preprocessed_path = model_path.replace(".onnx", "_preprocessed.onnx")
    shape_inference.quant_pre_process(model_path, preprocessed_path, skip_symbolic_shape=False)
    
    print(f"Loading calibration data from {data_path}...")
    X_test = np.load(data_path)
    
    session = ort.InferenceSession(preprocessed_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Use first 10,000 samples for calibration
    calib_data = X_test[:10000]
    data_reader = DataReader(calib_data, input_name)
    
    print(f"Quantizing to INT8 (Static): {output_path}")
    q.quantize_static(
        model_input=preprocessed_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        quant_format=q.QuantFormat.QOperator, 
        weight_type=q.QuantType.QInt8,
        activation_type=q.QuantType.QUInt8,
    )
    
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
        
    print("INT8 Static Quantization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to input FP32 ONNX model")
    parser.add_argument("--test-data", required=True, help="Path to test data (.npy) for INT8 calibration")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)
        
    fp16_path = model_path.replace("_fp32.onnx", "_fp16.onnx")
    int8_path = model_path.replace("_fp32.onnx", "_int8.onnx")
    
    if fp16_path == model_path:
        fp16_path = model_path.replace(".onnx", "_fp16.onnx")
        int8_path = model_path.replace(".onnx", "_int8.onnx")

    # 1. FP16 Quantization
    quantize_fp16(model_path, fp16_path)
    
    # 2. INT8 Static Quantization
    quantize_int8_static(model_path, int8_path, args.test_data)
    
    print("\nQuantization Summary:")
    print(f"  FP32 Size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print(f"  FP16 Size: {os.path.getsize(fp16_path) / 1024:.1f} KB")
    print(f"  INT8 Size: {os.path.getsize(int8_path) / 1024:.1f} KB")
