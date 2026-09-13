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

def _convert_gemm_to_matmul_add(model):
    """Convert Gemm nodes with constant weights into MatMul + Add so MatMulNBitsQuantizer can quantize them."""
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}
    new_nodes = []
    
    for node in graph.node:
        if node.op_type == "Gemm":
            transB = 0
            alpha = 1.0
            beta = 1.0
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
                elif attr.name == "alpha":
                    alpha = attr.f
                elif attr.name == "beta":
                    beta = attr.f
            
            A_name = node.input[0]
            B_name = node.input[1]
            C_name = node.input[2] if len(node.input) > 2 else None
            out_name = node.output[0]
            
            if B_name in initializers:
                b_tensor = onnx.numpy_helper.to_array(initializers[B_name])
                if transB:
                    b_tensor = b_tensor.T
                if alpha != 1.0:
                    b_tensor = b_tensor * alpha
                
                new_b_name = B_name + "_matmul_weight"
                new_b_init = onnx.numpy_helper.from_array(b_tensor, name=new_b_name)
                graph.initializer.remove(initializers[B_name])
                graph.initializer.append(new_b_init)
                initializers[new_b_name] = new_b_init
                
                matmul_out = out_name + "_matmul_out" if C_name else out_name
                matmul_node = onnx.helper.make_node(
                    "MatMul",
                    inputs=[A_name, new_b_name],
                    outputs=[matmul_out],
                    name=node.name + "_matmul"
                )
                new_nodes.append(matmul_node)
                
                if C_name:
                    add_node = onnx.helper.make_node(
                        "Add",
                        inputs=[matmul_out, C_name],
                        outputs=[out_name],
                        name=node.name + "_add"
                    )
                    new_nodes.append(add_node)
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)
            
    graph.ClearField("node")
    graph.node.extend(new_nodes)
    return model


def quantize_int4(model_path, output_path, data_path=None):
    """INT4 weight-only quantization using MatMulNBits (ONNX Runtime >= 1.17)."""
    print(f"Attempting INT4 quantization: {output_path}")
    try:
        model = onnx.load(model_path)
        # Ensure Gemm layers (e.g. PyTorch Linear layers) are represented as MatMul for 4-bit quantizer
        model = _convert_gemm_to_matmul_add(model)
        
        quant = None
        try:
            from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
            quant = MatMulNBitsQuantizer(
                model=model,
                block_size=32,
                is_symmetric=True,
                bits=4,
            )
        except (ImportError, AttributeError, TypeError):
            from onnxruntime.quantization import matmul_4bits_quantizer
            quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
                model=model,
                block_size=32,
                is_symmetric=True,
            )
            
        quant.process()
        quant.model.save_model_to_file(output_path, use_external_data_format=False)
        print("INT4 Quantization complete.")
        return True
    except (ImportError, AttributeError) as e:
        print(f"  [SKIP] INT4 quantization not available in this onnxruntime version: {e}")
        print(f"  Requires onnxruntime >= 1.17 with MatMul4BitsQuantizer / MatMulNBitsQuantizer support.")
        return False
    except Exception as e:
        print(f"  [ERROR] INT4 quantization failed: {e}")
        return False

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
    int4_path = model_path.replace("_fp32.onnx", "_int4.onnx")
    
    if fp16_path == model_path:
        fp16_path = model_path.replace(".onnx", "_fp16.onnx")
        int8_path = model_path.replace(".onnx", "_int8.onnx")
        int4_path = model_path.replace(".onnx", "_int4.onnx")

    # 1. FP16 Quantization
    quantize_fp16(model_path, fp16_path)
    
    # 2. INT8 Static Quantization
    quantize_int8_static(model_path, int8_path, args.test_data)
    
    # 3. INT4 Weight-Only Quantization
    has_int4 = quantize_int4(model_path, int4_path, args.test_data)
    
    print("\nQuantization Summary:")
    print(f"  FP32 Size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print(f"  FP16 Size: {os.path.getsize(fp16_path) / 1024:.1f} KB")
    print(f"  INT8 Size: {os.path.getsize(int8_path) / 1024:.1f} KB")
    if has_int4 and os.path.exists(int4_path):
        print(f"  INT4 Size: {os.path.getsize(int4_path) / 1024:.1f} KB")
