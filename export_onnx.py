import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.classifier import get_engine
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def export():
    print("=== Step 1: Loading BioClipEngine via get_engine() ===")
    t0 = time.time()
    engine = get_engine(device="cpu")
    visual = engine.classifier.model.visual
    visual.eval()
    print(f"Loaded visual model in {time.time() - t0:.2f}s")

    class NormalizedVisual(torch.nn.Module):
        def __init__(self, visual_mod):
            super().__init__()
            self.visual = visual_mod

        def forward(self, x):
            f = self.visual(x)
            return F.normalize(f, dim=-1)

    wrapped = NormalizedVisual(visual)
    wrapped.eval()

    output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(output_dir, exist_ok=True)
    onnx_fp32 = os.path.join(output_dir, "bioclip_visual_fp32.onnx")
    onnx_int8 = os.path.join(output_dir, "bioclip_visual_int8.onnx")

    dummy = torch.randn(1, 3, 224, 224)
    print(f"=== Step 2: Exporting to ONNX FP32 ({onnx_fp32}) ===")
    t1 = time.time()
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_fp32,
        input_names=["image"],
        output_names=["image_features"],
        dynamic_axes={"image": {0: "batch"}, "image_features": {0: "batch"}},
        opset_version=14,
        do_constant_folding=True,
        dynamo=False
    )
    fp32_size = os.path.getsize(onnx_fp32) / (1024 * 1024)
    print(f"ONNX FP32 exported in {time.time() - t1:.2f}s (Size: {fp32_size:.1f} MB)")

    print(f"=== Step 3: Quantizing to INT8 ({onnx_int8}) ===")
    t2 = time.time()
    quantize_dynamic(
        model_input=onnx_fp32,
        model_output=onnx_int8,
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True
    )
    int8_size = os.path.getsize(onnx_int8) / (1024 * 1024)
    print(f"ONNX INT8 saved in {time.time() - t2:.2f}s (Size: {int8_size:.1f} MB)")
    print(f"Reduction achieved: {((fp32_size - int8_size) / fp32_size) * 100:.1f}%!")

    print("=== Step 4: Verification of INT8 ONNX Engine ===")
    session = ort.InferenceSession(onnx_int8, providers=["CPUExecutionProvider"])
    ort_outs = session.run(["image_features"], {"image": dummy.numpy()})[0]
    with torch.no_grad():
        torch_outs = wrapped(dummy).numpy()
    cos_sim = np.dot(ort_outs[0], torch_outs[0])
    print(f"Cosine Similarity (INT8 ONNX vs PyTorch): {cos_sim:.5f}")
    if cos_sim > 0.98:
        print(" SUCCESS: Numerical accuracy verified (>98%)!")

if __name__ == "__main__":
    export()
