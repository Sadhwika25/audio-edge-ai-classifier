"""
Unit tests for Edge Quantization & Model Size calculations.
"""

import torch
from src.models import AudioCNN, EdgeDSCNN
from src.utils import quantize_model_dynamic, get_model_size_mb, benchmark_inference_latency


def test_dynamic_quantization():
    model = AudioCNN(num_classes=10)
    model.eval()

    quantized = quantize_model_dynamic(model)
    assert quantized is not None

    dummy_input = torch.randn(1, 1, 64, 101)
    with torch.no_grad():
        out_fp32 = model(dummy_input)
        out_int8 = quantized(dummy_input)

    assert out_int8.shape == out_fp32.shape
    # Check predictions are strongly aligned
    assert torch.argmax(out_fp32) == torch.argmax(out_int8)


def test_latency_benchmarking():
    model = EdgeDSCNN(num_classes=10)
    lat = benchmark_inference_latency(model, num_warmup=2, num_runs=5)

    assert "mean_ms" in lat
    assert lat["mean_ms"] > 0
    assert lat["max_ms"] >= lat["min_ms"]
