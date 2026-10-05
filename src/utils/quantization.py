"""
Post-Training Quantization (PTQ) & Edge Latency Benchmarking.
Quantizes FP32 neural models to INT8 for resource-constrained edge microcontrollers,
and profiles execution latency and memory savings.
"""

import os
import time
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """
    Applies PyTorch dynamic post-training quantization (PTQ) to INT8
    on Linear and Conv layers where applicable.
    """
    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def get_model_size_mb(model: nn.Module, filepath: str = "temp_model.pt") -> float:
    """
    Measures serialized model file size in Megabytes (MB).
    """
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    if os.path.exists(filepath):
        os.remove(filepath)
    return size_mb


def benchmark_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 64, 101),
    num_warmup: int = 20,
    num_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Profiles inference latency (mean, median, p95) in milliseconds.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()
    return {
        "mean_ms": sum(latencies_ms) / len(latencies_ms),
        "median_ms": latencies_ms[len(latencies_ms) // 2],
        "p95_ms": latencies_ms[int(0.95 * len(latencies_ms))],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
    }
