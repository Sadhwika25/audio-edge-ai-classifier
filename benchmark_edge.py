"""
Edge AI Inference Benchmark & Post-Training Quantization (PTQ) Profiler.
Measures latency, memory footprint, and compression ratio between FP32 and INT8 models.
"""

import argparse
import os
import torch

from src.models import AudioCNN, EdgeDSCNN
from src.data import ACOUSTIC_CLASSES
from src.utils import quantize_model_dynamic, get_model_size_mb, benchmark_inference_latency


def main():
    parser = argparse.ArgumentParser(description="Benchmark Edge AI Audio Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/edge_dscnn_best.pt")
    parser.add_argument("--num-runs", type=int, default=150)
    args = parser.parse_args()

    device = torch.device("cpu")
    num_classes = len(ACOUSTIC_CLASSES)

    # Load trained model or initialize fresh instance
    if os.path.exists(args.checkpoint):
        print(f"[*] Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model_type = ckpt.get("model_type", "edge_dscnn")
        model_fp32 = AudioCNN(num_classes=num_classes) if model_type == "audio_cnn" else EdgeDSCNN(num_classes=num_classes)
        model_fp32.load_state_dict(ckpt["model_state_dict"])
    else:
        print("[!] No checkpoint found; evaluating fresh EdgeDSCNN architecture...")
        model_fp32 = EdgeDSCNN(num_classes=num_classes)

    model_fp32.eval()

    # 1. Profile FP32 Model
    print("\n[*] Profiling Full Precision (FP32) Model on CPU...")
    size_fp32 = get_model_size_mb(model_fp32, "temp_fp32.pt")
    lat_fp32 = benchmark_inference_latency(model_fp32, num_runs=args.num_runs)

    # 2. Quantize to INT8
    print("[*] Performing Post-Training Dynamic Quantization (INT8)...")
    model_int8 = quantize_model_dynamic(model_fp32)
    size_int8 = get_model_size_mb(model_int8, "temp_int8.pt")
    lat_int8 = benchmark_inference_latency(model_int8, num_runs=args.num_runs)

    # 3. Output Benchmark Results Table
    size_reduction = ((size_fp32 - size_int8) / size_fp32) * 100.0 if size_fp32 > 0 else 0.0
    throughput_fp32 = 1000.0 / lat_fp32["mean_ms"]
    throughput_int8 = 1000.0 / lat_int8["mean_ms"]

    print("\n" + "=" * 70)
    print("        EDGE AI HARDWARE DEPLOYMENT BENCHMARK REPORT")
    print("=" * 70)
    print(f"{'Metric':<28}{'FP32 Baseline':<20}{'Quantized (INT8)':<20}")
    print("-" * 70)
    print(f"{'Model Size (MB)':<28}{size_fp32:<20.3f}{size_int8:<20.3f}")
    print(f"{'Memory Reduction (%)':<28}{'---':<20}{size_reduction:<20.1f}%")
    print(f"{'Mean Latency (ms)':<28}{lat_fp32['mean_ms']:<20.2f}{lat_int8['mean_ms']:<20.2f}")
    print(f"{'Median Latency (ms)':<28}{lat_fp32['median_ms']:<20.2f}{lat_int8['median_ms']:<20.2f}")
    print(f"{'P95 Latency (ms)':<28}{lat_fp32['p95_ms']:<20.2f}{lat_int8['p95_ms']:<20.2f}")
    print(f"{'Throughput (inferences/sec)':<28}{throughput_fp32:<20.1f}{throughput_int8:<20.1f}")
    print("=" * 70)
    print("[✓] Edge AI Profiling completed successfully.")


if __name__ == "__main__":
    main()
