"""
Real-Time Streaming Audio Simulation.
Simulates a live 16 kHz microphone input stream on edge hardware:
- Buffers incoming 16kHz audio frames
- Executes Voice Activity Detection (VAD)
- Computes STFT Log-Mel Spectrogram features
- Runs neural inference with confidence scoring and latency timing
"""

import os
import time
import torch
import numpy as np

from src.models import EdgeDSCNN
from src.dsp.features import AudioFeatureExtractor, voice_activity_detection
from src.data.synthetic_data import ACOUSTIC_CLASSES, generate_synthetic_acoustic_sample


def simulate_live_audio_stream(
    model_path: str = "checkpoints/edge_dscnn_best.pt",
    num_chunks: int = 8,
    sample_rate: int = 16000,
):
    print("=" * 68)
    print("      LIVE AUDIO STREAMING INFERENCE SIMULATOR (16 kHz)")
    print("=" * 68)

    device = torch.device("cpu")
    num_classes = len(ACOUSTIC_CLASSES)
    model = EdgeDSCNN(num_classes=num_classes)

    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[*] Loaded trained model: {model_path}")
    else:
        print("[!] Checkpoint not found; running with initialized model weights.")

    model.eval()
    feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)

    print("\nStreaming Simulation Starting...\n")
    print(f"{'Chunk':<8}{'VAD Status':<14}{'Top Prediction':<18}{'Confidence':<14}{'Latency':<10}")
    print("-" * 68)

    for i in range(num_chunks):
        # Pick a simulated acoustic event
        simulated_class = i % num_classes
        waveform = generate_synthetic_acoustic_sample(simulated_class, sample_rate=sample_rate)

        t_start = time.perf_counter()

        # Step 1: Voice Activity Detection (Short-time energy threshold)
        active_frames = voice_activity_detection(waveform, energy_threshold_db=-30.0)
        is_active = (active_frames.sum().item() / active_frames.numel()) > 0.15

        if not is_active:
            vad_str = "SILENCE / IDLE"
            pred_class = "---"
            confidence_str = "---"
        else:
            vad_str = "ACTIVE AUDIO"
            # Step 2: DSP Feature Extraction (STFT Log-Mel)
            with torch.no_grad():
                spec = feature_extractor(waveform.unsqueeze(0))
                # Step 3: Neural Inference
                logits = model(spec)
                probs = torch.softmax(logits, dim=-1)
                confidence, pred_idx = torch.max(probs, dim=-1)

            pred_class = ACOUSTIC_CLASSES[pred_idx.item()]
            confidence_str = f"{confidence.item() * 100.0:.1f}%"

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        print(f"#{i + 1:<7}{vad_str:<14}{pred_class:<18}{confidence_str:<14}{t_elapsed:.2f} ms")
        time.sleep(0.08)  # simulate buffer delay

    print("-" * 68)
    print("[✓] Audio stream simulation completed.")


if __name__ == "__main__":
    simulate_live_audio_stream()
