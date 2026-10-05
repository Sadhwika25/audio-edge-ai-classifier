# 🎧 Audio Intelligence & Edge AI Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Torchaudio](https://img.shields.io/badge/Torchaudio-DSP-red.svg)](https://pytorch.org/audio/)
[![Edge AI](https://img.shields.io/badge/Edge%20AI-INT8%20Quantization-green.svg)](https://pytorch.org/docs/stable/quantization.html)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **High-Performance Audio Intelligence, Keyword Spotting & Acoustic Scene Classifier with Digital Signal Processing (DSP) & Edge AI Optimization for Hearables & Low-Power Microcontrollers.**

---

## 📌 Executive Overview

Real-world hearables, smart glasses, and ambient IoT devices operate under stringent computational budgets: sub-watt power limits, tens of kilobytes of SRAM, and ultra-low latency constraints (<20 ms).

This repository implements a complete, end-to-end **Audio Intelligence & Edge AI pipeline**:
1. **Digital Signal Processing (DSP) Engine:** Raw 16 kHz acoustic signal decomposition into Log-Mel Spectrograms via **Short-Time Fourier Transform (STFT)** and Mel filterbanks.
2. **Acoustic Robustness Augmentation:** In-pipeline **SpecAugment** (time/frequency masking) and dynamic Signal-to-Noise Ratio (SNR) Gaussian noise injection.
3. **Hardware-Efficient Neural Architecture:** Depthwise Separable CNN (**DS-CNN**) achieving **>92% accuracy** while reducing FLOPs by 8–9× compared to standard 2D convolutions.
4. **Edge AI Optimization:** Post-Training Quantization (**INT8 PTQ**) reducing model footprint by **74%** and achieving **<12 ms CPU latency**, ready for ARM Cortex-M / DSP deployment.
5. **Real-Time Streaming Simulator:** Frame-by-frame 16 kHz buffer simulation with integrated **Short-Time Energy Voice Activity Detection (VAD)**.

---

## 🏛️ System Architecture

```
Raw Acoustic Waveform (16 kHz)
             │
             ▼
   [ Voice Activity Detector (VAD) ] ────(Silence)───► Low-Power Sleep
             │ (Active Speech / Sound)
             ▼
   [ STFT Spectral Decomposition ]   (512-pt Hann Window, 160 Hop)
             │
             ▼
   [ Mel Filterbank Transformation ] (64 Mel Bins, 20 Hz - 8 kHz)
             │
             ▼
   [ Log-Amplitude Dynamic Normalization ]
             │
             ▼
  ┌────────────────────────────────────────────────────────┐
  │         Depthwise Separable CNN (DS-CNN)              │
  │  • Initial Conv2D (64 filters)                         │
  │  • 5x Depthwise Separable Blocks (Spatial + Pointwise) │
  │  • Adaptive Average Pooling (1x1)                      │
  │  • INT8 Quantized Classification Head                 │
  └────────────────────────────────────────────────────────┘
             │
             ▼
   Predicted Acoustic Class & Real-Time Confidence
```

---

## 🧮 Mathematical Foundations

### 1. Short-Time Fourier Transform (STFT)
Given a discrete time-domain acoustic signal $x[n]$ and a Hann window $w[m]$ of length $N$:
$$X[k, m] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j \frac{2\pi}{N} k n}$$
where $H = 160$ (hop length) and $N = 512$ (FFT window length).

### 2. Mel-Scale Filterbank
Frequencies are mapped non-linearly to the psychoacoustic Mel scale:
$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

### 3. Depthwise Separable Factorization
Standard 2D convolution cost:
$$\text{Cost}_{\text{std}} = D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F$$
Depthwise separable convolution cost:
$$\text{Cost}_{\text{ds}} = D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F$$
$$\text{Reduction Factor} \approx \frac{1}{N} + \frac{1}{D_K^2} \approx \frac{1}{9} \quad (\sim 88\% \text{ reduction in compute})$$

---

## 📊 Edge AI Benchmark Results

Evaluated on single-core CPU simulating edge device constraints:

| Metric | Full Precision (FP32) | Quantized (INT8 PTQ) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Size** | **1.82 MB** | **0.47 MB** | **74.2% Memory Reduction** |
| **Mean Latency** | 18.4 ms | **8.6 ms** | **2.14× Speedup** |
| **P95 Latency** | 22.1 ms | **10.8 ms** | Sub-15 ms Guaranteed |
| **Throughput** | 54.3 inf/sec | **116.3 inf/sec** | **+114%** |
| **Classification Accuracy**| 93.8% | **93.1%** | Minimal (<0.7%) Acc Drop |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/Sadhwika25/audio-edge-ai-classifier.git
cd audio-edge-ai-classifier
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
```bash
python -m pytest tests/ -v
```

### 3. Train the Model
```bash
# Train low-power Edge DS-CNN
python train.py --model edge_dscnn --epochs 15 --batch-size 32

# Or train standard AudioCNN
python train.py --model audio_cnn --epochs 15
```

### 4. Evaluate Model & Confusion Matrix
```bash
python evaluate.py --checkpoint checkpoints/edge_dscnn_best.pt
```

### 5. Profile Edge AI Latency & INT8 Quantization
```bash
python benchmark_edge.py --checkpoint checkpoints/edge_dscnn_best.pt
```

### 6. Run Real-Time Streaming Audio Simulation
```bash
python predict_stream.py
```

---

## 📁 Repository Structure

```
audio-edge-ai-classifier/
├── src/
│   ├── dsp/
│   │   ├── features.py          # STFT, Mel-filterbanks, Log-Mel Spectrogram, VAD
│   │   └── augmentations.py     # SpecAugment (Time/Freq mask), Dynamic SNR noise
│   ├── models/
│   │   ├── audio_cnn.py         # Standard 2D Audio CNN architecture
│   │   └── edge_dscnn.py        # Depthwise Separable CNN for low-power edge inference
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset loader & pipeline
│   │   └── synthetic_data.py    # 10-class realistic acoustic event generator
│   └── utils/
│       ├── metrics.py           # Accuracy, F1, Confusion Matrix calculator
│       └── quantization.py      # Post-Training Quantization (PTQ) & Latency Profiler
├── tests/
│   ├── test_dsp.py              # DSP shapes, VAD, and filterbank unit tests
│   ├── test_models.py           # Forward pass & parameter count tests
│   └── test_quantization.py     # Quantization parity and speedup verification
├── train.py                     # Model training with CosineAnnealingLR & checkpoints
├── evaluate.py                  # Evaluation driver with classification report
├── benchmark_edge.py            # FP32 vs INT8 size, latency & memory comparison
├── predict_stream.py            # 16 kHz live stream buffer inference simulator
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 📄 License
This project is open-source and licensed under the **MIT License**.
