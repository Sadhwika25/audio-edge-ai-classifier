"""
Acoustic Dataset Generator for Audio Intelligence.
Generates multi-class acoustic signals with realistic spectral signatures:
Harmonic speech formants, frequency-modulated sirens, impulse bursts, and engine hums.
"""

import math
import numpy as np
import torch


ACOUSTIC_CLASSES = [
    "speech_yes",
    "speech_no",
    "speech_stop",
    "siren_alert",
    "dog_bark",
    "car_horn",
    "engine_hum",
    "footsteps",
    "glass_break",
    "ambient_noise",
]


def generate_synthetic_acoustic_sample(
    class_idx: int,
    sample_rate: int = 16000,
    duration: float = 1.0,
) -> torch.Tensor:
    """
    Generates a 1-second audio sample with distinct spectral characteristics
    based on the target acoustic class.
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    
    # Base background noise (low level)
    signal = np.random.normal(0, 0.005, num_samples)
    
    if class_idx == 0:  # speech_yes (two distinct formant resonances)
        env = np.exp(-((t - 0.5) ** 2) / 0.05)
        formant1 = np.sin(2 * np.pi * 300 * t)
        formant2 = 0.5 * np.sin(2 * np.pi * 1200 * t)
        signal += env * (formant1 + formant2)
        
    elif class_idx == 1:  # speech_no (low-frequency resonance with rapid decay)
        env = np.exp(-((t - 0.4) ** 2) / 0.03)
        formant = np.sin(2 * np.pi * 450 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)
        signal += env * formant
        
    elif class_idx == 2:  # speech_stop (plosive burst followed by fricative noise)
        burst_idx = int(0.3 * sample_rate)
        signal[burst_idx : burst_idx + 80] += np.random.normal(0, 0.6, 80)
        env = np.exp(-((t - 0.45) ** 2) / 0.02)
        signal += env * (np.sin(2 * np.pi * 600 * t) + np.sin(2 * np.pi * 1800 * t))
        
    elif class_idx == 3:  # siren_alert (sinusoidal frequency modulation ~700Hz to 1500Hz)
        fm = 1000.0 + 400.0 * np.sin(2 * np.pi * 2.5 * t)
        phase = 2 * np.pi * np.cumsum(fm) / sample_rate
        signal += 0.4 * np.sin(phase)
        
    elif class_idx == 4:  # dog_bark (repetitive acoustic bursts with harmonic decay)
        for burst_time in [0.2, 0.6]:
            env = np.exp(-((t - burst_time) ** 2) / 0.008)
            bark = np.sin(2 * np.pi * 750 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t)
            signal += env * bark
            
    elif class_idx == 5:  # car_horn (dual-tone high amplitude dissonance 400Hz + 500Hz)
        env = (t >= 0.2) & (t <= 0.8)
        horn = 0.4 * (np.sin(2 * np.pi * 415 * t) + np.sin(2 * np.pi * 495 * t))
        signal += env * horn
        
    elif class_idx == 6:  # engine_hum (low frequency fundamental ~80Hz + rich harmonics)
        hum = (
            np.sin(2 * np.pi * 85 * t)
            + 0.6 * np.sin(2 * np.pi * 170 * t)
            + 0.3 * np.sin(2 * np.pi * 255 * t)
        )
        signal += 0.3 * hum
        
    elif class_idx == 7:  # footsteps (rhythmic impulsive low-pass filtered taps)
        for tap_time in [0.15, 0.45, 0.75]:
            idx = int(tap_time * sample_rate)
            tap_len = 240
            if idx + tap_len < num_samples:
                signal[idx : idx + tap_len] += 0.35 * np.hanning(tap_len) * np.sin(2 * np.pi * 120 * np.linspace(0, 0.015, tap_len))
                
    elif class_idx == 8:  # glass_break (sharp transient high-frequency burst + exponentially decaying noise)
        burst_idx = int(0.2 * sample_rate)
        decay = np.exp(-15 * np.maximum(0, t - 0.2))
        high_freq_noise = np.random.normal(0, 0.5, num_samples) * decay
        signal += high_freq_noise
        
    else:  # ambient_noise (pink-ish ambient floor)
        signal += np.random.normal(0, 0.05, num_samples)

    # Normalize amplitude to [-0.95, 0.95]
    peak = np.max(np.abs(signal)) + 1e-8
    signal = 0.9 * (signal / peak)

    return torch.from_numpy(signal.astype(np.float32))
