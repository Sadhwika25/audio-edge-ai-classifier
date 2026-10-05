"""
Unit tests for Digital Signal Processing (DSP) & Augmentations.
"""

import pytest
import torch

from src.dsp.features import AudioFeatureExtractor, compute_stft_spectrum, voice_activity_detection
from src.dsp.augmentations import SpecAugment, WaveformAugmenter


def test_feature_extractor_output_shape():
    batch_size = 4
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    
    waveform = torch.randn(batch_size, num_samples)
    extractor = AudioFeatureExtractor(
        sample_rate=sample_rate,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
    )
    
    mel_spec = extractor(waveform)
    
    assert mel_spec.dim() == 4
    assert mel_spec.size(0) == batch_size
    assert mel_spec.size(1) == 1
    assert mel_spec.size(2) == 64
    assert mel_spec.size(3) == 101  # (16000 / 160) + 1


def test_stft_spectrum_computation():
    waveform = torch.randn(2, 16000)
    spectrum = compute_stft_spectrum(waveform, n_fft=512, hop_length=160, win_length=400)
    assert spectrum.size(1) == (512 // 2) + 1  # 257 frequency bins
    assert spectrum.size(2) == 101


def test_voice_activity_detection():
    # Silent signal
    silence = torch.zeros(16000)
    active_silence = voice_activity_detection(silence, energy_threshold_db=-20.0)
    assert active_silence.sum().item() == 0

    # Strong sinusoidal signal
    t = torch.linspace(0, 1, 16000)
    loud_tone = torch.sin(2 * 3.14159 * 440 * t)
    active_tone = voice_activity_detection(loud_tone, energy_threshold_db=-20.0)
    assert active_tone.sum().item() > 0


def test_spec_augment():
    spec = torch.randn(2, 1, 64, 101)
    augmenter = SpecAugment(freq_mask_param=8, time_mask_param=12)
    aug_spec = augmenter(spec)
    assert aug_spec.shape == spec.shape
    # Check that zeros have been introduced by masking
    assert (aug_spec == 0).sum().item() > 0


def test_waveform_augmenter():
    waveform = torch.randn(16000)
    noisy = WaveformAugmenter.add_noise(waveform, snr_db_range=(10, 20))
    assert noisy.shape == waveform.shape
    assert not torch.equal(noisy, waveform)

    shifted = WaveformAugmenter.random_time_shift(waveform)
    assert shifted.shape == waveform.shape
