"""
Acoustic Data Augmentation Pipeline.
Implements SpecAugment (Time & Frequency Masking),
Dynamic Signal-to-Noise Ratio (SNR) Noise Injection, and Random Time Shifting.
"""

import math
import random
import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpecAugment(nn.Module):
    """
    Applies frequency and time masking directly to spectrograms
    as described in Google's SpecAugment paper (Park et al., 2019).
    """

    def __init__(
        self,
        freq_mask_param: int = 12,
        time_mask_param: int = 25,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: Tensor of shape (..., freq, time)
        Returns:
            masked_spec: Augmented spectrogram tensor
        """
        augmented = spec.clone()
        for _ in range(self.n_freq_masks):
            augmented = self.freq_masking(augmented)
        for _ in range(self.n_time_masks):
            augmented = self.time_masking(augmented)
        return augmented


class WaveformAugmenter:
    """
    Applies waveform-level augmentations:
    - Additive noise at targeted SNR levels (dB)
    - Random circular time-shift
    - Random amplitude gain
    """

    @staticmethod
    def add_noise(waveform: torch.Tensor, snr_db_range: tuple = (5.0, 20.0)) -> torch.Tensor:
        """
        Adds Gaussian white noise matching a random SNR from snr_db_range.
        """
        target_snr = random.uniform(*snr_db_range)
        signal_power = torch.mean(waveform ** 2, dim=-1, keepdim=True) + 1e-12
        noise = torch.randn_like(waveform)
        noise_power = torch.mean(noise ** 2, dim=-1, keepdim=True) + 1e-12

        # SNR = 10 * log10(P_signal / P_noise) => P_noise_target = P_signal / 10^(SNR/10)
        target_noise_power = signal_power / (10.0 ** (target_snr / 10.0))
        scaled_noise = noise * torch.sqrt(target_noise_power / noise_power)
        return waveform + scaled_noise

    @staticmethod
    def random_time_shift(waveform: torch.Tensor, max_shift_ratio: float = 0.2) -> torch.Tensor:
        """
        Randomly shifts waveform in time with zero-padding or wrap-around.
        """
        length = waveform.shape[-1]
        max_shift = int(length * max_shift_ratio)
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(waveform, shifts=shift, dims=-1)

    @staticmethod
    def random_gain(waveform: torch.Tensor, min_gain_db: float = -6.0, max_gain_db: float = 3.0) -> torch.Tensor:
        """
        Randomly scales volume within given dB range.
        """
        gain_db = random.uniform(min_gain_db, max_gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        return waveform * gain_linear
