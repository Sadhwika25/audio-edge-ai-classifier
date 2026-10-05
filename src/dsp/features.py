"""
Digital Signal Processing (DSP) & Acoustic Feature Extraction Module.
Implements STFT, Mel-filterbanks, Log-Mel Spectrogram transformation,
and basic energy-based Voice Activity Detection (VAD).
"""

import math
import torch
import torch.nn as nn
import torchaudio.transforms as T


class AudioFeatureExtractor(nn.Module):
    """
    Extracts time-frequency representations (Log-Mel Spectrograms)
    from raw 1D acoustic audio signals.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        power: float = 2.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Torchaudio MelSpectrogram module
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=power,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )

        # Amplitude to Decibels (Log-Mel conversion)
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape (batch, time) or (batch, 1, time)
        Returns:
            log_mel: Tensor of shape (batch, 1, n_mels, frames)
        """
        if waveform.dim() == 2:
            # (batch, time) -> add channel dimension
            pass
        elif waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)

        # Compute Mel Spectrogram
        mel_spec = self.mel_spectrogram(waveform)  # (batch, n_mels, frames)

        # Convert to logarithmic decibel scale
        log_mel = self.amplitude_to_db(mel_spec)

        # Normalize per-sample (mean 0, std 1)
        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + 1e-6
        norm_log_mel = (log_mel - mean) / std

        # Return with single channel dimension: (batch, 1, n_mels, frames)
        return norm_log_mel.unsqueeze(1)


def compute_stft_spectrum(
    waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> torch.Tensor:
    """
    Computes standard complex STFT magnitude spectrum.
    """
    window = torch.hann_window(win_length, device=waveform.device)
    stft_res = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    )
    magnitude = torch.abs(stft_res)
    return magnitude


def voice_activity_detection(
    waveform: torch.Tensor,
    frame_length: int = 320,  # 20ms at 16kHz
    hop_length: int = 160,    # 10ms at 16kHz
    energy_threshold_db: float = -35.0,
) -> torch.Tensor:
    """
    Simple Short-Time Energy based Voice Activity Detector (VAD).
    Returns boolean tensor indicating active speech frames.
    """
    # Unfold into overlapping frames
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    frames = waveform.unfold(dimension=-1, size=frame_length, step=hop_length)
    frame_energy = torch.sum(frames ** 2, dim=-1) + 1e-10
    energy_db = 10 * torch.log10(frame_energy)
    
    active_frames = energy_db > energy_threshold_db
    return active_frames
