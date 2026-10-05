"""
Audio Dataset & Data Loading Pipeline with In-Memory Caching for Instant Training.
"""

from typing import Optional, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader

from ..dsp.features import AudioFeatureExtractor
from ..dsp.augmentations import SpecAugment, WaveformAugmenter
from .synthetic_data import ACOUSTIC_CLASSES, generate_synthetic_acoustic_sample


class AcousticDataset(Dataset):
    """
    Dataset returning (log_mel_spectrogram, label).
    Pre-caches base waveforms in RAM for blazing-fast training.
    """

    def __init__(
        self,
        num_samples: int = 600,
        sample_rate: int = 16000,
        duration: float = 1.0,
        is_train: bool = True,
        augment: bool = True,
        feature_extractor: Optional[AudioFeatureExtractor] = None,
    ):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = duration
        self.is_train = is_train
        self.augment = augment
        self.classes = ACOUSTIC_CLASSES
        self.num_classes = len(self.classes)

        self.feature_extractor = feature_extractor or AudioFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=64,
        )
        self.spec_augment = SpecAugment() if (is_train and augment) else None

        # Pre-generate base acoustic waveforms into memory
        self.samples = []
        self.labels = []
        for i in range(num_samples):
            lbl = i % self.num_classes
            wf = generate_synthetic_acoustic_sample(
                class_idx=lbl,
                sample_rate=self.sample_rate,
                duration=self.duration,
            )
            self.samples.append(wf)
            self.labels.append(lbl)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        waveform = self.samples[idx].clone()
        label = self.labels[idx]

        # Apply waveform-level augmentations on training
        if self.is_train and self.augment:
            waveform = WaveformAugmenter.random_time_shift(waveform)
            waveform = WaveformAugmenter.random_gain(waveform)
            if torch.rand(1).item() > 0.4:
                waveform = WaveformAugmenter.add_noise(waveform, snr_db_range=(10.0, 25.0))

        # Extract normalized Log-Mel Spectrogram: (1, n_mels, time_frames)
        with torch.no_grad():
            mel_spec = self.feature_extractor(waveform.unsqueeze(0)).squeeze(0)

        # Apply SpecAugment time & frequency masking
        if self.spec_augment is not None:
            mel_spec = self.spec_augment(mel_spec)

        return mel_spec, label


def get_dataloaders(
    train_samples: int = 600,
    val_samples: int = 150,
    batch_size: int = 32,
    sample_rate: int = 16000,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns train and validation PyTorch DataLoaders.
    """
    train_dataset = AcousticDataset(
        num_samples=train_samples,
        sample_rate=sample_rate,
        is_train=True,
        augment=True,
    )
    val_dataset = AcousticDataset(
        num_samples=val_samples,
        sample_rate=sample_rate,
        is_train=False,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
