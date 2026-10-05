"""
Depthwise Separable Convolutional Neural Network (DS-CNN).
Optimized for ultra-low latency & memory-efficient Edge AI audio inference
on microcontrollers, hearables, and edge processors.
"""

import torch
import torch.nn as nn


class DepthwiseSeparableConv2d(nn.Module):
    """
    Splits standard convolution into:
    1. Depthwise convolution (spatial filtering per channel)
    2. Pointwise convolution (1x1 conv across channels)
    Reduces parameter count and FLOPs by 8-9x compared to standard 3x3 conv.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EdgeDSCNN(nn.Module):
    """
    DS-CNN architecture for low-power edge keyword spotting / audio event classification.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        
        # Standard initial 2D convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        # Stack of Depthwise Separable Conv layers
        self.ds_blocks = nn.Sequential(
            DepthwiseSeparableConv2d(base_channels, base_channels, stride=1),
            DepthwiseSeparableConv2d(base_channels, base_channels, stride=2),
            DepthwiseSeparableConv2d(base_channels, base_channels * 2, stride=1),
            DepthwiseSeparableConv2d(base_channels * 2, base_channels * 2, stride=2),
            DepthwiseSeparableConv2d(base_channels * 2, base_channels * 2, stride=1),
        )
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.ds_blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
