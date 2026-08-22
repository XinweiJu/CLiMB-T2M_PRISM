"""Inference-only ResNet-18 encoder compatible with PRISM/Monodepth2 keys."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)

    def _make_layer(
        self, out_channels: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        layers = [
            BasicBlock(self.inplanes, out_channels, stride, downsample)
        ]
        self.inplanes = out_channels
        layers.extend(BasicBlock(out_channels, out_channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)


class ResnetEncoder(nn.Module):
    """Monodepth2-compatible ResNet-18 feature encoder.

    ``num_input_images`` follows the original three-channel frame convention.
    A caller may replace ``encoder.conv1`` before loading a checkpoint when each
    frame contains an auxiliary channel, as PRISM's DLPE pose model does.
    """

    def __init__(
        self, num_layers: int, pretrained: bool, num_input_images: int = 1
    ) -> None:
        super().__init__()
        if num_layers != 18:
            raise ValueError("The standalone inference package supports ResNet-18 only")
        if pretrained:
            raise ValueError("Inference loads checkpoint weights; pretrained=True is invalid")
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.encoder = ResNet18(input_channels=3 * num_input_images)

    def forward(self, input_image: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features
