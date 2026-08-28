"""Compact vessel-aware 3D U-Net for the two TopAneu26 tasks."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in range(min(channels, 8), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(inputs)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, channels: Sequence[int]) -> None:
        super().__init__()
        self.stem = nn.Conv3d(in_channels, channels[0], 3, padding=1, bias=False)
        self.stages = nn.ModuleList(
            [nn.Sequential(ResidualBlock(value), ResidualBlock(value)) for value in channels]
        )
        self.down = nn.ModuleList(
            [
                nn.Conv3d(source, target, 2, stride=2, bias=False)
                for source, target in pairwise(channels)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        features = []
        value = self.stem(inputs)
        for level, stage in enumerate(self.stages):
            value = stage(value)
            features.append(value)
            if level < len(self.down):
                value = self.down[level](value)
        return features


class Decoder(nn.Module):
    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        levels = list(range(len(channels) - 2, -1, -1))
        self.levels = levels
        self.up = nn.ModuleList(
            [
                nn.ConvTranspose3d(channels[level + 1], channels[level], 2, stride=2)
                for level in levels
            ]
        )
        self.fuse = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(channels[level] * 2, channels[level], 1, bias=False),
                    ResidualBlock(channels[level]),
                )
                for level in levels
            ]
        )

    def forward(self, encoded: Sequence[torch.Tensor]) -> dict[int, torch.Tensor]:
        value = encoded[-1]
        decoded: dict[int, torch.Tensor] = {}
        for index, level in enumerate(self.levels):
            value = self.up[index](value)
            skip = encoded[level]
            if value.shape[2:] != skip.shape[2:]:
                value = F.interpolate(
                    value, size=skip.shape[2:], mode="trilinear", align_corners=False
                )
            value = self.fuse[index](torch.cat([value, skip], dim=1))
            decoded[level] = value
        return decoded


def weighted_pool(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = F.interpolate(weights, size=features.shape[2:], mode="trilinear", align_corners=False)
    weights = weights.clamp_min(1e-4)
    return (features * weights).sum((2, 3, 4)) / weights.sum((2, 3, 4))


class RNSASurrogate(nn.Module):
    """One-pass surrogate of the winner's vessel-first fine model.

    Input channels are normalized angiography, modality, and global z/y/x coordinates.
    The vessel branch is auxiliary during training; predictions require only the image.
    """

    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 16,
        levels: int = 4,
        vessel_classes: int = 37,
        location_classes: int = 52,
    ) -> None:
        super().__init__()
        if levels < 3:
            raise ValueError("levels must be at least 3")
        channels = [base_channels * 2**index for index in range(levels)]
        self.encoder = Encoder(in_channels, channels)
        self.decoder = Decoder(channels)
        self.aneurysm_head = nn.Conv3d(channels[0], 1, 1)
        self.location_head = nn.Conv3d(channels[0], location_classes + 1, 1)
        self.vessel_head = nn.Conv3d(channels[1], vessel_classes, 1)
        pooled_channels = channels[-1] * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_channels),
            nn.Linear(pooled_channels, channels[-1]),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )
        self.location_presence = nn.Linear(channels[-1], location_classes)
        self.aneurysm_presence = nn.Linear(channels[-1], 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        full = decoded[0]
        half = decoded[1]
        aneurysm_logits = self.aneurysm_head(full)
        location_logits = self.location_head(full)
        vessel_logits = self.vessel_head(half)

        vessel_probability = 1.0 - torch.softmax(vessel_logits.float(), dim=1)[:, :1]
        bottleneck = encoded[-1]
        anatomy = weighted_pool(bottleneck, vessel_probability)
        average = F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
        maximum = F.adaptive_max_pool3d(bottleneck, 1).flatten(1)
        pooled = self.classifier(torch.cat([anatomy, average, maximum], dim=1))
        return {
            "aneurysm_logits": aneurysm_logits,
            "location_logits": location_logits,
            "vessel_logits": vessel_logits,
            "location_presence_logits": self.location_presence(pooled),
            "aneurysm_presence_logits": self.aneurysm_presence(pooled),
        }
