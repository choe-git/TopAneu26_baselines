"""Small 3D candidate objectness classifier."""

from __future__ import annotations

import torch
from torch import nn

from .refiner_data import REFINER_METADATA_FEATURES


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class RefinerBlock(nn.Module):
    def __init__(self, source: int, target: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(source, target, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(target), target),
            nn.SiLU(inplace=True),
            nn.Conv3d(target, target, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(target), target),
        )
        self.skip = (
            nn.Conv3d(source, target, 1, stride=stride, bias=False)
            if source != target or stride != 1
            else nn.Identity()
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layers(inputs) + self.skip(inputs))


class CandidateObjectnessRefiner(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 12,
        metadata_features: int = REFINER_METADATA_FEATURES,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4]
        self.encoder = nn.Sequential(
            RefinerBlock(in_channels, channels[0]),
            RefinerBlock(channels[0], channels[1], stride=2),
            RefinerBlock(channels[1], channels[2], stride=2),
            RefinerBlock(channels[2], channels[2], stride=2),
        )
        self.image_projection = nn.Sequential(
            nn.Linear(channels[-1] * 2, channels[-1]),
            nn.SiLU(inplace=True),
        )
        self.metadata_projection = nn.Sequential(
            nn.LayerNorm(metadata_features),
            nn.Linear(metadata_features, channels[-1]),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(channels[-1] * 2),
            nn.Dropout(dropout),
            nn.Linear(channels[-1] * 2, 1),
        )

    def forward(
        self, image: torch.Tensor, metadata: torch.Tensor
    ) -> torch.Tensor:
        features = self.encoder(image)
        average = features.mean(dim=(2, 3, 4))
        maximum = features.amax(dim=(2, 3, 4))
        image_features = self.image_projection(torch.cat([average, maximum], dim=1))
        metadata_features = self.metadata_projection(metadata)
        return self.head(
            torch.cat([image_features, metadata_features], dim=1)
        ).squeeze(1)
