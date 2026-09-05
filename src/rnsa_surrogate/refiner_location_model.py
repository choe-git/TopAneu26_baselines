"""Joint objectness and 53-way location candidate refiner."""

from __future__ import annotations

import torch
from torch import nn

from .refiner_data import REFINER_METADATA_FEATURES
from .refiner_model import RefinerBlock


class CandidateLocationRefiner(nn.Module):
    """Refine stage-1 candidates without changing their component masks."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 12,
        metadata_features: int = REFINER_METADATA_FEATURES,
        embedding_channels: int = 16,
        location_classes: int = 52,
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
        self.stage1_class_embedding = nn.Embedding(
            location_classes + 1, embedding_channels
        )
        fused_channels = channels[-1] * 2 + embedding_channels
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_channels),
            nn.Dropout(dropout),
            nn.Linear(fused_channels, channels[-1]),
            nn.SiLU(inplace=True),
        )
        self.objectness_head = nn.Linear(channels[-1], 1)
        self.location_head = nn.Linear(channels[-1], location_classes + 1)

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        stage1_class: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features = self.encoder(image)
        image_features = self.image_projection(
            torch.cat(
                [
                    features.mean(dim=(2, 3, 4)),
                    features.amax(dim=(2, 3, 4)),
                ],
                dim=1,
            )
        )
        metadata_features = self.metadata_projection(metadata)
        class_features = self.stage1_class_embedding(stage1_class)
        fused = self.fusion(
            torch.cat([image_features, metadata_features, class_features], dim=1)
        )
        return {
            "objectness_logits": self.objectness_head(fused).squeeze(1),
            "location_logits": self.location_head(fused),
        }
