"""Candidate-centred 3D ROI segmentation, objectness and location refiner."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .data import extract_patch
from .refiner_candidates import candidate_coordinates
from .refiner_data import CandidateROIDataset
from .refiner_model import RefinerBlock


@lru_cache(maxsize=8)
def _cached_instances(cache_root: str, cache_directory: str) -> np.ndarray:
    return np.load(
        Path(cache_root) / cache_directory / "instances.npy", mmap_mode="r"
    )


def roi_start(record: dict[str, Any], roi_size: Sequence[int]) -> tuple[int, int, int]:
    center = tuple(int(round(float(value))) for value in record["center_zyx"])
    return tuple(
        center_value - int(size) // 2
        for center_value, size in zip(center, roi_size, strict=True)
    )


class CandidateROIRefinementDataset(CandidateROIDataset):
    """Adds the GT instance matched by a stage-1 component as a dense target."""

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(item)
        record = self.records[int(item)]
        case = self.cases[str(record["case_id"])]
        target = np.zeros(self.roi_size, dtype=np.float32)
        valid = np.zeros(self.roi_size, dtype=np.float32)
        start = roi_start(record, self.roi_size)
        shape = tuple(int(value) for value in case["shape_zyx"])
        source_slices = tuple(
            slice(max(origin, 0), min(origin + size, available))
            for origin, size, available in zip(
                start, self.roi_size, shape, strict=True
            )
        )
        destination_slices = tuple(
            slice(max(-origin, 0), max(-origin, 0) + max(
                min(origin + size, available) - max(origin, 0), 0
            ))
            for origin, size, available in zip(
                start, self.roi_size, shape, strict=True
            )
        )
        valid[destination_slices] = 1.0
        if int(record["target"]) > 0:
            coordinates = candidate_coordinates(
                record["_artifact_path"], int(record["artifact_index"])
            )
            instances = _cached_instances(self.cache_root, str(case["cache_dir"]))
            values = np.asarray(instances[tuple(coordinates.T)], dtype=np.int64)
            foreground = values[values > 0]
            if foreground.size:
                matched_instance = int(np.argmax(np.bincount(foreground)))
                instance_roi, _ = extract_patch(
                    instances,
                    tuple(int(round(float(value))) for value in record["center_zyx"]),
                    self.roi_size,
                    pad_value=0,
                )
                target = (instance_roi == matched_instance).astype(np.float32)
        if bool(sample["flipped"]):
            target = np.flip(target, axis=-1).copy()
            valid = np.flip(valid, axis=-1).copy()
        sample["target_mask"] = torch.from_numpy(target[None])
        sample["valid_mask"] = torch.from_numpy(valid[None])
        return sample


class CandidateROIRefiner(nn.Module):
    """Compact residual 3D U-Net with classification auxiliaries."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 8,
        metadata_features: int = 11,
        embedding_channels: int = 16,
        location_classes: int = 52,
        dropout: float = 0.15,
        location_prior_logit: float = 0.0,
    ) -> None:
        super().__init__()
        c0, c1, c2, c3 = (base_channels * value for value in (1, 2, 4, 8))
        self.enc0 = RefinerBlock(in_channels, c0)
        self.enc1 = RefinerBlock(c0, c1, stride=2)
        self.enc2 = RefinerBlock(c1, c2, stride=2)
        self.bottleneck = RefinerBlock(c2, c3, stride=2)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = RefinerBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = RefinerBlock(c1 + c1, c1)
        self.up0 = nn.ConvTranspose3d(c1, c0, 2, stride=2)
        self.dec0 = RefinerBlock(c0 + c0, c0)
        self.mask_head = nn.Conv3d(c0, 1, 1)
        self.image_projection = nn.Sequential(
            nn.Linear(c3 * 2, c3), nn.SiLU(inplace=True)
        )
        self.metadata_projection = nn.Sequential(
            nn.LayerNorm(metadata_features),
            nn.Linear(metadata_features, c3),
            nn.SiLU(inplace=True),
        )
        self.stage1_class_embedding = nn.Embedding(
            location_classes + 1, embedding_channels
        )
        fused_channels = c3 * 2 + embedding_channels
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_channels),
            nn.Dropout(dropout),
            nn.Linear(fused_channels, c3),
            nn.SiLU(inplace=True),
        )
        self.objectness_head = nn.Linear(c3, 1)
        self.location_head = nn.Linear(c3, location_classes + 1)
        self.location_prior_logit = float(location_prior_logit)

    def forward(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        stage1_class: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x0 = self.enc0(image)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.bottleneck(x2)
        d2 = self.dec2(torch.cat([self.up2(x3), x2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), x1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), x0], dim=1))
        pooled = self.image_projection(torch.cat(
            [x3.mean(dim=(2, 3, 4)), x3.amax(dim=(2, 3, 4))], dim=1
        ))
        fused = self.fusion(torch.cat(
            [
                pooled,
                self.metadata_projection(metadata),
                self.stage1_class_embedding(stage1_class),
            ],
            dim=1,
        ))
        location_logits = self.location_head(fused)
        if self.location_prior_logit:
            prior = torch.zeros_like(location_logits)
            prior.scatter_(
                1, stage1_class[:, None],
                torch.full(
                    (stage1_class.shape[0], 1),
                    self.location_prior_logit,
                    dtype=location_logits.dtype,
                    device=location_logits.device,
                ),
            )
            location_logits = location_logits + prior
        return {
            "mask_logits": self.mask_head(d0),
            "objectness_logits": self.objectness_head(fused).squeeze(1),
            "location_logits": location_logits,
        }
