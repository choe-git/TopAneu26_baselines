"""Lazy fixed-size ROIs for stage-1 hard-negative objectness learning."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_cache_index
from .data import extract_patch
from .refiner_candidates import CANDIDATE_VERSION, candidate_coordinates


REFINER_METADATA_FEATURES = 11


def load_candidate_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("candidate_version") != CANDIDATE_VERSION:
        raise ValueError(
            f"Unsupported candidate manifest version: "
            f"{payload.get('candidate_version')}"
        )
    payload["manifest_path"] = str(path)
    return payload


def manifest_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    root = Path(manifest["manifest_path"]).parent
    records = []
    for source in manifest["candidates"]:
        record = dict(source)
        record["_artifact_path"] = str((root / record["artifact"]).resolve())
        record["_generator_fold"] = int(manifest["fold"])
        records.append(record)
    return records


@lru_cache(maxsize=8)
def _cached_image(cache_root: str, cache_directory: str) -> np.ndarray:
    return np.load(
        Path(cache_root) / cache_directory / "image.npy", mmap_mode="r"
    )


def candidate_input(
    image: np.ndarray,
    coordinates: np.ndarray,
    record: dict[str, Any],
    shape_zyx: Sequence[int],
    modality: str,
    roi_size: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    roi_size = tuple(int(value) for value in roi_size)
    center = tuple(int(round(float(value))) for value in record["center_zyx"])
    image_roi, start = extract_patch(image, center, roi_size, pad_value=-1.0)
    candidate_roi = np.zeros(roi_size, dtype=np.float32)
    local = coordinates.astype(np.int64) - np.asarray(start, dtype=np.int64)
    inside = np.all(
        (local >= 0) & (local < np.asarray(roi_size, dtype=np.int64)), axis=1
    )
    if np.any(inside):
        candidate_roi[tuple(local[inside].T)] = 1.0
    inputs = np.stack(
        [image_roi.astype(np.float32, copy=False), candidate_roi], axis=0
    )
    shape = np.asarray(shape_zyx, dtype=np.float32)
    extent = np.asarray(record["extent_zyx"], dtype=np.float32) / np.maximum(shape, 1)
    normalized_center = (
        2.0 * np.asarray(record["center_zyx"], dtype=np.float32)
        / np.maximum(shape - 1, 1)
        - 1.0
    )
    metadata = np.asarray(
        [
            float(record["stage1_score"]),
            float(record["stage1_score_max"]),
            float(record["stage1_score_mean"]),
            np.log1p(float(record["voxels"])) / 12.0,
            *extent.tolist(),
            *normalized_center.tolist(),
            1.0 if modality == "mr" else -1.0,
        ],
        dtype=np.float32,
    )
    if metadata.shape != (REFINER_METADATA_FEATURES,):
        raise RuntimeError(f"Unexpected metadata shape: {metadata.shape}")
    return inputs, metadata


class CandidateROIDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        cache_dir: str | Path,
        records: list[dict[str, Any]],
        roi_size: Sequence[int] = (48, 64, 64),
        augment: bool = False,
        seed: int = 2026,
    ) -> None:
        index = load_cache_index(cache_dir)
        self.cache_root = str(Path(index["index_path"]).parent)
        self.cases = {str(case["case_id"]): case for case in index["cases"]}
        self.location_swap = np.asarray(index["location_lr_swap"], dtype=np.int64)
        self.records = records
        self.roi_size = tuple(int(value) for value in roi_size)
        self.augment = bool(augment)
        self.seed = int(seed)
        if not self.records:
            raise ValueError("Candidate ROI dataset is empty")
        if any(value <= 0 or value % 8 for value in self.roi_size):
            raise ValueError(f"roi_size must be positive and divisible by 8: {roi_size}")
        for record in self.records:
            case_id = str(record["case_id"])
            if case_id not in self.cases:
                raise ValueError(f"Candidate references unknown cache case: {case_id}")
            if int(record["_generator_fold"]) < 0:
                raise ValueError(f"Invalid generator fold for {record['candidate_id']}")

    @property
    def positive_indices(self) -> np.ndarray:
        return np.asarray(
            [index for index, record in enumerate(self.records) if record["target"]],
            dtype=np.int64,
        )

    @property
    def negative_indices(self) -> np.ndarray:
        return np.asarray(
            [index for index, record in enumerate(self.records) if not record["target"]],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        record = self.records[int(item)]
        case = self.cases[str(record["case_id"])]
        image = _cached_image(self.cache_root, str(case["cache_dir"]))
        coordinates = candidate_coordinates(
            record["_artifact_path"], int(record["artifact_index"])
        )
        inputs, metadata = candidate_input(
            image,
            coordinates,
            record,
            case["shape_zyx"],
            str(case["modality"]),
            self.roi_size,
        )
        if self.augment:
            # DataLoader seeds NumPy independently in every worker.  Using that
            # stream keeps augmentation reproducible for a run while avoiding
            # the exact same transform whenever a sampled candidate reappears.
            inputs[0] = inputs[0] * float(np.random.uniform(0.9, 1.1)) + float(
                np.random.uniform(-0.08, 0.08)
            )
            if np.random.random() < 0.15:
                inputs[0] += np.random.normal(0, 0.02, self.roi_size).astype(
                    np.float32
                )
            if np.random.random() < 0.5:
                inputs = np.flip(inputs, axis=-1).copy()
                metadata[9] *= -1.0
                stage1_class = int(
                    self.location_swap[int(record["stage1_class"])]
                )
                target_class = int(
                    self.location_swap[int(record["target_class"])]
                )
            else:
                stage1_class = int(record["stage1_class"])
                target_class = int(record["target_class"])
        else:
            stage1_class = int(record["stage1_class"])
            target_class = int(record["target_class"])
        return {
            "image": torch.from_numpy(inputs.astype(np.float32, copy=False)),
            "metadata": torch.from_numpy(metadata),
            "target": torch.tensor(float(record["target"]), dtype=torch.float32),
            "target_class": torch.tensor(target_class, dtype=torch.long),
            "stage1_class": torch.tensor(stage1_class, dtype=torch.long),
            "index": torch.tensor(int(item), dtype=torch.long),
        }
