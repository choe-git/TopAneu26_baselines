"""Sparse stage-1 component artifacts for the objectness refiner."""

from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import find_objects, label


CANDIDATE_VERSION = 1
COMPONENT_STRUCTURE = np.ones((3, 3, 3), dtype=np.uint8)


def extract_candidate_records(
    case_id: str,
    segmentation: np.ndarray,
    binary_probability: np.ndarray,
    truth: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Describe retained stage-1 components and store only their sparse voxels."""
    if not (
        segmentation.shape == binary_probability.shape == truth.shape
    ):
        raise ValueError(
            f"Candidate geometry mismatch for {case_id}: "
            f"{segmentation.shape}, {binary_probability.shape}, {truth.shape}"
        )
    components, _ = label(segmentation > 0, structure=COMPONENT_STRUCTURE)
    records: list[dict[str, Any]] = []
    coordinate_blocks: list[np.ndarray] = []
    offsets = [0]
    for component_id, region in enumerate(find_objects(components), start=1):
        if region is None:
            continue
        local = components[region] == component_id
        local_coordinates = np.argwhere(local)
        if local_coordinates.size == 0:
            continue
        lower = np.asarray([value.start for value in region], dtype=np.int32)
        coordinates = local_coordinates.astype(np.int32) + lower
        values = binary_probability[tuple(coordinates.T)]
        predicted_values = segmentation[tuple(coordinates.T)]
        truth_values = truth[tuple(coordinates.T)]
        foreground_truth = truth_values[truth_values > 0]
        stage1_counts = np.bincount(predicted_values, minlength=53)
        stage1_counts[0] = 0
        stage1_class = int(np.argmax(stage1_counts))
        target_class = 0
        if foreground_truth.size:
            target_counts = np.bincount(foreground_truth, minlength=53)
            target_class = int(np.argmax(target_counts[1:53]) + 1)
        upper = np.asarray([value.stop for value in region], dtype=np.int32)
        record_index = len(records)
        records.append(
            {
                "candidate_id": f"{case_id}:{record_index:04d}",
                "case_id": case_id,
                "artifact_index": record_index,
                "bbox_zyx": [lower.tolist(), upper.tolist()],
                "center_zyx": coordinates.mean(axis=0).tolist(),
                "extent_zyx": (upper - lower).tolist(),
                "voxels": int(coordinates.shape[0]),
                "stage1_class": stage1_class,
                "stage1_score": float(
                    0.7 * float(values.max()) + 0.3 * float(values.mean())
                ),
                "stage1_score_max": float(values.max()),
                "stage1_score_mean": float(values.mean()),
                "target": int(target_class > 0),
                "target_class": target_class,
                "overlap_voxels": int(foreground_truth.size),
            }
        )
        coordinate_blocks.append(coordinates)
        offsets.append(offsets[-1] + coordinates.shape[0])
    coordinates = (
        np.concatenate(coordinate_blocks, axis=0)
        if coordinate_blocks
        else np.empty((0, 3), dtype=np.int32)
    )
    return records, coordinates, np.asarray(offsets, dtype=np.int64)


def atomic_save_candidate_artifact(
    path: str | Path,
    coordinates: np.ndarray,
    offsets: np.ndarray,
    vessel_rois: np.ndarray | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            payload = {
                "coordinates": np.asarray(coordinates, dtype=np.int32),
                "offsets": np.asarray(offsets, dtype=np.int64),
            }
            if vessel_rois is not None:
                payload["vessel_rois"] = np.asarray(vessel_rois, dtype=np.uint8)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def candidate_coordinates(path: str | Path, index: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        offsets = artifact["offsets"]
        if not 0 <= index < len(offsets) - 1:
            raise IndexError(f"Candidate index {index} is absent from {path}")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return np.asarray(artifact["coordinates"][start:stop], dtype=np.int32)


@lru_cache(maxsize=64)
def _candidate_vessel_rois(path: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as artifact:
        if "vessel_rois" not in artifact:
            raise ValueError(f"Candidate artifact has no vessel context: {path}")
        return np.asarray(artifact["vessel_rois"], dtype=np.uint8)


def candidate_vessel_roi(path: str | Path, index: int) -> np.ndarray:
    return _candidate_vessel_rois(str(Path(path).resolve()))[index]
