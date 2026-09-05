"""Lightweight vessel-topology refiner for aneurysm component locations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, find_objects, label


def _feature(
    component: np.ndarray,
    region: tuple[slice, slice, slice],
    vessel: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    lower = np.asarray([value.start for value in region], dtype=int)
    upper = np.asarray([value.stop for value in region], dtype=int)
    margin = 12
    crop_lower = np.maximum(lower - margin, 0)
    crop_upper = np.minimum(upper + margin, np.asarray(shape))
    crop = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(crop_lower, crop_upper, strict=True)
    )
    expanded = np.zeros(tuple(crop_upper - crop_lower), dtype=bool)
    insert = tuple(
        slice(int(start - crop_start), int(stop - crop_start))
        for start, stop, crop_start in zip(lower, upper, crop_lower, strict=True)
    )
    expanded[insert] = component
    vessel_crop = np.asarray(vessel[crop])
    distance = distance_transform_edt(~expanded)
    anatomy = []
    for radius in (2.0, 6.0, 12.0):
        nearby = distance <= radius
        labels = vessel_crop[nearby]
        counts = np.bincount(labels, minlength=37).astype(np.float64)
        foreground = counts[1:37]
        anatomy.extend((foreground / max(foreground.sum(), 1.0)).tolist())
        anatomy.append(float(foreground.sum() / max(labels.size, 1)))
    local_coordinates = np.argwhere(component)
    center = lower + local_coordinates.mean(axis=0)
    normalized_center = 2.0 * center / np.maximum(np.asarray(shape) - 1, 1) - 1.0
    extent = (upper - lower).astype(np.float64) / np.maximum(np.asarray(shape), 1)
    size = np.asarray([np.log1p(float(component.sum())) / 16.0])
    return np.concatenate(
        [np.asarray(anatomy), normalized_center, extent, size]
    ).astype(np.float32)


class VesselKNNRefiner:
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        neighbors: int = 3,
        geometry_weight: float = 0.5,
    ) -> None:
        self.mean = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale < 1e-5] = 1.0
        self.features = (features - self.mean) / self.scale
        self.features[:, -7:] *= geometry_weight
        self.labels = labels.astype(np.int64)
        self.neighbors = int(neighbors)
        self.geometry_weight = float(geometry_weight)
        self.class_counts = np.bincount(self.labels, minlength=53).astype(np.float64)

    @classmethod
    def fit(
        cls,
        index: dict[str, Any],
        cache_root: Path,
        train_ids: set[str],
        neighbors: int = 3,
        geometry_weight: float = 0.5,
    ) -> VesselKNNRefiner:
        features = []
        labels = []
        for case in index["cases"]:
            if str(case["case_id"]) not in train_ids:
                continue
            case_dir = cache_root / case["cache_dir"]
            instances = np.load(case_dir / "instances.npy", mmap_mode="r")
            vessel = np.load(case_dir / "vessel.npy", mmap_mode="r")
            shape = tuple(int(value) for value in case["shape_zyx"])
            for component in case.get("components", []):
                lower = component["bbox_zyx"][0]
                upper = component["bbox_zyx"][1]
                region = tuple(
                    slice(int(start), int(stop))
                    for start, stop in zip(lower, upper, strict=True)
                )
                mask = np.asarray(instances[region]) == int(component["instance_id"])
                features.append(_feature(mask, region, vessel, shape))
                labels.append(int(component["class_id"]))
        if not features:
            raise ValueError("No training components for vessel refiner")
        return cls(
            np.stack(features),
            np.asarray(labels),
            neighbors,
            geometry_weight,
        )

    def predict(self, feature: np.ndarray) -> int:
        value = (feature - self.mean) / self.scale
        value[-7:] *= self.geometry_weight
        distances = np.square(self.features - value).mean(axis=1)
        count = min(self.neighbors, len(distances))
        selected = np.argpartition(distances, count - 1)[:count]
        votes = np.zeros(53, dtype=np.float64)
        for index in selected:
            class_id = int(self.labels[index])
            votes[class_id] += 1.0 / (
                (distances[index] + 1e-4)
                * np.sqrt(max(self.class_counts[class_id], 1.0))
            )
        return int(np.argmax(votes[1:]) + 1)

    def refine(self, segmentation: np.ndarray, vessel: np.ndarray) -> np.ndarray:
        components, _ = label(
            segmentation > 0, structure=np.ones((3, 3, 3), dtype=np.uint8)
        )
        refined = np.zeros_like(segmentation, dtype=np.uint8)
        shape = tuple(int(value) for value in segmentation.shape)
        for component_id, region in enumerate(find_objects(components), start=1):
            if region is None:
                continue
            component = components[region] == component_id
            class_id = self.predict(_feature(component, region, vessel, shape))
            local = refined[region]
            local[component] = class_id
        return refined
