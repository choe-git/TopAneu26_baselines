"""Direct-from-release patch dataset with anatomy-safe left/right augmentation."""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .cache import load_cache_index


def read_split(path: str | Path, selected: str) -> list[str]:
    if selected not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {selected}")
    cases = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row[selected] == "1":
                cases.append(row["case_id"])
    if not cases:
        raise ValueError(f"No {selected} cases in {path}")
    return sorted(cases)


def left_right_lut(mapping_path: str | Path) -> np.ndarray:
    labels = json.loads(Path(mapping_path).read_text(encoding="utf-8"))["labels"]
    by_name = {str(name): int(value) for name, value in labels.items()}
    lut = np.arange(max(by_name.values()) + 1, dtype=np.int64)
    for name, value in by_name.items():
        if name.startswith("R-"):
            counterpart = "L-" + name[2:]
        elif name.startswith("L-"):
            counterpart = "R-" + name[2:]
        else:
            counterpart = name
        if counterpart not in by_name:
            raise ValueError(f"Missing left/right counterpart for {name!r}")
        lut[value] = by_name[counterpart]
    if not np.array_equal(lut[lut], np.arange(len(lut))):
        raise ValueError(f"Left/right mapping is not involutive: {mapping_path}")
    return lut


def extract_patch(
    array: np.ndarray,
    center_zyx: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    pad_value: float = 0,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    start = tuple(
        center - size // 2 for center, size in zip(center_zyx, patch_size, strict=True)
    )
    output = np.full(patch_size, pad_value, dtype=array.dtype)
    source_slices = []
    destination_slices = []
    for origin, length, available in zip(start, patch_size, array.shape, strict=True):
        source_start = max(origin, 0)
        source_end = min(origin + length, available)
        destination_start = max(-origin, 0)
        destination_end = destination_start + max(source_end - source_start, 0)
        source_slices.append(slice(source_start, source_end))
        destination_slices.append(slice(destination_start, destination_end))
    output[tuple(destination_slices)] = array[tuple(source_slices)]
    return output, start


def coordinate_patch(
    start_zyx: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    axes = []
    for start, length, full_length in zip(
        start_zyx, patch_size, shape_zyx, strict=True
    ):
        denominator = max(full_length - 1, 1)
        axes.append(
            (2.0 * (np.arange(length) + start) / denominator - 1.0).astype(np.float32)
        )
    z, y, x = np.meshgrid(*axes, indexing="ij")
    return np.stack([z, y, x])


def _normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image has no finite voxels")
    lower, upper = np.percentile(finite, (0.5, 99.5))
    image = np.nan_to_num(
        image, nan=float(lower), posinf=float(upper), neginf=float(lower)
    )
    image = np.clip(image, lower, upper)
    mean = float(image.mean())
    standard_deviation = max(float(image.std()), 1e-6)
    return ((image - mean) / standard_deviation).astype(np.float32)


@lru_cache(maxsize=2)
def _load_case(
    data_root: str, case_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_root)
    paths = (
        root / "images" / f"{case_id}_0000.nii.gz",
        root / "location_masks" / f"{case_id}.nii.gz",
        root / "vessel_masks" / f"{case_id}.nii.gz",
    )
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(f"Missing case files: {missing}")
    image = _normalize(
        np.asarray(nib.load(paths[0]).dataobj, dtype=np.float32).transpose(2, 1, 0)
    )
    location = np.asarray(nib.load(paths[1]).dataobj, dtype=np.uint8).transpose(2, 1, 0)
    vessel = np.asarray(nib.load(paths[2]).dataobj, dtype=np.uint8).transpose(2, 1, 0)
    if image.shape != location.shape or image.shape != vessel.shape:
        raise ValueError(
            f"Geometry mismatch for {case_id}: {image.shape}, {location.shape}, {vessel.shape}"
        )
    return image, location, vessel


class TopAneuPatchDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        data_root: str | Path,
        split_csv: str | Path,
        split: str,
        patch_size: tuple[int, int, int] = (64, 96, 96),
        samples: int = 1000,
        positive_fraction: float = 0.7,
        augment: bool = False,
        seed: int = 2026,
    ) -> None:
        self.data_root = str(Path(data_root).resolve())
        self.cases = read_split(split_csv, split)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.samples = int(samples)
        self.positive_fraction = float(positive_fraction)
        self.augment = bool(augment)
        self.seed = int(seed)
        root = Path(self.data_root)
        self.location_swap = left_right_lut(root / "location_mapping.json")
        self.vessel_swap = left_right_lut(root / "vessel_mapping.json")
        self.positive_cases = []
        for case_id in self.cases:
            payload = json.loads(
                (root / "location_jsons" / f"{case_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            if payload.get("locations"):
                self.positive_cases.append(case_id)
        if not 0 <= self.positive_fraction <= 1:
            raise ValueError("positive_fraction must be in [0, 1]")
        divisor = 2**3
        if any(size % divisor for size in self.patch_size):
            raise ValueError(
                f"patch_size must be divisible by {divisor}: {self.patch_size}"
            )

    def __len__(self) -> int:
        return self.samples

    @staticmethod
    def _random_center(
        shape: tuple[int, int, int], rng: np.random.Generator
    ) -> tuple[int, int, int]:
        return tuple(int(rng.integers(max(size, 1))) for size in shape)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(item))
        positive = bool(self.positive_cases) and rng.random() < self.positive_fraction
        pool = self.positive_cases if positive else self.cases
        case_id = pool[int(rng.integers(len(pool)))]
        image, location, vessel = _load_case(self.data_root, case_id)

        if positive:
            points = np.argwhere(location > 0)
            center = points[int(rng.integers(len(points)))].astype(int)
            jitter = np.asarray(
                [
                    rng.integers(-max(size // 8, 1), max(size // 8, 1) + 1)
                    for size in self.patch_size
                ]
            )
            center_zyx = tuple((center + jitter).tolist())
        elif rng.random() < 0.5 and np.any(vessel > 0):
            points = np.argwhere(vessel > 0)
            center_zyx = tuple(points[int(rng.integers(len(points)))].tolist())
        else:
            center_zyx = self._random_center(image.shape, rng)

        image_patch, start = extract_patch(
            image, center_zyx, self.patch_size, pad_value=-3.0
        )
        location_patch, _ = extract_patch(location, center_zyx, self.patch_size)
        vessel_patch, _ = extract_patch(vessel, center_zyx, self.patch_size)
        coordinates = coordinate_patch(start, self.patch_size, image.shape)
        modality_value = 1.0 if "_mr_" in case_id else -1.0
        modality = np.full((1, *self.patch_size), modality_value, dtype=np.float32)
        inputs = np.concatenate([image_patch[None], modality, coordinates]).astype(
            np.float32
        )

        if self.augment:
            scale = float(rng.uniform(0.9, 1.1))
            shift = float(rng.uniform(-0.1, 0.1))
            inputs[0] = inputs[0] * scale + shift
            if rng.random() < 0.15:
                inputs[0] += rng.normal(0, 0.03, self.patch_size).astype(np.float32)
            if rng.random() < 0.5:
                inputs = np.flip(inputs, axis=-1).copy()
                inputs[4] *= -1.0
                location_patch = self.location_swap[
                    np.flip(location_patch, axis=-1)
                ].copy()
                vessel_patch = self.vessel_swap[np.flip(vessel_patch, axis=-1)].copy()

        presence = np.zeros(52, dtype=np.float32)
        foreground_labels = np.unique(location_patch)
        foreground_labels = foreground_labels[
            (foreground_labels >= 1) & (foreground_labels <= 52)
        ]
        presence[foreground_labels - 1] = 1.0
        return {
            "image": torch.from_numpy(inputs),
            "location": torch.from_numpy(location_patch.astype(np.int64, copy=False)),
            "vessel": torch.from_numpy(vessel_patch.astype(np.int64, copy=False)),
            "location_presence": torch.from_numpy(presence),
            "aneurysm_presence": torch.tensor(float(foreground_labels.size > 0)),
        }


@lru_cache(maxsize=8)
def _load_cached_case(
    cache_root: str, cache_directory: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_dir = Path(cache_root) / cache_directory
    return (
        np.load(case_dir / "image.npy", mmap_mode="r"),
        np.load(case_dir / "location.npy", mmap_mode="r"),
        np.load(case_dir / "vessel.npy", mmap_mode="r"),
    )


class CachedTopAneuPatchDataset(Dataset[dict[str, torch.Tensor]]):
    """Lesion-balanced patches drawn only from a completed physical cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        patch_size: tuple[int, int, int] = (64, 96, 96),
        samples: int = 1000,
        positive_fraction: float = 0.7,
        vessel_negative_fraction: float = 0.5,
        augment: bool = False,
        seed: int = 2026,
    ) -> None:
        index = load_cache_index(cache_dir)
        self.cache_root = str(Path(index["index_path"]).parent)
        self.cases = [case for case in index["cases"] if case["split"] == split]
        if not self.cases:
            raise ValueError(f"No {split} cases in cache")
        self.patch_size = tuple(int(value) for value in patch_size)
        self.samples = int(samples)
        self.positive_fraction = float(positive_fraction)
        self.vessel_negative_fraction = float(vessel_negative_fraction)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.location_swap = np.asarray(index["location_lr_swap"], dtype=np.int64)
        self.vessel_swap = np.asarray(index["vessel_lr_swap"], dtype=np.int64)
        self.instances = [
            (case, component)
            for case in self.cases
            for component in case.get("components", [])
        ]
        class_counts = np.bincount(
            [int(component["class_id"]) for _, component in self.instances],
            minlength=53,
        )
        weights = np.asarray(
            [
                1.0
                / np.sqrt(max(class_counts[int(component["class_id"])], 1))
                / max(float(component["voxels"]), 1.0) ** 0.25
                for _, component in self.instances
            ],
            dtype=np.float64,
        )
        self.instance_probabilities = (
            weights / weights.sum() if weights.size else weights
        )
        self.vessel_cases = [
            case for case in self.cases if case.get("vessel_points_zyx")
        ]
        if not 0 <= self.positive_fraction <= 1:
            raise ValueError("positive_fraction must be in [0, 1]")
        if not 0 <= self.vessel_negative_fraction <= 1:
            raise ValueError("vessel_negative_fraction must be in [0, 1]")
        divisor = 2**3
        if any(size % divisor for size in self.patch_size):
            raise ValueError(
                f"patch_size must be divisible by {divisor}: {self.patch_size}"
            )

    def __len__(self) -> int:
        return self.samples

    @staticmethod
    def _random_center(
        shape: tuple[int, int, int], rng: np.random.Generator
    ) -> tuple[int, int, int]:
        return tuple(int(rng.integers(size)) for size in shape)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + int(item))
        positive = bool(self.instances) and rng.random() < self.positive_fraction
        if positive:
            instance_index = int(
                rng.choice(len(self.instances), p=self.instance_probabilities)
            )
            case, component = self.instances[instance_index]
            center = np.asarray(component["center_zyx"], dtype=int)
            jitter = np.asarray(
                [
                    rng.integers(-max(size // 8, 1), max(size // 8, 1) + 1)
                    for size in self.patch_size
                ]
            )
            center_zyx = tuple((center + jitter).tolist())
        else:
            use_vessel = (
                self.vessel_cases and rng.random() < self.vessel_negative_fraction
            )
            case = (
                self.vessel_cases[int(rng.integers(len(self.vessel_cases)))]
                if use_vessel
                else self.cases[int(rng.integers(len(self.cases)))]
            )
            if use_vessel:
                points = case["vessel_points_zyx"]
                center_zyx = tuple(
                    int(value) for value in points[int(rng.integers(len(points)))]
                )
            else:
                center_zyx = self._random_center(tuple(case["shape_zyx"]), rng)

        image, location, vessel = _load_cached_case(
            self.cache_root, str(case["cache_dir"])
        )
        image_patch, start = extract_patch(
            image, center_zyx, self.patch_size, pad_value=-1.0
        )
        location_patch, _ = extract_patch(location, center_zyx, self.patch_size)
        vessel_patch, _ = extract_patch(vessel, center_zyx, self.patch_size)
        coordinates = coordinate_patch(start, self.patch_size, tuple(case["shape_zyx"]))
        modality_value = 1.0 if case["modality"] == "mr" else -1.0
        modality = np.full((1, *self.patch_size), modality_value, dtype=np.float32)
        inputs = np.concatenate(
            [image_patch.astype(np.float32, copy=False)[None], modality, coordinates]
        )

        if self.augment:
            inputs[0] = inputs[0] * float(rng.uniform(0.9, 1.1)) + float(
                rng.uniform(-0.1, 0.1)
            )
            if rng.random() < 0.15:
                inputs[0] += rng.normal(0, 0.03, self.patch_size).astype(np.float32)
            if rng.random() < 0.5:
                inputs = np.flip(inputs, axis=-1).copy()
                inputs[4] *= -1.0
                location_patch = self.location_swap[
                    np.flip(location_patch, axis=-1)
                ].copy()
                vessel_patch = self.vessel_swap[np.flip(vessel_patch, axis=-1)].copy()

        presence = np.zeros(52, dtype=np.float32)
        foreground_labels = np.unique(location_patch)
        foreground_labels = foreground_labels[
            (foreground_labels >= 1) & (foreground_labels <= 52)
        ]
        presence[foreground_labels - 1] = 1.0
        return {
            "image": torch.from_numpy(inputs.astype(np.float32, copy=False)),
            "location": torch.from_numpy(location_patch.astype(np.int64, copy=False)),
            "vessel": torch.from_numpy(vessel_patch.astype(np.int64, copy=False)),
            "vessel_valid": torch.tensor(float(case.get("vessel_valid", True))),
            "location_presence": torch.from_numpy(presence),
            "aneurysm_presence": torch.tensor(float(foreground_labels.size > 0)),
        }
