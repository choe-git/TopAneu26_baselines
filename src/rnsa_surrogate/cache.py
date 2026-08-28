"""Build and validate a reusable physical-space TopAneu training cache."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm

CACHE_VERSION = 1
REQUIRED_SOURCE_DIRECTORIES = ("images", "location_masks", "location_jsons", "vessel_masks")


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_csv(path: str | Path) -> dict[str, str]:
    path = Path(path)
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = ["case_id", "train", "val", "test"]
        if reader.fieldnames != expected:
            raise ValueError(f"Expected split columns {expected}, got {reader.fieldnames}")
        for row in reader:
            selected = [name for name in ("train", "val", "test") if row[name] == "1"]
            if len(selected) != 1 or row["case_id"] in result:
                raise ValueError(f"Invalid split row: {row}")
            result[row["case_id"]] = selected[0]
    if not result:
        raise ValueError(f"Split CSV contains no cases: {path}")
    return result


def case_id_from_image(path: str | Path) -> str:
    suffix = "_0000.nii.gz"
    name = Path(path).name
    if not name.endswith(suffix):
        raise ValueError(f"Expected *{suffix}: {name}")
    return name.removesuffix(suffix)


def case_domain(case_id: str) -> tuple[str, str]:
    tokens = case_id.split("_")
    return (tokens[1].lower(), tokens[2].lower()) if len(tokens) >= 4 else ("unknown", "unknown")


def patient_id(case_id: str) -> str:
    tokens = case_id.split("_")
    return "_".join(tokens[:4]) if len(tokens) >= 4 else case_id


def load_zyx(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    image = nib.load(path)
    if image.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {image.shape}: {path}")
    array = np.asanyarray(image.dataobj).transpose(2, 1, 0)
    metadata = {
        "shape_zyx": list(array.shape),
        "spacing_xyz": [float(value) for value in image.header.get_zooms()[:3]],
        "affine": image.affine.tolist(),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
    }
    return np.asarray(array), metadata


def spacing_zyx(metadata: dict[str, Any]) -> tuple[float, float, float]:
    x, y, z = metadata["spacing_xyz"]
    return float(z), float(y), float(x)


def normalize_angiography(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError("Image contains no finite values")
    low, high = np.percentile(finite, (0.5, 99.5))
    if not np.isfinite(low + high) or high <= low:
        low, high = float(finite.min()), float(finite.max())
    scale = max(float(high - low), 1e-6)
    normalized = np.clip((image - low) / scale, 0.0, 1.0) * 2.0 - 1.0
    normalized[~np.isfinite(normalized)] = -1.0
    return normalized.astype(np.float32), {"low": float(low), "high": float(high)}


def resample_zyx(
    array: np.ndarray,
    source_spacing_zyx: Sequence[float],
    target_spacing_zyx: Sequence[float],
    order: int,
) -> np.ndarray:
    factors = np.asarray(source_spacing_zyx, dtype=np.float64) / np.asarray(
        target_spacing_zyx, dtype=np.float64
    )
    if np.allclose(factors, 1.0, atol=1e-4):
        return np.asarray(array).copy()
    return ndimage.zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)


def resize_to_shape(array: np.ndarray, shape_zyx: Sequence[int], order: int) -> np.ndarray:
    target = tuple(int(value) for value in shape_zyx)
    if array.shape == target:
        return np.asarray(array).copy()
    factors = np.asarray(target, dtype=np.float64) / np.asarray(array.shape, dtype=np.float64)
    resized = ndimage.zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)
    result = np.zeros(target, dtype=resized.dtype)
    common = tuple(
        min(source, destination) for source, destination in zip(resized.shape, target, strict=True)
    )
    slices = tuple(slice(0, value) for value in common)
    result[slices] = resized[slices]
    return result


def _same_geometry(first: dict[str, Any], second: dict[str, Any]) -> bool:
    if first["shape_zyx"] != second["shape_zyx"]:
        return False
    first_affine = np.asarray(first["affine"], dtype=np.float64)
    second_affine = np.asarray(second["affine"], dtype=np.float64)
    shape_xyz = tuple(reversed(first["shape_zyx"]))
    corners = np.asarray(
        [
            [x, y, z, 1.0]
            for x in (0, max(shape_xyz[0] - 1, 0))
            for y in (0, max(shape_xyz[1] - 1, 0))
            for z in (0, max(shape_xyz[2] - 1, 0))
        ]
    )
    displacement = corners @ (first_affine - second_affine).T
    return float(np.linalg.norm(displacement[:, :3], axis=1).max()) <= 0.05


def _preserve_component_centers(original: np.ndarray, resampled: np.ndarray) -> int:
    scale = np.asarray(resampled.shape, dtype=np.float64) / np.asarray(
        original.shape, dtype=np.float64
    )
    structure = ndimage.generate_binary_structure(3, 1)
    restored = 0
    for class_id in (int(value) for value in np.unique(original) if value != 0):
        binary = original == class_id
        connected, count = ndimage.label(binary, structure=structure)
        centers = ndimage.center_of_mass(binary, connected, range(1, count + 1))
        for center in centers:
            mapped = np.rint((np.asarray(center) + 0.5) * scale - 0.5).astype(int)
            mapped = np.clip(mapped, 0, np.asarray(resampled.shape) - 1)
            lower, upper = np.maximum(mapped - 1, 0), np.minimum(mapped + 2, resampled.shape)
            local = tuple(
                slice(int(low), int(high)) for low, high in zip(lower, upper, strict=True)
            )
            if np.any(resampled[local] == class_id):
                continue
            destination = tuple(int(value) for value in mapped)
            if resampled[destination] != 0:
                background = np.argwhere(resampled[local] == 0)
                if background.size:
                    destination = tuple(int(value) for value in background[0] + lower)
            resampled[destination] = class_id
            restored += 1
    return restored


def _instances(location: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    instance_map = np.zeros(location.shape, dtype=np.uint16)
    records: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(3, 1)
    next_id = 1
    for class_id in (int(value) for value in np.unique(location) if value != 0):
        connected, count = ndimage.label(location == class_id, structure=structure)
        for local_id in range(1, count + 1):
            coordinates = np.argwhere(connected == local_id)
            if coordinates.size == 0:
                continue
            instance_map[connected == local_id] = next_id
            records.append(
                {
                    "instance_id": next_id,
                    "class_id": class_id,
                    "center_zyx": coordinates.mean(axis=0).round().astype(int).tolist(),
                    "bbox_zyx": [
                        coordinates.min(axis=0).tolist(),
                        (coordinates.max(axis=0) + 1).tolist(),
                    ],
                    "voxels": len(coordinates),
                }
            )
            next_id += 1
    return instance_map, records


def _sample_vessel_points(vessel: np.ndarray, seed: int, limit: int = 4096) -> list[list[int]]:
    flat = np.flatnonzero(vessel > 0)
    if flat.size == 0:
        return []
    generator = np.random.default_rng(seed)
    if flat.size > limit:
        flat = generator.choice(flat, size=limit, replace=False)
    return np.column_stack(np.unravel_index(flat, vessel.shape)).astype(int).tolist()


def _mapping(source_root: Path, filename: str) -> tuple[dict[str, int], list[int]]:
    labels = json.loads((source_root / filename).read_text(encoding="utf-8"))["labels"]
    labels = {str(name): int(value) for name, value in labels.items()}
    lut = list(range(max(labels.values()) + 1))
    for name, value in labels.items():
        counterpart = (
            "L-" + name[2:]
            if name.startswith("R-")
            else "R-" + name[2:]
            if name.startswith("L-")
            else name
        )
        if counterpart not in labels:
            raise ValueError(f"Missing LR counterpart for {name!r} in {filename}")
        lut[value] = labels[counterpart]
    if any(lut[lut[index]] != index for index in range(len(lut))):
        raise ValueError(f"LR permutation is not involutive: {filename}")
    return labels, lut


def build_cache(
    source_root: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    target_spacing_zyx: Sequence[float] = (0.6, 0.6, 0.6),
    overwrite: bool = False,
) -> Path:
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    target_spacing = tuple(float(value) for value in target_spacing_zyx)
    if len(target_spacing) != 3 or not all(
        np.isfinite(value) and value > 0 for value in target_spacing
    ):
        raise ValueError(f"Invalid target spacing: {target_spacing}")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    missing_directories = [
        name for name in REQUIRED_SOURCE_DIRECTORIES if not (source_root / name).is_dir()
    ]
    if missing_directories:
        raise FileNotFoundError(f"Invalid source root {source_root}; missing {missing_directories}")

    images = sorted((source_root / "images").glob("*_0000.nii.gz"))
    split = read_split_csv(split_csv)
    image_cases = {case_id_from_image(path) for path in images}
    if image_cases != set(split):
        raise ValueError(
            f"Split/data mismatch: missing_in_split={sorted(image_cases - set(split))}, "
            f"missing_images={sorted(set(split) - image_cases)}"
        )
    index_path = output_dir / "index.json"
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"Cache already exists: {index_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Cache output is non-empty: {output_dir}")
    if overwrite:
        index_path.unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    location_mapping, location_lr_swap = _mapping(source_root, "location_mapping.json")
    vessel_mapping, vessel_lr_swap = _mapping(source_root, "vessel_mapping.json")
    cases = []
    for case_index, image_path in enumerate(tqdm(images, desc="Building 0.6 mm cache")):
        case_id = case_id_from_image(image_path)
        location_path = source_root / "location_masks" / f"{case_id}.nii.gz"
        vessel_path = source_root / "vessel_masks" / f"{case_id}.nii.gz"
        json_path = source_root / "location_jsons" / f"{case_id}.json"
        for required in (location_path, json_path):
            if not required.is_file():
                raise FileNotFoundError(required)

        image, image_meta = load_zyx(image_path)
        location, location_meta = load_zyx(location_path)
        if not _same_geometry(image_meta, location_meta):
            raise ValueError(f"Image/location geometry mismatch: {case_id}")
        source_spacing = spacing_zyx(image_meta)
        normalized, intensity_stats = normalize_angiography(image)
        normalized = resample_zyx(normalized, source_spacing, target_spacing, 1).astype(np.float16)
        original_location = np.asarray(location, dtype=np.uint8)
        location = resample_zyx(original_location, source_spacing, target_spacing, 0).astype(
            np.uint8
        )
        preserved_components = _preserve_component_centers(original_location, location)

        vessel_valid = vessel_path.is_file()
        if vessel_valid:
            vessel, vessel_meta = load_zyx(vessel_path)
            if not _same_geometry(image_meta, vessel_meta):
                raise ValueError(f"Image/vessel geometry mismatch: {case_id}")
            vessel = resample_zyx(vessel, source_spacing, target_spacing, 0).astype(np.uint8)
        else:
            vessel = np.zeros(location.shape, dtype=np.uint8)
        if normalized.shape != location.shape or vessel.shape != location.shape:
            raise RuntimeError(f"Resampling shape mismatch: {case_id}")

        instances, components = _instances(location)
        case_dir = output_dir / "cases" / case_id
        atomic_save_npy(case_dir / "image.npy", normalized)
        atomic_save_npy(case_dir / "location.npy", location)
        atomic_save_npy(case_dir / "vessel.npy", vessel)
        atomic_save_npy(case_dir / "instances.npy", instances)
        center, modality = case_domain(case_id)
        json_locations = json.loads(json_path.read_text(encoding="utf-8")).get("locations", [])
        record = {
            "case_id": case_id,
            "patient_id": patient_id(case_id),
            "center": center,
            "modality": modality,
            "split": split[case_id],
            "shape_zyx": list(location.shape),
            "source_spacing_zyx": list(source_spacing),
            "target_spacing_zyx": list(target_spacing),
            "original_metadata": image_meta,
            "intensity_stats": intensity_stats,
            "json_locations": [int(value) for value in json_locations],
            "components": components,
            "preserved_components": preserved_components,
            "vessel_valid": vessel_valid,
            "vessel_points_zyx": _sample_vessel_points(vessel, 2026 + case_index),
            "cache_dir": str(Path("cases") / case_id),
        }
        atomic_json_dump(record, case_dir / "meta.json")
        cases.append(record)

    payload = {
        "cache_version": CACHE_VERSION,
        "source_root": str(source_root),
        "split_csv": str(Path(split_csv).resolve()),
        "split_csv_sha256": sha256_file(split_csv),
        "target_spacing_zyx": list(target_spacing),
        "location_mapping": location_mapping,
        "vessel_mapping": vessel_mapping,
        "location_lr_swap": location_lr_swap,
        "vessel_lr_swap": vessel_lr_swap,
        "cases": cases,
    }
    atomic_json_dump(payload, index_path)
    return index_path


def load_cache_index(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    index_path = requested if requested.name == "index.json" else requested / "index.json"
    if not index_path.is_file():
        partial = sum(1 for item in (index_path.parent / "cases").glob("*") if item.is_dir())
        raise FileNotFoundError(
            f"Missing cache index {index_path}; partial case directories={partial}"
        )
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("cache_version") != CACHE_VERSION:
        raise ValueError(f"Unsupported cache version: {payload.get('cache_version')}")
    payload["index_path"] = str(index_path.resolve())
    return payload


def validate_cache(path: str | Path, deep: bool = False) -> dict[str, Any]:
    payload = load_cache_index(path)
    root = Path(payload["index_path"]).parent
    errors = []
    split_counts = {name: 0 for name in ("train", "val", "test")}
    restored = 0
    for case in payload["cases"]:
        split_counts[case["split"]] += 1
        restored += int(case.get("preserved_components", 0))
        case_dir = root / case["cache_dir"]
        for filename in ("image.npy", "location.npy", "vessel.npy", "instances.npy", "meta.json"):
            if not (case_dir / filename).is_file():
                errors.append(f"{case['case_id']}: missing {filename}")
        if deep and not errors:
            arrays = {
                name: np.load(case_dir / f"{name}.npy", mmap_mode="r")
                for name in ("image", "location", "vessel", "instances")
            }
            expected = tuple(case["shape_zyx"])
            if any(array.shape != expected for array in arrays.values()):
                errors.append(f"{case['case_id']}: shape mismatch")
            if arrays["image"].dtype != np.float16 or arrays["location"].dtype != np.uint8:
                errors.append(f"{case['case_id']}: dtype mismatch")
            if not np.isfinite(arrays["image"]).all():
                errors.append(f"{case['case_id']}: non-finite image")
    report = {
        "cache_version": payload["cache_version"],
        "index": payload["index_path"],
        "index_sha256": sha256_file(payload["index_path"]),
        "target_spacing_zyx": payload["target_spacing_zyx"],
        "cases": len(payload["cases"]),
        "split_counts": split_counts,
        "preserved_components": restored,
        "errors": errors,
    }
    if errors:
        raise ValueError("Invalid cache: " + "; ".join(errors[:8]))
    return report
