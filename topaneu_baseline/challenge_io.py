from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


NIFTI_SUFFIX = ".nii.gz"
IMAGE_SUFFIX = "_0000.nii.gz"


@dataclass(frozen=True)
class OutputValidation:
    case_id: str
    locations: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str


def case_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    if name.endswith(IMAGE_SUFFIX):
        return name[: -len(IMAGE_SUFFIX)]
    if name.endswith(NIFTI_SUFFIX):
        return name[: -len(NIFTI_SUFFIX)]
    raise ValueError(f"Expected a .nii.gz file, got: {name}")


def load_label_mapping(path: str | Path) -> dict[str, int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"Missing non-empty 'labels' mapping in {path}")
    parsed = {str(name): int(value) for name, value in labels.items()}
    if parsed.get("background") != 0:
        raise ValueError("Location mapping must define background as 0")
    values = sorted(parsed.values())
    if values != list(range(max(values) + 1)):
        raise ValueError(f"Location labels must be contiguous from 0: {values}")
    return parsed


def locations_from_mask(
    mask: np.ndarray,
    *,
    max_label: int,
    min_component_voxels: int = 1,
    min_component_mm3: float = 0.0,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
) -> list[int]:
    """Convert a multiclass mask into the Task 1 top-level integer list."""
    if min_component_voxels < 1:
        raise ValueError("min_component_voxels must be at least 1")
    voxel_volume = float(np.prod(tuple(float(v) for v in spacing)))
    min_voxels_from_volume = int(np.ceil(float(min_component_mm3) / voxel_volume)) if min_component_mm3 > 0 else 1
    minimum = max(int(min_component_voxels), min_voxels_from_volume)

    locations: list[int] = []
    for value in np.unique(mask):
        label = int(value)
        if label == 0:
            continue
        if not 1 <= label <= max_label:
            raise ValueError(f"Mask contains label {label}, expected 0..{max_label}")
        binary = mask == label
        if minimum == 1:
            locations.append(label)
            continue
        components, count = ndimage.label(binary)
        if count == 0:
            continue
        sizes = np.bincount(components.ravel())[1:]
        if bool(np.any(sizes >= minimum)):
            locations.append(label)
    return sorted(locations)


def _as_uint8_labels(image: nib.spatialimages.SpatialImage, max_label: int) -> np.ndarray:
    data = np.asanyarray(image.dataobj)
    if not np.all(np.isfinite(data)):
        raise ValueError("Prediction contains NaN or infinite values")
    rounded = np.rint(data)
    if not np.array_equal(data, rounded):
        raise ValueError("Prediction mask contains non-integer values")
    minimum = int(rounded.min(initial=0))
    maximum = int(rounded.max(initial=0))
    if minimum < 0 or maximum > max_label:
        raise ValueError(f"Prediction labels must be within 0..{max_label}, got {minimum}..{maximum}")
    return rounded.astype(np.uint8, copy=False)


def write_challenge_outputs(
    prediction_path: str | Path,
    reference_image_path: str | Path,
    task1_output_path: str | Path,
    task2_output_path: str | Path,
    *,
    max_label: int,
    min_component_voxels: int = 1,
    min_component_mm3: float = 0.0,
) -> OutputValidation:
    prediction_path = Path(prediction_path)
    reference_image_path = Path(reference_image_path)
    task1_output_path = Path(task1_output_path)
    task2_output_path = Path(task2_output_path)

    prediction = nib.load(str(prediction_path))
    reference = nib.load(str(reference_image_path))
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for {prediction_path.name}: prediction {prediction.shape}, reference {reference.shape}"
        )
    if not np.allclose(prediction.affine, reference.affine, rtol=1e-5, atol=1e-4):
        raise ValueError(f"Affine mismatch for {prediction_path.name}")

    mask = _as_uint8_labels(prediction, max_label)
    spacing = tuple(float(v) for v in reference.header.get_zooms()[:3])
    locations = locations_from_mask(
        mask,
        max_label=max_label,
        min_component_voxels=min_component_voxels,
        min_component_mm3=min_component_mm3,
        spacing=spacing,
    )

    task1_output_path.parent.mkdir(parents=True, exist_ok=True)
    task2_output_path.parent.mkdir(parents=True, exist_ok=True)
    task1_output_path.write_text(json.dumps(locations, indent=2) + "\n", encoding="utf-8")

    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(mask, reference.affine, header=header)
    output.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output, str(task2_output_path))

    return validate_challenge_outputs(
        reference_image_path,
        task1_output_path,
        task2_output_path,
        max_label=max_label,
    )


def validate_challenge_outputs(
    reference_image_path: str | Path,
    task1_output_path: str | Path,
    task2_output_path: str | Path,
    *,
    max_label: int,
) -> OutputValidation:
    reference = nib.load(str(reference_image_path))
    output = nib.load(str(task2_output_path))
    if output.shape != reference.shape:
        raise ValueError(f"Task 2 output shape {output.shape} does not match input {reference.shape}")
    if not np.allclose(output.affine, reference.affine, rtol=1e-5, atol=1e-4):
        raise ValueError("Task 2 output affine does not match the input image")
    if np.dtype(output.get_data_dtype()) != np.dtype(np.uint8):
        raise ValueError(f"Task 2 output must be uint8, got {output.get_data_dtype()}")
    mask = _as_uint8_labels(output, max_label)

    payload = json.loads(Path(task1_output_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or any(isinstance(v, bool) or not isinstance(v, int) for v in payload):
        raise ValueError("Task 1 output must be a top-level JSON list of integers")
    locations = tuple(int(v) for v in payload)
    if tuple(sorted(set(locations))) != locations:
        raise ValueError("Task 1 locations must be sorted and unique")
    invalid = [v for v in locations if not 1 <= v <= max_label]
    if invalid:
        raise ValueError(f"Task 1 output contains invalid locations: {invalid}")

    return OutputValidation(
        case_id=case_id_from_path(reference_image_path),
        locations=locations,
        shape=tuple(int(v) for v in mask.shape),
        dtype=str(mask.dtype),
    )
