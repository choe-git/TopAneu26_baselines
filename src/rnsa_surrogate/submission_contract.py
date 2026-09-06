"""Pure validation helpers for the official TopAneu submission interfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


N_CLASSES = 52
INPUT_INTERFACES = {
    "head-ct-angiography": (Path("images/head-ct-angio"), "ct"),
    "head-mr-angiography": (Path("images/head-mr-angio"), "mr"),
}
TASK1_OUTPUT = Path("detected-aneurysm-locations.json")
TASK2_OUTPUT_DIRECTORY = Path("images/aneurysm-segmentation")


def input_interface(inputs: Any) -> tuple[str, Path]:
    """Return modality and image directory for one image-only GC socket."""
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise ValueError("TopAneu submission requires exactly one input socket")
    try:
        slug = inputs[0]["socket"]["slug"]
    except (KeyError, TypeError) as error:
        raise ValueError("Invalid /input/inputs.json structure") from error
    if slug not in INPUT_INTERFACES:
        raise ValueError(f"Unsupported or non-image input socket: {slug!r}")
    directory, modality = INPUT_INTERFACES[slug]
    return modality, directory


def load_input_contract(input_root: str | Path) -> tuple[str, Path]:
    input_root = Path(input_root)
    payload = json.loads((input_root / "inputs.json").read_text(encoding="utf-8"))
    modality, relative_directory = input_interface(payload)
    directory = input_root / relative_directory
    candidates = sorted(
        path
        for suffix in ("*.mha", "*.tif", "*.tiff")
        for path in directory.glob(suffix)
        if path.is_file()
    )
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one image in {directory}, found {len(candidates)}"
        )
    return modality, candidates[0]


def validate_task1_locations(values: Any) -> list[int]:
    if not isinstance(values, list):
        raise TypeError("Task 1 output must be a JSON list")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"Task 1 labels must be integers, got {value!r}")
        label = int(value)
        if not 1 <= label <= N_CLASSES:
            raise ValueError(f"Task 1 labels must be in [1, {N_CLASSES}]: {label}")
        normalized.append(label)
    if len(normalized) != len(set(normalized)):
        raise ValueError("Task 1 labels must be unique")
    return normalized


def validate_task2_array(
    prediction_zyx: np.ndarray,
    expected_shape_zyx: Sequence[int] | None = None,
) -> np.ndarray:
    prediction = np.asarray(prediction_zyx)
    if prediction.ndim != 3:
        raise ValueError(f"Task 2 output must be 3D, got {prediction.shape}")
    if prediction.dtype != np.uint8:
        raise TypeError(f"Task 2 output must be uint8, got {prediction.dtype}")
    if expected_shape_zyx is not None and prediction.shape != tuple(
        int(value) for value in expected_shape_zyx
    ):
        raise ValueError(
            f"Task 2 geometry shape mismatch: {prediction.shape} != "
            f"{tuple(expected_shape_zyx)}"
        )
    if prediction.size and int(prediction.max()) > N_CLASSES:
        raise ValueError(f"Task 2 labels must be in [0, {N_CLASSES}]")
    return prediction


def copy_task2_geometry(
    prediction_zyx: np.ndarray,
    reference_image: Any,
    simpleitk: Any | None = None,
) -> Any:
    """Create a uint8 SimpleITK image with the exact input physical geometry."""
    if simpleitk is None:
        try:
            import SimpleITK as simpleitk  # type: ignore[no-redef]
        except ImportError as error:
            raise RuntimeError("SimpleITK is required by the submission adapter") from error
    expected_shape = tuple(reversed(tuple(int(v) for v in reference_image.GetSize())))
    prediction = validate_task2_array(prediction_zyx, expected_shape)
    output = simpleitk.GetImageFromArray(prediction)
    output.CopyInformation(reference_image)
    return output


def resolve_inference_amp(
    requested: str, device_type: str, bf16_supported: bool
) -> str:
    """Resolve a portable inference autocast mode; ``none`` means FP32."""
    requested = str(requested).lower()
    if requested not in {"bf16", "fp16", "fp32", "none"}:
        raise ValueError(f"Unsupported AMP mode: {requested}")
    if device_type != "cuda" or requested in {"fp32", "none"}:
        return "none"
    if requested == "bf16" and not bf16_supported:
        return "fp16"
    return requested
