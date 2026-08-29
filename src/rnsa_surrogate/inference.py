"""Memory-conscious sliding-window inference for arbitrary 3D volumes."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from scipy import ndimage


def window_starts(length: int, patch: int, overlap: float) -> list[int]:
    if patch <= 0 or not 0 <= overlap < 1:
        raise ValueError("patch must be positive and overlap must be in [0, 1)")
    if length <= patch:
        return [0]
    stride = max(round(patch * (1.0 - overlap)), 1)
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def _pad_to_patch(
    image: np.ndarray, patch_size: Sequence[int]
) -> tuple[np.ndarray, tuple[slice, ...]]:
    padding = []
    crop = []
    for size, patch in zip(image.shape, patch_size, strict=True):
        total = max(int(patch) - int(size), 0)
        before = total // 2
        after = total - before
        padding.append((before, after))
        crop.append(slice(before, before + size))
    return np.pad(image, padding, constant_values=-3.0), tuple(crop)


def _coordinate_volume(shape: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(-1.0, 1.0, size, dtype=np.float32) for size in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"))


def component_postprocess(
    binary_probability: np.ndarray,
    class_confidence: np.ndarray,
    best_class: np.ndarray,
    mask_threshold: float,
    class_threshold: float,
    min_component_voxels: int = 3,
    component_probability_threshold: float = 0.55,
    component_class_threshold: float = 0.25,
    component_top_fraction: float = 0.25,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Filter voxel candidates and assign exactly one location to each component."""
    if min_component_voxels <= 0:
        raise ValueError("min_component_voxels must be positive")
    if not 0.0 < component_top_fraction <= 1.0:
        raise ValueError("component_top_fraction must be in (0, 1]")
    for name, value in (
        ("mask_threshold", mask_threshold),
        ("class_threshold", class_threshold),
        ("component_probability_threshold", component_probability_threshold),
        ("component_class_threshold", component_class_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    candidate = (binary_probability >= mask_threshold) & (
        class_confidence >= class_threshold
    )
    connected, _ = ndimage.label(
        candidate, structure=ndimage.generate_binary_structure(3, 1)
    )
    segmentation = np.zeros(best_class.shape, dtype=np.uint8)
    records: list[dict[str, Any]] = []
    for component_id, bounds in enumerate(ndimage.find_objects(connected), start=1):
        if bounds is None:
            continue
        local_connected = connected[bounds]
        component = local_connected == component_id
        voxel_count = int(np.count_nonzero(component))
        binary_values = binary_probability[bounds][component]
        top_count = max(math.ceil(voxel_count * component_top_fraction), 1)
        binary_score = float(
            np.partition(binary_values, -top_count)[-top_count:].mean()
        )

        labels = best_class[bounds][component].astype(np.int64, copy=False)
        weights = binary_values * class_confidence[bounds][component]
        label_scores = np.bincount(labels, weights=weights, minlength=53)
        label_scores[0] = 0.0
        location = int(label_scores.argmax())
        class_score = float(
            label_scores[location] / max(float(binary_values.sum()), 1e-8)
        )
        accepted = (
            voxel_count >= min_component_voxels
            and binary_score >= component_probability_threshold
            and class_score >= component_class_threshold
            and location > 0
        )
        if accepted:
            local_segmentation = segmentation[bounds]
            local_segmentation[component] = location
        records.append(
            {
                "component_id": component_id,
                "voxels": voxel_count,
                "binary_score": binary_score,
                "class_score": class_score,
                "location": location,
                "accepted": accepted,
            }
        )
    return segmentation, records


@torch.inference_mode()
def sliding_window_predict(
    model: torch.nn.Module,
    image_zyx: np.ndarray,
    modality: str,
    patch_size: Sequence[int],
    device: torch.device,
    overlap: float = 0.5,
    amp_dtype: torch.dtype | None = None,
    mask_threshold: float = 0.45,
    class_threshold: float = 0.15,
    presence_threshold: float = 0.35,
    min_component_voxels: int = 3,
    component_probability_threshold: float = 0.55,
    component_class_threshold: float = 0.25,
    component_top_fraction: float = 0.25,
) -> tuple[np.ndarray, list[int], dict[str, Any]]:
    patch_size = tuple(int(value) for value in patch_size)
    image, crop = _pad_to_patch(image_zyx.astype(np.float32, copy=False), patch_size)
    coordinates = _coordinate_volume(image.shape)
    binary_sum = np.zeros(image.shape, dtype=np.float32)
    binary_count = np.zeros(image.shape, dtype=np.uint16)
    best_class_score = np.zeros(image.shape, dtype=np.float32)
    best_class = np.zeros(image.shape, dtype=np.uint8)
    global_scores = np.zeros(52, dtype=np.float32)
    modality_value = 1.0 if modality.lower() == "mr" else -1.0

    starts = [
        window_starts(size, patch, overlap)
        for size, patch in zip(image.shape, patch_size, strict=True)
    ]
    model.eval()
    for z in starts[0]:
        for y in starts[1]:
            for x in starts[2]:
                region = (
                    slice(z, z + patch_size[0]),
                    slice(y, y + patch_size[1]),
                    slice(x, x + patch_size[2]),
                )
                image_patch = image[region][None]
                coordinate_patch = coordinates[(slice(None), *region)]
                modality_patch = np.full(
                    (1, *patch_size), modality_value, dtype=np.float32
                )
                inputs = np.concatenate([image_patch, modality_patch, coordinate_patch])
                tensor = torch.from_numpy(inputs[None]).to(device)
                with torch.autocast(
                    device.type, dtype=amp_dtype, enabled=amp_dtype is not None
                ):
                    outputs = model(tensor)

                binary = (
                    torch.sigmoid(outputs["aneurysm_logits"])[0, 0]
                    .float()
                    .cpu()
                    .numpy()
                )
                global_probability = torch.sigmoid(outputs["location_presence_logits"])[
                    0
                ].float()
                global_scores = np.maximum(
                    global_scores, global_probability.cpu().numpy()
                )
                location = torch.softmax(outputs["location_logits"], dim=1)[
                    0, 1:
                ].float()
                location = location * (
                    0.5 + 0.5 * global_probability[:, None, None, None]
                )
                class_score, class_index = location.max(dim=0)
                class_score_array = class_score.cpu().numpy()
                class_array = (class_index + 1).to(torch.uint8).cpu().numpy()

                binary_sum[region] += binary
                binary_count[region] += 1
                replace = class_score_array > best_class_score[region]
                best_class_score[region][replace] = class_score_array[replace]
                best_class[region][replace] = class_array[replace]

    binary_probability = binary_sum / np.maximum(binary_count, 1)
    binary_probability = binary_probability[crop]
    best_class_score = best_class_score[crop]
    best_class = best_class[crop]
    segmentation, components = component_postprocess(
        binary_probability,
        best_class_score,
        best_class,
        mask_threshold,
        class_threshold,
        min_component_voxels,
        component_probability_threshold,
        component_class_threshold,
        component_top_fraction,
    )
    task1 = sorted(
        int(value)
        for value in np.unique(segmentation)
        if value > 0 and global_scores[int(value) - 1] >= presence_threshold
    )
    diagnostics = {
        "binary_probability": binary_probability,
        "class_confidence": best_class_score,
        "global_location_scores": global_scores,
        "components": components,
    }
    return segmentation, task1, diagnostics
