"""Memory-conscious sliding-window inference for arbitrary 3D volumes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


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
) -> tuple[np.ndarray, list[int], dict[str, np.ndarray]]:
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
    segmentation = best_class.copy()
    segmentation[
        (binary_probability < mask_threshold) | (best_class_score < class_threshold)
    ] = 0
    segmentation = segmentation[crop]
    binary_probability = binary_probability[crop]
    best_class_score = best_class_score[crop]
    task1 = {
        int(value) for value in np.flatnonzero(global_scores >= presence_threshold) + 1
    }
    task1.update(int(value) for value in np.unique(segmentation) if value > 0)
    diagnostics = {
        "binary_probability": binary_probability,
        "class_confidence": best_class_score,
        "global_location_scores": global_scores,
    }
    return segmentation, sorted(task1), diagnostics
