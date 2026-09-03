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
    presence_top_k: int = 3,
    presence_evidence_voxels: int = 64,
) -> tuple[np.ndarray, list[int], dict[str, np.ndarray]]:
    return ensemble_sliding_window_predict(
        [model],
        image_zyx,
        modality,
        patch_size,
        device,
        overlap=overlap,
        amp_dtype=amp_dtype,
        mask_threshold=mask_threshold,
        class_threshold=class_threshold,
        presence_threshold=presence_threshold,
        presence_top_k=presence_top_k,
        presence_evidence_voxels=presence_evidence_voxels,
    )


@torch.inference_mode()
def ensemble_sliding_window_predict(
    models: Sequence[torch.nn.Module],
    image_zyx: np.ndarray,
    modality: str,
    patch_size: Sequence[int],
    device: torch.device,
    overlap: float = 0.5,
    amp_dtype: torch.dtype | None = None,
    mask_threshold: float = 0.45,
    class_threshold: float = 0.15,
    presence_threshold: float = 0.35,
    presence_top_k: int = 3,
    presence_evidence_voxels: int = 64,
    tta_left_right: bool = False,
    location_lr_swap: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[int], dict[str, np.ndarray]]:
    """Soft-vote fold probabilities, optionally with left-right flip TTA.

    Task 1 uses top-k patch evidence instead of a single-patch maximum. Location
    labels use a memory-conscious confidence-weighted overlap consensus rather
    than allocating a 52-channel probability volume.
    """
    if not models:
        raise ValueError("At least one model is required")
    if tta_left_right and location_lr_swap is None:
        raise ValueError("location_lr_swap is required for left-right TTA")
    if presence_top_k <= 0 or presence_evidence_voxels <= 0:
        raise ValueError(
            "presence_top_k and presence_evidence_voxels must be positive"
        )
    patch_size = tuple(int(value) for value in patch_size)
    image, crop = _pad_to_patch(image_zyx.astype(np.float32, copy=False), patch_size)
    coordinates = _coordinate_volume(image.shape)
    binary_sum = np.zeros(image.shape, dtype=np.float32)
    binary_count = np.zeros(image.shape, dtype=np.uint16)
    class_vote_margin = np.zeros(image.shape, dtype=np.float32)
    best_class = np.zeros(image.shape, dtype=np.uint8)
    global_patch_scores: list[np.ndarray] = []
    aneurysm_patch_scores: list[float] = []
    modality_value = 1.0 if modality.lower() == "mr" else -1.0
    tta_variants = (False, True) if tta_left_right else (False,)
    members = len(models) * len(tta_variants)
    if location_lr_swap is not None:
        swap = np.asarray(location_lr_swap, dtype=np.int64)
        if swap.shape[0] < 53:
            raise ValueError("location_lr_swap must include labels 0..52")
        location_restore = torch.as_tensor(
            swap[1:53] - 1, device=device, dtype=torch.long
        )
    else:
        location_restore = None

    starts = [
        window_starts(size, patch, overlap)
        for size, patch in zip(image.shape, patch_size, strict=True)
    ]
    for model in models:
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
                binary_probability_patch = None
                location_probability_patch = None
                global_probability_patch = None
                aneurysm_probability_patch = None
                for model in models:
                    for flipped in tta_variants:
                        member_input = tensor
                        if flipped:
                            member_input = torch.flip(tensor, dims=(-1,)).clone()
                            member_input[:, 4].mul_(-1.0)
                        with torch.autocast(
                            device.type,
                            dtype=amp_dtype,
                            enabled=amp_dtype is not None,
                        ):
                            outputs = model(member_input)
                        member_binary = torch.sigmoid(
                            outputs["aneurysm_logits"]
                        )[0, 0].float()
                        member_location = torch.softmax(
                            outputs["location_logits"][:, 1:53], dim=1
                        )[0].float()
                        member_global = torch.sigmoid(
                            outputs["location_presence_logits"]
                        )[0].float()
                        member_aneurysm = torch.sigmoid(
                            outputs["aneurysm_presence_logits"]
                        )[0, 0].float()
                        if flipped:
                            assert location_restore is not None
                            member_binary = torch.flip(member_binary, dims=(-1,))
                            member_location = torch.flip(
                                member_location, dims=(-1,)
                            )[location_restore]
                            member_global = member_global[location_restore]
                        if binary_probability_patch is None:
                            binary_probability_patch = member_binary
                            location_probability_patch = member_location
                            global_probability_patch = member_global
                            aneurysm_probability_patch = member_aneurysm
                        else:
                            binary_probability_patch += member_binary
                            location_probability_patch += member_location
                            global_probability_patch += member_global
                            aneurysm_probability_patch += member_aneurysm
                assert binary_probability_patch is not None
                assert location_probability_patch is not None
                assert global_probability_patch is not None
                assert aneurysm_probability_patch is not None
                binary_probability_patch /= members
                location_probability_patch /= members
                global_probability_patch /= members
                aneurysm_probability_patch /= members
                binary = binary_probability_patch.cpu().numpy()
                global_probability = global_probability_patch
                evidence_voxels = min(presence_evidence_voxels, binary.size)
                binary_evidence = float(
                    np.partition(binary.reshape(-1), -evidence_voxels)[
                        -evidence_voxels:
                    ].mean()
                )
                aneurysm_probability = float(aneurysm_probability_patch.item())
                patch_gate = float(np.sqrt(binary_evidence * aneurysm_probability))
                global_patch_scores.append(
                    global_probability.cpu().numpy() * patch_gate
                )
                aneurysm_patch_scores.append(aneurysm_probability)
                location = location_probability_patch * (
                    0.5 + 0.5 * global_probability[:, None, None, None]
                )
                class_score, class_index = location.max(dim=0)
                class_score_array = class_score.cpu().numpy()
                class_array = (class_index + 1).to(torch.uint8).cpu().numpy()

                binary_sum[region] += binary
                binary_count[region] += 1
                current_class = best_class[region]
                current_margin = class_vote_margin[region]
                empty = current_class == 0
                same = empty | (current_class == class_array)
                current_class[empty] = class_array[empty]
                current_margin[same] += class_score_array[same]
                different = ~same
                current_margin[different] -= class_score_array[different]
                switch = different & (current_margin < 0)
                current_class[switch] = class_array[switch]
                current_margin[switch] *= -1.0

    binary_probability = binary_sum / np.maximum(binary_count, 1)
    class_confidence = class_vote_margin / np.maximum(binary_count, 1)
    patch_score_array = np.stack(global_patch_scores)
    top_k = min(presence_top_k, patch_score_array.shape[0])
    global_scores = np.partition(patch_score_array, -top_k, axis=0)[
        -top_k:
    ].mean(axis=0)
    global_aneurysm_score = float(
        np.mean(sorted(aneurysm_patch_scores)[-top_k:])
    )
    segmentation = best_class.copy()
    segmentation[
        (binary_probability < mask_threshold) | (class_confidence < class_threshold)
    ] = 0
    segmentation = segmentation[crop]
    binary_probability = binary_probability[crop]
    class_confidence = class_confidence[crop]
    task1 = {
        int(value) for value in np.flatnonzero(global_scores >= presence_threshold) + 1
    }
    diagnostics = {
        "binary_probability": binary_probability,
        "class_confidence": class_confidence,
        "global_location_scores": global_scores,
        "global_aneurysm_score": np.asarray(global_aneurysm_score, dtype=np.float32),
        "presence_top_k": np.asarray(top_k, dtype=np.int64),
        "ensemble_members": np.asarray(members, dtype=np.int64),
    }
    return segmentation, sorted(task1), diagnostics
