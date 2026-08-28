"""Losses for sparse aneurysms and auxiliary vessel anatomy."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    axes = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(axes)
    denominator = probability.sum(axes) + target.sum(axes)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def focal_tversky_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    axes = tuple(range(2, logits.ndim))
    true_positive = (probability * target).sum(axes)
    false_positive = (probability * (1.0 - target)).sum(axes)
    false_negative = ((1.0 - probability) * target).sum(axes)
    score = (true_positive + 1.0) / (
        true_positive + 0.3 * false_positive + 0.7 * false_negative + 1.0
    )
    return (1.0 - score).mean()


def asymmetric_multilabel_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(logits.dtype)
    positive = torch.sigmoid(logits)
    negative = (1.0 - positive + 0.05).clamp(max=1.0)
    likelihood = target * torch.log(positive.clamp_min(1e-8))
    likelihood += (1.0 - target) * torch.log(negative.clamp_min(1e-8))
    probability_target = positive * target + negative * (1.0 - target)
    focusing = (1.0 - probability_target).pow(4.0 * (1.0 - target))
    return -(focusing * likelihood).mean()


def vessel_loss(
    logits: torch.Tensor, target: torch.Tensor, valid_cases: torch.Tensor | None = None
) -> torch.Tensor:
    target = F.interpolate(target[:, None].float(), size=logits.shape[2:], mode="nearest")[
        :, 0
    ].long()
    if valid_cases is None:
        valid_cases = torch.ones(target.shape[0], device=target.device)
    valid_cases = valid_cases.to(logits.dtype)
    valid_target = target[valid_cases > 0]
    if valid_target.numel() == 0:
        return logits.sum() * 0.0
    counts = torch.bincount(valid_target.flatten(), minlength=logits.shape[1]).float()
    weights = (counts + 1.0).rsqrt()
    weights[0] *= 0.05
    per_voxel = F.cross_entropy(logits, target, weight=weights, reduction="none")
    case_mask = valid_cases.view(-1, 1, 1, 1)
    cross_entropy = (per_voxel * case_mask).sum() / (
        case_mask.sum() * per_voxel.shape[1] * per_voxel.shape[2] * per_voxel.shape[3]
    ).clamp_min(1.0)
    foreground_probability = 1.0 - torch.softmax(logits.float(), dim=1)[:, :1]
    foreground_logits = torch.logit(foreground_probability.clamp(1e-5, 1.0 - 1e-5))
    foreground_target = (target > 0).float()[:, None]
    foreground_probability = torch.sigmoid(foreground_logits) * case_mask[:, None]
    foreground_target = foreground_target * case_mask[:, None]
    axes = (2, 3, 4)
    intersection = (foreground_probability * foreground_target).sum(axes)
    denominator = foreground_probability.sum(axes) + foreground_target.sum(axes)
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))[valid_cases > 0].mean()
    return cross_entropy + dice


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    binary_target = (batch["location"] > 0).float()[:, None]
    binary = focal_tversky_loss(outputs["aneurysm_logits"], binary_target)
    binary += F.binary_cross_entropy_with_logits(outputs["aneurysm_logits"], binary_target)

    flat_location = batch["location"].long()
    counts = torch.bincount(
        flat_location.flatten(), minlength=outputs["location_logits"].shape[1]
    ).float()
    class_weights = (counts + 1.0).rsqrt()
    class_weights[0] *= 0.05
    location = F.cross_entropy(outputs["location_logits"], flat_location, weight=class_weights)
    vessel = vessel_loss(outputs["vessel_logits"], batch["vessel"], batch.get("vessel_valid"))
    location_presence = asymmetric_multilabel_loss(
        outputs["location_presence_logits"], batch["location_presence"]
    )
    presence = F.binary_cross_entropy_with_logits(
        outputs["aneurysm_presence_logits"].flatten(), batch["aneurysm_presence"].float()
    )

    total = (
        weights.get("aneurysm", 1.0) * binary
        + weights.get("location_seg", 0.5) * location
        + weights.get("vessel", 0.1) * vessel
        + weights.get("location_presence", 0.1) * location_presence
        + weights.get("aneurysm_presence", 0.05) * presence
    )
    values: dict[str, Any] = {
        "aneurysm": binary,
        "location_seg": location,
        "vessel": vessel,
        "location_presence": location_presence,
        "aneurysm_presence": presence,
        "total": total,
    }
    return total, {name: float(value.detach()) for name, value in values.items()}
