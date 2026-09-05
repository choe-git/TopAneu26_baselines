"""Losses for sparse aneurysms and auxiliary vessel anatomy."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

# Exact location labels are grouped without changing the model's 53-way output.
# Group logits are obtained with logsumexp, which is equivalent to summing the
# member probabilities after the original softmax.
TERRITORY_GROUPS: tuple[tuple[int, ...], ...] = (
    (0,),
    tuple(range(1, 18)),  # vertebrobasilar
    tuple(range(18, 22)),  # PCA
    tuple(range(22, 36)),  # ICA
    tuple(range(36, 45)),  # ACA / Acom
    tuple(range(45, 53)),  # MCA
)
RIGHT_LABELS = (
    1,
    3,
    5,
    9,
    11,
    13,
    15,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    37,
    39,
    41,
    43,
    45,
    47,
    49,
    51,
)
LEFT_LABELS = (
    2,
    4,
    6,
    10,
    12,
    14,
    16,
    19,
    21,
    23,
    25,
    27,
    29,
    31,
    33,
    35,
    38,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
)
MIDLINE_LABELS = (7, 8, 17, 36)
SIDE_GROUPS: tuple[tuple[int, ...], ...] = (
    (0,),
    RIGHT_LABELS,
    LEFT_LABELS,
    MIDLINE_LABELS,
)


def _label_lookup(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    lookup = [0] * 53
    for group_index, labels in enumerate(groups):
        for label in labels:
            lookup[label] = group_index
    return tuple(lookup)


TERRITORY_LOOKUP = _label_lookup(TERRITORY_GROUPS)
SIDE_LOOKUP = _label_lookup(SIDE_GROUPS)


def _aggregate_class_logits(
    logits: torch.Tensor, groups: tuple[tuple[int, ...], ...]
) -> torch.Tensor:
    return torch.stack(
        [torch.logsumexp(logits[:, labels], dim=1) for labels in groups], dim=1
    )


def _group_voxel_target(target: torch.Tensor, lookup: tuple[int, ...]) -> torch.Tensor:
    lookup_tensor = torch.as_tensor(lookup, device=target.device, dtype=torch.long)
    return lookup_tensor[target]


def _balanced_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(target.flatten(), minlength=logits.shape[1]).float()
    class_weights = (counts + 1.0).rsqrt()
    class_weights[0] *= 0.05
    return F.cross_entropy(logits, target, weight=class_weights)


def _foreground_location_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Classify location only where an aneurysm exists.

    Background is handled by the independent binary decoder. This prevents the
    52 rare location classes from competing with hundreds of thousands of
    background voxels in the same softmax.
    """
    foreground = target > 0
    if not torch.any(foreground):
        return logits.sum() * 0.0
    foreground_logits = logits[:, 1:53].movedim(1, -1)[foreground]
    foreground_target = target[foreground] - 1
    return F.cross_entropy(foreground_logits, foreground_target)


def _positive_weighted_bce(
    logits: torch.Tensor, target: torch.Tensor, positive_weight: float = 10.0
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=logits.new_tensor(positive_weight),
    )


def _group_presence(
    logits: torch.Tensor,
    target: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine exact-label presence into differentiable group-level presence."""
    grouped_logits = []
    grouped_target = []
    for labels in groups:
        indices = [label - 1 for label in labels]
        member_logits = logits[:, indices].float()
        # Log-mean-exp is a smooth maximum whose zero point does not depend on
        # the number of exact labels in the group.
        normalizer = member_logits.new_tensor(len(indices)).log()
        grouped_logits.append(torch.logsumexp(member_logits, dim=1) - normalizer)
        grouped_target.append(target[:, indices].amax(dim=1))
    return torch.stack(grouped_logits, dim=1), torch.stack(grouped_target, dim=1)


def _hierarchy_weight(
    weights: dict[str, float],
    name: str,
    default: float,
    legacy_name: str,
    legacy_fraction: float,
) -> float:
    if name in weights:
        return weights[name]
    if legacy_name in weights:
        return weights[legacy_name] * legacy_fraction
    return default


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    axes = tuple(range(2, logits.ndim))
    intersection = (probability * target).sum(axes)
    denominator = probability.sum(axes) + target.sum(axes)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def focal_tversky_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    axes = tuple(range(2, logits.ndim))
    true_positive = (probability * target).sum(axes)
    false_positive = (probability * (1.0 - target)).sum(axes)
    false_negative = ((1.0 - probability) * target).sum(axes)
    score = (true_positive + 1.0) / (
        true_positive + 0.3 * false_positive + 0.7 * false_negative + 1.0
    )
    return (1.0 - score).pow(gamma).mean()


def asymmetric_multilabel_loss(
    logits: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
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
    target = F.interpolate(
        target[:, None].float(), size=logits.shape[2:], mode="nearest"
    )[:, 0].long()
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
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))[
        valid_cases > 0
    ].mean()
    return cross_entropy + dice


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or {}
    binary_target = (batch["location"] > 0).float()[:, None]
    binary = focal_tversky_loss(outputs["aneurysm_logits"], binary_target)
    binary += soft_dice_loss(outputs["aneurysm_logits"], binary_target)
    binary += _positive_weighted_bce(
        outputs["aneurysm_logits"],
        binary_target,
        positive_weight=float(weights.get("aneurysm_positive_weight", 10.0)),
    )

    flat_location = batch["location"].long()
    exact_location = _foreground_location_cross_entropy(
        outputs["location_logits"], flat_location
    )
    location_territory_weight = _hierarchy_weight(
        weights, "location_territory", 0.15, "location_seg", 0.30
    )
    location_side_weight = _hierarchy_weight(
        weights, "location_side", 0.10, "location_seg", 0.20
    )
    zero = outputs["location_logits"].sum() * 0.0
    territory_location = zero
    if location_territory_weight > 0:
        territory_location = _balanced_cross_entropy(
            _aggregate_class_logits(outputs["location_logits"], TERRITORY_GROUPS),
            _group_voxel_target(flat_location, TERRITORY_LOOKUP),
        )
    side_location = zero
    if location_side_weight > 0:
        side_location = _balanced_cross_entropy(
            _aggregate_class_logits(outputs["location_logits"], SIDE_GROUPS),
            _group_voxel_target(flat_location, SIDE_LOOKUP),
        )
    vessel = vessel_loss(
        outputs["vessel_logits"], batch["vessel"], batch.get("vessel_valid")
    )
    exact_location_presence = _positive_weighted_bce(
        outputs["location_presence_logits"],
        batch["location_presence"].to(outputs["location_presence_logits"].dtype),
        positive_weight=float(weights.get("location_presence_positive_weight", 4.0)),
    )
    presence_territory_weight = _hierarchy_weight(
        weights,
        "location_presence_territory",
        0.03,
        "location_presence",
        0.30,
    )
    presence_side_weight = _hierarchy_weight(
        weights,
        "location_presence_side",
        0.02,
        "location_presence",
        0.20,
    )
    territory_location_presence = zero
    if presence_territory_weight > 0:
        territory_presence_logits, territory_presence_target = _group_presence(
            outputs["location_presence_logits"],
            batch["location_presence"],
            TERRITORY_GROUPS[1:],
        )
        territory_location_presence = asymmetric_multilabel_loss(
            territory_presence_logits, territory_presence_target
        )
    side_location_presence = zero
    if presence_side_weight > 0:
        side_presence_logits, side_presence_target = _group_presence(
            outputs["location_presence_logits"],
            batch["location_presence"],
            SIDE_GROUPS[1:],
        )
        side_location_presence = asymmetric_multilabel_loss(
            side_presence_logits, side_presence_target
        )
    presence = F.binary_cross_entropy_with_logits(
        outputs["aneurysm_presence_logits"].flatten(),
        batch["aneurysm_presence"].float(),
    )
    component_location = zero
    if "component_location_logits" in outputs:
        component_target = batch["component_location"].long()
        component_values = F.cross_entropy(
            outputs["component_location_logits"],
            component_target,
            reduction="none",
            label_smoothing=float(
                weights.get("component_location_label_smoothing", 0.05)
            ),
        )
        component_weights = torch.where(
            component_target > 0,
            torch.ones_like(component_values),
            torch.full_like(
                component_values,
                float(weights.get("component_location_negative_weight", 0.25)),
            ),
        )
        component_location = (
            (component_values * component_weights).sum()
            / component_weights.sum().clamp_min(1.0)
        )

    sphere = zero
    has_sphere_output = "sphere_logits" in outputs
    has_sphere_target = "aneurysm_sphere" in batch
    if has_sphere_output != has_sphere_target:
        raise ValueError("sphere output and target must either both be present or absent")
    if has_sphere_output:
        sphere_target = batch["aneurysm_sphere"].float()[:, None]
        sphere = _positive_weighted_bce(
            outputs["sphere_logits"],
            sphere_target,
            positive_weight=float(weights.get("sphere_positive_weight", 10.0)),
        )
        sphere += focal_tversky_loss(
            outputs["sphere_logits"],
            sphere_target,
            gamma=float(weights.get("sphere_focal_gamma", 4.0 / 3.0)),
        )

    total = (
        weights.get("aneurysm", 1.0) * binary
        + _hierarchy_weight(weights, "location_exact", 0.25, "location_seg", 0.50)
        * exact_location
        + location_territory_weight * territory_location
        + location_side_weight * side_location
        + weights.get("vessel", 0.1) * vessel
        + _hierarchy_weight(
            weights,
            "location_presence_exact",
            0.05,
            "location_presence",
            0.50,
        )
        * exact_location_presence
        + presence_territory_weight * territory_location_presence
        + presence_side_weight * side_location_presence
        + weights.get("aneurysm_presence", 0.05) * presence
        + weights.get("component_location", 0.0) * component_location
        + weights.get("sphere", 0.0) * sphere
    )
    values: dict[str, Any] = {
        "aneurysm": binary,
        "location_exact": exact_location,
        "location_territory": territory_location,
        "location_side": side_location,
        "vessel": vessel,
        "location_presence_exact": exact_location_presence,
        "location_presence_territory": territory_location_presence,
        "location_presence_side": side_location_presence,
        "aneurysm_presence": presence,
        "component_location": component_location,
        "sphere": sphere,
        "total": total,
    }
    return total, {name: float(value.detach()) for name, value in values.items()}
