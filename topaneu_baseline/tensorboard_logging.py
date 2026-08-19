from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

from .challenge_metrics import ClassMetrics


def challenge_metric_scalars(
    split: str,
    summary: Mapping[str, object],
    per_class: Sequence[ClassMetrics],
    *,
    stage: str = "final",
) -> dict[str, float]:
    """Flatten finite challenge metrics into stable TensorBoard tag names."""
    prefix = f"{split}/{stage}"
    scalars: dict[str, float] = {}
    for group_name in ("macro", "micro_classification"):
        group = summary.get(group_name)
        if not isinstance(group, Mapping):
            continue
        for name, value in group.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                scalars[f"{prefix}/{group_name}/{name}"] = float(value)

    field_names = {
        "precision": "Precision",
        "recall": "Recall",
        "mcc": "MCC",
        "dice": "Dice",
        "volsim": "VolSim",
        "hd95": "HD95",
    }
    for item in per_class:
        for field_name, display_name in field_names.items():
            value = float(getattr(item, field_name))
            if math.isfinite(value):
                scalars[f"{prefix}/per_class/class_{item.label:02d}/{display_name}"] = value
    return scalars


def write_challenge_metrics_tensorboard(
    log_dir: str | Path,
    split: str,
    summary: Mapping[str, object],
    per_class: Sequence[ClassMetrics],
    *,
    step: int = 0,
) -> None:
    """Append final full-volume challenge metrics to a TensorBoard event file."""
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        for tag, value in challenge_metric_scalars(split, summary, per_class).items():
            writer.add_scalar(tag, value, global_step=step)
        writer.flush()
    finally:
        writer.close()
