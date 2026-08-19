from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
from scipy import ndimage

from .challenge_io import case_id_from_path, locations_from_mask


@dataclass(frozen=True)
class ClassMetrics:
    label: int
    support: int
    predicted_positive: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    mcc: float
    dice: float
    volsim: float
    hd95: float


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return _safe_ratio(tn * tp - fn * fp, denominator)


def _nanmean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    return float(array[finite].mean()) if finite.any() else float("nan")


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    if not a.any() or not b.any():
        return np.empty(0, dtype=np.float64)

    union_points = np.argwhere(a | b)
    lower = np.maximum(union_points.min(axis=0) - 1, 0)
    upper = np.minimum(union_points.max(axis=0) + 2, np.asarray(a.shape))
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    a_crop = a[slices]
    b_crop = b[slices]

    structure = ndimage.generate_binary_structure(a.ndim, 1)
    a_surface = a_crop ^ ndimage.binary_erosion(a_crop, structure=structure, border_value=0)
    b_surface = b_crop ^ ndimage.binary_erosion(b_crop, structure=structure, border_value=0)
    distance_to_a = ndimage.distance_transform_edt(~a_surface, sampling=tuple(float(v) for v in spacing))
    distance_to_b = ndimage.distance_transform_edt(~b_surface, sampling=tuple(float(v) for v in spacing))
    return np.concatenate((distance_to_b[a_surface], distance_to_a[b_surface])).astype(np.float64, copy=False)


def segmentation_metrics(a: np.ndarray, b: np.ndarray, spacing: Sequence[float]) -> tuple[float, float, float]:
    a_count = int(a.sum())
    b_count = int(b.sum())
    denominator = a_count + b_count
    if denominator == 0:
        return float("nan"), float("nan"), float("nan")
    intersection = int(np.logical_and(a, b).sum())
    dice = 2.0 * intersection / denominator
    volsim = 1.0 - abs(a_count - b_count) / denominator
    distances = _surface_distances(a, b, spacing)
    hd95 = float(np.percentile(distances, 95)) if distances.size else float("nan")
    return float(dice), float(volsim), hd95


def _load_prediction_locations(
    case_id: str,
    mask: np.ndarray,
    spacing: Sequence[float],
    prediction_json_dir: Path | None,
    max_label: int,
    min_component_voxels: int,
    min_component_mm3: float,
) -> set[int]:
    if prediction_json_dir is not None:
        path = prediction_json_dir / f"{case_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Task 1 prediction must be a list: {path}")
        locations = {int(v) for v in payload}
        invalid = sorted(v for v in locations if not 1 <= v <= max_label)
        if invalid:
            raise ValueError(f"Invalid Task 1 labels in {path}: {invalid}")
        return locations
    return set(
        locations_from_mask(
            mask,
            max_label=max_label,
            min_component_voxels=min_component_voxels,
            min_component_mm3=min_component_mm3,
            spacing=spacing,
        )
    )


def evaluate_prediction_masks(
    prediction_paths: Sequence[str | Path],
    ground_truth_dir: str | Path,
    *,
    max_label: int,
    prediction_json_dir: str | Path | None = None,
    min_component_voxels: int = 1,
    min_component_mm3: float = 0.0,
) -> tuple[dict[str, object], list[ClassMetrics], list[dict[str, object]]]:
    ground_truth_dir = Path(ground_truth_dir)
    json_dir = Path(prediction_json_dir) if prediction_json_dir is not None else None
    predictions: dict[str, Path] = {}
    for raw_path in prediction_paths:
        path = Path(raw_path)
        case_id = case_id_from_path(path)
        if case_id in predictions:
            raise ValueError(f"Duplicate prediction for {case_id}: {predictions[case_id]} and {path}")
        predictions[case_id] = path
    if not predictions:
        raise ValueError("No prediction masks were provided")

    present_true = np.zeros((len(predictions), max_label), dtype=bool)
    present_pred = np.zeros_like(present_true)
    dice_values: list[list[float]] = [[] for _ in range(max_label)]
    volsim_values: list[list[float]] = [[] for _ in range(max_label)]
    hd95_values: list[list[float]] = [[] for _ in range(max_label)]
    per_case: list[dict[str, object]] = []

    for row, case_id in enumerate(sorted(predictions)):
        pred_image = nib.load(str(predictions[case_id]))
        truth_path = ground_truth_dir / f"{case_id}.nii.gz"
        if not truth_path.exists():
            raise FileNotFoundError(f"Missing ground-truth mask: {truth_path}")
        truth_image = nib.load(str(truth_path))
        if pred_image.shape != truth_image.shape:
            raise ValueError(f"Shape mismatch for {case_id}: {pred_image.shape} vs {truth_image.shape}")
        if not np.allclose(pred_image.affine, truth_image.affine, rtol=1e-5, atol=1e-4):
            raise ValueError(f"Affine mismatch for {case_id}")

        prediction = np.rint(np.asanyarray(pred_image.dataobj)).astype(np.uint8)
        truth = np.rint(np.asanyarray(truth_image.dataobj)).astype(np.uint8)
        if int(prediction.max(initial=0)) > max_label or int(truth.max(initial=0)) > max_label:
            raise ValueError(f"Label outside 0..{max_label} in {case_id}")
        spacing = tuple(float(v) for v in truth_image.header.get_zooms()[:3])
        true_locations = {int(v) for v in np.unique(truth) if int(v) > 0}
        pred_locations = _load_prediction_locations(
            case_id,
            prediction,
            spacing,
            json_dir,
            max_label,
            min_component_voxels,
            min_component_mm3,
        )
        for value in true_locations:
            present_true[row, value - 1] = True
        for value in pred_locations:
            present_pred[row, value - 1] = True

        case_dice: list[float] = []
        case_volsim: list[float] = []
        case_hd95: list[float] = []
        for label in sorted(true_locations | {int(v) for v in np.unique(prediction) if int(v) > 0}):
            dice, volsim, hd95 = segmentation_metrics(truth == label, prediction == label, spacing)
            dice_values[label - 1].append(dice)
            volsim_values[label - 1].append(volsim)
            hd95_values[label - 1].append(hd95)
            case_dice.append(dice)
            case_volsim.append(volsim)
            case_hd95.append(hd95)
        per_case.append(
            {
                "case_id": case_id,
                "true_locations": sorted(true_locations),
                "predicted_locations": sorted(pred_locations),
                "Dice": _nanmean(case_dice),
                "VolSim": _nanmean(case_volsim),
                "HD95": _nanmean(case_hd95),
            }
        )

    per_class: list[ClassMetrics] = []
    for index in range(max_label):
        y_true = present_true[:, index]
        y_pred = present_pred[:, index]
        tp = int(np.logical_and(y_true, y_pred).sum())
        fp = int(np.logical_and(~y_true, y_pred).sum())
        tn = int(np.logical_and(~y_true, ~y_pred).sum())
        fn = int(np.logical_and(y_true, ~y_pred).sum())
        per_class.append(
            ClassMetrics(
                label=index + 1,
                support=int(y_true.sum()),
                predicted_positive=int(y_pred.sum()),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                precision=_safe_ratio(tp, tp + fp),
                recall=_safe_ratio(tp, tp + fn),
                mcc=_mcc(tp, fp, tn, fn),
                dice=_nanmean(dice_values[index]),
                volsim=_nanmean(volsim_values[index]),
                hd95=_nanmean(hd95_values[index]),
            )
        )

    total_tp = int(np.logical_and(present_true, present_pred).sum())
    total_fp = int(np.logical_and(~present_true, present_pred).sum())
    total_tn = int(np.logical_and(~present_true, ~present_pred).sum())
    total_fn = int(np.logical_and(present_true, ~present_pred).sum())
    summary: dict[str, object] = {
        "num_cases": len(predictions),
        "num_classes": max_label,
        "macro": {
            "Precision": _nanmean(item.precision for item in per_class),
            "Recall": _nanmean(item.recall for item in per_class),
            "MCC": _nanmean(item.mcc for item in per_class),
            "Dice": _nanmean(item.dice for item in per_class),
            "VolSim": _nanmean(item.volsim for item in per_class),
            "HD95": _nanmean(item.hd95 for item in per_class),
        },
        "micro_classification": {
            "Precision": _safe_ratio(total_tp, total_tp + total_fp),
            "Recall": _safe_ratio(total_tp, total_tp + total_fn),
            "MCC": _mcc(total_tp, total_fp, total_tn, total_fn),
            "tp": total_tp,
            "fp": total_fp,
            "tn": total_tn,
            "fn": total_fn,
        },
        "undefined_values": "NaN values are excluded from macro means; HD95 is undefined when either mask is empty.",
    }
    return summary, per_class, per_case


def write_evaluation(
    output_dir: str | Path,
    summary: dict[str, object],
    per_class: Sequence[ClassMetrics],
    per_case: Sequence[dict[str, object]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(per_class[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in per_class)
    with (output_dir / "per_case.json").open("w", encoding="utf-8") as handle:
        json.dump(list(per_case), handle, indent=2, allow_nan=True)
        handle.write("\n")
