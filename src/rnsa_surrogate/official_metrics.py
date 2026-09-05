"""TopAneu-26 metrics ported from the official challenge evaluators.

Reference implementations:
https://github.com/Bangulli/TopAneu-26/tree/main/eval
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

N_CLASSES = 52
EPSILON = 1e-6


def _validate_labels(labels: Sequence[int]) -> list[int]:
    values = [int(value) for value in labels]
    invalid = sorted({value for value in values if not 1 <= value <= N_CLASSES})
    if invalid:
        raise ValueError(f"Labels must be in [1, {N_CLASSES}], got {invalid}")
    return values


def _location_counts(labels: Sequence[int]) -> np.ndarray:
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for label in _validate_labels(labels):
        counts[label - 1] += 1
    return counts


def task1_case_counts(
    ground_truth: Sequence[int], prediction: Sequence[int]
) -> np.ndarray:
    """Return official per-class TP/FP/FN/TN for one Task 1 case."""
    gt = _location_counts(ground_truth)
    pred = _location_counts(prediction)
    tp = np.minimum(pred, gt)
    fp = np.maximum(0, pred - gt)
    fn = np.maximum(0, gt - pred)
    tn = int(gt.sum()) - (tp + fn)
    return np.stack((tp, fp, fn, tn), axis=1)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask)


def dice_score(first: np.ndarray, second: np.ndarray) -> float:
    if not np.any(first) and not np.any(second):
        return 1.0
    if not np.any(first) or not np.any(second):
        return 0.0
    intersection = np.bitwise_and(first, second).sum()
    return float(2 * intersection / (first.sum() + second.sum() + EPSILON))


def volumetric_similarity(first: np.ndarray, second: np.ndarray) -> float:
    if not np.any(first) and not np.any(second):
        return 1.0
    if not np.any(first) or not np.any(second):
        return 0.0
    first_size, second_size = int(first.sum()), int(second.sum())
    return float(
        1.0 - abs(first_size - second_size) / (first_size + second_size + EPSILON)
    )


def normalized_hd95(first: np.ndarray, second: np.ndarray) -> float:
    """Official voxel-coordinate HD95 normalized by the volume diagonal."""
    diagonal = float(np.linalg.norm(first.shape))
    if not np.any(first) and not np.any(second):
        return 0.0
    if not np.any(first) or not np.any(second):
        return 1.0
    first_coordinates = np.argwhere(_surface(first))
    second_coordinates = np.argwhere(_surface(second))
    first_to_second = cKDTree(second_coordinates).query(first_coordinates)[0]
    second_to_first = cKDTree(first_coordinates).query(second_coordinates)[0]
    distance = max(
        np.percentile(first_to_second, 95),
        np.percentile(second_to_first, 95),
    )
    return float(distance / diagonal)


def task2_case_metrics(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return official Task 2 counts and DSC/HD95/VS sums for one case."""
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: GT={ground_truth.shape}; prediction={prediction.shape}"
        )
    for name, array in (("ground truth", ground_truth), ("prediction", prediction)):
        invalid = np.unique(array[(array < 0) | (array > N_CLASSES)])
        if invalid.size:
            raise ValueError(f"Invalid {name} labels: {invalid.tolist()}")

    counts = np.zeros((N_CLASSES, 4), dtype=np.int64)
    segmentation = np.zeros((N_CLASSES, 3), dtype=np.float64)
    ground_truth_classes = {int(value) for value in np.unique(ground_truth) if value}
    prediction_classes = {int(value) for value in np.unique(prediction) if value}
    number_of_aneurysms = len(ground_truth_classes)
    counts[:, 3] = number_of_aneurysms

    # Classes absent from both volumes are official true negatives with zero
    # segmentation contribution. Avoid creating 104 full-volume boolean arrays
    # per case when only a handful of location classes are normally present.
    for class_id in sorted(ground_truth_classes | prediction_classes):
        class_index = class_id - 1
        in_ground_truth = class_id in ground_truth_classes
        in_prediction = class_id in prediction_classes
        gt_mask = ground_truth == class_id
        prediction_mask = prediction == class_id
        true_positive = bool(
            in_ground_truth
            and in_prediction
            and np.bitwise_and(gt_mask, prediction_mask).sum() > 0
        )
        tp = int(true_positive)
        fp = int(in_prediction and not true_positive)
        fn = int(in_ground_truth and not true_positive)
        tn = number_of_aneurysms - (tp + fn)
        counts[class_index] = tp, fp, fn, tn

        if in_ground_truth or in_prediction:
            segmentation[class_index] = (
                dice_score(gt_mask, prediction_mask),
                normalized_hd95(gt_mask, prediction_mask),
                volumetric_similarity(gt_mask, prediction_mask),
            )
    return counts, segmentation


def task2_case_counts(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> np.ndarray:
    """Return official Task 2 detection counts without costly surface metrics."""
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: GT={ground_truth.shape}; prediction={prediction.shape}"
        )
    counts = np.zeros((N_CLASSES, 4), dtype=np.int64)
    ground_truth_classes = {int(value) for value in np.unique(ground_truth) if value}
    prediction_classes = {int(value) for value in np.unique(prediction) if value}
    number_of_aneurysms = len(ground_truth_classes)
    counts[:, 3] = number_of_aneurysms
    for class_id in sorted(ground_truth_classes | prediction_classes):
        class_index = class_id - 1
        in_ground_truth = class_id in ground_truth_classes
        in_prediction = class_id in prediction_classes
        true_positive = bool(
            in_ground_truth
            and in_prediction
            and np.any(
                (ground_truth == class_id) & (prediction == class_id)
            )
        )
        tp = int(true_positive)
        fp = int(in_prediction and not true_positive)
        fn = int(in_ground_truth and not true_positive)
        tn = number_of_aneurysms - (tp + fn)
        counts[class_index] = tp, fp, fn, tn
    return counts


def _classification_summary(counts: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, dict[str, float | int]] = {}
    precision_values, recall_values, mcc_values = [], [], []
    for class_index, (tp, fp, fn, tn) in enumerate(counts, start=1):
        tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
        precision = tp / (tp + fp + EPSILON)
        recall = tp / (tp + fn + EPSILON)
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + EPSILON
        mcc = (tp * tn - fn * fp) / denominator
        precision_values.append(precision)
        recall_values.append(recall)
        mcc_values.append(mcc)
        per_class[str(class_index)] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": float(precision),
            "recall": float(recall),
            "mcc": float(mcc),
        }
    return {
        "macro": {
            "precision": float(np.mean(precision_values)),
            "recall": float(np.mean(recall_values)),
            "mcc": float(np.mean(mcc_values)),
        },
        "per_class": per_class,
    }


def summarize_task1(case_counts: Sequence[np.ndarray]) -> dict[str, Any]:
    if not case_counts:
        raise ValueError("At least one Task 1 case is required")
    counts = np.sum(np.stack(case_counts), axis=0)
    return _classification_summary(counts)


def summarize_task2(
    case_counts: Sequence[np.ndarray], case_segmentation: Sequence[np.ndarray]
) -> dict[str, Any]:
    if not case_counts or len(case_counts) != len(case_segmentation):
        raise ValueError("Task 2 counts and segmentation metrics must align")
    counts = np.sum(np.stack(case_counts), axis=0)
    segmentation_sums = np.sum(np.stack(case_segmentation), axis=0)
    summary = _classification_summary(counts)
    macro_dice, macro_hd95, macro_volsim = [], [], []
    for class_index in range(N_CLASSES):
        denominator = float(counts[class_index, :3].sum()) + EPSILON
        dice, hd95, volsim = segmentation_sums[class_index] / denominator
        record = summary["per_class"][str(class_index + 1)]
        record.update(
            {
                "dice": float(dice),
                "hd95": float(hd95),
                "volumetric_similarity": float(volsim),
            }
        )
        macro_dice.append(dice)
        macro_hd95.append(hd95)
        macro_volsim.append(volsim)
    summary["macro"].update(
        {
            "dice": float(np.mean(macro_dice)),
            "hd95": float(np.mean(macro_hd95)),
            "volumetric_similarity": float(np.mean(macro_volsim)),
        }
    )
    return summary
