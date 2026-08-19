"""Evaluate TopAneu predictions with the official Task 1 and Task 2 metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

N_CLASSES = 52
EPS = 1e-6


def load_mask(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def hd95(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Official normalized HD95: voxel distance divided by the volume diagonal."""
    if not reference.any() or not prediction.any():
        return 1.0

    reference_surface = reference & ~binary_erosion(reference)
    prediction_surface = prediction & ~binary_erosion(prediction)
    reference_points = np.argwhere(reference_surface)
    prediction_points = np.argwhere(prediction_surface)
    ref_to_pred = cKDTree(prediction_points).query(reference_points)[0]
    pred_to_ref = cKDTree(reference_points).query(prediction_points)[0]
    distance = max(np.percentile(ref_to_pred, 95), np.percentile(pred_to_ref, 95))
    return float(distance / np.linalg.norm(reference.shape))


def segmentation_scores(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    reference_size = int(reference.sum())
    prediction_size = int(prediction.sum())
    if reference_size == 0 or prediction_size == 0:
        return 0.0, 0.0, 1.0

    intersection = int((reference & prediction).sum())
    dice = 2 * intersection / (reference_size + prediction_size + EPS)
    volume_similarity = 1 - abs(reference_size - prediction_size) / (
        reference_size + prediction_size + EPS
    )
    return float(dice), float(volume_similarity), hd95(reference, prediction)


def classification_scores(counts: np.ndarray) -> dict[str, list[float] | float]:
    tp, fp, fn, tn = counts.T
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + EPS
    mcc = (tp * tn - fn * fp) / denominator
    return {
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "mcc": float(mcc.mean()),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "mcc_per_class": mcc.tolist(),
    }


def evaluate_predictions(prediction_dir: Path, reference_dir: Path, cases: list[str]) -> dict:
    """Return 52-class averages using the challenge's official aggregation rules."""
    task1_counts = np.zeros((N_CLASSES, 4), dtype=np.float64)
    task2_counts = np.zeros((N_CLASSES, 4), dtype=np.float64)
    segmentation = np.zeros((N_CLASSES, 3), dtype=np.float64)

    for case in cases:
        reference = load_mask(reference_dir / f"{case}.nii.gz")
        prediction = load_mask(prediction_dir / f"{case}.nii.gz")
        if reference.shape != prediction.shape:
            raise ValueError(f"Shape mismatch for {case}: {reference.shape} != {prediction.shape}")

        reference_labels = set(np.unique(reference).tolist()) - {0}
        prediction_labels = set(np.unique(prediction).tolist()) - {0}
        n_aneurysms = len(reference_labels)

        for label in range(1, N_CLASSES + 1):
            index = label - 1
            in_reference = label in reference_labels
            in_prediction = label in prediction_labels

            task1_tp = int(in_reference and in_prediction)
            task1_fp = int(in_prediction and not in_reference)
            task1_fn = int(in_reference and not in_prediction)
            task1_counts[index] += (task1_tp, task1_fp, task1_fn, n_aneurysms - task1_tp - task1_fn)

            reference_class = reference == label
            prediction_class = prediction == label
            task2_tp = int(in_reference and in_prediction and (reference_class & prediction_class).any())
            task2_fp = int(in_prediction and not task2_tp)
            task2_fn = int(in_reference and not task2_tp)
            task2_counts[index] += (task2_tp, task2_fp, task2_fn, n_aneurysms - task2_tp - task2_fn)

            if in_reference or in_prediction:
                segmentation[index] += segmentation_scores(reference_class, prediction_class)

    task1 = classification_scores(task1_counts)
    task2 = classification_scores(task2_counts)
    evaluated = task2_counts[:, :3].sum(axis=1) + EPS
    dice = segmentation[:, 0] / evaluated
    volume_similarity = segmentation[:, 1] / evaluated
    distance = segmentation[:, 2] / evaluated
    task2.update({
        "dice": float(dice.mean()),
        "volume_similarity": float(volume_similarity.mean()),
        "hd95": float(distance.mean()),
        "dice_per_class": dice.tolist(),
        "volume_similarity_per_class": volume_similarity.tolist(),
        "hd95_per_class": distance.tolist(),
    })
    return {"task1": task1, "task2": task2}


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2) + "\n")


def metric_summary(metrics: dict) -> dict:
    """Remove per-class arrays for concise TensorBoard and terminal output."""
    return {
        task: {name: value for name, value in values.items() if not name.endswith("_per_class")}
        for task, values in metrics.items()
    }
