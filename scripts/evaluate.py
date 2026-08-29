"""Evaluate one cached split with a trained RNSA surrogate checkpoint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from rnsa_surrogate.cache import (
    atomic_json_dump,
    atomic_save_npy,
    load_cache_index,
    sha256_file,
)
from rnsa_surrogate.inference import sliding_window_predict
from rnsa_surrogate.model import RNSASurrogate
from rnsa_surrogate.run_layout import BaselineRunLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="Shared experiment root"
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--checkpoint", type=Path, help="Override baseline/checkpoint_best.pth"
    )
    parser.add_argument(
        "--output", type=Path, help="Override baseline/evaluation/SPLIT"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.45)
    parser.add_argument("--class-threshold", type=float, default=0.15)
    parser.add_argument("--presence-threshold", type=float, default=0.35)
    parser.add_argument("--min-component-voxels", type=int, default=3)
    parser.add_argument("--component-probability-threshold", type=float, default=0.55)
    parser.add_argument("--component-class-threshold", type=float, default=0.25)
    parser.add_argument("--component-top-fraction", type=float, default=0.25)
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0,
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "iou": safe_divide(tp, tp + fp + fn),
    }


def task1_counts(truth: set[int], prediction: set[int]) -> tuple[int, int, int, int]:
    labels = set(range(1, 53))
    return (
        len(truth & prediction),
        len(prediction - truth),
        len(truth - prediction),
        len(labels - truth - prediction),
    )


def confusion_metrics(confusion: np.ndarray) -> dict[str, Any]:
    total = int(confusion.sum())
    per_class: dict[str, dict[str, float | int]] = {}
    active = []
    for class_id in range(1, confusion.shape[0]):
        tp = int(confusion[class_id, class_id])
        fp = int(confusion[:, class_id].sum() - tp)
        fn = int(confusion[class_id, :].sum() - tp)
        tn = total - tp - fp - fn
        if tp + fp + fn > 0:
            active.append(class_id)
            per_class[str(class_id)] = binary_metrics(tp, fp, fn, tn)

    macro = {
        name: float(np.mean([per_class[str(label)][name] for label in active]))
        if active
        else 0.0
        for name in ("precision", "recall", "mcc", "dice", "iou")
    }
    true_counts = confusion.sum(axis=1).astype(np.float64)
    predicted_counts = confusion.sum(axis=0).astype(np.float64)
    correct = float(np.trace(confusion))
    samples = float(confusion.sum())
    numerator = correct * samples - float(np.dot(true_counts, predicted_counts))
    denominator = math.sqrt(
        max(samples**2 - float(np.dot(predicted_counts, predicted_counts)), 0.0)
        * max(samples**2 - float(np.dot(true_counts, true_counts)), 0.0)
    )
    return {
        "active_classes": active,
        "macro": macro,
        "multiclass_mcc": numerator / denominator if denominator > 0 else 0.0,
        "voxel_accuracy": safe_divide(correct, samples),
        "per_class": per_class,
    }


def lesion_counts(
    prediction: np.ndarray, instances: np.ndarray
) -> tuple[int, int, int]:
    ground_truth_ids = {int(value) for value in np.unique(instances) if value > 0}
    recalled_ids = {
        int(value) for value in np.unique(instances[prediction > 0]) if value > 0
    }
    predicted_components, predicted_count = ndimage.label(
        prediction > 0, structure=ndimage.generate_binary_structure(3, 1)
    )
    false_positives = 0
    for component_id in range(1, predicted_count + 1):
        if not np.any(instances[predicted_components == component_id] > 0):
            false_positives += 1
    return len(recalled_ids), len(ground_truth_ids), false_positives


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[RNSASurrogate, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage", "baseline") != "baseline":
        raise ValueError(f"Not a baseline checkpoint: {checkpoint.get('stage')}")
    config = checkpoint["config"]
    model = RNSASurrogate(**config["model"])
    state = dict(checkpoint["model"])
    if "ema" in checkpoint:
        for name, value in checkpoint["ema"]["shadow"].items():
            state[name] = value.to(dtype=state[name].dtype)
    model.load_state_dict(state)
    return model.to(device).eval(), config


def main() -> None:
    args = parse_args()
    layout = BaselineRunLayout.from_root(args.run_dir)
    checkpoint_path = (args.checkpoint or layout.checkpoint).resolve()
    output_dir = (args.output or layout.baseline / "evaluation" / args.split).resolve()
    metrics_path = output_dir / "metrics.json"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation already completed: {metrics_path}")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    model, config = load_model(checkpoint_path, device)
    amp_name = str(config["train"].get("amp", "none"))
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
        "none": None,
    }[amp_name]
    if device.type != "cuda":
        amp_dtype = None

    cache_path = layout.cache
    if not (cache_path / "index.json").is_file():
        inputs_path = layout.baseline / "inputs.json"
        if not inputs_path.is_file():
            raise FileNotFoundError(
                f"Missing cache and training inputs metadata: {cache_path}"
            )
        cache_path = Path(
            json.loads(inputs_path.read_text(encoding="utf-8"))["cache_index"]
        ).parent
    cache_index = load_cache_index(cache_path)
    cases = [case for case in cache_index["cases"] if case["split"] == args.split]
    if not cases:
        raise ValueError(f"Cache contains no {args.split!r} cases")
    cache_root = Path(cache_index["index_path"]).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    location_dir = output_dir / "locations"

    task1_totals = np.zeros(4, dtype=np.int64)
    location_confusion = np.zeros((53, 53), dtype=np.int64)
    binary_totals = np.zeros(4, dtype=np.int64)
    lesion_recalled = lesion_total = lesion_false_positives = 0
    per_case = []

    for case in tqdm(cases, desc=f"Evaluating {args.split}"):
        case_id = case["case_id"]
        case_dir = cache_root / case["cache_dir"]
        image = np.load(case_dir / "image.npy", mmap_mode="r").astype(np.float32)
        truth = np.load(case_dir / "location.npy", mmap_mode="r")
        instances = np.load(case_dir / "instances.npy", mmap_mode="r")
        prediction, predicted_locations, _ = sliding_window_predict(
            model,
            image,
            case["modality"],
            config["data"]["patch_size"],
            device,
            overlap=args.overlap,
            amp_dtype=amp_dtype,
            mask_threshold=args.mask_threshold,
            class_threshold=args.class_threshold,
            presence_threshold=args.presence_threshold,
            min_component_voxels=args.min_component_voxels,
            component_probability_threshold=args.component_probability_threshold,
            component_class_threshold=args.component_class_threshold,
            component_top_fraction=args.component_top_fraction,
        )
        truth_array = np.asarray(truth, dtype=np.uint8)
        prediction = np.asarray(prediction, dtype=np.uint8)
        truth_locations = {int(value) for value in np.unique(truth_array) if value > 0}
        predicted_location_set = set(predicted_locations)
        task1_case_counts = task1_counts(truth_locations, predicted_location_set)
        task1_totals += task1_case_counts

        location_confusion += np.bincount(
            truth_array.ravel().astype(np.int64) * 53 + prediction.ravel(),
            minlength=53 * 53,
        ).reshape(53, 53)
        truth_binary = truth_array > 0
        prediction_binary = prediction > 0
        tp = int(np.count_nonzero(truth_binary & prediction_binary))
        fp = int(np.count_nonzero(~truth_binary & prediction_binary))
        fn = int(np.count_nonzero(truth_binary & ~prediction_binary))
        tn = int(np.count_nonzero(~truth_binary & ~prediction_binary))
        binary_totals += (tp, fp, fn, tn)
        recalled, total, false_positives = lesion_counts(prediction, instances)
        lesion_recalled += recalled
        lesion_total += total
        lesion_false_positives += false_positives

        case_payload = {
            "case_id": case_id,
            "modality": case["modality"],
            "task1_truth": sorted(truth_locations),
            "task1_prediction": sorted(predicted_location_set),
            "task1": binary_metrics(*task1_case_counts),
            "task2_binary": binary_metrics(tp, fp, fn, tn),
            "lesions": {
                "recalled": recalled,
                "total": total,
                "false_positive_components": false_positives,
            },
        }
        per_case.append(case_payload)
        atomic_json_dump(per_case, output_dir / "per_case_metrics.json")
        if args.save_predictions:
            atomic_save_npy(prediction_dir / f"{case_id}.npy", prediction)
            atomic_json_dump(
                sorted(predicted_location_set), location_dir / f"{case_id}.json"
            )

    task1 = binary_metrics(*(int(value) for value in task1_totals))
    task2_binary = binary_metrics(*(int(value) for value in binary_totals))
    payload = {
        "split": args.split,
        "cases": len(cases),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "cache_index": cache_index["index_path"],
        "cache_index_sha256": sha256_file(cache_index["index_path"]),
        "thresholds": {
            "overlap": args.overlap,
            "mask": args.mask_threshold,
            "class": args.class_threshold,
            "presence": args.presence_threshold,
            "min_component_voxels": args.min_component_voxels,
            "component_probability": args.component_probability_threshold,
            "component_class": args.component_class_threshold,
            "component_top_fraction": args.component_top_fraction,
        },
        "task1_case_label": task1,
        "task2_binary_voxel": task2_binary,
        "task2_location_voxel": confusion_metrics(location_confusion),
        "lesion_detection": {
            "recalled": lesion_recalled,
            "total": lesion_total,
            "recall": safe_divide(lesion_recalled, lesion_total),
            "false_positive_components": lesion_false_positives,
            "false_positives_per_case": safe_divide(lesion_false_positives, len(cases)),
        },
    }
    atomic_json_dump(payload, metrics_path)
    print(f"Evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
