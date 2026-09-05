"""Estimate the location-classification value of vessel anatomy on held-out folds."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from rnsa_surrogate.cache import atomic_json_dump, load_cache_index
from rnsa_surrogate.official_metrics import summarize_task1, task1_case_counts
from rnsa_surrogate.run_layout import BaselineRunLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def component_feature(
    case_dir: Path, component: dict[str, Any], shape: tuple[int, int, int]
) -> np.ndarray:
    vessel = np.load(case_dir / "vessel.npy", mmap_mode="r")
    instances = np.load(case_dir / "instances.npy", mmap_mode="r")
    lower = np.asarray(component["bbox_zyx"][0], dtype=int)
    upper = np.asarray(component["bbox_zyx"][1], dtype=int)
    margin = 12
    crop_lower = np.maximum(lower - margin, 0)
    crop_upper = np.minimum(upper + margin, np.asarray(shape))
    crop = tuple(
        slice(int(start), int(stop))
        for start, stop in zip(crop_lower, crop_upper, strict=True)
    )
    component_mask = instances[crop] == int(component["instance_id"])
    vessel_crop = np.asarray(vessel[crop])
    distance = distance_transform_edt(~component_mask)
    anatomy = []
    for radius in (2.0, 6.0, 12.0):
        region = distance <= radius
        labels = vessel_crop[region]
        counts = np.bincount(labels, minlength=37).astype(np.float64)
        foreground = counts[1:37]
        anatomy.extend((foreground / max(foreground.sum(), 1.0)).tolist())
        anatomy.append(float(foreground.sum() / max(labels.size, 1)))

    center = np.asarray(component["center_zyx"], dtype=np.float64)
    normalized_center = 2.0 * center / np.maximum(np.asarray(shape) - 1, 1) - 1.0
    extent = (upper - lower).astype(np.float64) / np.maximum(np.asarray(shape), 1)
    size = np.asarray([np.log1p(float(component["voxels"])) / 16.0])
    return np.concatenate(
        [np.asarray(anatomy), normalized_center, extent, size]
    ).astype(np.float32)


def weighted_knn(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    neighbors: int,
    geometry_weight: float,
) -> np.ndarray:
    mean = train_features.mean(axis=0)
    scale = train_features.std(axis=0)
    scale[scale < 1e-5] = 1.0
    train = (train_features - mean) / scale
    validation = (validation_features - mean) / scale
    train[:, -7:] *= geometry_weight
    validation[:, -7:] *= geometry_weight
    predictions = []
    class_counts = np.bincount(train_labels, minlength=53).astype(np.float64)
    for feature in validation:
        distances = np.square(train - feature).mean(axis=1)
        selected = np.argpartition(distances, min(neighbors, len(distances)) - 1)[
            :neighbors
        ]
        votes = np.zeros(53, dtype=np.float64)
        for index in selected:
            label = int(train_labels[index])
            votes[label] += 1.0 / (
                (distances[index] + 1e-4) * np.sqrt(max(class_counts[label], 1.0))
            )
        predictions.append(int(np.argmax(votes[1:]) + 1))
    return np.asarray(predictions, dtype=np.int64)


def main() -> None:
    args = parse_args()
    layout = BaselineRunLayout.from_root(args.run_dir)
    cache_dir = layout.cache
    if not (cache_dir / "index.json").is_file():
        inputs = json.loads((layout.baseline / "inputs.json").read_text(encoding="utf-8"))
        cache_dir = Path(inputs["cache_index"]).parent
    index = load_cache_index(cache_dir)
    cache_root = Path(index["index_path"]).parent
    folds = json.loads(layout.fold_manifest.read_text(encoding="utf-8"))
    if not 0 <= args.fold < int(folds["n_folds"]):
        raise ValueError(f"Invalid fold: {args.fold}")
    validation_ids = set(folds["folds"][str(args.fold)])
    development_ids = set(folds["case_to_fold"])
    train_ids = development_ids - validation_ids

    features = []
    labels = []
    case_ids = []
    truth_by_case: dict[str, list[int]] = {}
    selected_cases = [
        case for case in index["cases"] if case["case_id"] in development_ids
    ]
    for case in tqdm(selected_cases, desc="Extracting vessel-location features"):
        case_id = str(case["case_id"])
        truth_by_case[case_id] = [int(value) for value in case["json_locations"]]
        case_dir = cache_root / case["cache_dir"]
        shape = tuple(int(value) for value in case["shape_zyx"])
        for component in case.get("components", []):
            features.append(component_feature(case_dir, component, shape))
            labels.append(int(component["class_id"]))
            case_ids.append(case_id)
    feature_array = np.stack(features)
    label_array = np.asarray(labels, dtype=np.int64)
    case_array = np.asarray(case_ids)
    train_mask = np.asarray([case_id in train_ids for case_id in case_ids])
    validation_mask = np.asarray([case_id in validation_ids for case_id in case_ids])
    if not np.any(train_mask) or not np.any(validation_mask):
        raise ValueError("Fold contains no train or validation components")

    results = []
    for neighbors in (1, 3, 5, 9):
        for geometry_weight in (0.5, 1.0, 2.0, 4.0):
            prediction = weighted_knn(
                feature_array[train_mask],
                label_array[train_mask],
                feature_array[validation_mask],
                neighbors,
                geometry_weight,
            )
            predictions_by_case: dict[str, list[int]] = defaultdict(list)
            for case_id, class_id in zip(
                case_array[validation_mask], prediction, strict=True
            ):
                predictions_by_case[str(case_id)].append(int(class_id))
            counts = [
                task1_case_counts(
                    truth_by_case[case_id], predictions_by_case.get(case_id, [])
                )
                for case_id in sorted(validation_ids)
            ]
            summary = summarize_task1(counts)
            macro = summary["macro"]
            results.append(
                {
                    "neighbors": neighbors,
                    "geometry_weight": geometry_weight,
                    "instance_accuracy": float(
                        np.mean(prediction == label_array[validation_mask])
                    ),
                    "selection_score": float(
                        np.mean(
                            [macro["precision"], macro["recall"], macro["mcc"]]
                        )
                    ),
                    "official_task1": summary,
                }
            )

    best = max(results, key=lambda item: item["selection_score"])
    output = args.output or (
        layout.ensemble
        / "ablation"
        / "vessel_location_oracle"
        / f"fold_{args.fold}"
        / "metrics.json"
    )
    atomic_json_dump(
        {
            "fold": args.fold,
            "oracle_input": "ground-truth aneurysm components and organizer vessel masks",
            "train_cases": len(train_ids),
            "validation_cases": len(validation_ids),
            "train_components": int(train_mask.sum()),
            "validation_components": int(validation_mask.sum()),
            "best": best,
            "experiments": results,
        },
        output,
    )
    print(f"Vessel-location oracle: {output}")


if __name__ == "__main__":
    main()
