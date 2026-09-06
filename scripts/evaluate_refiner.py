"""Apply held-out objectness refiners and compute official OOF metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label
from torch.utils.data import DataLoader
from tqdm import tqdm

from rnsa_surrogate.cache import (
    atomic_json_dump,
    atomic_save_npy,
    load_cache_index,
    load_zyx,
    resize_to_shape,
    sha256_file,
)
from rnsa_surrogate.official_metrics import (
    summarize_task1,
    summarize_task2,
    task1_case_counts,
    task2_case_metrics,
)
from rnsa_surrogate.refiner_candidates import (
    COMPONENT_STRUCTURE,
    candidate_coordinates,
)
from rnsa_surrogate.refiner_data import (
    CandidateROIDataset,
    load_candidate_manifest,
    manifest_records,
)
from rnsa_surrogate.refiner_model import CandidateObjectnessRefiner
from rnsa_surrogate.run_layout import BaselineRunLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--candidate-variant",
        default="candidates",
        help="Read OOF manifests from baseline/refiner/NAME",
    )
    parser.add_argument(
        "--refiner-variant",
        default="refiner",
        help="Read checkpoints below baseline/NAME and isolate default output",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def binary_metrics(values: np.ndarray) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(value) for value in values)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "mcc": (tp * tn - fp * fn) / denominator if denominator else 0.0,
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
    }


def component_counts(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_components, truth_count = label(truth, structure=COMPONENT_STRUCTURE)
    prediction_components, prediction_count = label(
        prediction, structure=COMPONENT_STRUCTURE
    )
    detected = {
        int(value) for value in np.unique(truth_components[prediction]) if value
    }
    overlapping = {
        int(value) for value in np.unique(prediction_components[truth]) if value
    }
    return np.asarray(
        [truth_count, prediction_count, len(detected), len(overlapping)],
        dtype=np.int64,
    )


@torch.inference_mode()
def predict_records(
    model: CandidateObjectnessRefiner,
    dataset: CandidateROIDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    result: dict[str, float] = {}
    model.eval()
    for batch in loader:
        logits = model(
            batch["image"].to(device, non_blocking=True),
            batch["metadata"].to(device, non_blocking=True),
        )
        probabilities = torch.sigmoid(logits).cpu().numpy()
        for index, probability in zip(
            batch["index"].numpy(), probabilities, strict=True
        ):
            candidate_id = str(dataset.records[int(index)]["candidate_id"])
            result[candidate_id] = float(probability)
    return result


def main() -> None:
    args = parse_args()
    layout = BaselineRunLayout.from_root(args.run_dir)
    candidate_root = layout.refiner_candidates_for(args.candidate_variant)
    refiner_folds = layout.refiner_folds_for(args.refiner_variant)
    output = (
        args.output
        or layout.refiner_evaluation_for(args.refiner_variant)
    ).resolve()
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(metrics_path)
    cache_index = load_cache_index(layout.cache)
    cache_root = Path(cache_index["index_path"]).parent
    cache_sha = sha256_file(cache_index["index_path"])
    folds_path = layout.fold_manifest.resolve()
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_sha = sha256_file(folds_path)
    if folds.get("cache_index_sha256") != cache_sha:
        raise ValueError("Fold manifest and cache SHA256 differ")
    selected = args.folds or list(range(int(folds["n_folds"])))
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)

    records_by_case: dict[str, list[dict[str, Any]]] = {}
    probabilities: dict[str, float] = {}
    checkpoint_paths = []
    checkpoint_hashes = []
    manifest_paths = []
    manifest_hashes = []
    threshold_by_fold: dict[int, float] = {}
    for fold in selected:
        manifest_path = (
            candidate_root / "oof" / f"fold_{fold}" / "manifest.json"
        )
        manifest = load_candidate_manifest(manifest_path)
        manifest_variant = str(manifest.get("candidate_variant", "candidates"))
        if manifest_variant != args.candidate_variant:
            raise ValueError(
                f"Candidate variant mismatch: requested {args.candidate_variant}, "
                f"manifest records {manifest_variant}: {manifest_path}"
            )
        if int(manifest["fold"]) != fold:
            raise ValueError(f"Candidate manifest fold mismatch: {manifest_path}")
        if manifest["cache_index_sha256"] != cache_sha:
            raise ValueError(f"Candidate cache mismatch: {manifest_path}")
        if manifest["fold_manifest_sha256"] != fold_sha:
            raise ValueError(f"Candidate fold provenance mismatch: {manifest_path}")
        records = manifest_records(manifest)
        expected_ids = set(folds["folds"][str(fold)])
        if set(manifest["case_ids"]) != expected_ids:
            raise ValueError(f"Candidate cases differ from fold {fold}")
        for record in records:
            case_id = str(record["case_id"])
            if int(record["_generator_fold"]) != int(
                folds["case_to_fold"][case_id]
            ):
                raise ValueError(f"Leaked candidate generator for {case_id}")
            records_by_case.setdefault(case_id, []).append(record)
        checkpoint_path = (
            refiner_folds / f"fold_{fold}" / "checkpoint_best.pth"
        ).resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if checkpoint.get("stage") != "objectness_refiner":
            raise ValueError(f"Not a refiner checkpoint: {checkpoint_path}")
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"Refiner checkpoint fold mismatch: {checkpoint_path}")
        if checkpoint.get("candidate_variant", "candidates") != args.candidate_variant:
            raise ValueError(
                f"Refiner checkpoint candidate variant mismatch: {checkpoint_path}"
            )
        if checkpoint.get("refiner_variant", "refiner") != args.refiner_variant:
            raise ValueError(
                f"Refiner checkpoint output variant mismatch: {checkpoint_path}"
            )
        if checkpoint["cache_index_sha256"] != cache_sha:
            raise ValueError(f"Refiner checkpoint cache mismatch: {checkpoint_path}")
        if checkpoint["fold_manifest_sha256"] != fold_sha:
            raise ValueError(f"Refiner checkpoint fold provenance mismatch")
        actual_manifest_sha = sha256_file(manifest_path)
        expected_manifest_sha = checkpoint.get(
            "candidate_manifest_sha256s", {}
        ).get(str(fold))
        if (
            expected_manifest_sha is not None
            and expected_manifest_sha != actual_manifest_sha
        ):
            raise ValueError(
                f"Refiner checkpoint candidate provenance mismatch: {manifest_path}"
            )
        model = CandidateObjectnessRefiner(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        if records:
            dataset = CandidateROIDataset(
                layout.cache,
                records,
                checkpoint["roi_size"],
                augment=False,
            )
            probabilities.update(
                predict_records(model, dataset, device, args.batch_size)
            )
        threshold_by_fold[fold] = (
            float(args.threshold)
            if args.threshold is not None
            else float(checkpoint["selection_threshold"])
        )
        checkpoint_paths.append(str(checkpoint_path))
        checkpoint_hashes.append(sha256_file(checkpoint_path))
        manifest_paths.append(str(manifest_path.resolve()))
        manifest_hashes.append(actual_manifest_sha)

    selected_case_ids = {
        str(case_id)
        for case_id, fold in folds["case_to_fold"].items()
        if int(fold) in selected
    }
    case_by_id = {str(case["case_id"]): case for case in cache_index["cases"]}
    source_root = Path(cache_index["source_root"]).resolve()
    location_root = source_root / "location_masks"
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    location_dir = output / "locations"
    task1_counts_list = []
    task2_counts_list = []
    task2_segmentation_list = []
    binary_total = np.zeros(4, dtype=np.int64)
    component_total = np.zeros(4, dtype=np.int64)
    per_case = []
    kept_total = 0
    candidate_total = 0
    for case_id in tqdm(sorted(selected_case_ids), desc="Refined OOF evaluation"):
        case = case_by_id[case_id]
        fold = int(folds["case_to_fold"][case_id])
        threshold = threshold_by_fold[fold]
        cache_prediction = np.zeros(tuple(case["shape_zyx"]), dtype=np.uint8)
        case_records = records_by_case.get(case_id, [])
        case_scores = []
        kept = []
        for record in case_records:
            score = probabilities[str(record["candidate_id"])]
            keep = score >= threshold
            case_scores.append(
                {
                    "candidate_id": record["candidate_id"],
                    "stage1_score": float(record["stage1_score"]),
                    "refiner_score": score,
                    "kept": keep,
                    "stage1_class": int(record["stage1_class"]),
                }
            )
            if not keep:
                continue
            coordinates = candidate_coordinates(
                record["_artifact_path"], int(record["artifact_index"])
            )
            cache_prediction[tuple(coordinates.T)] = int(record["stage1_class"])
            kept.append(int(record["stage1_class"]))
        predicted_locations = sorted(set(kept))
        candidate_total += len(case_records)
        kept_total += len(kept)
        ground_truth, _ = load_zyx(location_root / f"{case_id}.nii.gz")
        prediction = resize_to_shape(cache_prediction, ground_truth.shape, order=0)
        prediction = np.asarray(prediction, dtype=np.uint8)
        ground_truth = np.asarray(ground_truth, dtype=np.uint8)
        task1_counts_list.append(
            task1_case_counts(case["json_locations"], predicted_locations)
        )
        task2_counts, task2_segmentation = task2_case_metrics(
            ground_truth, prediction
        )
        task2_counts_list.append(task2_counts)
        task2_segmentation_list.append(task2_segmentation)
        truth_binary = ground_truth > 0
        prediction_binary = prediction > 0
        binary_total += (
            int(np.count_nonzero(truth_binary & prediction_binary)),
            int(np.count_nonzero(~truth_binary & prediction_binary)),
            int(np.count_nonzero(truth_binary & ~prediction_binary)),
            int(np.count_nonzero(~truth_binary & ~prediction_binary)),
        )
        component_total += component_counts(truth_binary, prediction_binary)
        per_case.append(
            {
                "case_id": case_id,
                "oof_fold": fold,
                "source_split": case["split"],
                "candidate_count": len(case_records),
                "kept_count": len(kept),
                "task1_truth": [int(value) for value in case["json_locations"]],
                "task1_prediction": predicted_locations,
                "candidates": case_scores,
            }
        )
        atomic_json_dump(per_case, output / "per_case_metrics.json")
        if args.save_predictions:
            atomic_save_npy(prediction_dir / f"{case_id}.npy", prediction)
            atomic_json_dump(predicted_locations, location_dir / f"{case_id}.json")
    payload = {
        "split": "oof",
        "cases": len(selected_case_ids),
        "mode": "stage1_oof_plus_heldout_objectness_refiner",
        "candidate_variant": args.candidate_variant,
        "refiner_variant": args.refiner_variant,
        "folds": selected,
        "checkpoints": checkpoint_paths,
        "checkpoint_sha256s": checkpoint_hashes,
        "candidate_manifests": manifest_paths,
        "candidate_manifest_sha256s": manifest_hashes,
        "cache_index": cache_index["index_path"],
        "cache_index_sha256": cache_sha,
        "fold_manifest": str(folds_path),
        "fold_manifest_sha256": fold_sha,
        "threshold_by_fold": {
            str(key): value for key, value in threshold_by_fold.items()
        },
        "candidate_count": candidate_total,
        "kept_count": kept_total,
        "location_policy": "retain stage1 component class; prune mask only",
        "organizer_vessel_input": False,
        "official_task1": summarize_task1(task1_counts_list),
        "official_task2": summarize_task2(
            task2_counts_list, task2_segmentation_list
        ),
        "diagnostics": {
            "task2_binary_voxel": binary_metrics(binary_total),
            "task2_component_objectness": {
                "ground_truth_components": int(component_total[0]),
                "predicted_components": int(component_total[1]),
                "detected_ground_truth_components": int(component_total[2]),
                "false_negative_ground_truth_components": int(
                    component_total[0] - component_total[2]
                ),
                "overlapping_prediction_components": int(component_total[3]),
                "false_positive_prediction_components": int(
                    component_total[1] - component_total[3]
                ),
                "sensitivity": safe_divide(
                    int(component_total[2]), int(component_total[0])
                ),
                "precision": safe_divide(
                    int(component_total[3]), int(component_total[1])
                ),
            },
        },
    }
    atomic_json_dump(payload, metrics_path)
    print(f"Refined official-equivalent OOF metrics: {metrics_path}")


if __name__ == "__main__":
    main()
