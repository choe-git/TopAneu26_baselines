"""Select one global OOF threshold and evaluate relabeled candidate masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_refiner import binary_metrics, component_counts
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
    task2_case_counts,
    task2_case_metrics,
)
from rnsa_surrogate.refiner_candidates import candidate_coordinates
from rnsa_surrogate.refiner_data import (
    CandidateROIDataset,
    load_candidate_manifest,
    manifest_records,
)
from rnsa_surrogate.refiner_location_model import CandidateLocationRefiner
from rnsa_surrogate.run_layout import BaselineRunLayout
from train import official_selection_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    parser.add_argument("--selection-task", choices=("task1", "task2"), default="task2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def predict_records(
    model: CandidateLocationRefiner,
    dataset: CandidateROIDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float | int]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    result: dict[str, dict[str, float | int]] = {}
    model.eval()
    for batch in loader:
        outputs = model(
            batch["image"].to(device, non_blocking=True),
            batch["metadata"].to(device, non_blocking=True),
            batch["stage1_class"].to(device, non_blocking=True),
        )
        objectness = torch.sigmoid(outputs["objectness_logits"]).cpu().numpy()
        location_probability = torch.softmax(
            outputs["location_logits"][:, 1:53], dim=1
        )
        confidence, location = location_probability.max(dim=1)
        for index, score, class_id, class_confidence in zip(
            batch["index"].numpy(),
            objectness,
            location.cpu().numpy() + 1,
            confidence.cpu().numpy(),
            strict=True,
        ):
            candidate_id = str(dataset.records[int(index)]["candidate_id"])
            result[candidate_id] = {
                "objectness": float(score),
                "location_class": int(class_id),
                "location_confidence": float(class_confidence),
            }
    return result


def evaluate_threshold(
    threshold: float,
    case_ids: list[str],
    cases: dict[str, dict[str, Any]],
    folds: dict[str, Any],
    records_by_case: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, float | int]],
    source_root: Path,
    ground_truth_cache: dict[str, np.ndarray],
    full_segmentation_metrics: bool = True,
    save_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task1_counts_list = []
    task2_counts_list = []
    task2_segmentation_list = []
    binary_total = np.zeros(4, dtype=np.int64)
    component_total = np.zeros(4, dtype=np.int64)
    per_case = []
    location_root = source_root / "location_masks"
    for case_id in tqdm(case_ids, desc=f"Official threshold {threshold:.2f}", leave=False):
        case = cases[case_id]
        cache_prediction = np.zeros(tuple(case["shape_zyx"]), dtype=np.uint8)
        candidate_details = []
        retained_classes = []
        for record in records_by_case.get(case_id, []):
            prediction = predictions[str(record["candidate_id"])]
            keep = float(prediction["objectness"]) >= threshold
            refined_class = int(prediction["location_class"])
            candidate_details.append(
                {
                    "candidate_id": record["candidate_id"],
                    "objectness": float(prediction["objectness"]),
                    "location_class": refined_class,
                    "location_confidence": float(prediction["location_confidence"]),
                    "stage1_class": int(record["stage1_class"]),
                    "kept": keep,
                }
            )
            if not keep:
                continue
            coordinates = candidate_coordinates(
                record["_artifact_path"], int(record["artifact_index"])
            )
            cache_prediction[tuple(coordinates.T)] = refined_class
            retained_classes.append(refined_class)
        predicted_locations = sorted(set(retained_classes))
        if case_id not in ground_truth_cache:
            loaded_truth, _ = load_zyx(location_root / f"{case_id}.nii.gz")
            ground_truth_cache[case_id] = np.asarray(
                loaded_truth, dtype=np.uint8
            )
        ground_truth = ground_truth_cache[case_id]
        prediction_mask = resize_to_shape(
            cache_prediction, ground_truth.shape, order=0
        ).astype(np.uint8)
        task1_counts_list.append(
            task1_case_counts(case["json_locations"], predicted_locations)
        )
        if full_segmentation_metrics:
            counts, segmentation = task2_case_metrics(
                ground_truth, prediction_mask
            )
        else:
            counts = task2_case_counts(ground_truth, prediction_mask)
            segmentation = np.zeros((52, 3), dtype=np.float64)
        task2_counts_list.append(counts)
        task2_segmentation_list.append(segmentation)
        truth_binary = ground_truth > 0
        prediction_binary = prediction_mask > 0
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
                "oof_fold": int(folds["case_to_fold"][case_id]),
                "source_split": case["split"],
                "task1_truth": [int(value) for value in case["json_locations"]],
                "task1_prediction": predicted_locations,
                "candidates": candidate_details,
            }
        )
        if save_root is not None:
            atomic_save_npy(save_root / "predictions" / f"{case_id}.npy", prediction_mask)
            atomic_json_dump(
                predicted_locations,
                save_root / "locations" / f"{case_id}.json",
            )
    task1 = summarize_task1(task1_counts_list)
    task2 = summarize_task2(task2_counts_list, task2_segmentation_list)
    scores = official_selection_scores(task1, task2)
    metrics = {
        "threshold": threshold,
        "official_task1": task1,
        "official_task2": task2,
        "selection_scores": scores,
        "diagnostics": {
            "task2_binary_voxel": binary_metrics(binary_total),
            "task2_component_objectness": {
                "ground_truth_components": int(component_total[0]),
                "predicted_components": int(component_total[1]),
                "detected_ground_truth_components": int(component_total[2]),
                "overlapping_prediction_components": int(component_total[3]),
                "false_negative_ground_truth_components": int(
                    component_total[0] - component_total[2]
                ),
                "false_positive_prediction_components": int(
                    component_total[1] - component_total[3]
                ),
            },
        },
    }
    return metrics, per_case


def main() -> None:
    args = parse_args()
    if not args.thresholds or any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        raise ValueError("Thresholds must be in [0, 1]")
    layout = BaselineRunLayout.from_root(args.run_dir)
    output = (
        args.output
        or layout.ensemble / "evaluation" / "oof_refiner_location"
    ).resolve()
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(metrics_path)
    cache_index = load_cache_index(layout.cache)
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
    scores: dict[str, dict[str, float | int]] = {}
    checkpoint_paths, checkpoint_hashes = [], []
    manifest_paths, manifest_hashes = [], []
    for fold in selected:
        if not 0 <= fold < int(folds["n_folds"]):
            raise ValueError(f"Invalid fold: {fold}")
        manifest_path = (
            layout.refiner_candidates / "oof" / f"fold_{fold}" / "manifest.json"
        )
        manifest = load_candidate_manifest(manifest_path)
        if (
            int(manifest["fold"]) != fold
            or manifest["cache_index_sha256"] != cache_sha
            or manifest["fold_manifest_sha256"] != fold_sha
        ):
            raise ValueError(f"Candidate provenance mismatch: {manifest_path}")
        if set(manifest["case_ids"]) != set(folds["folds"][str(fold)]):
            raise ValueError(f"Candidate cases differ from fold {fold}")
        records = manifest_records(manifest)
        for record in records:
            case_id = str(record["case_id"])
            generator = int(record["generator_fold"])
            if generator != fold or generator != int(folds["case_to_fold"][case_id]):
                raise ValueError(f"Leaked candidate generator for {case_id}")
            records_by_case.setdefault(case_id, []).append(record)
        checkpoint_path = (
            layout.refiner_location_folds / f"fold_{fold}" / "checkpoint_best.pth"
        ).resolve()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("stage") != "candidate_location_refiner"
            or int(checkpoint.get("fold", -1)) != fold
            or checkpoint["cache_index_sha256"] != cache_sha
            or checkpoint["fold_manifest_sha256"] != fold_sha
        ):
            raise ValueError(f"Refiner checkpoint provenance mismatch: {checkpoint_path}")
        manifest_sha = sha256_file(manifest_path)
        if checkpoint["candidate_manifest_sha256s"].get(str(fold)) != manifest_sha:
            raise ValueError(
                f"Refiner checkpoint/candidate manifest mismatch: {manifest_path}"
            )
        model = CandidateLocationRefiner(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        if records:
            dataset = CandidateROIDataset(
                layout.cache, records, checkpoint["roi_size"], augment=False
            )
            scores.update(predict_records(model, dataset, device, args.batch_size))
        checkpoint_paths.append(str(checkpoint_path))
        checkpoint_hashes.append(sha256_file(checkpoint_path))
        manifest_paths.append(str(manifest_path.resolve()))
        manifest_hashes.append(manifest_sha)
    selected_cases = sorted(
        str(case_id)
        for case_id, fold in folds["case_to_fold"].items()
        if int(fold) in selected
    )
    cases = {str(case["case_id"]): case for case in cache_index["cases"]}
    source_root = Path(cache_index["source_root"]).resolve()
    sweep = []
    ground_truth_cache: dict[str, np.ndarray] = {}
    for threshold in sorted(set(float(value) for value in args.thresholds)):
        metrics, _ = evaluate_threshold(
            threshold,
            selected_cases,
            cases,
            folds,
            records_by_case,
            scores,
            source_root,
            ground_truth_cache,
            False,
        )
        sweep.append(metrics)
    best = max(
        sweep,
        key=lambda item: (
            float(item["selection_scores"][args.selection_task]),
            float(item[f"official_{args.selection_task}"]["macro"]["mcc"]),
            -abs(float(item["threshold"]) - 0.5),
        ),
    )
    final_metrics, per_case = evaluate_threshold(
        float(best["threshold"]),
        selected_cases,
        cases,
        folds,
        records_by_case,
        scores,
        source_root,
        ground_truth_cache,
        True,
        output if args.save_predictions else None,
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(per_case, output / "per_case_metrics.json")
    atomic_json_dump(
        [
            {
                "threshold": item["threshold"],
                "selection_scores": item["selection_scores"],
                "official_task1_macro": item["official_task1"]["macro"],
                "official_task2_macro": item["official_task2"]["macro"],
            }
            for item in sweep
        ],
        output / "threshold_sweep.json",
    )
    payload = {
        "split": "oof",
        "cases": len(selected_cases),
        "folds": selected,
        "mode": "heldout_53way_candidate_refiner",
        "global_threshold_selection": {
            "task": args.selection_task,
            "threshold": best["threshold"],
            "score": best["selection_scores"][args.selection_task],
            "note": "one threshold selected from combined held-out-fold predictions",
        },
        "location_policy": "retain stage1 component mask; relabel with refiner argmax",
        "organizer_vessel_input": False,
        "checkpoints": checkpoint_paths,
        "checkpoint_sha256s": checkpoint_hashes,
        "candidate_manifests": manifest_paths,
        "candidate_manifest_sha256s": manifest_hashes,
        "cache_index": cache_index["index_path"],
        "cache_index_sha256": cache_sha,
        "fold_manifest": str(folds_path),
        "fold_manifest_sha256": fold_sha,
        **final_metrics,
    }
    atomic_json_dump(payload, metrics_path)
    print(f"Location-refined official OOF metrics: {metrics_path}")


if __name__ == "__main__":
    main()
