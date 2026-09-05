"""Paste dense ROI-refiner predictions back and score leakage-safe OOF cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import binary_dilation
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
from rnsa_surrogate.refiner_data import load_candidate_manifest, manifest_records
from rnsa_surrogate.roi_refiner import (
    CandidateROIRefinementDataset,
    CandidateROIRefiner,
    roi_start,
)
from rnsa_surrogate.run_layout import BaselineRunLayout
from train import official_selection_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", default="roi_refiner")
    parser.add_argument("--candidate-variant", default="candidates")
    parser.add_argument("--folds", type=int, nargs="+", default=[4])
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument(
        "--objectness-thresholds", type=float, nargs="+", default=[0.2, 0.35, 0.5]
    )
    parser.add_argument(
        "--mask-thresholds", type=float, nargs="+", default=[0.25, 0.35, 0.5]
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--support-radius-voxels", type=int, default=5)
    parser.add_argument("--use-refined-location", action="store_true")
    parser.add_argument(
        "--relabel-confidence-threshold",
        type=float,
        default=0.0,
        help="Use the refined class only at or above this confidence",
    )
    parser.add_argument(
        "--fast-only",
        action="store_true",
        help="Write the cache-space detection sweep without costly surface metrics",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def predict_records(
    model: CandidateROIRefiner,
    dataset: CandidateROIRefinementDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda"
    )
    predictions: dict[str, dict[str, Any]] = {}
    model.eval()
    for batch in tqdm(loader, desc="ROI refiner inference", leave=False):
        outputs = model(
            batch["image"].to(device, non_blocking=True),
            batch["metadata"].to(device, non_blocking=True),
            batch["stage1_class"].to(device, non_blocking=True),
        )
        objectness = torch.sigmoid(outputs["objectness_logits"]).float().cpu().numpy()
        masks = (
            torch.sigmoid(outputs["mask_logits"]).mul(255).round()
            .clamp(0, 255).to(torch.uint8).cpu().numpy()
        )
        location_probability = torch.softmax(outputs["location_logits"][:, 1:53], 1)
        confidence, location = location_probability.max(1)
        for index, score, mask, class_id, class_confidence in zip(
            batch["index"].numpy(), objectness, masks,
            location.cpu().numpy() + 1, confidence.cpu().numpy(), strict=True
        ):
            record = dataset.records[int(index)]
            predictions[str(record["candidate_id"])] = {
                "objectness": float(score),
                "mask_probability_u8": mask[0],
                "location_class": int(class_id),
                "location_confidence": float(class_confidence),
            }
    return predictions


def paste_case(
    case: dict[str, Any],
    records: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    roi_size: tuple[int, int, int],
    objectness_threshold: float,
    mask_threshold: float,
    support_radius: int,
    use_refined_location: bool,
    relabel_confidence_threshold: float,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    shape = tuple(int(value) for value in case["shape_zyx"])
    output = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)
    touched_coordinates: list[np.ndarray] = []
    roi_shape = np.asarray(roi_size)
    for record in records:
        prediction = predictions[str(record["candidate_id"])]
        objectness = float(prediction["objectness"])
        if objectness < objectness_threshold:
            continue
        from rnsa_surrogate.refiner_candidates import candidate_coordinates
        coordinates = candidate_coordinates(
            record["_artifact_path"], int(record["artifact_index"])
        )
        start = np.asarray(roi_start(record, roi_size), dtype=np.int64)
        candidate_roi = np.zeros(roi_size, dtype=bool)
        local = coordinates.astype(np.int64) - start
        inside = np.all((local >= 0) & (local < roi_shape), axis=1)
        candidate_roi[tuple(local[inside].T)] = True
        support = binary_dilation(
            candidate_roi, iterations=max(int(support_radius), 0)
        )
        probability = np.asarray(
            prediction["mask_probability_u8"], dtype=np.float32
        ) / 255.0
        keep = (probability >= mask_threshold) & support
        local_coordinates = np.argwhere(keep)
        if not local_coordinates.size:
            continue
        global_coordinates = local_coordinates + start
        valid = np.all(
            (global_coordinates >= 0)
            & (global_coordinates < np.asarray(shape, dtype=np.int64)), axis=1
        )
        local_coordinates = local_coordinates[valid]
        global_coordinates = global_coordinates[valid]
        if not global_coordinates.size:
            continue
        voxel_confidence = objectness * probability[tuple(local_coordinates.T)]
        existing = confidence[tuple(global_coordinates.T)]
        replace = voxel_confidence > existing
        global_coordinates = global_coordinates[replace]
        voxel_confidence = voxel_confidence[replace]
        class_id = (
            int(prediction["location_class"])
            if use_refined_location
            and float(prediction["location_confidence"]) >= relabel_confidence_threshold
            else int(record["stage1_class"])
        )
        output[tuple(global_coordinates.T)] = class_id
        confidence[tuple(global_coordinates.T)] = voxel_confidence
        touched_coordinates.append(global_coordinates)
    touched = (
        np.concatenate(touched_coordinates, axis=0)
        if touched_coordinates else np.empty((0, 3), dtype=np.int64)
    )
    if touched.size:
        touched = np.unique(touched, axis=0)
        touched = touched[output[tuple(touched.T)] > 0]
    locations = sorted(
        int(value) for value in np.unique(output[tuple(touched.T)])
        if value > 0
    ) if touched.size else []
    return output, locations, touched


def sparse_task2_case_counts(
    case: dict[str, Any],
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    predicted_locations: list[int],
    touched: np.ndarray,
) -> np.ndarray:
    """Official class detection counts without scanning the full cache volume."""
    ground_truth_classes = {
        int(component["class_id"]) for component in case["components"]
    }
    prediction_classes = set(predicted_locations)
    number_of_aneurysms = len(ground_truth_classes)
    counts = np.zeros((52, 4), dtype=np.int64)
    counts[:, 3] = number_of_aneurysms
    sparse_truth = (
        ground_truth[tuple(touched.T)] if touched.size
        else np.empty(0, dtype=np.uint8)
    )
    sparse_prediction = (
        prediction[tuple(touched.T)] if touched.size
        else np.empty(0, dtype=np.uint8)
    )
    for class_id in sorted(ground_truth_classes | prediction_classes):
        in_ground_truth = class_id in ground_truth_classes
        in_prediction = class_id in prediction_classes
        true_positive = bool(
            in_ground_truth and in_prediction
            and np.any((sparse_truth == class_id) & (sparse_prediction == class_id))
        )
        tp = int(true_positive)
        fn = int(in_ground_truth and not true_positive)
        counts[class_id - 1] = (
            tp, int(in_prediction and not true_positive), fn,
            number_of_aneurysms - (tp + fn)
        )
    return counts


def evaluate_pair(
    case_ids: list[str],
    cases: dict[str, dict[str, Any]],
    folds: dict[str, Any],
    records_by_case: dict[str, list[dict[str, Any]]],
    predictions: dict[str, dict[str, Any]],
    roi_sizes: dict[int, tuple[int, int, int]],
    source_root: Path,
    cache_root: Path,
    objectness_threshold: float,
    mask_threshold: float,
    support_radius: int,
    use_refined_location: bool,
    relabel_confidence_threshold: float,
    full: bool,
    truth_cache: dict[str, np.ndarray],
    save_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task1_counts, task2_counts, task2_segmentations = [], [], []
    binary_total, component_total, per_case = np.zeros(4, dtype=np.int64), np.zeros(4, dtype=np.int64), []
    for case_id in tqdm(case_ids, desc=f"OOF obj={objectness_threshold:.2f} mask={mask_threshold:.2f}", leave=False):
        case = cases[case_id]
        fold = int(folds["case_to_fold"][case_id])
        cache_prediction, locations, touched = paste_case(
            case, records_by_case.get(case_id, []), predictions, roi_sizes[fold],
            objectness_threshold, mask_threshold, support_radius,
            use_refined_location, relabel_confidence_threshold
        )
        task1_counts.append(task1_case_counts(case["json_locations"], locations))
        if full:
            key = f"native:{case_id}"
            if key not in truth_cache:
                truth, _ = load_zyx(source_root / "location_masks" / f"{case_id}.nii.gz")
                truth_cache[key] = np.asarray(truth, dtype=np.uint8)
            truth = truth_cache[key]
            native_prediction = resize_to_shape(cache_prediction, truth.shape, order=0).astype(np.uint8)
            counts, segmentation = task2_case_metrics(truth, native_prediction)
        else:
            key = f"cache:{case_id}"
            if key not in truth_cache:
                truth_cache[key] = np.load(
                    cache_root / str(case["cache_dir"]) / "location.npy",
                    mmap_mode="r",
                )
            truth = truth_cache[key]
            native_prediction = cache_prediction
            counts = sparse_task2_case_counts(
                case, truth, cache_prediction, locations, touched
            )
            segmentation = np.zeros((52, 3), dtype=np.float64)
        task2_counts.append(counts)
        task2_segmentations.append(segmentation)
        if full:
            truth_binary, prediction_binary = truth > 0, native_prediction > 0
            binary_total += (
                int(np.count_nonzero(truth_binary & prediction_binary)),
                int(np.count_nonzero(~truth_binary & prediction_binary)),
                int(np.count_nonzero(truth_binary & ~prediction_binary)),
                int(np.count_nonzero(~truth_binary & ~prediction_binary)),
            )
            component_total += component_counts(truth_binary, prediction_binary)
        per_case.append({
            "case_id": case_id, "oof_fold": fold,
            "task1_truth": case["json_locations"], "task1_prediction": locations,
        })
        if save_root is not None:
            atomic_save_npy(save_root / "predictions" / f"{case_id}.npy", native_prediction)
    task1 = summarize_task1(task1_counts)
    task2 = summarize_task2(task2_counts, task2_segmentations)
    metrics = {
        "objectness_threshold": objectness_threshold,
        "mask_threshold": mask_threshold,
        "official_task1": task1,
        "official_task2": task2,
        "selection_scores": official_selection_scores(task1, task2),
        "diagnostics": {
            "task2_binary_voxel": binary_metrics(binary_total),
            "component_counts": component_total.tolist(),
        },
    }
    if not full:
        macro = task2["macro"]
        metrics["detection_proxy"] = float(
            (macro["precision"] + macro["recall"] + macro["mcc"]) / 3
        )
    return metrics, per_case


def main() -> None:
    args = parse_args()
    if not args.variant.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--variant must be a simple directory name")
    if not args.candidate_variant.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--candidate-variant must be a simple directory name")
    if any(not 0 <= value <= 1 for value in args.objectness_thresholds + args.mask_thresholds):
        raise ValueError("Thresholds must be in [0, 1]")
    layout = BaselineRunLayout.from_root(args.run_dir)
    variant_root = layout.baseline / args.variant
    output = (args.output or variant_root / "oof_evaluation").resolve()
    if (output / "metrics.json").exists() and not args.overwrite:
        raise FileExistsError(output / "metrics.json")
    cache_index = load_cache_index(layout.cache)
    cases = {str(case["case_id"]): case for case in cache_index["cases"]}
    cache_sha = sha256_file(cache_index["index_path"])
    folds_path = layout.fold_manifest.resolve()
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    fold_sha = sha256_file(folds_path)
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    records_by_case: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    roi_sizes: dict[int, tuple[int, int, int]] = {}
    checkpoint_paths, manifest_paths = [], []
    for fold in args.folds:
        manifest_path = (
            layout.refiner / args.candidate_variant
            / "oof" / f"fold_{fold}" / "manifest.json"
        )
        manifest = load_candidate_manifest(manifest_path)
        if int(manifest["fold"]) != fold or manifest["cache_index_sha256"] != cache_sha or manifest["fold_manifest_sha256"] != fold_sha:
            raise ValueError(f"Candidate provenance mismatch: {manifest_path}")
        records = manifest_records(manifest)
        for record in records:
            if int(record["generator_fold"]) != fold:
                raise ValueError(f"Leaked candidate generator: {record['candidate_id']}")
            records_by_case.setdefault(str(record["case_id"]), []).append(record)
        checkpoint_path = variant_root / "folds" / f"fold_{fold}" / "checkpoint_best.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "roi_refiner" or int(checkpoint["fold"]) != fold or checkpoint["cache_index_sha256"] != cache_sha or checkpoint["fold_manifest_sha256"] != fold_sha:
            raise ValueError(f"ROI refiner checkpoint provenance mismatch: {checkpoint_path}")
        if checkpoint["candidate_manifest_sha256s"][str(fold)] != sha256_file(manifest_path):
            raise ValueError(f"ROI refiner manifest mismatch: {manifest_path}")
        model = CandidateROIRefiner(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        roi_size = tuple(int(value) for value in checkpoint["roi_size"])
        roi_sizes[fold] = roi_size
        dataset = CandidateROIRefinementDataset(
            layout.cache, records, roi_size, augment=False,
            vessel_context=bool(checkpoint.get("vessel_context", False)),
        )
        predictions.update(predict_records(model, dataset, device, args.batch_size))
        checkpoint_paths.append(str(checkpoint_path.resolve()))
        manifest_paths.append(str(manifest_path.resolve()))
    case_ids = sorted(
        case_id for case_id, fold in folds["case_to_fold"].items()
        if int(fold) in args.folds
    )
    source_root = Path(cache_index["source_root"]).resolve()
    sweep = []
    truth_cache: dict[str, np.ndarray] = {}
    for objectness_threshold in sorted(set(args.objectness_thresholds)):
        for mask_threshold in sorted(set(args.mask_thresholds)):
            metrics, _ = evaluate_pair(
                case_ids, cases, folds, records_by_case, predictions, roi_sizes,
                source_root, Path(cache_index["index_path"]).parent,
                objectness_threshold, mask_threshold,
                support_radius=args.support_radius_voxels,
                use_refined_location=args.use_refined_location,
                relabel_confidence_threshold=args.relabel_confidence_threshold,
                full=False, truth_cache=truth_cache,
            )
            sweep.append(metrics)
    best = max(sweep, key=lambda item: (item["detection_proxy"], -abs(item["objectness_threshold"] - 0.35), -abs(item["mask_threshold"] - 0.35)))
    if args.fast_only:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(sweep, output / "threshold_sweep.json")
        atomic_json_dump(
            {
                "split": "oof", "cases": len(case_ids), "folds": args.folds,
                "mode": "stage1_oof_candidate_to_dense_roi_refiner_fast_sweep",
                "location_policy": "refined" if args.use_refined_location else "stage1",
                "relabel_confidence_threshold": args.relabel_confidence_threshold,
                "best": best,
                "note": "cache-space detection-only screening; no Dice, HD95 or VS",
            },
            output / "fast_metrics.json",
        )
        print(f"ROI-refined fast OOF sweep: {output / 'fast_metrics.json'}")
        return
    final, per_case = evaluate_pair(
        case_ids, cases, folds, records_by_case, predictions, roi_sizes, source_root,
        Path(cache_index["index_path"]).parent,
        float(best["objectness_threshold"]), float(best["mask_threshold"]),
        support_radius=args.support_radius_voxels,
        use_refined_location=args.use_refined_location,
        relabel_confidence_threshold=args.relabel_confidence_threshold,
        full=True, truth_cache=truth_cache,
        save_root=output if args.save_predictions else None,
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(sweep, output / "threshold_sweep.json")
    atomic_json_dump(per_case, output / "per_case_metrics.json")
    payload = {
        "split": "oof", "cases": len(case_ids), "folds": args.folds,
        "mode": "stage1_oof_candidate_to_dense_roi_refiner",
        "location_policy": "refined" if args.use_refined_location else "stage1",
        "relabel_confidence_threshold": args.relabel_confidence_threshold,
        "support_radius_voxels": args.support_radius_voxels,
        "threshold_selection": {
            "criterion": "mean task2 precision, recall, MCC on combined heldout predictions",
            "objectness": best["objectness_threshold"],
            "mask": best["mask_threshold"],
        },
        "checkpoints": checkpoint_paths, "candidate_manifests": manifest_paths,
        "cache_index_sha256": cache_sha, "fold_manifest_sha256": fold_sha,
        **final,
    }
    atomic_json_dump(payload, output / "metrics.json")
    print(f"ROI-refined official OOF metrics: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
