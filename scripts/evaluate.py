"""Evaluate a cached split using the official TopAneu-26 metric protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from tqdm import tqdm

from rnsa_surrogate.cache import (
    atomic_json_dump,
    atomic_save_npy,
    load_cache_index,
    load_zyx,
    resize_to_shape,
    sha256_file,
)
from rnsa_surrogate.inference import ensemble_sliding_window_predict
from rnsa_surrogate.model import RNSASurrogate
from rnsa_surrogate.official_metrics import (
    summarize_task1,
    summarize_task2,
    task1_case_counts,
    task2_case_metrics,
)
from rnsa_surrogate.run_layout import BaselineRunLayout
from rnsa_surrogate.vessel_refiner import VesselKNNRefiner

OFFICIAL_EVALUATION_URL = "https://github.com/Bangulli/TopAneu-26/tree/main/eval"


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
        "--ensemble-folds",
        type=int,
        nargs="+",
        help="Soft-vote RUN_DIR/baseline/folds/fold_N/checkpoint_best.pth",
    )
    parser.add_argument(
        "--oof",
        action="store_true",
        help=(
            "Evaluate each non-test case with only its held-out fold model. "
            "Requires --ensemble-folds and never soft-votes across folds."
        ),
    )
    parser.add_argument(
        "--tta-left-right",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Average original and left-right flipped probabilities",
    )
    parser.add_argument(
        "--output", type=Path, help="Override baseline/evaluation/SPLIT"
    )
    parser.add_argument(
        "--source", type=Path, help="Override original TopAneu release root"
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.45)
    parser.add_argument("--class-threshold", type=float, default=0.15)
    parser.add_argument("--presence-threshold", type=float, default=0.35)
    parser.add_argument(
        "--presence-top-k",
        type=int,
        default=3,
        help="Average the strongest K gated patches for each Task 1 class",
    )
    parser.add_argument(
        "--presence-evidence-voxels",
        type=int,
        default=64,
        help="Number of strongest aneurysm voxels used for patch evidence",
    )
    parser.add_argument("--minimum-component-voxels", type=int, default=5)
    parser.add_argument("--maximum-components", type=int, default=5)
    parser.add_argument(
        "--component-location-weight",
        type=float,
        default=0.0,
        help="Fuse categorical patch-head votes into component location labels",
    )
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--oracle-vessel-refiner-fold",
        type=int,
        help=(
            "OOF-only diagnostic: relabel predicted components with cached organizer "
            "vessels and a fold-trained kNN refiner"
        ),
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
        "mcc": float((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0,
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "iou": safe_divide(tp, tp + fp + fn),
    }


def component_objectness(
    truth_binary: np.ndarray, prediction_binary: np.ndarray
) -> dict[str, float | int]:
    """Measure lesion discovery without requiring the correct location class."""
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    truth_components, truth_count = connected_components(
        truth_binary, structure=structure
    )
    prediction_components, prediction_count = connected_components(
        prediction_binary, structure=structure
    )
    detected_truth = {
        int(value)
        for value in np.unique(truth_components[prediction_binary])
        if value
    }
    overlapping_predictions = {
        int(value)
        for value in np.unique(prediction_components[truth_binary])
        if value
    }
    return {
        "ground_truth_components": int(truth_count),
        "predicted_components": int(prediction_count),
        "detected_ground_truth_components": len(detected_truth),
        "false_negative_ground_truth_components": int(truth_count)
        - len(detected_truth),
        "overlapping_prediction_components": len(overlapping_predictions),
        "false_positive_prediction_components": int(prediction_count)
        - len(overlapping_predictions),
        "sensitivity": safe_divide(len(detected_truth), int(truth_count)),
        "precision": safe_divide(len(overlapping_predictions), int(prediction_count)),
    }


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[RNSASurrogate, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage", "baseline") not in {"baseline", "aneurysm_fold"}:
        raise ValueError(f"Not an aneurysm checkpoint: {checkpoint.get('stage')}")
    config = checkpoint["config"]
    model = RNSASurrogate(**config["model"])
    state = dict(checkpoint["model"])
    if "ema" in checkpoint:
        for name, value in checkpoint["ema"]["shadow"].items():
            state[name] = value.to(dtype=state[name].dtype)
    model.load_state_dict(state)
    return model.to(device).eval(), config


def resolve_cache(layout: BaselineRunLayout) -> Path:
    if (layout.cache / "index.json").is_file():
        return layout.cache
    inputs_path = layout.baseline / "inputs.json"
    if not inputs_path.is_file():
        raise FileNotFoundError(
            f"Missing cache and training inputs metadata: {layout.cache}"
        )
    return Path(
        json.loads(inputs_path.read_text(encoding="utf-8"))["cache_index"]
    ).parent


def main() -> None:
    args = parse_args()
    layout = BaselineRunLayout.from_root(args.run_dir)
    if args.checkpoint is not None and args.ensemble_folds is not None:
        raise ValueError("Use either --checkpoint or --ensemble-folds")
    if args.oof and args.ensemble_folds is None:
        raise ValueError("--oof requires --ensemble-folds")
    if args.oof and args.checkpoint is not None:
        raise ValueError("--oof cannot be combined with --checkpoint")
    if args.oracle_vessel_refiner_fold is not None and not args.oof:
        raise ValueError("--oracle-vessel-refiner-fold is restricted to --oof")
    if args.ensemble_folds is not None:
        if len(set(args.ensemble_folds)) != len(args.ensemble_folds):
            raise ValueError("--ensemble-folds contains duplicates")
        checkpoint_paths = [
            (
                layout.folds
                / f"fold_{fold}"
                / "checkpoint_best.pth"
            ).resolve()
            for fold in args.ensemble_folds
        ]
        if args.oracle_vessel_refiner_fold is not None:
            default_output = (
                layout.ensemble
                / "ablation"
                / "vessel_refiner_oracle"
                / f"fold_{args.oracle_vessel_refiner_fold}"
            )
        else:
            default_output = layout.ensemble / "evaluation" / (
                "oof" if args.oof else args.split
            )
    else:
        checkpoint_paths = [(args.checkpoint or layout.checkpoint).resolve()]
        default_output = layout.baseline / "evaluation" / args.split
    output_dir = (args.output or default_output).resolve()
    metrics_path = output_dir / "metrics.json"
    missing_checkpoints = [path for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(missing_checkpoints[0])
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"Evaluation already completed: {metrics_path}")

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    loaded = [load_model(path, device) for path in checkpoint_paths]
    models = [item[0] for item in loaded]
    config = loaded[0][1]
    if any(item[1]["model"] != config["model"] for item in loaded[1:]):
        raise ValueError("Ensemble checkpoints use different model configurations")
    amp_name = str(config["train"].get("amp", "none"))
    amp_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": None,
        "none": None,
    }[amp_name]
    if device.type != "cuda":
        amp_dtype = None

    cache_index = load_cache_index(resolve_cache(layout))
    fold_to_model: dict[int, RNSASurrogate] = {}
    case_to_fold: dict[str, int] = {}
    if args.oof:
        fold_manifest = json.loads(layout.fold_manifest.read_text(encoding="utf-8"))
        case_to_fold = {
            str(case_id): int(fold)
            for case_id, fold in fold_manifest["case_to_fold"].items()
        }
        assert args.ensemble_folds is not None
        fold_to_model = dict(zip(args.ensemble_folds, models, strict=True))
        cases = [
            case
            for case in cache_index["cases"]
            if case_to_fold.get(str(case["case_id"])) in fold_to_model
        ]
        evaluation_split = "oof"
    else:
        cases = [case for case in cache_index["cases"] if case["split"] == args.split]
        evaluation_split = args.split
    if not cases:
        raise ValueError(f"Cache contains no {evaluation_split!r} cases")
    cache_root = Path(cache_index["index_path"]).parent
    vessel_refiner = None
    if args.oracle_vessel_refiner_fold is not None:
        refiner_fold = int(args.oracle_vessel_refiner_fold)
        if args.ensemble_folds != [refiner_fold]:
            raise ValueError(
                "Oracle vessel refiner requires exactly its matching OOF fold model"
            )
        development_ids = set(case_to_fold)
        refiner_validation_ids = {
            case_id for case_id, fold in case_to_fold.items() if fold == refiner_fold
        }
        vessel_refiner = VesselKNNRefiner.fit(
            cache_index,
            cache_root,
            development_ids - refiner_validation_ids,
        )
    source_root = (args.source or Path(cache_index["source_root"])).resolve()
    location_mask_root = source_root / "location_masks"
    if not location_mask_root.is_dir():
        raise FileNotFoundError(
            f"Official-grid evaluation requires original masks: {location_mask_root}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    location_dir = output_dir / "locations"
    task1_counts_per_case = []
    task2_counts_per_case = []
    task2_segmentation_per_case = []
    binary_totals = np.zeros(4, dtype=np.int64)
    component_totals = np.zeros(4, dtype=np.int64)
    per_case = []

    for case in tqdm(cases, desc=f"Evaluating {evaluation_split}"):
        case_id = case["case_id"]
        case_fold = case_to_fold.get(str(case_id)) if args.oof else None
        case_models = [fold_to_model[case_fold]] if case_fold is not None else models
        case_dir = cache_root / case["cache_dir"]
        image = np.load(case_dir / "image.npy", mmap_mode="r").astype(np.float32)
        cache_prediction, predicted_locations, inference_diagnostics = (
            ensemble_sliding_window_predict(
                case_models,
                image,
                case["modality"],
                config["data"]["patch_size"],
                device,
                overlap=args.overlap,
                amp_dtype=amp_dtype,
                mask_threshold=args.mask_threshold,
                class_threshold=args.class_threshold,
                presence_threshold=args.presence_threshold,
                presence_top_k=args.presence_top_k,
                presence_evidence_voxels=args.presence_evidence_voxels,
                minimum_component_voxels=args.minimum_component_voxels,
                maximum_components=args.maximum_components,
                component_location_weight=args.component_location_weight,
                tta_left_right=args.tta_left_right,
                location_lr_swap=cache_index["location_lr_swap"],
            )
        )
        if vessel_refiner is not None:
            vessel = np.load(case_dir / "vessel.npy", mmap_mode="r")
            cache_prediction = vessel_refiner.refine(cache_prediction, vessel)
            predicted_locations = sorted(
                int(value) for value in np.unique(cache_prediction) if value > 0
            )

        ground_truth, ground_truth_metadata = load_zyx(
            location_mask_root / f"{case_id}.nii.gz"
        )
        expected_shape = tuple(case["original_metadata"]["shape_zyx"])
        if ground_truth.shape != expected_shape:
            raise ValueError(
                f"Original shape mismatch for {case_id}: "
                f"index={expected_shape}, mask={ground_truth.shape}"
            )
        prediction = resize_to_shape(cache_prediction, ground_truth.shape, order=0)
        prediction = np.asarray(prediction, dtype=np.uint8)
        ground_truth = np.asarray(ground_truth, dtype=np.uint8)

        task1_counts_per_case.append(
            task1_case_counts(case["json_locations"], predicted_locations)
        )
        task2_counts, task2_segmentation = task2_case_metrics(ground_truth, prediction)
        task2_counts_per_case.append(task2_counts)
        task2_segmentation_per_case.append(task2_segmentation)

        truth_binary = ground_truth > 0
        prediction_binary = prediction > 0
        tp = int(np.count_nonzero(truth_binary & prediction_binary))
        fp = int(np.count_nonzero(~truth_binary & prediction_binary))
        fn = int(np.count_nonzero(truth_binary & ~prediction_binary))
        tn = int(np.count_nonzero(~truth_binary & ~prediction_binary))
        binary_totals += tp, fp, fn, tn
        objectness = component_objectness(truth_binary, prediction_binary)
        component_totals += (
            objectness["ground_truth_components"],
            objectness["predicted_components"],
            objectness["detected_ground_truth_components"],
            objectness["overlapping_prediction_components"],
        )
        per_case.append(
            {
                "case_id": case_id,
                "oof_fold": case_fold,
                "source_split": case["split"],
                "modality": case["modality"],
                "original_shape_zyx": list(ground_truth.shape),
                "original_spacing_xyz": ground_truth_metadata["spacing_xyz"],
                "task1_truth": [int(value) for value in case["json_locations"]],
                "task1_prediction": [int(value) for value in predicted_locations],
                "task1_location_scores": [
                    float(value)
                    for value in inference_diagnostics["global_location_scores"]
                ],
                "task1_patch_location_scores": [
                    float(value)
                    for value in inference_diagnostics["patch_location_scores"]
                ],
                "aneurysm_presence_score": float(
                    inference_diagnostics["global_aneurysm_score"]
                ),
                "task2_binary_diagnostic": binary_metrics(tp, fp, fn, tn),
                "task2_component_objectness": objectness,
            }
        )
        atomic_json_dump(per_case, output_dir / "per_case_metrics.json")
        if args.save_predictions:
            atomic_save_npy(prediction_dir / f"{case_id}.npy", prediction)
            atomic_json_dump(
                [int(value) for value in predicted_locations],
                location_dir / f"{case_id}.json",
            )

    payload = {
        "split": evaluation_split,
        "cases": len(cases),
        "checkpoint": (
            str(checkpoint_paths[0]) if len(checkpoint_paths) == 1 else None
        ),
        "checkpoint_sha256": (
            sha256_file(checkpoint_paths[0])
            if len(checkpoint_paths) == 1
            else None
        ),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "checkpoint_sha256s": [sha256_file(path) for path in checkpoint_paths],
        "cache_index": cache_index["index_path"],
        "cache_index_sha256": sha256_file(cache_index["index_path"]),
        "source_root": str(source_root),
        "evaluation_protocol": {
            "name": "TopAneu-26 official metric port",
            "reference": OFFICIAL_EVALUATION_URL,
            "classes": 52,
            "prediction_geometry": "original source mask grid",
            "ranking_note": "Grand Challenge computes mean ranks across submissions",
        },
        "thresholds": {
            "overlap": args.overlap,
            "mask": args.mask_threshold,
            "class": args.class_threshold,
            "presence": args.presence_threshold,
            "presence_top_k": args.presence_top_k,
            "presence_evidence_voxels": args.presence_evidence_voxels,
            "minimum_component_voxels": args.minimum_component_voxels,
            "maximum_components": args.maximum_components,
            "component_location_weight": args.component_location_weight,
        },
        "inference": {
            "soft_voting_folds": args.ensemble_folds,
            "mode": "held_out_fold_per_case" if args.oof else "ensemble",
            "models_loaded": len(models),
            "models_per_case": 1 if args.oof else len(models),
            "models": 1 if args.oof else len(models),
            "left_right_tta": args.tta_left_right,
            "probability_members_per_case": (
                1 if args.oof else len(models)
            ) * (2 if args.tta_left_right else 1),
            "probability_members": (
                1 if args.oof else len(models)
            ) * (2 if args.tta_left_right else 1),
            "location_probability": "conditional softmax over classes 1..52",
            "location_overlap": "component-level weighted vote",
            "task1_aggregation": "retained aneurysm component labels",
            "oracle_vessel_refiner": (
                {
                    "fold": args.oracle_vessel_refiner_fold,
                    "neighbors": 3,
                    "geometry_weight": 0.5,
                    "warning": "OOF diagnostic uses organizer vessel masks; not deployable",
                }
                if vessel_refiner is not None
                else None
            ),
        },
        "official_task1": summarize_task1(task1_counts_per_case),
        "official_task2": summarize_task2(
            task2_counts_per_case, task2_segmentation_per_case
        ),
        "diagnostics": {
            "task2_binary_voxel": binary_metrics(
                *(int(value) for value in binary_totals)
            ),
            "task2_component_objectness": {
                "ground_truth_components": int(component_totals[0]),
                "predicted_components": int(component_totals[1]),
                "detected_ground_truth_components": int(component_totals[2]),
                "false_negative_ground_truth_components": int(
                    component_totals[0] - component_totals[2]
                ),
                "overlapping_prediction_components": int(component_totals[3]),
                "false_positive_prediction_components": int(
                    component_totals[1] - component_totals[3]
                ),
                "sensitivity": safe_divide(
                    int(component_totals[2]), int(component_totals[0])
                ),
                "precision": safe_divide(
                    int(component_totals[3]), int(component_totals[1])
                ),
            },
        },
    }
    atomic_json_dump(payload, metrics_path)
    print(f"Official-equivalent evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
