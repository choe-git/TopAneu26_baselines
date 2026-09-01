"""Evaluate a cached split using the official TopAneu-26 metric protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
        "--save-predictions", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
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
        default_output = layout.ensemble / "evaluation" / args.split
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
    cases = [case for case in cache_index["cases"] if case["split"] == args.split]
    if not cases:
        raise ValueError(f"Cache contains no {args.split!r} cases")
    cache_root = Path(cache_index["index_path"]).parent
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
    per_case = []

    for case in tqdm(cases, desc=f"Evaluating {args.split}"):
        case_id = case["case_id"]
        case_dir = cache_root / case["cache_dir"]
        image = np.load(case_dir / "image.npy", mmap_mode="r").astype(np.float32)
        cache_prediction, predicted_locations, _ = ensemble_sliding_window_predict(
            models,
            image,
            case["modality"],
            config["data"]["patch_size"],
            device,
            overlap=args.overlap,
            amp_dtype=amp_dtype,
            mask_threshold=args.mask_threshold,
            class_threshold=args.class_threshold,
            presence_threshold=args.presence_threshold,
            tta_left_right=args.tta_left_right,
            location_lr_swap=cache_index["location_lr_swap"],
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
        per_case.append(
            {
                "case_id": case_id,
                "modality": case["modality"],
                "original_shape_zyx": list(ground_truth.shape),
                "original_spacing_xyz": ground_truth_metadata["spacing_xyz"],
                "task1_truth": [int(value) for value in case["json_locations"]],
                "task1_prediction": [int(value) for value in predicted_locations],
                "task2_binary_diagnostic": binary_metrics(tp, fp, fn, tn),
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
        "split": args.split,
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
        },
        "inference": {
            "soft_voting_folds": args.ensemble_folds,
            "models": len(models),
            "left_right_tta": args.tta_left_right,
            "probability_members": len(models) * (2 if args.tta_left_right else 1),
        },
        "official_task1": summarize_task1(task1_counts_per_case),
        "official_task2": summarize_task2(
            task2_counts_per_case, task2_segmentation_per_case
        ),
        "diagnostics": {
            "task2_binary_voxel": binary_metrics(
                *(int(value) for value in binary_totals)
            )
        },
    }
    atomic_json_dump(payload, metrics_path)
    print(f"Official-equivalent evaluation metrics: {metrics_path}")


if __name__ == "__main__":
    main()
