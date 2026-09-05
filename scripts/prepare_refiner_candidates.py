"""Generate leakage-safe stage-1 OOF components for refiner training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rnsa_surrogate.cache import (
    atomic_json_dump,
    load_cache_index,
    sha256_file,
)
from rnsa_surrogate.data import extract_patch
from rnsa_surrogate.inference import ensemble_sliding_window_predict
from rnsa_surrogate.model import RNSASurrogate
from rnsa_surrogate.refiner_candidates import (
    CANDIDATE_VERSION,
    atomic_save_candidate_artifact,
    extract_candidate_records,
)
from rnsa_surrogate.run_layout import BaselineRunLayout
from rnsa_surrogate.runtime import config_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--tta-left-right", action="store_true")
    parser.add_argument("--mask-threshold", type=float)
    parser.add_argument("--presence-threshold", type=float)
    parser.add_argument("--minimum-component-voxels", type=int)
    parser.add_argument("--maximum-components", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--candidate-variant", default="candidates")
    parser.add_argument("--with-vessel-context", action="store_true")
    parser.add_argument("--vessel-roi-size", type=int, nargs=3, default=[48, 64, 64])
    return parser.parse_args()


def load_stage1(
    path: Path, expected_fold: int, device: torch.device
) -> tuple[RNSASurrogate, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "aneurysm_fold":
        raise ValueError(f"Not a fold checkpoint: {path}")
    if int(checkpoint.get("fold", -1)) != expected_fold:
        raise ValueError(
            f"Checkpoint fold {checkpoint.get('fold')} cannot generate fold "
            f"{expected_fold} OOF candidates"
        )
    config = checkpoint["config"]
    model = RNSASurrogate(**config["model"])
    state = dict(checkpoint["model"])
    if "ema" in checkpoint:
        state.update(
            {
                name: value.to(dtype=state[name].dtype)
                for name, value in checkpoint["ema"]["shadow"].items()
            }
        )
    model.load_state_dict(state)
    return model.to(device).eval(), config, checkpoint


def main() -> None:
    args = parse_args()
    if not args.candidate_variant.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--candidate-variant must be a simple directory name")
    layout = BaselineRunLayout.from_root(args.run_dir)
    cache_index = load_cache_index(layout.cache)
    cache_sha = sha256_file(cache_index["index_path"])
    fold_manifest_path = layout.fold_manifest.resolve()
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if fold_manifest.get("cache_index_sha256") != cache_sha:
        raise ValueError("Fold manifest and current cache have different SHA256")
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    selected = args.folds or list(range(int(fold_manifest["n_folds"])))
    case_by_id = {str(case["case_id"]): case for case in cache_index["cases"]}
    cache_root = Path(cache_index["index_path"]).parent

    for fold in selected:
        if not 0 <= fold < int(fold_manifest["n_folds"]):
            raise ValueError(f"Invalid fold: {fold}")
        checkpoint_path = (
            layout.folds / f"fold_{fold}" / "checkpoint_best.pth"
        ).resolve()
        model, config, checkpoint = load_stage1(checkpoint_path, fold, device)
        if checkpoint.get("cache_index_sha256") != cache_sha:
            raise ValueError(f"Fold {fold} checkpoint was trained from another cache")
        candidate_root = layout.refiner / args.candidate_variant
        output = candidate_root / "oof" / f"fold_{fold}"
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(manifest_path)
        output.mkdir(parents=True, exist_ok=True)
        case_ids = [str(value) for value in fold_manifest["folds"][str(fold)]]
        for case_id in case_ids:
            assigned = int(fold_manifest["case_to_fold"][case_id])
            if assigned != fold:
                raise AssertionError(
                    f"{case_id} belongs to fold {assigned}, not generator fold {fold}"
                )
        settings = config.get("validation", {})
        inference_settings = {
            "overlap": float(settings.get("overlap", 0.5)),
            "mask_threshold": float(settings.get("mask_threshold", 0.45)),
            "presence_threshold": float(settings.get("presence_threshold", 0.35)),
            "presence_top_k": int(settings.get("presence_top_k", 3)),
            "presence_evidence_voxels": int(
                settings.get("presence_evidence_voxels", 64)
            ),
            "minimum_component_voxels": int(
                settings.get("minimum_component_voxels", 5)
            ),
            "maximum_components": int(settings.get("maximum_components", 5)),
            "component_location_weight": float(
                settings.get("component_location_weight", 0.0)
            ),
            "tta_left_right": bool(args.tta_left_right),
        }
        if args.mask_threshold is not None:
            inference_settings["mask_threshold"] = float(args.mask_threshold)
        if args.presence_threshold is not None:
            inference_settings["presence_threshold"] = float(
                args.presence_threshold
            )
        if args.minimum_component_voxels is not None:
            inference_settings["minimum_component_voxels"] = int(
                args.minimum_component_voxels
            )
        if args.maximum_components is not None:
            inference_settings["maximum_components"] = int(
                args.maximum_components
            )
        if not 0.0 < inference_settings["mask_threshold"] < 1.0:
            raise ValueError("mask_threshold must be between zero and one")
        if not 0.0 < inference_settings["presence_threshold"] < 1.0:
            raise ValueError("presence_threshold must be between zero and one")
        if inference_settings["minimum_component_voxels"] < 1:
            raise ValueError("minimum_component_voxels must be positive")
        if inference_settings["maximum_components"] < 1:
            raise ValueError("maximum_components must be positive")
        amp_name = str(config["train"].get("amp", "none"))
        amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": None,
            "none": None,
        }[amp_name]
        if device.type != "cuda":
            amp_dtype = None
        all_records: list[dict[str, Any]] = []
        positives = 0
        for case_id in tqdm(case_ids, desc=f"Refiner OOF candidates fold {fold}"):
            case = case_by_id[case_id]
            case_dir = cache_root / case["cache_dir"]
            image = np.load(case_dir / "image.npy", mmap_mode="r").astype(np.float32)
            truth = np.load(case_dir / "location.npy", mmap_mode="r")
            segmentation, _, diagnostics = ensemble_sliding_window_predict(
                [model],
                image,
                str(case["modality"]),
                config["data"]["patch_size"],
                device,
                overlap=inference_settings["overlap"],
                amp_dtype=amp_dtype,
                mask_threshold=inference_settings["mask_threshold"],
                presence_threshold=inference_settings["presence_threshold"],
                presence_top_k=inference_settings["presence_top_k"],
                presence_evidence_voxels=inference_settings[
                    "presence_evidence_voxels"
                ],
                minimum_component_voxels=inference_settings[
                    "minimum_component_voxels"
                ],
                maximum_components=inference_settings["maximum_components"],
                component_location_weight=inference_settings[
                    "component_location_weight"
                ],
                tta_left_right=args.tta_left_right,
                location_lr_swap=cache_index["location_lr_swap"],
                return_vessel_segmentation=args.with_vessel_context,
            )
            records, coordinates, offsets = extract_candidate_records(
                case_id,
                segmentation,
                diagnostics["binary_probability"],
                np.asarray(truth),
            )
            artifact = Path("cases") / f"{case_id}.npz"
            vessel_rois = None
            if args.with_vessel_context:
                vessel_prediction = diagnostics["vessel_segmentation"]
                vessel_rois = np.stack([
                    extract_patch(
                        vessel_prediction,
                        tuple(int(round(float(v))) for v in record["center_zyx"]),
                        tuple(args.vessel_roi_size),
                        pad_value=0,
                    )[0]
                    for record in records
                ]) if records else np.empty((0, *args.vessel_roi_size), dtype=np.uint8)
            atomic_save_candidate_artifact(
                output / artifact, coordinates, offsets, vessel_rois
            )
            for record in records:
                record["artifact"] = artifact.as_posix()
                record["generator_fold"] = fold
                record["vessel_context"] = bool(args.with_vessel_context)
                if int(record["generator_fold"]) != int(
                    fold_manifest["case_to_fold"][case_id]
                ):
                    raise AssertionError("OOF generator leakage detected")
            all_records.extend(records)
            positives += sum(int(record["target"]) for record in records)
        payload = {
            "candidate_version": CANDIDATE_VERSION,
            "mode": "stage1_oof",
            "fold": fold,
            "case_ids": case_ids,
            "cases": len(case_ids),
            "candidates": all_records,
            "candidate_count": len(all_records),
            "positive_candidates": positives,
            "negative_candidates": len(all_records) - positives,
            "cache_index": cache_index["index_path"],
            "cache_index_sha256": cache_sha,
            "fold_manifest": str(fold_manifest_path),
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "stage1_checkpoint": str(checkpoint_path),
            "stage1_checkpoint_sha256": sha256_file(checkpoint_path),
            "stage1_checkpoint_epoch": int(checkpoint["epoch"]),
            "inference": inference_settings,
            "candidate_variant": args.candidate_variant,
            "vessel_context": bool(args.with_vessel_context),
            "vessel_roi_size": list(args.vessel_roi_size),
            "inference_sha256": config_digest(inference_settings),
            "leakage_guard": (
                "Every case is generated only by the stage1 checkpoint whose "
                "held-out fold equals the case fold."
            ),
        }
        atomic_json_dump(payload, manifest_path)
        print(
            f"Fold {fold}: {len(all_records)} candidates "
            f"({positives} positive) -> {manifest_path}"
        )


if __name__ == "__main__":
    main()
