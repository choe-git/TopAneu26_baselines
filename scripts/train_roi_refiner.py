"""Train a leakage-safe candidate-centred dense ROI refiner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rnsa_surrogate.cache import atomic_json_dump, load_cache_index, sha256_file
from rnsa_surrogate.refiner_data import manifest_records
from rnsa_surrogate.roi_refiner import (
    CandidateROIRefinementDataset,
    CandidateROIRefiner,
    VESSEL_CONTEXT_NONE,
    VESSEL_CONTEXT_ORACLE,
    VESSEL_CONTEXT_STAGE1,
    validate_vessel_context_records,
)
from rnsa_surrogate.run_layout import BaselineRunLayout
from rnsa_surrogate.runtime import (
    append_log,
    config_digest,
    environment_payload,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    seed_everything,
    write_status,
)
from train_refiner import BalancedCandidateSampler, classification_metrics
from train_refiner_location import validated_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--variant", default="roi_refiner")
    parser.add_argument("--candidate-variant", default="candidates")
    parser.add_argument("--vessel-context", action="store_true")
    parser.add_argument(
        "--oracle-organizer-vessel-context",
        action="store_true",
        help=(
            "DIAGNOSTIC ONLY: permit cache/vessel.npy organizer ground truth. "
            "The resulting checkpoint is marked nondeployable."
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def segmentation_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positive: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_weight = torch.as_tensor(8.0, device=logits.device, dtype=logits.dtype)
    bce_voxels = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    bce_per_case = (bce_voxels * valid).sum((1, 2, 3, 4)) / valid.sum(
        (1, 2, 3, 4)
    ).clamp_min(1)
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probability * target).sum((1, 2, 3, 4))
    predicted = probability.sum((1, 2, 3, 4))
    truth = target.sum((1, 2, 3, 4))
    dice = (2 * intersection + 1) / (predicted + truth + 1)
    false_positive = (probability * (1 - target)).sum((1, 2, 3, 4))
    false_negative = ((1 - probability) * target).sum((1, 2, 3, 4))
    alpha = float(settings.get("tversky_alpha", 0.3))
    beta = float(settings.get("tversky_beta", 0.7))
    tversky = (intersection + 1) / (
        intersection + alpha * false_positive + beta * false_negative + 1
    )
    positive_float = positive.float()
    positive_count = positive_float.sum().clamp_min(1)
    negative_float = 1 - positive_float
    negative_count = negative_float.sum().clamp_min(1)
    positive_loss = (
        float(settings.get("mask_bce_weight", 0.35))
        * (bce_per_case * positive_float).sum() / positive_count
        + float(settings.get("mask_dice_weight", 0.35))
        * ((1 - dice) * positive_float).sum() / positive_count
        + float(settings.get("mask_tversky_weight", 0.30))
        * ((1 - tversky) * positive_float).sum() / positive_count
    )
    negative_loss = (bce_per_case * negative_float).sum() / negative_count
    mask_loss = positive_loss + 0.25 * negative_loss
    return mask_loss, positive_loss, negative_loss, dice


def run_epoch(
    model: CandidateROIRefiner,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    settings: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    values: dict[str, list[float]] = {
        name: [] for name in ("total", "mask", "objectness", "location")
    }
    targets, probabilities = [], []
    location_correct = location_count = 0
    dice_sum = dice_count = 0.0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc="ROI refiner train" if training else "ROI refiner val", leave=False)
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            stage1_class = batch["stage1_class"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_class = batch["target_class"].to(device, non_blocking=True)
            target_mask = batch["target_mask"].to(device, non_blocking=True)
            valid_mask = batch["valid_mask"].to(device, non_blocking=True)
            positive = target_class > 0
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                outputs = model(image, metadata, stage1_class)
                mask_loss, _, _, dice = segmentation_losses(
                    outputs["mask_logits"], target_mask, valid_mask, positive, settings
                )
                objectness_loss = F.binary_cross_entropy_with_logits(
                    outputs["objectness_logits"], target
                )
                location_loss = (
                    F.cross_entropy(
                        outputs["location_logits"][positive], target_class[positive]
                    )
                    if torch.any(positive)
                    else outputs["location_logits"].sum() * 0
                )
                loss = (
                    float(settings.get("mask_weight", 1.0)) * mask_loss
                    + float(settings.get("objectness_weight", 0.35)) * objectness_loss
                    + float(settings.get("location_weight", 0.35)) * location_loss
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ROI refiner loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            probability = torch.sigmoid(outputs["objectness_logits"])
            if torch.any(positive):
                predicted_class = outputs["location_logits"][positive, 1:53].argmax(1) + 1
                location_correct += int((predicted_class == target_class[positive]).sum())
                location_count += int(positive.sum())
                dice_sum += float(dice[positive].sum())
                dice_count += float(positive.sum())
            for name, value in (
                ("total", loss), ("mask", mask_loss),
                ("objectness", objectness_loss), ("location", location_loss)
            ):
                values[name].append(float(value.detach()))
            targets.append(target.detach().cpu().numpy())
            probabilities.append(probability.detach().float().cpu().numpy())
            progress.set_postfix(loss=f"{values['total'][-1]:.4f}")
    candidate = classification_metrics(
        np.concatenate(targets),
        np.concatenate(probabilities),
        float(settings.get("fixed_objectness_threshold", 0.35)),
    )
    location_accuracy = float(location_correct / location_count) if location_count else 0.0
    mask_dice = float(dice_sum / dice_count) if dice_count else 0.0
    selection_score = float((candidate["mcc"] + location_accuracy + mask_dice) / 3)
    return {
        "loss": float(np.mean(values["total"])),
        "mask_loss": float(np.mean(values["mask"])),
        "objectness_loss": float(np.mean(values["objectness"])),
        "location_loss": float(np.mean(values["location"])),
        "metrics": {
            **candidate,
            "location_accuracy": location_accuracy,
            "mask_dice": mask_dice,
            "selection_score": selection_score,
            "positive_candidates": location_count,
        },
    }


def main() -> None:
    args = parse_args()
    if args.oracle_organizer_vessel_context and not args.vessel_context:
        raise ValueError(
            "--oracle-organizer-vessel-context requires --vessel-context"
        )
    if not args.variant.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--variant must be a simple directory name")
    if not args.candidate_variant.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--candidate-variant must be a simple directory name")
    layout = BaselineRunLayout.from_root(args.run_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config["roi_refiner"])
    if args.epochs is not None:
        settings["epochs"] = int(args.epochs)
    if args.smoke_test:
        settings.update(epochs=1, train_samples=4, num_workers=0, batch_size=1)
    folds_path = layout.fold_manifest.resolve()
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    if not 0 <= args.fold < int(folds["n_folds"]):
        raise ValueError(f"Invalid fold: {args.fold}")
    cache_index = load_cache_index(layout.cache)
    cache_sha, fold_sha = sha256_file(cache_index["index_path"]), sha256_file(folds_path)
    candidate_root = layout.refiner / args.candidate_variant
    manifests = validated_manifests(
        layout, folds, cache_sha, fold_sha, candidate_root
    )
    train_records = [
        record for fold, manifest in enumerate(manifests) if fold != args.fold
        for record in manifest_records(manifest)
    ]
    val_records = manifest_records(manifests[args.fold])
    if {r["case_id"] for r in train_records} & {r["case_id"] for r in val_records}:
        raise AssertionError("ROI refiner train/validation case leakage")
    vessel_context_source = VESSEL_CONTEXT_NONE
    if args.vessel_context:
        vessel_context_source = validate_vessel_context_records(
            [*train_records, *val_records],
            True,
            allow_oracle=args.oracle_organizer_vessel_context,
        )
        for manifest in manifests:
            declared = str(manifest.get("vessel_context_source", ""))
            if vessel_context_source == VESSEL_CONTEXT_STAGE1:
                if declared != VESSEL_CONTEXT_STAGE1:
                    raise ValueError(
                        f"Manifest {manifest['manifest_path']} does not explicitly "
                        "declare vessel_context_source=stage1_prediction"
                    )
                if bool(manifest.get("organizer_vessel_input", False)):
                    raise ValueError(
                        f"Manifest {manifest['manifest_path']} used organizer vessel input"
                    )
            elif not args.oracle_organizer_vessel_context:
                raise ValueError("Organizer vessel context requires explicit oracle mode")
    organizer_vessel_input = vessel_context_source == VESSEL_CONTEXT_ORACLE
    deployment_eligible = not organizer_vessel_input
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    seed = int(config["experiment"].get("seed", 2026)) + 3000 + args.fold
    seed_everything(seed)
    roi_size = tuple(settings.get("roi_size", (48, 64, 64)))
    train_dataset = CandidateROIRefinementDataset(
        layout.cache, train_records, roi_size, augment=True, seed=seed,
        vessel_context=args.vessel_context,
        allow_oracle_vessel_context=args.oracle_organizer_vessel_context,
    )
    val_dataset = CandidateROIRefinementDataset(
        layout.cache, val_records, roi_size, augment=False, seed=seed + 10_000_000,
        vessel_context=args.vessel_context,
        allow_oracle_vessel_context=args.oracle_organizer_vessel_context,
    )
    sampler = BalancedCandidateSampler(
        train_dataset, int(settings.get("train_samples", 768)),
        float(settings.get("positive_fraction", 0.65)), seed
    )
    workers = int(settings.get("num_workers", 2))
    loader_options = dict(
        batch_size=int(settings.get("batch_size", 2)), num_workers=workers,
        pin_memory=device.type == "cuda", persistent_workers=workers > 0
    )
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model_config = dict(
        in_channels=4 if args.vessel_context else 2,
        base_channels=int(settings.get("base_channels", 8)),
        metadata_features=11,
        embedding_channels=int(settings.get("embedding_channels", 16)),
        location_classes=52, dropout=float(settings.get("dropout", 0.15))
        , location_prior_logit=float(settings.get("location_prior_logit", 0.0))
    )
    model = CandidateROIRefiner(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings.get("learning_rate", 2e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4))
    )
    epochs = int(settings.get("epochs", 16))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp_dtype = torch.bfloat16 if device.type == "cuda" and settings.get("amp") == "bf16" else None
    manifest_hashes = {
        str(manifest["fold"]): sha256_file(manifest["manifest_path"])
        for manifest in manifests
    }
    contract = {
        "stage": "roi_refiner", "fold": args.fold, "settings": settings,
        "model": model_config, "cache_index_sha256": cache_sha,
        "fold_manifest_sha256": fold_sha,
        "candidate_manifest_sha256s": manifest_hashes,
        "candidate_variant": args.candidate_variant,
        "vessel_context": args.vessel_context,
    }
    # Keep the legacy no-vessel contract byte-for-byte compatible for resume.
    # Vessel-enabled legacy checkpoints are intentionally rejected because
    # their organizer-vs-prediction provenance cannot be proven.
    if args.vessel_context:
        contract.update(
            vessel_context_source=vessel_context_source,
            organizer_vessel_input=organizer_vessel_input,
            deployment_eligible=deployment_eligible,
        )
    contract_sha = config_digest(contract)
    output = layout.baseline / args.variant / "folds" / f"fold_{args.fold}"
    resume = args.resume.resolve() if args.resume else None
    start_epoch, best_score = 0, float("-inf")
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(config, output / "config.json")
        atomic_json_dump(environment_payload(), output / "environment.json")
        atomic_json_dump(
            {
                "fold": args.fold, "train_candidates": len(train_records),
                "validation_candidates": len(val_records),
                "cache_index_sha256": cache_sha,
                "fold_manifest_sha256": fold_sha,
                "candidate_manifest_sha256s": manifest_hashes,
                "candidate_variant": args.candidate_variant,
                "vessel_context": args.vessel_context,
                "vessel_context_source": vessel_context_source,
                "organizer_vessel_input": organizer_vessel_input,
                "deployment_eligible": deployment_eligible,
                "selection": "maximum fixed-threshold mean(candidate MCC, positive location accuracy, positive mask Dice)",
                "contract_sha256": contract_sha,
            },
            output / "inputs.json",
        )
    else:
        if resume.parent != output.resolve():
            raise ValueError(f"Resume checkpoint must be inside {output}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "roi_refiner" or checkpoint["contract_sha256"] != contract_sha:
            raise ValueError("ROI refiner resume contract differs")
        if bool(checkpoint.get("vessel_context", False)) and (
            checkpoint.get("vessel_context_source") != vessel_context_source
            or bool(checkpoint.get("deployment_eligible", False))
            != deployment_eligible
        ):
            raise ValueError(
                "Vessel-enabled resume checkpoint has missing or incompatible provenance"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        restore_rng_state(checkpoint.get("rng"))
        start_epoch, best_score = int(checkpoint["epoch"]), float(checkpoint["best_validation_score"])
    atomic_json_dump(
        {"stage": "roi_refiner", "parameters": sum(p.numel() for p in model.parameters()), "model": model_config},
        output / "model.json",
    )
    atomic_json_dump(
        {
            "stage": "roi_refiner",
            "fold": args.fold,
            "candidate_variant": args.candidate_variant,
            "candidate_manifest_sha256s": manifest_hashes,
            "cache_index_sha256": cache_sha,
            "fold_manifest_sha256": fold_sha,
            "vessel_context": args.vessel_context,
            "vessel_context_source": vessel_context_source,
            "organizer_vessel_input": organizer_vessel_input,
            "deployment_eligible": deployment_eligible,
        },
        output / "provenance.json",
    )
    writer = SummaryWriter(
        layout.tensorboard / args.variant / "folds" / f"fold_{args.fold}",
        purge_step=start_epoch + 1,
    )
    status = output / "status.json"
    write_status(status, "running", fold=args.fold, device=str(device))
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            started = perf_counter()
            sampler.set_epoch(epoch - 1)
            train_result = run_epoch(model, train_loader, device, amp_dtype, settings, optimizer)
            val_result = run_epoch(model, val_loader, device, amp_dtype, settings)
            score = float(val_result["metrics"]["selection_score"])
            improved = score > best_score
            best_score = max(best_score, score)
            scheduler.step()
            state = {
                "stage": "roi_refiner", "fold": args.fold, "epoch": epoch,
                "model": model.state_dict(), "model_config": model_config,
                "roi_size": list(roi_size), "settings": settings,
                "candidate_variant": args.candidate_variant,
                "vessel_context": args.vessel_context,
                "vessel_context_source": vessel_context_source,
                "organizer_vessel_input": organizer_vessel_input,
                "deployment_eligible": deployment_eligible,
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "best_validation_score": best_score, "contract_sha256": contract_sha,
                "candidate_manifest_sha256s": manifest_hashes,
                "cache_index_sha256": cache_sha, "fold_manifest_sha256": fold_sha,
                "rng": rng_state(),
            }
            if improved:
                save_checkpoint(state, output / "checkpoint_best.pth")
            save_checkpoint(state, output / "checkpoint_latest.pth")
            for split, result in (("train", train_result), ("val", val_result)):
                atomic_json_dump({"epoch": epoch, "split": split, **result}, output / "metrics" / split / f"epoch_{epoch:04d}.json")
                writer.add_scalar(f"loss/{split}", result["loss"], epoch)
                for name, value in result["metrics"].items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"metrics/{split}/{name}", value, epoch)
            writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            append_log(
                output / "training_log.txt",
                f"Epoch {epoch}/{epochs}: train={train_result['loss']:.6f}, val={val_result['loss']:.6f}, "
                f"score={score:.4f}, P/R/MCC={val_result['metrics']['precision']:.4f}/"
                f"{val_result['metrics']['recall']:.4f}/{val_result['metrics']['mcc']:.4f}, "
                f"loc={val_result['metrics']['location_accuracy']:.4f}, dice={val_result['metrics']['mask_dice']:.4f}, "
                f"time={perf_counter() - started:.1f}s",
            )
            writer.flush()
        write_status(status, "completed", fold=args.fold, epochs=epochs, best_validation_score=best_score)
    except BaseException as error:
        write_status(status, "failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
