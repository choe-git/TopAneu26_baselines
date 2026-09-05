"""Train one held-out fold of the joint candidate location refiner."""

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
from rnsa_surrogate.refiner_data import (
    CandidateROIDataset,
    load_candidate_manifest,
    manifest_records,
)
from rnsa_surrogate.refiner_location_model import CandidateLocationRefiner
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, help="Override refiner_location.epochs")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def validated_manifests(
    layout: BaselineRunLayout,
    folds: dict[str, Any],
    cache_sha: str,
    fold_sha: str,
) -> list[dict[str, Any]]:
    manifests = []
    for fold in range(int(folds["n_folds"])):
        path = layout.refiner_candidates / "oof" / f"fold_{fold}" / "manifest.json"
        manifest = load_candidate_manifest(path)
        if int(manifest["fold"]) != fold:
            raise ValueError(f"Candidate manifest fold mismatch: {path}")
        if manifest["cache_index_sha256"] != cache_sha:
            raise ValueError(f"Candidate manifest cache mismatch: {path}")
        if manifest["fold_manifest_sha256"] != fold_sha:
            raise ValueError(f"Candidate fold provenance mismatch: {path}")
        if set(manifest["case_ids"]) != set(folds["folds"][str(fold)]):
            raise ValueError(f"Candidate manifest cases differ from fold {fold}")
        for record in manifest["candidates"]:
            case_id = str(record["case_id"])
            generator = int(record["generator_fold"])
            if generator != fold or generator != int(folds["case_to_fold"][case_id]):
                raise ValueError(f"Leaked candidate generator for {case_id}")
        manifests.append(manifest)
    return manifests


def run_epoch(
    model: CandidateLocationRefiner,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    threshold: float,
    objectness_weight: float,
    location_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_values, objectness_values, location_values = [], [], []
    targets, probabilities = [], []
    location_correct = 0
    location_count = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(
            loader,
            desc="Location refiner train" if training else "Location refiner val",
            leave=False,
        )
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            stage1_class = batch["stage1_class"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            target_class = batch["target_class"].to(device, non_blocking=True)
            with torch.autocast(
                device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                outputs = model(image, metadata, stage1_class)
                objectness_loss = F.binary_cross_entropy_with_logits(
                    outputs["objectness_logits"], target
                )
                positive = target_class > 0
                location_loss = (
                    F.cross_entropy(outputs["location_logits"][positive], target_class[positive])
                    if torch.any(positive)
                    else outputs["location_logits"].sum() * 0.0
                )
                loss = (
                    objectness_weight * objectness_loss
                    + location_weight * location_loss
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite location refiner loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            probability = torch.sigmoid(outputs["objectness_logits"])
            if torch.any(positive):
                predicted_class = outputs["location_logits"][positive, 1:53].argmax(
                    dim=1
                ) + 1
                location_correct += int(
                    (predicted_class == target_class[positive]).sum().item()
                )
                location_count += int(positive.sum().item())
            total_values.append(float(loss.detach()))
            objectness_values.append(float(objectness_loss.detach()))
            location_values.append(float(location_loss.detach()))
            targets.append(target.detach().cpu().numpy())
            probabilities.append(probability.detach().cpu().numpy())
            progress.set_postfix(loss=f"{total_values[-1]:.4f}")
    metrics = classification_metrics(
        np.concatenate(targets), np.concatenate(probabilities), threshold
    )
    metrics["location_accuracy"] = (
        float(location_correct / location_count) if location_count else 0.0
    )
    metrics["location_candidates"] = location_count
    return {
        "loss": float(np.mean(total_values)),
        "objectness_loss": float(np.mean(objectness_values)),
        "location_loss": float(np.mean(location_values)),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    layout = BaselineRunLayout.from_root(args.run_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config.get("refiner_location", {}))
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        settings["epochs"] = int(args.epochs)
    if args.smoke_test:
        settings.update(epochs=1, train_samples=4, num_workers=0)
    config["refiner_location"] = settings
    folds_path = layout.fold_manifest.resolve()
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    if not 0 <= args.fold < int(folds["n_folds"]):
        raise ValueError(f"Invalid fold: {args.fold}")
    cache_index = load_cache_index(layout.cache)
    cache_sha = sha256_file(cache_index["index_path"])
    fold_sha = sha256_file(folds_path)
    if folds.get("cache_index_sha256") != cache_sha:
        raise ValueError("Fold manifest and cache SHA256 differ")
    manifests = validated_manifests(layout, folds, cache_sha, fold_sha)
    train_records = [
        record
        for fold, manifest in enumerate(manifests)
        if fold != args.fold
        for record in manifest_records(manifest)
    ]
    val_records = manifest_records(manifests[args.fold])
    train_ids = {str(record["case_id"]) for record in train_records}
    val_ids = {str(record["case_id"]) for record in val_records}
    if train_ids & val_ids:
        raise AssertionError("Location refiner train/validation case leakage")

    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    seed = int(config["experiment"].get("seed", 2026)) + 2000 + args.fold
    seed_everything(seed)
    roi_size = tuple(settings.get("roi_size", (48, 64, 64)))
    train_dataset = CandidateROIDataset(
        layout.cache, train_records, roi_size, augment=True, seed=seed
    )
    val_dataset = CandidateROIDataset(
        layout.cache, val_records, roi_size, augment=False, seed=seed + 10_000_000
    )
    sampler = BalancedCandidateSampler(
        train_dataset,
        int(settings.get("train_samples", max(len(train_records), 256))),
        float(settings.get("positive_fraction", 0.5)),
        seed,
    )
    workers = int(settings.get("num_workers", 2))
    options: dict[str, Any] = {
        "batch_size": int(settings.get("batch_size", 8)),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **options)
    val_loader = DataLoader(val_dataset, shuffle=False, **options)
    model_config = {
        "in_channels": 2,
        "base_channels": int(settings.get("base_channels", 12)),
        "metadata_features": 11,
        "embedding_channels": int(settings.get("embedding_channels", 16)),
        "location_classes": 52,
        "dropout": float(settings.get("dropout", 0.15)),
    }
    model = CandidateLocationRefiner(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 2e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    epochs = int(settings.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    threshold = float(settings.get("fixed_threshold", 0.5))
    objectness_weight = float(settings.get("objectness_weight", 1.0))
    location_weight = float(settings.get("location_weight", 1.0))
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and str(settings.get("amp", "bf16")) == "bf16"
        else None
    )
    manifest_hashes = {
        str(manifest["fold"]): sha256_file(manifest["manifest_path"])
        for manifest in manifests
    }
    contract = {
        "stage": "candidate_location_refiner",
        "settings": settings,
        "model": model_config,
        "fold": args.fold,
        "cache_index_sha256": cache_sha,
        "fold_manifest_sha256": fold_sha,
        "candidate_manifest_sha256s": manifest_hashes,
    }
    contract_sha = config_digest(contract)
    output = layout.refiner_location_folds / f"fold_{args.fold}"
    tensorboard = (
        layout.refiner_location_tensorboard / "folds" / f"fold_{args.fold}"
    )
    resume = args.resume.resolve() if args.resume is not None else None
    start_epoch, best_validation_loss = 0, float("inf")
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(config, output / "config.json")
        atomic_json_dump(environment_payload(), output / "environment.json")
        atomic_json_dump(
            {
                "fold": args.fold,
                "train_folds": [
                    fold for fold in range(len(manifests)) if fold != args.fold
                ],
                "validation_fold": args.fold,
                "train_cases": len(train_ids),
                "validation_cases": len(val_ids),
                "train_candidates": len(train_records),
                "validation_candidates": len(val_records),
                "cache_index": cache_index["index_path"],
                "cache_index_sha256": cache_sha,
                "fold_manifest": str(folds_path),
                "fold_manifest_sha256": fold_sha,
                "candidate_manifest_sha256s": manifest_hashes,
                "contract_sha256": contract_sha,
                "selection": (
                    f"minimum validation multitask loss; candidate metrics use "
                    f"fixed threshold {threshold:.2f}"
                ),
            },
            output / "inputs.json",
        )
    else:
        if resume.parent != output.resolve():
            raise ValueError(f"Resume checkpoint must be inside {output}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "candidate_location_refiner":
            raise ValueError(f"Not a location refiner checkpoint: {resume}")
        if checkpoint["contract_sha256"] != contract_sha:
            raise ValueError("Location refiner resume contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        restore_rng_state(checkpoint.get("rng"))
    atomic_json_dump(
        {
            "stage": "candidate_location_refiner",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "model": model_config,
        },
        output / "model.json",
    )
    atomic_json_dump(
        {
            "cache_index_sha256": cache_sha,
            "fold_manifest_sha256": fold_sha,
            "candidate_manifest_sha256s": manifest_hashes,
            "heldout_fold": args.fold,
            "organizer_vessel_input": False,
        },
        output / "provenance.json",
    )
    writer = SummaryWriter(tensorboard, purge_step=start_epoch + 1)
    status = output / "status.json"
    write_status(status, "running", fold=args.fold, device=str(device))
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            started = perf_counter()
            sampler.set_epoch(epoch - 1)
            train_result = run_epoch(
                model, train_loader, device, amp_dtype, threshold,
                objectness_weight, location_weight, optimizer
            )
            val_result = run_epoch(
                model, val_loader, device, amp_dtype, threshold,
                objectness_weight, location_weight
            )
            improved = val_result["loss"] < best_validation_loss
            best_validation_loss = min(
                best_validation_loss, float(val_result["loss"])
            )
            scheduler.step()
            state = {
                "stage": "candidate_location_refiner",
                "fold": args.fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "model_config": model_config,
                "roi_size": list(roi_size),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_validation_loss": best_validation_loss,
                "fixed_threshold": threshold,
                "contract_sha256": contract_sha,
                "candidate_manifest_sha256s": manifest_hashes,
                "cache_index_sha256": cache_sha,
                "fold_manifest_sha256": fold_sha,
                "rng": rng_state(),
            }
            if improved:
                save_checkpoint(state, output / "checkpoint_best.pth")
            save_checkpoint(state, output / "checkpoint_latest.pth")
            for split, result in (("train", train_result), ("val", val_result)):
                atomic_json_dump(
                    {"epoch": epoch, "split": split, **result},
                    output / "metrics" / split / f"epoch_{epoch:04d}.json",
                )
                writer.add_scalar(f"loss/{split}", result["loss"], epoch)
                writer.add_scalar(
                    f"loss_components/{split}/objectness",
                    result["objectness_loss"],
                    epoch,
                )
                writer.add_scalar(
                    f"loss_components/{split}/location",
                    result["location_loss"],
                    epoch,
                )
            for name in ("precision", "recall", "mcc", "location_accuracy"):
                writer.add_scalar(
                    f"candidate/val/{name}",
                    float(val_result["metrics"][name]),
                    epoch,
                )
            writer.add_scalar(
                "optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )
            append_log(
                output / "training_log.txt",
                f"Epoch {epoch}/{epochs}: train={train_result['loss']:.6f}, "
                f"val={val_result['loss']:.6f}, "
                f"P/R/MCC={val_result['metrics']['precision']:.4f}/"
                f"{val_result['metrics']['recall']:.4f}/"
                f"{val_result['metrics']['mcc']:.4f}, "
                f"location_acc={val_result['metrics']['location_accuracy']:.4f}, "
                f"time={perf_counter() - started:.1f}s",
            )
            writer.flush()
        write_status(
            status,
            "completed",
            fold=args.fold,
            epochs=epochs,
            best_validation_loss=best_validation_loss,
            checkpoint=str(output / "checkpoint_best.pth"),
        )
    except BaseException as error:
        write_status(
            status, "failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
