"""Train one leakage-safe fold of the stage-2 objectness refiner."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rnsa_surrogate.cache import atomic_json_dump, load_cache_index, sha256_file
from rnsa_surrogate.refiner_data import (
    CandidateROIDataset,
    load_candidate_manifest,
    manifest_records,
)
from rnsa_surrogate.refiner_model import CandidateObjectnessRefiner
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


class BalancedCandidateSampler(Sampler[int]):
    def __init__(
        self,
        dataset: CandidateROIDataset,
        samples: int,
        positive_fraction: float,
        seed: int,
    ) -> None:
        self.positive = dataset.positive_indices
        self.negative = dataset.negative_indices
        if not len(self.positive) or not len(self.negative):
            raise ValueError("Refiner training requires positive and negative candidates")
        self.samples = int(samples)
        self.positive_fraction = float(positive_fraction)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        positive_count = int(round(self.samples * self.positive_fraction))
        values = np.concatenate(
            [
                rng.choice(self.positive, positive_count, replace=True),
                rng.choice(
                    self.negative, self.samples - positive_count, replace=True
                ),
            ]
        )
        rng.shuffle(values)
        return iter(int(value) for value in values)

    def __len__(self) -> int:
        return self.samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--candidate-variant",
        default="candidates",
        help="Read OOF manifests from baseline/refiner/NAME",
    )
    parser.add_argument(
        "--refiner-variant",
        default="refiner",
        help="Write checkpoints below baseline/NAME and TensorBoard below NAME",
    )
    return parser.parse_args()


def classification_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    truth = targets > 0.5
    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(mcc),
        "score": float(np.mean([precision, recall, mcc])),
    }


def run_epoch(
    model: CandidateObjectnessRefiner,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    threshold: float,
    optimizer: torch.optim.Optimizer | None = None,
    tune_threshold: bool = False,
) -> tuple[float, dict[str, float | int], float]:
    training = optimizer is not None
    model.train(training)
    losses, targets, probabilities = [], [], []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc="Refiner train" if training else "Refiner val")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            metadata = batch["metadata"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            with torch.autocast(
                device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                logits = model(image, metadata)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite refiner loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            losses.append(float(loss.detach()))
            targets.append(target.detach().cpu().numpy())
            probabilities.append(
                torch.sigmoid(logits).detach().float().cpu().numpy()
            )
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
    target_array = np.concatenate(targets)
    probability_array = np.concatenate(probabilities)
    selected_threshold = threshold
    metrics = classification_metrics(target_array, probability_array, threshold)
    if tune_threshold:
        candidates = np.linspace(0.05, 0.95, 19)
        ranked = [
            (classification_metrics(target_array, probability_array, float(value)), value)
            for value in candidates
        ]
        metrics, selected_threshold = max(
            ranked,
            key=lambda item: (
                float(item[0]["score"]),
                float(item[0]["mcc"]),
                -abs(float(item[1]) - 0.5),
            ),
        )
    return float(np.mean(losses)), metrics, float(selected_threshold)


def main() -> None:
    args = parse_args()
    layout = BaselineRunLayout.from_root(args.run_dir)
    candidate_root = layout.refiner_candidates_for(args.candidate_variant)
    refiner_folds = layout.refiner_folds_for(args.refiner_variant)
    refiner_tensorboard = layout.refiner_tensorboard_for(args.refiner_variant)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = dict(config.get("refiner", {}))
    if args.smoke_test:
        settings.update(epochs=1, train_samples=4, num_workers=0)
    fold_manifest_path = layout.fold_manifest.resolve()
    fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if not 0 <= args.fold < int(fold_manifest["n_folds"]):
        raise ValueError(f"Invalid fold: {args.fold}")
    cache_index = load_cache_index(layout.cache)
    cache_sha = sha256_file(cache_index["index_path"])
    fold_sha = sha256_file(fold_manifest_path)
    if fold_manifest.get("cache_index_sha256") != cache_sha:
        raise ValueError("Fold manifest and current cache have different SHA256")

    manifests = []
    for fold in range(int(fold_manifest["n_folds"])):
        path = candidate_root / "oof" / f"fold_{fold}" / "manifest.json"
        manifest = load_candidate_manifest(path)
        manifest_variant = str(manifest.get("candidate_variant", "candidates"))
        if manifest_variant != args.candidate_variant:
            raise ValueError(
                f"Candidate variant mismatch: requested {args.candidate_variant}, "
                f"manifest records {manifest_variant}: {path}"
            )
        if int(manifest["fold"]) != fold:
            raise ValueError(f"Candidate manifest fold mismatch: {path}")
        if manifest["cache_index_sha256"] != cache_sha:
            raise ValueError(f"Candidate manifest cache mismatch: {path}")
        if manifest["fold_manifest_sha256"] != fold_sha:
            raise ValueError(f"Candidate manifest fold provenance mismatch: {path}")
        expected_cases = set(fold_manifest["folds"][str(fold)])
        if set(manifest["case_ids"]) != expected_cases:
            raise ValueError(f"Candidate manifest cases differ from fold {fold}")
        for record in manifest["candidates"]:
            case_id = str(record["case_id"])
            generator_fold = int(record["generator_fold"])
            if generator_fold != int(fold_manifest["case_to_fold"][case_id]):
                raise ValueError(f"Leaked candidate generator for {case_id}")
        manifests.append(manifest)
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
        raise AssertionError("Refiner train/validation case leakage")

    seed = int(config["experiment"].get("seed", 2026)) + 1000 + args.fold
    seed_everything(seed)
    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    roi_size = tuple(settings.get("roi_size", (48, 64, 64)))
    train_dataset = CandidateROIDataset(
        layout.cache, train_records, roi_size, augment=True, seed=seed
    )
    val_dataset = CandidateROIDataset(
        layout.cache, val_records, roi_size, augment=False, seed=seed + 10_000_000
    )
    sampler = BalancedCandidateSampler(
        train_dataset,
        int(settings.get("train_samples", max(len(train_dataset), 256))),
        float(settings.get("positive_fraction", 0.5)),
        seed,
    )
    workers = int(settings.get("num_workers", 2))
    loader_options: dict[str, Any] = {
        "batch_size": int(settings.get("batch_size", 8)),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    model_config = {
        "in_channels": 2,
        "base_channels": int(settings.get("base_channels", 12)),
        "metadata_features": 11,
        "dropout": float(settings.get("dropout", 0.15)),
    }
    model = CandidateObjectnessRefiner(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 2e-4)),
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    epochs = int(settings.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    threshold = float(settings.get("selection_threshold", 0.5))
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
        "refiner": settings,
        "model": model_config,
        "fold": args.fold,
        "cache_index_sha256": cache_sha,
        "fold_manifest_sha256": fold_sha,
        "candidate_manifest_sha256s": manifest_hashes,
    }
    # Preserve the exact legacy resume contract for the canonical defaults.
    if args.candidate_variant != "candidates" or args.refiner_variant != "refiner":
        contract["candidate_variant"] = args.candidate_variant
        contract["refiner_variant"] = args.refiner_variant
    contract_sha = config_digest(contract)
    output = refiner_folds / f"fold_{args.fold}"
    tensorboard = refiner_tensorboard / "folds" / f"fold_{args.fold}"
    resume = args.resume.resolve() if args.resume is not None else None
    start_epoch, best_score = 0, float("-inf")
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
                "candidate_manifest_sha256s": manifest_hashes,
                "candidate_variant": args.candidate_variant,
                "refiner_variant": args.refiner_variant,
                "cache_index": cache_index["index_path"],
                "cache_index_sha256": cache_sha,
                "fold_manifest": str(fold_manifest_path),
                "fold_manifest_sha256": fold_sha,
                "contract_sha256": contract_sha,
            },
            output / "inputs.json",
        )
    else:
        if resume.parent != output.resolve():
            raise ValueError(f"Resume checkpoint must be inside {output}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "objectness_refiner":
            raise ValueError(f"Not a refiner checkpoint: {resume}")
        if checkpoint.get("candidate_variant", "candidates") != args.candidate_variant:
            raise ValueError("Resume checkpoint candidate variant differs")
        if checkpoint.get("refiner_variant", "refiner") != args.refiner_variant:
            raise ValueError("Resume checkpoint refiner variant differs")
        if checkpoint["contract_sha256"] != contract_sha:
            raise ValueError("Refiner resume contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_score"])
        restore_rng_state(checkpoint.get("rng"))
    atomic_json_dump(
        {
            "stage": "objectness_refiner",
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
            "candidate_variant": args.candidate_variant,
            "refiner_variant": args.refiner_variant,
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
            train_loss, train_metrics, _ = run_epoch(
                model, train_loader, device, amp_dtype, threshold, optimizer
            )
            val_loss, val_metrics, selected_threshold = run_epoch(
                model, val_loader, device, amp_dtype, threshold,
                tune_threshold=True,
            )
            score = float(val_metrics["score"])
            improved = score > best_score
            best_score = max(best_score, score)
            state = {
                "stage": "objectness_refiner",
                "fold": args.fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "model_config": model_config,
                "roi_size": list(roi_size),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_score": best_score,
                "selection_threshold": selected_threshold,
                "contract_sha256": contract_sha,
                "candidate_manifest_sha256s": manifest_hashes,
                "candidate_variant": args.candidate_variant,
                "refiner_variant": args.refiner_variant,
                "cache_index_sha256": cache_sha,
                "fold_manifest_sha256": fold_sha,
                "rng": rng_state(),
            }
            if improved:
                save_checkpoint(state, output / "checkpoint_best.pth")
            save_checkpoint(state, output / "checkpoint_latest.pth")
            atomic_json_dump(
                {
                    "epoch": epoch,
                    "split": "train",
                    "loss": train_loss,
                    "metrics": train_metrics,
                },
                output / "metrics" / "train" / f"epoch_{epoch:04d}.json",
            )
            atomic_json_dump(
                {
                    "epoch": epoch,
                    "split": "val",
                    "loss": val_loss,
                    "metrics": val_metrics,
                },
                output / "metrics" / "val" / f"epoch_{epoch:04d}.json",
            )
            writer.add_scalar("loss/refiner/train", train_loss, epoch)
            writer.add_scalar("loss/refiner/val", val_loss, epoch)
            for name in ("precision", "recall", "mcc", "score"):
                writer.add_scalar(
                    f"candidate/val/{name}", float(val_metrics[name]), epoch
                )
            writer.add_scalar(
                "candidate/val/selection_threshold", selected_threshold, epoch
            )
            writer.add_scalar(
                "optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )
            scheduler.step()
            append_log(
                output / "training_log.txt",
                f"Epoch {epoch}/{epochs}: train={train_loss:.6f}, "
                f"val={val_loss:.6f}, score={score:.6f}, "
                f"P/R/MCC={val_metrics['precision']:.4f}/"
                f"{val_metrics['recall']:.4f}/{val_metrics['mcc']:.4f}, "
                f"threshold={selected_threshold:.2f}, "
                f"time={perf_counter() - started:.1f}s",
            )
            writer.flush()
        write_status(
            status,
            "completed",
            fold=args.fold,
            epochs=epochs,
            best_score=best_score,
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
