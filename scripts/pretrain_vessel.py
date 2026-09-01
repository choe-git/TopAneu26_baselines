"""Fold-safe vessel U-Net pretraining before aneurysm fine-tuning."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rnsa_surrogate.cache import atomic_json_dump, sha256_file, validate_cache
from rnsa_surrogate.data import CachedTopAneuPatchDataset
from rnsa_surrogate.losses import vessel_loss
from rnsa_surrogate.model import VesselPretrainUNet
from rnsa_surrogate.run_layout import BaselineRunLayout
from rnsa_surrogate.runtime import (
    ExponentialMovingAverage,
    append_log,
    environment_payload,
    rng_state,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    write_status,
)


class EpochSampler(Sampler[int]):
    def __init__(self, length: int) -> None:
        self.length = length
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.epoch * self.length, (self.epoch + 1) * self.length))

    def __len__(self) -> int:
        return self.length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def make_loader(
    dataset: CachedTopAneuPatchDataset,
    config: dict[str, Any],
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    workers = int(config["data"].get("num_workers", 0))
    options: dict[str, Any] = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "sampler": sampler,
        "shuffle": False,
        "drop_last": sampler is not None,
    }
    if workers > 0:
        options["prefetch_factor"] = int(config["data"].get("prefetch_factor", 1))
    return DataLoader(dataset, **options)


def run_epoch(
    model: VesselPretrainUNet,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ExponentialMovingAverage | None = None,
    accumulate: int = 1,
    grad_clip: float = 12.0,
) -> float:
    training = optimizer is not None
    model.train(training)
    values = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.enable_grad() if training else torch.no_grad():
        progress = tqdm(loader, desc="Vessel train" if training else "Vessel val", leave=False)
        for step, batch in enumerate(progress):
            image = batch["image"].to(device, non_blocking=True)
            vessel = batch["vessel"].to(device, non_blocking=True)
            valid = batch["vessel_valid"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                outputs = model(image)
                loss = vessel_loss(outputs["vessel_logits"], vessel, valid)
                loss += 0.5 * vessel_loss(
                    outputs["vessel_half_logits"], vessel, valid
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite vessel pretraining loss")
            if training:
                assert scaler is not None
                scaler.scale(loss / accumulate).backward()
                update = (step + 1) % accumulate == 0 or step + 1 == len(loader)
                if update:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema is not None:
                        ema.update(model)
            value = float(loss.detach())
            values.append(value)
            progress.set_postfix(loss=f"{value:.4f}")
    return float(sum(values) / len(values))


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    settings = config.get("vessel_pretrain", {})
    layout = BaselineRunLayout.from_root(args.run_dir)
    cache_report = validate_cache(layout.cache, deep=False)
    fold_manifest_path = layout.root / "folds.json"
    folds = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if not 0 <= args.fold < int(folds["n_folds"]):
        raise ValueError(f"Invalid fold {args.fold}")
    val_ids = set(folds["folds"][str(args.fold)])
    train_ids = set(folds["case_to_fold"]) - val_ids
    output = layout.root / "vessel_pretrain" / f"fold_{args.fold}"
    resume_path = args.resume.resolve() if args.resume is not None else None
    if resume_path is not None:
        if resume_path.parent != output.resolve():
            raise ValueError(f"Resume checkpoint must be inside {output}")
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(output)
        output.mkdir(parents=True)

    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested)
    seed = int(config["experiment"].get("seed", 2026)) + args.fold
    seed_everything(seed)
    patch_size = tuple(config["data"]["patch_size"])
    train_dataset = CachedTopAneuPatchDataset(
        layout.cache,
        split=None,
        case_ids=train_ids,
        patch_size=patch_size,
        samples=int(settings.get("train_samples", config["data"]["train_samples"])),
        positive_fraction=0.0,
        vessel_negative_fraction=float(settings.get("vessel_fraction", 0.9)),
        augment=True,
        seed=seed,
    )
    val_dataset = CachedTopAneuPatchDataset(
        layout.cache,
        split=None,
        case_ids=val_ids,
        patch_size=patch_size,
        samples=int(settings.get("val_samples", config["data"]["val_samples"])),
        positive_fraction=0.0,
        vessel_negative_fraction=1.0,
        augment=False,
        seed=seed + 10_000_000,
    )
    sampler = EpochSampler(len(train_dataset))
    train_loader = make_loader(train_dataset, config, device, sampler)
    val_loader = make_loader(val_dataset, config, device)
    model = VesselPretrainUNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("learning_rate", 1e-4)),
        weight_decay=float(config["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(settings.get("epochs", 30))
    warmup_epochs = int(settings.get("warmup_epochs", 3))

    def learning_rate_scale(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)
    dtype = torch.bfloat16 if config["train"].get("amp") == "bf16" else None
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    ema = ExponentialMovingAverage(model, float(config["train"].get("ema_decay", 0.999)))
    accumulate = int(config["train"].get("accumulate_steps", 1))
    start_epoch = 0
    best = float("inf")
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage") != "vessel_pretrain":
            raise ValueError(f"Not a vessel pretraining checkpoint: {resume_path}")
        if int(checkpoint["fold"]) != args.fold:
            raise ValueError("Resume checkpoint fold mismatch")
        if checkpoint["cache_index_sha256"] != cache_report["index_sha256"]:
            raise ValueError("Resume cache provenance differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.load_state_dict(checkpoint["ema"], device)
        start_epoch = int(checkpoint["epoch"])
        best = float(checkpoint["best_validation"])
        restore_rng_state(checkpoint.get("rng"))
    writer = SummaryWriter(
        layout.root / "tensorboard" / "vessel_pretrain" / f"fold_{args.fold}",
        purge_step=start_epoch + 1,
    )
    if resume_path is None:
        atomic_json_dump(config, output / "config.json")
        atomic_json_dump(environment_payload(), output / "environment.json")
        atomic_json_dump(
            {
                "fold": args.fold,
                "fold_manifest": str(fold_manifest_path),
                "fold_manifest_sha256": sha256_file(fold_manifest_path),
                "cache_index_sha256": cache_report["index_sha256"],
                "train_cases": len(train_ids),
                "val_cases": len(val_ids),
            },
            output / "inputs.json",
        )
    status = output / "status.json"
    write_status(status, "running", fold=args.fold, device=str(device))
    try:
        for epoch in range(start_epoch + 1, epochs + 1):
            started = perf_counter()
            sampler.set_epoch(epoch - 1)
            train_loss = run_epoch(
                model,
                train_loader,
                device,
                dtype,
                optimizer,
                scaler,
                ema,
                accumulate,
                float(config["train"].get("grad_clip", 12.0)),
            )
            with ema.average_parameters(model):
                val_loss = run_epoch(model, val_loader, device, dtype)
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar(
                "optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )
            scheduler.step()
            state = {
                "stage": "vessel_pretrain",
                "fold": args.fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation": min(best, val_loss),
                "config": config,
                "cache_index_sha256": cache_report["index_sha256"],
                "rng": rng_state(),
            }
            if val_loss < best:
                best = val_loss
                save_checkpoint(state, output / "checkpoint_best.pth")
            save_checkpoint(state, output / "checkpoint_latest.pth")
            append_log(
                output / "training_log.txt",
                f"Epoch {epoch}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}, "
                f"time={perf_counter() - started:.1f}s",
            )
            writer.flush()
        write_status(
            status,
            "completed",
            fold=args.fold,
            epochs=epochs,
            best_validation=best,
            checkpoint=str(output / "checkpoint_best.pth"),
        )
    except BaseException as error:
        write_status(status, "failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
