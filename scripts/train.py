"""Train RNSA surrogate exclusively from a completed physical-space cache."""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rnsa_surrogate.cache import (
    atomic_json_dump,
    load_cache_index,
    sha256_file,
    validate_cache,
)
from rnsa_surrogate.data import CachedTopAneuPatchDataset
from rnsa_surrogate.losses import multitask_loss
from rnsa_surrogate.model import RNSASurrogate
from rnsa_surrogate.run_layout import BaselineRunLayout, create_legacy_run
from rnsa_surrogate.runtime import (
    ExponentialMovingAverage,
    append_log,
    config_digest,
    environment_payload,
    restore_rng_state,
    rng_state,
    save_checkpoint,
    seed_everything,
    write_status,
)


class EpochIndexSampler(Sampler[int]):
    """Give the deterministic dataset a fresh, resumable index range each epoch."""

    def __init__(self, length: int) -> None:
        self.length = int(length)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        start = self.epoch * self.length
        return iter(range(start, start + self.length))

    def __len__(self) -> int:
        return self.length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--run-dir", type=Path, help="Shared timestamped cache/baseline root"
    )
    destination.add_argument(
        "--output-root", type=Path, help="Legacy name/timestamp output root"
    )
    parser.add_argument("--cache", type=Path, help="Override RUN_DIR/cache")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"))
    parser.add_argument(
        "--resume", type=Path, help="Resume checkpoint_latest.pth in place"
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def training_contract(config: dict[str, Any]) -> dict[str, Any]:
    train = config["train"]
    return {
        "model": config["model"],
        "loss": config["loss"],
        "data": {
            key: config["data"][key]
            for key in (
                "target_spacing_zyx",
                "patch_size",
                "positive_fraction",
                "vessel_negative_fraction",
                "train_samples",
                "val_samples",
                "test_samples",
            )
        },
        "train": {
            key: train[key]
            for key in (
                "batch_size",
                "accumulate_steps",
                "learning_rate",
                "weight_decay",
                "warmup_epochs",
                "amp",
                "ema_decay",
            )
        },
    }


def make_dataset(
    config: dict[str, Any],
    cache_dir: Path,
    split: str,
    samples: int,
    augment: bool,
    seed: int,
) -> CachedTopAneuPatchDataset:
    data = config["data"]
    return CachedTopAneuPatchDataset(
        cache_dir=cache_dir,
        split=split,
        patch_size=tuple(data["patch_size"]),
        samples=samples,
        positive_fraction=float(data["positive_fraction"]),
        vessel_negative_fraction=float(data["vessel_negative_fraction"]),
        augment=augment,
        seed=seed,
    )


def make_loader(
    dataset: CachedTopAneuPatchDataset,
    config: dict[str, Any],
    device: torch.device,
    sampler: Sampler[int] | None = None,
    workers: int | None = None,
) -> DataLoader:
    workers = int(config["data"]["num_workers"] if workers is None else workers)
    options: dict[str, Any] = {
        "batch_size": int(config["train"]["batch_size"]),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": workers > 0,
        "drop_last": sampler is not None,
    }
    if workers > 0:
        options["prefetch_factor"] = int(config["data"].get("prefetch_factor", 1))
    return DataLoader(dataset, sampler=sampler, shuffle=False, **options)


def amp_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "none":
        return None
    if device.type != "cuda" and name == "fp16":
        raise ValueError("fp16 autocast is only supported here on CUDA")
    return {"bf16": torch.bfloat16, "fp16": torch.float16}[name]


def scheduler_for(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def mean_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([item[key] for item in values])) for key in values[0]}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    dtype: torch.dtype | None,
    split: str,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
    accumulate: int = 1,
    grad_clip: float = 12.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    values = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        batches = tqdm(loader, desc=split.capitalize(), leave=False)
        for step, batch in enumerate(batches):
            batch = {
                key: value.to(device, non_blocking=True) for key, value in batch.items()
            }
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                outputs = model(batch["image"])
                loss, metrics = multitask_loss(outputs, batch, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {split} loss at step {step + 1}")
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
                    if scheduler is not None:
                        scheduler.step()
                    if ema is not None:
                        ema.update(model)
            values.append(metrics)
            batches.set_postfix(loss=f"{metrics['total']:.4f}")
    return mean_metrics(values)


def write_metrics(
    model_dir: Path, split: str, epoch: int, metrics: dict[str, float]
) -> None:
    atomic_json_dump(
        {"epoch": epoch, "split": split, "metrics": metrics},
        model_dir / "metrics" / split / f"epoch_{epoch:04d}.json",
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"Config root must be a mapping: {args.config}")
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.smoke_test:
        config["train"]["epochs"] = 1
        config["data"]["train_samples"] = 2
        config["data"]["val_samples"] = 2
        config["data"]["test_samples"] = 2
        config["data"]["num_workers"] = 0

    resume_path = args.resume.resolve() if args.resume is not None else None
    if args.run_dir is not None:
        layout = BaselineRunLayout.from_root(args.run_dir)
        run_root = layout.root
        cache_dir = (args.cache or layout.cache).resolve()
        model_dir = layout.baseline
        tensorboard_dir = layout.tensorboard
        if resume_path is not None and resume_path.parent != model_dir:
            raise ValueError(f"Resume checkpoint must be inside {model_dir}")
    else:
        if args.cache is None:
            raise ValueError("--cache is required when --run-dir is not used")
        cache_dir = args.cache.resolve()
        if resume_path is not None:
            model_dir = resume_path.parent
        else:
            model_dir = create_legacy_run(
                args.output_root or Path("runs"), str(config["experiment"]["name"])
            )
        run_root = model_dir
        tensorboard_dir = model_dir / "tensorboard"
    status_path = model_dir / "status.json"

    seed = int(config["experiment"].get("seed", 2026))
    seed_everything(seed)
    device = resolve_device(str(config["train"].get("device", "cuda")))
    cache_report = validate_cache(cache_dir, deep=False)
    cache_index = load_cache_index(cache_dir)
    if not np.allclose(
        cache_index["target_spacing_zyx"], config["data"]["target_spacing_zyx"]
    ):
        raise ValueError("Cache spacing differs from data.target_spacing_zyx")

    contract = training_contract(config)
    contract_sha256 = config_digest(contract)
    checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage", "baseline") != "baseline":
            raise ValueError(
                f"Checkpoint stage is not baseline: {checkpoint.get('stage')}"
            )
        if checkpoint["contract_sha256"] != contract_sha256:
            raise ValueError("Resume training contract differs from checkpoint")
        if checkpoint["cache_index_sha256"] != cache_report["index_sha256"]:
            raise ValueError("Resume cache provenance differs from checkpoint")

    if checkpoint is None:
        if args.run_dir is not None:
            if model_dir.exists() and any(model_dir.iterdir()):
                raise FileExistsError(f"Baseline run already exists: {model_dir}")
            run_root.mkdir(parents=True, exist_ok=True)
            model_dir.mkdir(parents=False, exist_ok=False)
        atomic_json_dump(config, model_dir / "config.json")
        atomic_json_dump(environment_payload(), model_dir / "environment.json")
        atomic_json_dump(
            {
                "cache_index": cache_report["index"],
                "cache_index_sha256": cache_report["index_sha256"],
                "config_source": str(args.config.resolve()),
                "config_source_sha256": sha256_file(args.config),
                "contract_sha256": contract_sha256,
            },
            model_dir / "inputs.json",
        )
    else:
        atomic_json_dump(config, model_dir / "config.json")
    atomic_json_dump(
        {
            "data": {
                "cache_index": cache_report["index"],
                "cache_index_sha256": cache_report["index_sha256"],
                "target_spacing_zyx": cache_index["target_spacing_zyx"],
            }
        },
        model_dir / "provenance.json",
    )
    train_samples = int(config["data"]["train_samples"])
    val_samples = int(config["data"]["val_samples"])
    test_samples = int(config["data"].get("test_samples", val_samples))
    train_dataset = make_dataset(config, cache_dir, "train", train_samples, True, seed)
    val_dataset = make_dataset(
        config, cache_dir, "val", val_samples, False, seed + 10_000_000
    )
    test_every = int(config["train"].get("test_every", 0))
    test_dataset = (
        make_dataset(config, cache_dir, "test", test_samples, False, seed + 20_000_000)
        if test_every > 0
        else None
    )
    train_sampler = EpochIndexSampler(len(train_dataset))
    train_loader = make_loader(train_dataset, config, device, sampler=train_sampler)
    val_loader = make_loader(val_dataset, config, device)
    test_loader = (
        make_loader(
            test_dataset,
            config,
            device,
            workers=int(config["data"].get("test_num_workers", 0)),
        )
        if test_dataset is not None
        else None
    )

    model = RNSASurrogate(**config["model"]).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    atomic_json_dump({"parameters": parameter_count}, model_dir / "model.json")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(config["train"]["epochs"])
    accumulate = int(config["train"]["accumulate_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulate)
    scheduler = scheduler_for(
        optimizer,
        epochs * updates_per_epoch,
        int(config["train"].get("warmup_epochs", 0)) * updates_per_epoch,
    )
    dtype = amp_dtype(str(config["train"].get("amp", "none")), device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and dtype == torch.float16
    )
    ema = ExponentialMovingAverage(
        model, float(config["train"].get("ema_decay", 0.999))
    )
    start_epoch, best_validation = 0, float("inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.load_state_dict(checkpoint["ema"], device)
        start_epoch = int(checkpoint["epoch"])
        best_validation = float(checkpoint["best_validation"])
        restore_rng_state(checkpoint.get("rng"))
        carried_best = (model_dir / "checkpoint_best.pth").is_file()
        if not carried_best:
            best_validation = float("inf")
        atomic_json_dump(
            {
                "checkpoint": str(resume_path),
                "start_epoch": start_epoch + 1,
                "carried_best": carried_best,
                "best_validation_reset": not carried_best,
                "in_place": True,
            },
            model_dir / "resume.json",
        )

    if start_epoch > epochs:
        raise ValueError(
            f"Checkpoint already passed epoch {start_epoch}; requested epochs={epochs}"
        )

    write_status(
        status_path, "running", device=str(device), resumed=checkpoint is not None
    )
    writer = SummaryWriter(tensorboard_dir, purge_step=start_epoch + 1)
    log_path = model_dir / "training_log.txt"
    try:
        for epoch_index in range(start_epoch, epochs):
            epoch = epoch_index + 1
            started = perf_counter()
            train_sampler.set_epoch(epoch_index)
            train_metrics = run_epoch(
                model,
                train_loader,
                device,
                config["loss"],
                dtype,
                "train",
                optimizer,
                scaler,
                scheduler,
                ema,
                accumulate,
                float(config["train"].get("grad_clip", 12.0)),
            )
            write_metrics(model_dir, "train", epoch, train_metrics)
            for name, value in train_metrics.items():
                writer.add_scalar(f"loss_components/train/{name}", value, epoch)
            writer.add_scalar("loss/train", train_metrics["total"], epoch)
            writer.add_scalar(
                "optimizer/learning_rate", optimizer.param_groups[0]["lr"], epoch
            )

            validate_every = int(config["train"].get("validate_every", 1))
            validation_metrics = None
            if epoch % validate_every == 0 or epoch == epochs:
                with ema.average_parameters(model):
                    validation_metrics = run_epoch(
                        model, val_loader, device, config["loss"], dtype, "validation"
                    )
                write_metrics(model_dir, "val", epoch, validation_metrics)
                for name, value in validation_metrics.items():
                    writer.add_scalar(f"loss_components/val/{name}", value, epoch)
                writer.add_scalar("loss/val", validation_metrics["total"], epoch)

            if test_loader is not None and (epoch % test_every == 0 or epoch == epochs):
                with ema.average_parameters(model):
                    test_metrics = run_epoch(
                        model, test_loader, device, config["loss"], dtype, "test"
                    )
                write_metrics(model_dir, "test", epoch, test_metrics)
                for name, value in test_metrics.items():
                    writer.add_scalar(f"loss_components/test/{name}", value, epoch)
                writer.add_scalar("loss/test", test_metrics["total"], epoch)

            state = {
                "stage": "baseline",
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation": best_validation,
                "config": config,
                "contract_sha256": contract_sha256,
                "cache_index_sha256": cache_report["index_sha256"],
                "rng": rng_state(),
            }
            if (
                validation_metrics is not None
                and validation_metrics["total"] < best_validation
            ):
                best_validation = validation_metrics["total"]
                state["best_validation"] = best_validation
                save_checkpoint(state, model_dir / "checkpoint_best.pth")
            if validation_metrics is not None or epoch == epochs:
                save_checkpoint(state, model_dir / "checkpoint_latest.pth")
            message = f"Epoch {epoch}/{epochs}: train_loss={train_metrics['total']:.6f}"
            if validation_metrics is not None:
                message += f", val_loss={validation_metrics['total']:.6f}"
            message += f", time={perf_counter() - started:.1f}s"
            append_log(log_path, message)
            writer.flush()

        write_status(
            status_path,
            "completed",
            epochs=epochs,
            best_validation=best_validation,
            checkpoint=str(model_dir / "checkpoint_best.pth"),
        )
    except BaseException as error:
        write_status(
            status_path, "failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        writer.close()
    print(f"Completed run: {model_dir}")


if __name__ == "__main__":
    main()
