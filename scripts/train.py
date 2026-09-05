"""Train RNSA surrogate exclusively from a completed physical-space cache."""

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
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rnsa_surrogate.cache import (
    atomic_json_dump,
    load_cache_index,
    load_zyx,
    resize_to_shape,
    sha256_file,
    validate_cache,
)
from rnsa_surrogate.data import CachedTopAneuPatchDataset
from rnsa_surrogate.inference import sliding_window_predict
from rnsa_surrogate.losses import multitask_loss
from rnsa_surrogate.model import RNSASurrogate, load_vessel_pretraining
from rnsa_surrogate.official_metrics import (
    summarize_task1,
    summarize_task2,
    task1_case_counts,
    task2_case_metrics,
)
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
    parser.add_argument("--fold", type=int, help="Cross-validation fold index")
    parser.add_argument(
        "--folds-file", type=Path, help="Override RUN_DIR/baseline/folds.json"
    )
    parser.add_argument(
        "--pretrained", type=Path, help="Override fold vessel checkpoint"
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
    split: str | None,
    samples: int,
    augment: bool,
    seed: int,
    case_ids: set[str] | None = None,
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
        case_ids=case_ids,
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


def official_selection_scores(
    task1: dict[str, Any], task2: dict[str, Any]
) -> dict[str, float]:
    """Direction-adjusted means of official metrics for checkpoint selection.

    The challenge score is a mean rank across submissions, so it cannot be computed
    during one training run. These composites only select checkpoints; every
    component is calculated with the official evaluator port.
    """
    task1_macro = task1["macro"]
    task2_macro = task2["macro"]
    return {
        "task1": float(
            np.mean(
                [
                    task1_macro["precision"],
                    task1_macro["recall"],
                    task1_macro["mcc"],
                ]
            )
        ),
        "task2": float(
            np.mean(
                [
                    task2_macro["precision"],
                    task2_macro["recall"],
                    task2_macro["mcc"],
                    task2_macro["dice"],
                    task2_macro["volumetric_similarity"],
                    1.0 - task2_macro["hd95"],
                ]
            )
        ),
    }


@torch.inference_mode()
def run_official_validation(
    model: torch.nn.Module,
    cache_index: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype | None,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run full-volume validation on the original mask grid."""
    settings = config.get("validation", {})
    cases = [
        case
        for case in cache_index["cases"]
        if (
            case["case_id"] in case_ids
            if case_ids is not None
            else case["split"] == "val"
        )
    ]
    max_cases = int(settings.get("max_cases", 0))
    if max_cases > 0:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("Cache contains no validation cases")

    cache_root = Path(cache_index["index_path"]).parent
    location_mask_root = Path(cache_index["source_root"]).resolve() / "location_masks"
    if not location_mask_root.is_dir():
        raise FileNotFoundError(
            "Official validation requires original masks: "
            f"{location_mask_root}"
        )

    task1_counts_per_case = []
    task2_counts_per_case = []
    task2_segmentation_per_case = []
    model.eval()
    for case in tqdm(cases, desc="Official validation", leave=False):
        case_dir = cache_root / case["cache_dir"]
        image = np.load(case_dir / "image.npy", mmap_mode="r").astype(np.float32)
        cache_prediction, predicted_locations, _ = sliding_window_predict(
            model,
            image,
            case["modality"],
            config["data"]["patch_size"],
            device,
            overlap=float(settings.get("overlap", 0.5)),
            amp_dtype=dtype,
            mask_threshold=float(settings.get("mask_threshold", 0.45)),
            class_threshold=float(settings.get("class_threshold", 0.15)),
            presence_threshold=float(settings.get("presence_threshold", 0.35)),
            presence_top_k=int(settings.get("presence_top_k", 3)),
            presence_evidence_voxels=int(
                settings.get("presence_evidence_voxels", 64)
            ),
            minimum_component_voxels=int(
                settings.get("minimum_component_voxels", 5)
            ),
            maximum_components=int(settings.get("maximum_components", 5)),
            component_location_weight=float(
                settings.get("component_location_weight", 0.0)
            ),
        )
        ground_truth, _ = load_zyx(
            location_mask_root / f"{case['case_id']}.nii.gz"
        )
        expected_shape = tuple(case["original_metadata"]["shape_zyx"])
        if ground_truth.shape != expected_shape:
            raise ValueError(
                f"Original shape mismatch for {case['case_id']}: "
                f"index={expected_shape}, mask={ground_truth.shape}"
            )
        prediction = resize_to_shape(cache_prediction, ground_truth.shape, order=0)
        prediction = np.asarray(prediction, dtype=np.uint8)
        ground_truth = np.asarray(ground_truth, dtype=np.uint8)
        task1_counts_per_case.append(
            task1_case_counts(case["json_locations"], predicted_locations)
        )
        task2_counts, task2_segmentation = task2_case_metrics(
            ground_truth, prediction
        )
        task2_counts_per_case.append(task2_counts)
        task2_segmentation_per_case.append(task2_segmentation)

    task1 = summarize_task1(task1_counts_per_case)
    task2 = summarize_task2(task2_counts_per_case, task2_segmentation_per_case)
    scores = official_selection_scores(task1, task2)
    selection_task = str(settings.get("selection_task", "task2"))
    if selection_task not in scores:
        raise ValueError("validation.selection_task must be 'task1' or 'task2'")
    return {
        "cases": len(cases),
        "official_task1": task1,
        "official_task2": task2,
        "checkpoint_selection": {
            "task": selection_task,
            "score": scores[selection_task],
            "task1_score": scores["task1"],
            "task2_score": scores["task2"],
            "note": "direction-adjusted metric mean; official mean rank needs other submissions",
        },
        "thresholds": {
            "overlap": float(settings.get("overlap", 0.5)),
            "mask": float(settings.get("mask_threshold", 0.45)),
            "class": float(settings.get("class_threshold", 0.15)),
            "presence": float(settings.get("presence_threshold", 0.35)),
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
        },
    }


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
        config.setdefault("validation", {})["max_cases"] = 1

    resume_path = args.resume.resolve() if args.resume is not None else None
    if args.run_dir is not None:
        layout = BaselineRunLayout.from_root(args.run_dir)
        run_root = layout.root
        cache_dir = (args.cache or layout.cache).resolve()
        if args.fold is None:
            model_dir = layout.baseline
            tensorboard_dir = layout.tensorboard
        else:
            model_dir = layout.folds / f"fold_{args.fold}"
            tensorboard_dir = layout.tensorboard / "folds" / f"fold_{args.fold}"
        if resume_path is not None and resume_path.parent != model_dir:
            raise ValueError(f"Resume checkpoint must be inside {model_dir}")
    else:
        if args.fold is not None:
            raise ValueError("--fold requires --run-dir")
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

    seed = int(config["experiment"].get("seed", 2026)) + (args.fold or 0)
    seed_everything(seed)
    device = resolve_device(str(config["train"].get("device", "cuda")))
    cache_report = validate_cache(cache_dir, deep=False)
    cache_index = load_cache_index(cache_dir)
    if not np.allclose(
        cache_index["target_spacing_zyx"], config["data"]["target_spacing_zyx"]
    ):
        raise ValueError("Cache spacing differs from data.target_spacing_zyx")

    fold_train_ids: set[str] | None = None
    fold_val_ids: set[str] | None = None
    fold_manifest_path: Path | None = None
    pretrained_path: Path | None = None
    if args.fold is not None:
        fold_manifest_path = (args.folds_file or layout.fold_manifest).resolve()
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        if not 0 <= args.fold < int(fold_manifest["n_folds"]):
            raise ValueError(f"Invalid fold {args.fold}")
        fold_val_ids = set(fold_manifest["folds"][str(args.fold)])
        fold_train_ids = set(fold_manifest["case_to_fold"]) - fold_val_ids
        pretrained_path = (
            args.pretrained
            or (layout.vessel_pretrain / "checkpoint_best.pth")
        ).resolve()
        if resume_path is None and not pretrained_path.is_file():
            raise FileNotFoundError(
                f"Fold training requires vessel pretraining: {pretrained_path}"
            )

    contract = training_contract(config)
    if args.fold is not None:
        assert fold_manifest_path is not None
        contract["cross_validation"] = {
            "fold": args.fold,
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "pretrained_sha256": (
                sha256_file(pretrained_path) if pretrained_path is not None else None
            ),
        }
    contract_sha256 = config_digest(contract)
    checkpoint: dict[str, Any] | None = None
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected_stage = "aneurysm_fold" if args.fold is not None else "baseline"
        if checkpoint.get("stage", "baseline") != expected_stage:
            raise ValueError(
                f"Checkpoint stage is not {expected_stage}: {checkpoint.get('stage')}"
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
            model_dir.mkdir(parents=True, exist_ok=False)
        atomic_json_dump(config, model_dir / "config.json")
        atomic_json_dump(environment_payload(), model_dir / "environment.json")
        atomic_json_dump(
            {
                "cache_index": cache_report["index"],
                "cache_index_sha256": cache_report["index_sha256"],
                "config_source": str(args.config.resolve()),
                "config_source_sha256": sha256_file(args.config),
                "contract_sha256": contract_sha256,
                "fold": args.fold,
                "fold_manifest": (
                    str(fold_manifest_path) if fold_manifest_path is not None else None
                ),
                "pretrained_checkpoint": (
                    str(pretrained_path) if pretrained_path is not None else None
                ),
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
    train_dataset = make_dataset(
        config,
        cache_dir,
        "train" if fold_train_ids is None else None,
        train_samples,
        True,
        seed,
        fold_train_ids,
    )
    test_every = int(config["train"].get("test_every", 0))
    test_dataset = (
        make_dataset(config, cache_dir, "test", test_samples, False, seed + 20_000_000)
        if test_every > 0
        else None
    )
    train_sampler = EpochIndexSampler(len(train_dataset))
    train_loader = make_loader(train_dataset, config, device, sampler=train_sampler)
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
    if checkpoint is None and pretrained_path is not None:
        vessel_checkpoint = torch.load(
            pretrained_path, map_location="cpu", weights_only=False
        )
        if vessel_checkpoint.get("stage") != "vessel_pretrain":
            raise ValueError(f"Not a vessel pretraining checkpoint: {pretrained_path}")
        pretrained_state = dict(vessel_checkpoint["model"])
        if "ema" in vessel_checkpoint:
            pretrained_state.update(vessel_checkpoint["ema"]["shadow"])
        transfer = load_vessel_pretraining(model, pretrained_state)
        atomic_json_dump(
            {
                "checkpoint": str(pretrained_path),
                "checkpoint_sha256": sha256_file(pretrained_path),
                **transfer,
            },
            model_dir / "pretraining.json",
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    atomic_json_dump({"parameters": parameter_count}, model_dir / "model.json")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    epochs = int(
        config.get("ensemble", {}).get("fold_epochs", config["train"]["epochs"])
        if args.fold is not None
        else config["train"]["epochs"]
    )
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
    selection_task = str(config.get("validation", {}).get("selection_task", "task2"))
    if selection_task not in {"task1", "task2"}:
        raise ValueError("validation.selection_task must be 'task1' or 'task2'")
    validation_signature = config_digest(config.get("validation", {}))
    start_epoch = 0
    best_official_score = float("-inf")
    best_task1_score = float("-inf")
    best_task2_score = float("-inf")
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        ema.load_state_dict(checkpoint["ema"], device)
        start_epoch = int(checkpoint["epoch"])
        stored_selection_task = checkpoint.get("selection_task")
        stored_validation_signature = checkpoint.get("validation_signature")
        compatible_official_best = (
            stored_selection_task == selection_task
            and stored_validation_signature == validation_signature
        )
        if compatible_official_best:
            best_official_score = float(
                checkpoint.get("best_official_score", float("-inf"))
            )
            best_task1_score = float(
                checkpoint.get("best_task1_score", float("-inf"))
            )
            best_task2_score = float(
                checkpoint.get("best_task2_score", float("-inf"))
            )
        restore_rng_state(checkpoint.get("rng"))
        atomic_json_dump(
            {
                "checkpoint": str(resume_path),
                "start_epoch": start_epoch + 1,
                "selection_task": selection_task,
                "validation_signature": validation_signature,
                "official_best_reset": not compatible_official_best,
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
            official_validation = None
            if epoch % validate_every == 0 or epoch == epochs:
                with ema.average_parameters(model):
                    official_validation = run_official_validation(
                        model,
                        cache_index,
                        config,
                        device,
                        dtype,
                        fold_val_ids,
                    )
                atomic_json_dump(
                    {
                        "epoch": epoch,
                        "split": "val",
                        **official_validation,
                    },
                    model_dir
                    / "metrics"
                    / "official_val"
                    / f"epoch_{epoch:04d}.json",
                )
                for task_name in ("task1", "task2"):
                    macro = official_validation[f"official_{task_name}"]["macro"]
                    for name, value in macro.items():
                        writer.add_scalar(
                            f"official/val/{task_name}/{name}", value, epoch
                        )
                selection = official_validation["checkpoint_selection"]
                writer.add_scalar("metric/val", selection["score"], epoch)
                writer.add_scalar(
                    "official/val/checkpoint_selection", selection["score"], epoch
                )

            if test_loader is not None and (epoch % test_every == 0 or epoch == epochs):
                with ema.average_parameters(model):
                    test_metrics = run_epoch(
                        model, test_loader, device, config["loss"], dtype, "test"
                    )
                write_metrics(model_dir, "test", epoch, test_metrics)
                for name, value in test_metrics.items():
                    writer.add_scalar(f"loss_components/test/{name}", value, epoch)
                writer.add_scalar("loss/test", test_metrics["total"], epoch)

            selected_improved = False
            task1_improved = False
            task2_improved = False
            if official_validation is not None:
                selection = official_validation["checkpoint_selection"]
                selected_improved = selection["score"] > best_official_score
                task1_improved = selection["task1_score"] > best_task1_score
                task2_improved = selection["task2_score"] > best_task2_score
                best_official_score = max(
                    best_official_score, float(selection["score"])
                )
                best_task1_score = max(
                    best_task1_score, float(selection["task1_score"])
                )
                best_task2_score = max(
                    best_task2_score, float(selection["task2_score"])
                )

            state = {
                "stage": "aneurysm_fold" if args.fold is not None else "baseline",
                "fold": args.fold,
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "selection_task": selection_task,
                "validation_signature": validation_signature,
                "best_official_score": best_official_score,
                "best_task1_score": best_task1_score,
                "best_task2_score": best_task2_score,
                "config": config,
                "contract_sha256": contract_sha256,
                "cache_index_sha256": cache_report["index_sha256"],
                "rng": rng_state(),
            }
            if selected_improved:
                save_checkpoint(state, model_dir / "checkpoint_best.pth")
            if task1_improved:
                save_checkpoint(state, model_dir / "checkpoint_best_task1.pth")
            if task2_improved:
                save_checkpoint(state, model_dir / "checkpoint_best_task2.pth")
            if official_validation is not None or epoch == epochs:
                save_checkpoint(state, model_dir / "checkpoint_latest.pth")
            message = f"Epoch {epoch}/{epochs}: train_loss={train_metrics['total']:.6f}"
            if official_validation is not None:
                selection = official_validation["checkpoint_selection"]
                message += (
                    f", official_{selection_task}={selection['score']:.6f}, "
                    f"task1={selection['task1_score']:.6f}, "
                    f"task2={selection['task2_score']:.6f}"
                )
            message += f", time={perf_counter() - started:.1f}s"
            append_log(log_path, message)
            writer.flush()

        write_status(
            status_path,
            "completed",
            epochs=epochs,
            selection_task=selection_task,
            best_official_score=best_official_score,
            best_task1_score=best_task1_score,
            best_task2_score=best_task2_score,
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
