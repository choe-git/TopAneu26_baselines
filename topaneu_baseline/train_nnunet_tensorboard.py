from __future__ import annotations

import argparse
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnunetv2.run.run_training import run_training
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from topaneu_baseline.challenge_metrics import evaluate_prediction_masks
from topaneu_baseline.tensorboard_logging import challenge_metric_scalars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="nnU-Net v2 training with per-epoch TensorBoard logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_name_or_id")
    parser.add_argument("configuration")
    parser.add_argument("fold")
    parser.add_argument("-tr", default="nnUNetTrainer")
    parser.add_argument("-p", default="nnUNetPlans")
    parser.add_argument("-pretrained_weights", default=None)
    parser.add_argument("--npz", action="store_true")
    parser.add_argument("--c", action="store_true")
    parser.add_argument("-device", choices=("cuda", "cpu", "mps"), default="cuda")
    return parser.parse_args()


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def install_tensorboard_hooks() -> None:
    """Attach logging without changing nnU-Net's optimizer, loss, or checkpoints."""
    from torch.utils.tensorboard import SummaryWriter

    original_train_start = nnUNetTrainer.on_train_start
    original_epoch_end = nnUNetTrainer.on_epoch_end
    original_train_end = nnUNetTrainer.on_train_end

    def on_train_start(self: nnUNetTrainer) -> None:
        original_train_start(self)
        if self.local_rank != 0:
            return
        root = os.environ.get("TOPANEU_TENSORBOARD_ROOT")
        log_dir = Path(root) / f"fold_{self.fold}" if root else Path(self.output_folder) / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._topaneu_tb_log_dir = log_dir
        self._topaneu_tb_writer = SummaryWriter(log_dir=str(log_dir), purge_step=int(self.current_epoch))

    def run_training_with_interval_validation(self: nnUNetTrainer) -> None:
        interval = max(1, int(os.environ.get("TOPANEU_EVAL_EVERY", "10")))
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)

            # Epoch 1 establishes values required by nnU-Net's progress/checkpoint logger;
            # subsequent validation passes happen on epochs N, 2N, ... and the final epoch.
            run_validation = epoch == 0 or (epoch + 1) % interval == 0 or epoch + 1 == self.num_epochs
            self._topaneu_validation_ran = run_validation
            self._topaneu_validation_log = (epoch + 1) % interval == 0 or epoch + 1 == self.num_epochs
            if run_validation:
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = [
                        self.validation_step(next(self.dataloader_val))
                        for _ in range(self.num_val_iterations_per_epoch)
                    ]
                    self.on_validation_epoch_end(val_outputs)
            else:
                # nnU-Net's stock on_epoch_end expects one value at every index. Carrying
                # the last observation preserves checkpoint/resume compatibility without
                # pretending that a new validation pass ran.
                for key in ("val_losses", "mean_fg_dice", "dice_per_class_or_region"):
                    self.logger.log(key, self.logger.get_value(key, step=-1), self.current_epoch)
            self.on_epoch_end()
        self.on_train_end()

    def log_periodic_monitor_test(self: nnUNetTrainer, writer: Any, step: int) -> None:
        input_dir = os.environ.get("TOPANEU_MONITOR_TEST_INPUT")
        truth_dir = os.environ.get("TOPANEU_MONITOR_TEST_TRUTH")
        max_label = os.environ.get("TOPANEU_MONITOR_MAX_LABEL")
        if not input_dir or not truth_dir or not max_label or not Path(input_dir).is_dir():
            return

        cache_root = Path(os.environ["TOPANEU_TENSORBOARD_ROOT"]).parent / "monitor_test_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        self.print_to_log_file(f"Running monitor-test full-volume evaluation at epoch {step + 1}")
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            predictor.manual_initialization(
                self.network,
                self.plans_manager,
                self.configuration_manager,
                None,
                self.dataset_json,
                self.__class__.__name__,
                self.inference_allowed_mirroring_axes,
            )
            with tempfile.TemporaryDirectory(prefix=f"epoch_{step + 1:04d}_", dir=cache_root) as temp_dir:
                predictor.predict_from_files(input_dir, temp_dir, overwrite=True)
                predictions = sorted(Path(temp_dir).glob("*.nii.gz"))
                summary, per_class, _ = evaluate_prediction_masks(
                    predictions, truth_dir, max_label=int(max_label)
                )
                for tag, value in challenge_metric_scalars(
                    "monitor_test", summary, per_class, stage="periodic"
                ).items():
                    writer.add_scalar(tag, value, global_step=step)
                writer.flush()
        finally:
            self.set_deep_supervision_enabled(True)

    def on_epoch_end(self: nnUNetTrainer) -> None:
        step = int(self.current_epoch)
        original_epoch_end(self)
        if self.local_rank == 0 and hasattr(self, "_topaneu_tb_writer"):
            writer = self._topaneu_tb_writer
            scalar_keys = {
                "train/loss/total": "train_losses",
                "optimization/learning_rate": "lrs",
            }
            if getattr(self, "_topaneu_validation_log", True):
                scalar_keys.update(
                    {
                        "validation/patch/loss/total": "val_losses",
                        "validation/patch/dice/mean_foreground": "mean_fg_dice",
                        "validation/patch/dice/ema_foreground": "ema_fg_dice",
                    }
                )
            for tag, logger_key in scalar_keys.items():
                value = _finite(self.logger.get_value(logger_key, step=-1))
                if value is not None:
                    writer.add_scalar(tag, value, global_step=step)

            if getattr(self, "_topaneu_validation_log", True):
                dice_values = self.logger.get_value("dice_per_class_or_region", step=-1)
                for class_index, raw_value in enumerate(dice_values, start=1):
                    value = _finite(raw_value)
                    if value is not None:
                        writer.add_scalar(
                            f"validation/patch/dice/class_{class_index:02d}", value, global_step=step
                        )

            start = _finite(self.logger.get_value("epoch_start_timestamps", step=-1))
            end = _finite(self.logger.get_value("epoch_end_timestamps", step=-1))
            if start is not None and end is not None:
                writer.add_scalar("time/epoch_seconds", end - start, global_step=step)
            writer.flush()
            interval = max(1, int(os.environ.get("TOPANEU_EVAL_EVERY", "10")))
            if (step + 1) % interval == 0 or step + 1 == self.num_epochs:
                log_periodic_monitor_test(self, writer, step)

    def on_train_end(self: nnUNetTrainer) -> None:
        original_train_end(self)
        if self.local_rank == 0 and hasattr(self, "_topaneu_tb_writer"):
            writer = self._topaneu_tb_writer
            writer.flush()
            writer.close()
            marker = Path(self._topaneu_tb_log_dir) / "last_step.txt"
            marker.write_text(f"{int(self.current_epoch)}\n", encoding="utf-8")

    nnUNetTrainer.on_train_start = on_train_start
    nnUNetTrainer.run_training = run_training_with_interval_validation
    nnUNetTrainer.on_epoch_end = on_epoch_end
    nnUNetTrainer.on_train_end = on_train_end


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        torch.set_num_threads(multiprocessing.cpu_count())
    elif args.device == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    install_tensorboard_hooks()
    run_training(
        args.dataset_name_or_id,
        args.configuration,
        args.fold,
        trainer_class_name=args.tr,
        plans_identifier=args.p,
        pretrained_weights=args.pretrained_weights,
        num_gpus=1,
        export_validation_probabilities=args.npz,
        continue_training=args.c,
        device=torch.device(args.device),
    )


if __name__ == "__main__":
    main()
