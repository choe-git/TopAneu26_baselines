"""Minimal nnU-Net trainer loop with TopAneu evaluation and TensorBoard logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import time

import numpy as np
import torch
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as BaseTrainer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate_predictions, metric_summary, save_metrics

EVALUATION_INTERVAL = 10


class nnUNetTrainer(BaseTrainer):
    """Keep nnU-Net defaults; change only training cadence and logging."""

    def _read_cases(self) -> dict[str, list[str]]:
        path = Path(nnUNet_raw) / self.plans_manager.dataset_name / "split.csv"
        cases = {"train": [], "val": [], "test": []}
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                split = next(name for name in cases if row[name] == "1")
                cases[split].append(row["case_id"])
        return {name: sorted(values) for name, values in cases.items()}

    def _run_validation_loss(self) -> float:
        self.on_validation_epoch_start()
        outputs = [
            self.validation_step(next(self.dataloader_val))
            for _ in tqdm(
                range(self.num_val_iterations_per_epoch),
                desc="Validation loss",
                leave=False,
                unit="batch",
            )
        ]
        return float(np.mean([output["loss"] for output in outputs]))

    def _predict_and_evaluate(
        self,
        predictor: nnUNetPredictor,
        split: str,
        epoch: int,
        writer: SummaryWriter,
    ) -> dict:
        raw = Path(nnUNet_raw) / self.plans_manager.dataset_name
        image_folder = raw / ("imagesTs" if split == "test" else "imagesTr")
        label_folder = raw / ("labelsTs" if split == "test" else "labelsTr")
        prediction_folder = Path(self.output_folder) / "evaluation" / f"{split}_predictions"
        prediction_folder.mkdir(parents=True, exist_ok=True)

        inputs = [[str(image_folder / f"{case}_0000.nii.gz")] for case in self.cases[split]]
        outputs = [str(prediction_folder / case) for case in self.cases[split]]
        predictor.predict_from_files(
            inputs,
            outputs,
            overwrite=True,
            num_processes_preprocessing=min(default_num_processes, len(inputs)),
            num_processes_segmentation_export=min(default_num_processes, len(inputs)),
        )

        metrics = evaluate_predictions(prediction_folder, label_folder, self.cases[split])
        summary = metric_summary(metrics)
        save_metrics(
            metrics,
            Path(self.output_folder) / "evaluation" / "metrics" / f"epoch_{epoch:04d}" / f"{split}.json",
        )
        for task, values in summary.items():
            for name, value in values.items():
                writer.add_scalar(f"{split}/{task}/{name}", value, epoch)
        print(f"Epoch {epoch} {split} metrics:\n{json.dumps(summary, indent=2)}")
        return metrics

    def _run_full_evaluation(self, epoch: int, writer: SummaryWriter) -> dict[str, dict]:
        self.set_deep_supervision_enabled(False)
        predictor = nnUNetPredictor(
            tile_step_size=getattr(self, "inference_tile_step_size", 0.5),
            use_gaussian=True,
            use_mirroring=getattr(self, "inference_use_mirroring", True),
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            [self.network.state_dict()],
            self.dataset_json,
            "nnUNetTrainer",
            self.inference_allowed_mirroring_axes,
        )
        try:
            return {
                split: self._predict_and_evaluate(predictor, split, epoch, writer)
                for split in ("val", "test")
            }
        finally:
            self.set_deep_supervision_enabled(True)
            compute_gaussian.cache_clear()

    def _finish_epoch(self, started_at: float, val_loss: float | None) -> None:
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        train_loss = float(self.logger.get_value("train_losses", step=-1))
        message = f"Epoch {self.current_epoch + 1}: train_loss={train_loss:.4f}"
        if val_loss is not None:
            message += f", val_loss={val_loss:.4f}"
        self.print_to_log_file(f"{message}, time={time() - started_at:.1f}s")

        self.current_epoch += 1

    def run_training(self) -> None:
        self.on_train_start()
        self.cases = self._read_cases()
        writer = SummaryWriter(Path(nnUNet_results).parent / "tensorboard", purge_step=self.current_epoch + 1)
        final_test_metrics = None

        try:
            epochs = tqdm(
                range(self.current_epoch, self.num_epochs),
                initial=self.current_epoch,
                total=self.num_epochs,
                desc="Training",
                unit="epoch",
            )
            for _ in epochs:
                started_at = time()
                self.on_epoch_start()
                self.on_train_epoch_start()

                train_outputs = []
                batches = tqdm(
                    range(self.num_iterations_per_epoch),
                    desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs}",
                    leave=False,
                    unit="batch",
                )
                for _ in batches:
                    output = self.train_step(next(self.dataloader_train))
                    train_outputs.append(output)
                    batches.set_postfix(loss=f"{float(output['loss']):.4f}")
                self.on_train_epoch_end(train_outputs)

                epoch = self.current_epoch + 1
                train_loss = float(self.logger.get_value("train_losses", step=-1))
                writer.add_scalar("loss/train", train_loss, epoch)
                evaluated = epoch % EVALUATION_INTERVAL == 0
                val_loss = None
                if evaluated:
                    with torch.no_grad():
                        val_loss = self._run_validation_loss()
                    writer.add_scalar("loss/val", val_loss, epoch)
                    self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))
                    metrics = self._run_full_evaluation(epoch, writer)
                    final_test_metrics = metrics["test"]

                epochs.set_postfix(train_loss=f"{train_loss:.4f}")
                self._finish_epoch(started_at, val_loss)
                writer.flush()

            if final_test_metrics is None or self.current_epoch % EVALUATION_INTERVAL != 0:
                final_test_metrics = self._run_full_evaluation(self.current_epoch, writer)["test"]
            self.on_train_end()
        finally:
            writer.close()

        print("Final test metrics:")
        print(json.dumps(metric_summary(final_test_metrics), indent=2))
