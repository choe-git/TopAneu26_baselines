"""Minimal nnU-Net trainer loop with TopAneu evaluation and TensorBoard logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as BaseTrainer
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from torch import nn
from torch._dynamo import OptimizedModule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate_predictions, metric_summary, save_metrics

EVALUATION_INTERVAL = 10

TERRITORY_GROUPS = (
    tuple(range(1, 18)),
    tuple(range(18, 22)),
    tuple(range(22, 36)),
    tuple(range(36, 45)),
    tuple(range(45, 53)),
)
LATERALITY_GROUPS = (
    (1, 3, 5, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 37, 39, 41, 43, 45, 47, 49, 51),
    (2, 4, 6, 10, 12, 14, 16, 19, 21, 23, 25, 27, 29, 31, 33, 35, 38, 40, 42, 44, 46, 48, 50, 52),
    (7, 8, 17, 36),
)


def label_to_group(groups: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    result = [0] * 53
    for group_index, labels in enumerate(groups):
        for label in labels:
            result[label] = group_index
    return tuple(result)


class HierarchicalLoss(nn.Module):
    """Supervise foreground, territory, laterality, and the 52 official locations."""

    territory_by_label = label_to_group(TERRITORY_GROUPS)
    laterality_by_label = label_to_group(LATERALITY_GROUPS)

    @staticmethod
    def grouped_logits(logits: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
        return torch.stack(
            [torch.logsumexp(logits[:, [label - 1 for label in labels]], dim=1) for labels in groups],
            dim=1,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target[:, 0].long() if target.ndim == logits.ndim else target.long()
        foreground = target > 0

        binary_logits = torch.cat(
            (logits[:, :1], torch.logsumexp(logits[:, 1:], dim=1, keepdim=True)),
            dim=1,
        )
        binary_ce = F.cross_entropy(binary_logits, foreground.long())
        foreground_probability = torch.softmax(binary_logits, dim=1)[:, 1]
        intersection = (foreground_probability * foreground).sum()
        denominator = foreground_probability.sum() + foreground.sum()
        binary_dice = 1 - (2 * intersection + 1e-5) / (denominator + 1e-5)

        classification_losses = [binary_ce]
        if foreground.any():
            fine_logits = logits[:, 1:].movedim(1, -1)[foreground]
            fine_target = target[foreground] - 1
            classification_losses.append(F.cross_entropy(fine_logits, fine_target))

            territory_logits = self.grouped_logits(fine_logits, TERRITORY_GROUPS)
            territory_map = target.new_tensor(self.territory_by_label)
            classification_losses.append(F.cross_entropy(territory_logits, territory_map[target[foreground]]))

            laterality_logits = self.grouped_logits(fine_logits, LATERALITY_GROUPS)
            laterality_map = target.new_tensor(self.laterality_by_label)
            classification_losses.append(F.cross_entropy(laterality_logits, laterality_map[target[foreground]]))

        return binary_dice + torch.stack(classification_losses).mean()


class ClassBalancedDataLoader(nnUNetDataLoader):
    """Use nnU-Net crops while choosing the forced foreground class uniformly."""

    def __init__(self, *args, class_cases: dict[int, list[str]], **kwargs):
        super().__init__(*args, **kwargs)
        available = set(self.indices)
        self.class_cases = {
            label: sorted(available.intersection(cases))
            for label, cases in class_cases.items()
            if available.intersection(cases)
        }
        if not self.class_cases:
            raise ValueError("No foreground locations from class_cases.json occur in the training split.")
        self.foreground_labels = tuple(sorted(self.class_cases))
        self._desired_classes = []

    def get_indices(self):
        selected = list(super().get_indices())
        desired_classes = [None] * len(selected)
        for sample_index in range(len(selected)):
            if self.get_do_oversample(sample_index):
                label = int(np.random.choice(self.foreground_labels))
                selected[sample_index] = str(np.random.choice(self.class_cases[label]))
                desired_classes[sample_index] = label
        self._desired_classes = desired_classes
        return selected

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None, verbose=False):
        desired_class = self._desired_classes.pop(0) if self._desired_classes else overwrite_class
        return super().get_bbox(data_shape, force_fg, class_locations, desired_class, verbose)


class nnUNetTrainer(BaseTrainer):
    """Use nnU-Net v2 with TopAneu-aware loss, sampling, evaluation, and logging."""

    def _build_loss(self):
        loss = HierarchicalLoss()
        if self.enable_deep_supervision:
            weights = np.array([1 / (2**index) for index in range(len(self._get_deep_supervision_scales()))])
            weights[-1] = 1e-6 if self.is_ddp and not self._do_i_compile() else 0
            loss = DeepSupervisionWrapper(loss, weights / weights.sum())
        return loss

    def _read_class_cases(self) -> dict[int, list[str]]:
        path = Path(nnUNet_raw) / self.plans_manager.dataset_name / "class_cases.json"
        if not path.is_file():
            raise FileNotFoundError(f"Run prepare_dataset.py again to create {path.name}.")
        return {int(label): cases for label, cases in json.loads(path.read_text()).items()}

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch_size, mirror_axes = (
            self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        train_transforms = self.get_training_transforms(
            patch_size,
            rotation,
            deep_supervision_scales,
            mirror_axes,
            dummy_2d,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label,
        )
        dataset_train, dataset_val = self.get_tr_and_val_datasets()
        common = {
            "oversample_foreground_percent": self.oversample_foreground_percent,
            "sampling_probabilities": None,
            "pad_sides": None,
            "probabilistic_oversampling": self.probabilistic_oversampling,
        }
        loader_train = ClassBalancedDataLoader(
            dataset_train,
            self.batch_size,
            initial_patch_size,
            patch_size,
            self.label_manager,
            transforms=train_transforms,
            class_cases=self._read_class_cases(),
            **common,
        )
        loader_val = nnUNetDataLoader(
            dataset_val,
            self.batch_size,
            patch_size,
            patch_size,
            self.label_manager,
            transforms=val_transforms,
            **common,
        )

        processes = get_allowed_n_proc_DA()
        if processes == 0:
            train_augmenter = SingleThreadedAugmenter(loader_train, None)
            val_augmenter = SingleThreadedAugmenter(loader_val, None)
        else:
            train_augmenter = NonDetMultiThreadedAugmenter(
                data_loader=loader_train,
                transform=None,
                num_processes=processes,
                num_cached=max(6, processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            val_augmenter = NonDetMultiThreadedAugmenter(
                data_loader=loader_val,
                transform=None,
                num_processes=max(1, processes // 2),
                num_cached=max(3, processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        next(train_augmenter)
        next(val_augmenter)
        return train_augmenter, val_augmenter

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
        try:
            model = self.network.module if self.is_ddp else self.network
            if isinstance(model, OptimizedModule):
                model = model._orig_mod

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
                model,
                self.plans_manager,
                self.configuration_manager,
                [model.state_dict()],
                self.dataset_json,
                self.__class__.__name__,
                self.inference_allowed_mirroring_axes,
            )
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

    def _evaluation_complete(self, epoch: int) -> bool:
        folder = Path(self.output_folder) / "evaluation" / "metrics" / f"epoch_{epoch:04d}"
        return all((folder / f"{split}.json").is_file() for split in ("val", "test"))

    def run_training(self) -> None:
        self.on_train_start()
        self.cases = self._read_cases()
        writer = SummaryWriter(Path(nnUNet_results).parent / "tensorboard", purge_step=self.current_epoch + 1)
        final_test_metrics = None

        try:
            if (
                self.current_epoch > 0
                and self.current_epoch % EVALUATION_INTERVAL == 0
                and not self._evaluation_complete(self.current_epoch)
            ):
                print(f"Retrying incomplete epoch {self.current_epoch} evaluation")
                metrics = self._run_full_evaluation(self.current_epoch, writer)
                final_test_metrics = metrics["test"]
                writer.flush()

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
