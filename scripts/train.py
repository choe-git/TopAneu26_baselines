"""Train nnU-Net with the train/val rows defined in split.csv."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from prepare_dataset import DATASET_ID, DATASET_NAME, DEFAULT_SPLIT, read_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the TopAneu nnU-Net v2 baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared nnU-Net data/run directory")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT, help="Train/val/test split")
    parser.add_argument("--continue-training", action="store_true", help="Resume checkpoint_latest.pth")
    parser.add_argument("--smoke-test", action="store_true", help="Run the 10-epoch plumbing test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda", help="PyTorch device")
    return parser.parse_args()


def nnunet_paths(data_root: Path) -> dict[str, Path]:
    root = data_root.resolve()
    return {
        "nnUNet_raw": root / "nnUNet_raw",
        "nnUNet_preprocessed": root / "nnUNet_preprocessed",
        "nnUNet_results": root / "nnUNet_results",
    }


def write_nnunet_split(split: dict[str, str], preprocessed: Path) -> Path:
    dataset = preprocessed / f"Dataset{DATASET_ID}_{DATASET_NAME}"
    if not dataset.is_dir():
        raise FileNotFoundError(f"Run preprocess.sh first: {dataset}")

    output = dataset / "splits_final.json"
    content = [{
        "train": sorted(case for case, name in split.items() if name == "train"),
        "val": sorted(case for case, name in split.items() if name == "val"),
    }]
    output.write_text(json.dumps(content, indent=2) + "\n")
    return output


def validate_prepared_data(split: dict[str, str], raw: Path) -> None:
    dataset = raw / f"Dataset{DATASET_ID}_{DATASET_NAME}"
    training = {case for case, name in split.items() if name != "test"}
    testing = {case for case, name in split.items() if name == "test"}

    def image_cases(directory: str) -> set[str]:
        return {path.name.removesuffix("_0000.nii.gz") for path in (dataset / directory).glob("*.nii.gz")}

    def label_cases(directory: str) -> set[str]:
        return {path.name.removesuffix(".nii.gz") for path in (dataset / directory).glob("*.nii.gz")}

    if not (
        image_cases("imagesTr") == training
        and label_cases("labelsTr") == training
        and image_cases("imagesTs") == testing
        and label_cases("labelsTs") == testing
    ):
        raise ValueError("Prepared data does not match split.csv. Run prepare_dataset.py again.")


def main() -> None:
    args = parse_args()
    split = read_split(args.split_csv)
    paths = nnunet_paths(args.data_root)
    validate_prepared_data(split, paths["nnUNet_raw"])
    split_path = write_nnunet_split(split, paths["nnUNet_preprocessed"])

    os.environ.update({name: str(path) for name, path in paths.items()})
    print(f"Using split: {split_path}")

    import torch
    from nnunetv2.run.run_training import maybe_load_checkpoint
    from trainer import nnUNetTrainer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA GPU.")

    dataset = paths["nnUNet_preprocessed"] / f"Dataset{DATASET_ID}_{DATASET_NAME}"
    plans = json.loads((dataset / "nnUNetPlans.json").read_text())
    dataset_json = json.loads((dataset / "dataset.json").read_text())
    trainer = nnUNetTrainer(plans, "3d_fullres", 0, dataset_json, torch.device(args.device))
    if args.smoke_test:
        trainer.num_epochs = 10
        trainer.num_iterations_per_epoch = 1
        trainer.num_val_iterations_per_epoch = 1
        trainer.inference_tile_step_size = 1.0
        trainer.inference_use_mirroring = False
    maybe_load_checkpoint(trainer, args.continue_training, False)
    torch.backends.cudnn.benchmark = True
    trainer.run_training()


if __name__ == "__main__":
    main()
