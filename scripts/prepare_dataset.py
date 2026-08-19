"""Create the minimal nnU-Net v2 dataset for both TopAneu tasks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

DATASET_ID = 501
DATASET_NAME = "TopAneu"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "split.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Arrange TopAneu files in the minimal nnU-Net v2 format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="TopAneu release directory")
    parser.add_argument("--output", type=Path, required=True, help="Separate run/data directory")
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT, help="Train/val/test split")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild only Dataset501_TopAneu")
    return parser.parse_args()


def case_id(image_path: Path) -> str:
    suffix = "_0000.nii.gz"
    if not image_path.name.endswith(suffix):
        raise ValueError(f"Expected '*_0000.nii.gz': {image_path.name}")
    return image_path.name.removesuffix(suffix)


def read_split(path: Path) -> dict[str, str]:
    split = {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["case_id", *SPLIT_NAMES]:
            raise ValueError(f"Invalid split columns in {path}")
        for row in reader:
            case = row["case_id"]
            if any(row[name] not in {"0", "1"} for name in SPLIT_NAMES):
                raise ValueError(f"Split values must be 0 or 1 for {case}")
            selected = [name for name in SPLIT_NAMES if row[name] == "1"]
            if len(selected) != 1 or case in split:
                raise ValueError(f"Invalid split row for {case}")
            split[case] = selected[0]
    return split


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def dataset_json(labels: dict[str, int], num_training: int) -> dict:
    return {
        "channel_names": {"0": "angiography"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
    }


def main() -> None:
    args = parse_args()
    images = sorted((args.source / "images").glob("*_0000.nii.gz"))
    if not images:
        raise FileNotFoundError(f"No images found in {args.source / 'images'}")

    dataset = args.output / "nnUNet_raw" / f"Dataset{DATASET_ID}_{DATASET_NAME}"
    if dataset.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists: {dataset}. Use --overwrite to rebuild it.")
        shutil.rmtree(dataset)

    images_tr = dataset / "imagesTr"
    labels_tr = dataset / "labelsTr"
    images_ts = dataset / "imagesTs"
    labels_ts = dataset / "labelsTs"
    for directory in (images_tr, labels_tr, images_ts, labels_ts):
        directory.mkdir(parents=True, exist_ok=True)

    labels = json.loads((args.source / "location_mapping.json").read_text())["labels"]
    split = read_split(args.split_csv)
    image_cases = {case_id(path) for path in images}
    if image_cases != set(split):
        missing = sorted(image_cases - set(split))
        extra = sorted(set(split) - image_cases)
        raise ValueError(f"split.csv mismatch: missing={missing}, extra={extra}")

    for image_path in images:
        name = case_id(image_path)
        label_path = args.source / "location_masks" / f"{name}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing location mask for {name}")

        image_dir, label_dir = (images_ts, labels_ts) if split[name] == "test" else (images_tr, labels_tr)
        link_or_copy(image_path, image_dir / image_path.name)
        link_or_copy(label_path, label_dir / label_path.name)

    num_training = sum(value != "test" for value in split.values())
    (dataset / "dataset.json").write_text(json.dumps(dataset_json(labels, num_training), indent=2) + "\n")
    shutil.copy2(args.split_csv, dataset / "split.csv")
    counts = {name: sum(value == name for value in split.values()) for name in SPLIT_NAMES}
    print(f"Prepared {counts} in {dataset}")


if __name__ == "__main__":
    main()
