from __future__ import annotations

import argparse
import json
from pathlib import Path

from topaneu_baseline.nnunet_pipeline import (
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    inspect_dataset,
    prepare_nnunet_dataset,
    repository_layout,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT = repository_layout(PROJECT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect TopAneu and prepare an nnU-Net v2 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LAYOUT["data_root"])
    parser.add_argument("--workspace", type=Path, default=DEFAULT_LAYOUT["workspace"])
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-mode", choices=("holdout", "crossval"), default="holdout")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PROJECT_DIR / "split.csv",
        help="Editable hold-out CSV bundled in the execution directory.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "symlink", "copy"), default="auto")
    parser.add_argument("--test-images", type=Path, help="Optional directory of unlabeled .nii.gz scans")
    parser.add_argument(
        "--check-mask-values",
        action="store_true",
        help="Stream all masks and verify mask labels against location_jsons (slow but thorough)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inspection = inspect_dataset(args.data_root, check_mask_values=args.check_mask_values)
    if inspection["geometry_errors"] or inspection["annotation_errors"]:
        raise RuntimeError(json.dumps(inspection, indent=2))
    manifest = prepare_nnunet_dataset(
        args.data_root,
        args.workspace,
        dataset_id=args.dataset_id,
        dataset_name=args.dataset_name,
        n_splits=args.folds,
        seed=args.seed,
        link_mode=args.link_mode,
        test_image_dir=args.test_images,
        split_mode=args.split_mode,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_csv=args.split_csv,
    )
    print(json.dumps({"inspection": inspection, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
