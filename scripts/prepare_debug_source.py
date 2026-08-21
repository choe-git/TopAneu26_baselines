"""Create reproducible TopAneu subsets for overfit and mini tests."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from prepare_dataset import link_or_copy

PROFILES = {
    "overfit": (
        ("topaneu_center1_mr_308", "overfit_train", "train"),
        ("topaneu_center1_mr_308", "overfit_val", "val"),
        ("topaneu_center1_mr_308", "overfit_test", "test"),
    ),
    "mini": (
        ("topaneu_center4_ct_200", "topaneu_center4_ct_200", "train"),
        ("topaneu_center2_mr_085", "topaneu_center2_mr_085", "train"),
        ("topaneu_center1_mr_308", "topaneu_center1_mr_308", "train"),
        ("topaneu_center1_mr_001", "topaneu_center1_mr_001", "train"),
        ("topaneu_center1_mr_184", "topaneu_center1_mr_184", "val"),
        ("topaneu_center1_mr_064", "topaneu_center1_mr_064", "val"),
        ("topaneu_center1_mr_359", "topaneu_center1_mr_359", "test"),
        ("topaneu_center1_mr_061", "topaneu_center1_mr_061", "test"),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small, hard-linked TopAneu source directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="Original TopAneu release directory")
    parser.add_argument("--output", type=Path, required=True, help="New debug source directory")
    parser.add_argument("--profile", choices=PROFILES, required=True, help="Debug subset")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild the output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --source")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {args.output}. Use --overwrite to rebuild it.")
        shutil.rmtree(args.output)

    for directory in ("images", "location_masks", "location_jsons"):
        (args.output / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source / "location_mapping.json", args.output / "location_mapping.json")

    rows = []
    for source_case, output_case, split in PROFILES[args.profile]:
        files = (
            (
                args.source / "images" / f"{source_case}_0000.nii.gz",
                args.output / "images" / f"{output_case}_0000.nii.gz",
            ),
            (
                args.source / "location_masks" / f"{source_case}.nii.gz",
                args.output / "location_masks" / f"{output_case}.nii.gz",
            ),
            (
                args.source / "location_jsons" / f"{source_case}.json",
                args.output / "location_jsons" / f"{output_case}.json",
            ),
        )
        for source, destination in files:
            if not source.is_file():
                raise FileNotFoundError(source)
            link_or_copy(source, destination)
        rows.append([output_case, *("1" if name == split else "0" for name in ("train", "val", "test"))])

    split_path = args.output / "split.csv"
    with split_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("case_id", "train", "val", "test"))
        writer.writerows(rows)
    print(f"Created {args.profile} source at {args.output}")
    print(f"Split: {split_path}")


if __name__ == "__main__":
    main()
