"""Convert Task 2 masks into Task 1 submission lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def case_id(mask_path: Path) -> str:
    return mask_path.name.removesuffix(".nii.gz")


def main() -> None:
    args = parse_args()
    masks = sorted(args.masks.glob("*.nii.gz"))
    if not masks:
        raise FileNotFoundError(f"No masks found in {args.masks}")
    args.output.mkdir(parents=True, exist_ok=True)

    for mask_path in masks:
        values = np.asanyarray(nib.load(mask_path).dataobj)
        locations = np.unique(values).astype(int)
        locations = [int(label) for label in locations if label != 0]
        output_path = args.output / f"{case_id(mask_path)}.json"
        output_path.write_text(json.dumps(locations, indent=2) + "\n")

    print(f"Wrote {len(masks)} Task 1 JSON files to {args.output}")


if __name__ == "__main__":
    main()
