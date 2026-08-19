"""Run one nnU-Net v2 model and create outputs for TopAneu Tasks 1 and 2."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", default="0")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    task2_output = args.output / "task2_masks"
    task1_output = args.output / "task1_locations"
    task2_output.mkdir(parents=True, exist_ok=True)

    run([
        "nnUNetv2_predict_from_modelfolder",
        "-i", str(args.images),
        "-o", str(task2_output),
        "-f", str(args.fold),
        "-chk", "checkpoint_final.pth",
        "-m", str(args.model),
    ])
    run([
        sys.executable, str(Path(__file__).with_name("locations_from_masks.py")),
        "--masks", str(task2_output),
        "--output", str(task1_output),
    ])


if __name__ == "__main__":
    main()
