"""Run vessel pretraining and aneurysm fine-tuning for every ensemble fold."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from rnsa_surrogate.run_layout import BaselineRunLayout


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def completed(directory: Path) -> bool:
    status = directory / "status.json"
    if not status.is_file():
        return False
    return json.loads(status.read_text(encoding="utf-8")).get("status") == "completed"


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--save-predictions", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()

    layout = BaselineRunLayout.from_root(args.run_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ensemble = config.get("ensemble", {})
    python = sys.executable
    fold_manifest = layout.root / "folds.json"
    if not fold_manifest.is_file():
        run(
            [
                python,
                "scripts/prepare_folds.py",
                "--run-dir",
                str(layout.root),
                "--n-folds",
                str(int(ensemble.get("n_folds", 5))),
            ]
        )
    payload = json.loads(fold_manifest.read_text(encoding="utf-8"))
    selected = args.folds or list(range(int(payload["n_folds"])))
    invalid = [fold for fold in selected if not 0 <= fold < int(payload["n_folds"])]
    if invalid:
        raise ValueError(f"Invalid folds: {invalid}")

    for fold in selected:
        vessel_dir = layout.root / "vessel_pretrain" / f"fold_{fold}"
        if not completed(vessel_dir):
            command = [
                python,
                "scripts/pretrain_vessel.py",
                "--run-dir",
                str(layout.root),
                "--config",
                str(args.config),
                "--fold",
                str(fold),
                "--device",
                args.device,
            ]
            vessel_latest = vessel_dir / "checkpoint_latest.pth"
            if vessel_latest.is_file():
                command.extend(["--resume", str(vessel_latest)])
            run(command)
        fold_dir = layout.root / "folds" / f"fold_{fold}"
        if not completed(fold_dir):
            command = [
                python,
                "scripts/train.py",
                "--run-dir",
                str(layout.root),
                "--config",
                str(args.config),
                "--fold",
                str(fold),
                "--device",
                args.device,
            ]
            fold_latest = fold_dir / "checkpoint_latest.pth"
            if fold_latest.is_file():
                command.extend(["--resume", str(fold_latest)])
            run(command)

    if not args.skip_evaluation:
        command = [
            python,
            "scripts/evaluate.py",
            "--run-dir",
            str(layout.root),
            "--split",
            args.split,
            "--device",
            args.device,
            "--ensemble-folds",
            *(str(fold) for fold in selected),
            "--overwrite",
        ]
        if bool(ensemble.get("tta_left_right", True)):
            command.append("--tta-left-right")
        command.append(
            "--save-predictions"
            if args.save_predictions
            else "--no-save-predictions"
        )
        run(command)


if __name__ == "__main__":
    main()
