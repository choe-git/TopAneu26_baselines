from __future__ import annotations

import argparse
import json
from pathlib import Path

from topaneu_baseline.challenge_io import load_label_mapping
from topaneu_baseline.challenge_metrics import evaluate_prediction_masks, write_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TopAneu Task 1/2 outputs with challenge metrics")
    parser.add_argument(
        "--prediction-masks",
        type=Path,
        required=True,
        nargs="+",
        help="Prediction files or directories containing .nii.gz masks",
    )
    parser.add_argument("--ground-truth-masks", type=Path, required=True)
    parser.add_argument("--prediction-jsons", type=Path)
    parser.add_argument("--location-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument("--min-component-mm3", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for item in args.prediction_masks:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.nii.gz")))
        elif item.is_file():
            paths.append(item)
        else:
            raise FileNotFoundError(item)
    labels = load_label_mapping(args.location_mapping)
    summary, per_class, per_case = evaluate_prediction_masks(
        paths,
        args.ground_truth_masks,
        max_label=max(labels.values()),
        prediction_json_dir=args.prediction_jsons,
        min_component_voxels=args.min_component_voxels,
        min_component_mm3=args.min_component_mm3,
    )
    write_evaluation(args.output_dir, summary, per_class, per_case)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
