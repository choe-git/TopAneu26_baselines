from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from topaneu_baseline.challenge_io import IMAGE_SUFFIX, load_label_mapping, write_challenge_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert nnU-Net masks to TopAneu Task 1 and Task 2 outputs")
    parser.add_argument("--prediction-masks", type=Path, required=True)
    parser.add_argument("--input-images", type=Path, required=True)
    parser.add_argument("--location-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument("--min-component-mm3", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_label_mapping(args.location_mapping)
    max_label = max(labels.values())
    prediction_paths = sorted(args.prediction_masks.glob("*.nii.gz"))
    if not prediction_paths:
        raise FileNotFoundError(f"No nnU-Net predictions found in {args.prediction_masks}")

    task1_dir = args.output_dir / "task1"
    task2_dir = args.output_dir / "task2"
    manifest: list[dict[str, object]] = []
    for prediction_path in tqdm(prediction_paths, desc="challenge outputs"):
        case_id = prediction_path.name[: -len(".nii.gz")]
        image_path = args.input_images / f"{case_id}{IMAGE_SUFFIX}"
        if not image_path.exists():
            alternate = args.input_images / f"{case_id}.nii.gz"
            image_path = alternate if alternate.exists() else image_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing input image for {case_id}: {image_path}")
        result = write_challenge_outputs(
            prediction_path,
            image_path,
            task1_dir / f"{case_id}.json",
            task2_dir / f"{case_id}.nii.gz",
            max_label=max_label,
            min_component_voxels=args.min_component_voxels,
            min_component_mm3=args.min_component_mm3,
        )
        manifest.append(
            {
                "case_id": result.case_id,
                "locations": list(result.locations),
                "shape": list(result.shape),
                "dtype": result.dtype,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote and validated {len(manifest)} cases under {args.output_dir}")


if __name__ == "__main__":
    main()
