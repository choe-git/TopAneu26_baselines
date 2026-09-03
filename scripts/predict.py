"""Predict TopAneu Task 1 JSON and Task 2 NIfTI outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from rnsa_surrogate.cache import normalize_angiography, resample_zyx, resize_to_shape
from rnsa_surrogate.inference import sliding_window_predict
from rnsa_surrogate.model import RNSASurrogate
from rnsa_surrogate.run_layout import BaselineRunLayout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, help="Shared experiment root")
    parser.add_argument(
        "--checkpoint", type=Path, help="Override RUN_DIR/baseline/checkpoint_best.pth"
    )
    parser.add_argument("--output", type=Path, help="Override RUN_DIR/predictions")
    parser.add_argument("--modality", choices=("ct", "mr"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.45)
    parser.add_argument("--class-threshold", type=float, default=0.15)
    parser.add_argument("--presence-threshold", type=float, default=0.35)
    parser.add_argument("--presence-top-k", type=int, default=3)
    parser.add_argument("--presence-evidence-voxels", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = (
        BaselineRunLayout.from_root(args.run_dir) if args.run_dir is not None else None
    )
    checkpoint_path = args.checkpoint or (
        layout.checkpoint if layout is not None else None
    )
    output_dir = args.output or (layout.predictions if layout is not None else None)
    if checkpoint_path is None or output_dir is None:
        raise ValueError("Use --run-dir, or provide both --checkpoint and --output")
    checkpoint_path = checkpoint_path.resolve()
    output_dir = output_dir.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = RNSASurrogate(**config["model"])
    state = checkpoint["model"]
    if "ema" in checkpoint:
        state = dict(state)
        for name, value in checkpoint["ema"]["shadow"].items():
            state[name] = value.to(dtype=state[name].dtype)
    model.load_state_dict(state)
    model.to(device)

    source = nib.load(args.image)
    original = np.asarray(source.dataobj, dtype=np.float32).transpose(2, 1, 0)
    image, _ = normalize_angiography(original)
    source_spacing = tuple(
        float(value) for value in reversed(source.header.get_zooms()[:3])
    )
    target_spacing = tuple(
        float(value) for value in config["data"]["target_spacing_zyx"]
    )
    image = resample_zyx(image, source_spacing, target_spacing, order=1).astype(
        np.float32
    )
    modality = args.modality or ("mr" if "_mr_" in args.image.name.lower() else "ct")
    amp_name = str(config["train"].get("amp", "none"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[amp_name]
    segmentation, locations, _ = sliding_window_predict(
        model,
        image,
        modality,
        config["data"]["patch_size"],
        device,
        overlap=args.overlap,
        amp_dtype=amp_dtype,
        mask_threshold=args.mask_threshold,
        class_threshold=args.class_threshold,
        presence_threshold=args.presence_threshold,
        presence_top_k=args.presence_top_k,
        presence_evidence_voxels=args.presence_evidence_voxels,
    )

    task2_dir = output_dir / "task2_masks"
    task1_dir = output_dir / "task1_locations"
    task2_dir.mkdir(parents=True, exist_ok=True)
    task1_dir.mkdir(parents=True, exist_ok=True)
    case_id = args.image.name.removesuffix(".nii.gz").removesuffix("_0000")
    segmentation = resize_to_shape(segmentation, original.shape, order=0).astype(
        np.uint8
    )
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(
        nib.Nifti1Image(segmentation.transpose(2, 1, 0), source.affine, header),
        task2_dir / f"{case_id}.nii.gz",
    )
    (task1_dir / f"{case_id}.json").write_text(
        json.dumps(locations) + "\n", encoding="utf-8"
    )
    print(f"Task 1: {locations}")
    print(f"Task 2: {task2_dir / f'{case_id}.nii.gz'}")


if __name__ == "__main__":
    main()
