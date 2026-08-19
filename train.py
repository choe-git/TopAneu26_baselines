from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from topaneu_baseline.challenge_io import IMAGE_SUFFIX, load_label_mapping, write_challenge_outputs
from topaneu_baseline.challenge_metrics import evaluate_prediction_masks, write_evaluation
from topaneu_baseline.nnunet_pipeline import (
    DEFAULT_CONFIGURATION,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_PLANNER,
    DEFAULT_PLANS,
    DEFAULT_TRAINER,
    dataset_folder_name,
    find_crossval_predictions,
    inspect_dataset,
    nnunet_environment,
    nnunet_paths,
    prepare_nnunet_dataset,
    repository_layout,
    stage_inference_images,
)
from topaneu_baseline.tensorboard_logging import write_challenge_metrics_tensorboard


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LAYOUT = repository_layout(PROJECT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, train, validate, evaluate, and optionally test the TopAneu nnU-Net strong baseline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_LAYOUT["data_root"])
    parser.add_argument("--workspace", type=Path, default=DEFAULT_LAYOUT["workspace"])
    parser.add_argument("--dataset-id", type=int, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--split-mode", choices=("holdout", "crossval"), default="holdout")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=PROJECT_DIR / "split.csv",
        help="Editable hold-out CSV bundled in the execution directory.",
    )
    parser.add_argument("--folds", type=int, nargs="+", help="Defaults to fold 0 for holdout or all folds for crossval")
    parser.add_argument("--planner", default=DEFAULT_PLANNER)
    parser.add_argument("--plans", default=DEFAULT_PLANS)
    parser.add_argument("--trainer", default=DEFAULT_TRAINER)
    parser.add_argument("--configuration", default=DEFAULT_CONFIGURATION)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help="Run/log validation and monitor-test metrics every N epochs; train loss is logged every epoch",
    )
    parser.add_argument("--link-mode", choices=("auto", "hardlink", "symlink", "copy"), default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-images", type=Path, help="Optional unlabeled scans to predict after training")
    parser.add_argument("--min-component-voxels", type=int, default=1)
    parser.add_argument("--min-component-mm3", type=float, default=0.0)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="Do not run automatic held-out/external test inference")
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Also train fold 'all'; recommended for one-model final inference under the time limit",
    )
    parser.add_argument(
        "--inference-folds",
        nargs="+",
        help="Folds used for test inference (for example: all, or 0 1 2 3 4). Defaults to 'all' with --train-all.",
    )
    parser.add_argument("--resume", action="store_true", help="Continue existing fold checkpoints")
    parser.add_argument(
        "--save-validation-probabilities",
        action="store_true",
        help="Save very large validation softmax .npz files; only needed for configuration ensembling",
    )
    parser.add_argument("--verify-mask-values", action="store_true", help="Read every mask during data inspection")
    return parser.parse_args()


def run_command(command: Sequence[str], env: dict[str, str], log: list[list[str]]) -> None:
    executable = shutil.which(command[0], path=env.get("PATH"))
    if executable is None:
        suffix = ".exe" if os.name == "nt" else ""
        same_environment = Path(sys.executable).resolve().parent / f"{command[0]}{suffix}"
        if same_environment.exists():
            executable = str(same_environment)
    if executable is None:
        raise RuntimeError(
            f"Required command '{command[0]}' was not found. Install requirements.txt in the active environment."
        )
    rendered = [str(executable), *[str(value) for value in command[1:]]]
    print("\n$ " + " ".join(rendered), flush=True)
    log.append(rendered)
    subprocess.run(rendered, check=True, env=env)


def export_test_outputs(
    raw_predictions: Path,
    staged_images: Path,
    output_dir: Path,
    *,
    max_label: int,
    min_component_voxels: int,
    min_component_mm3: float,
) -> None:
    records: list[dict[str, object]] = []
    for prediction in sorted(raw_predictions.glob("*.nii.gz")):
        case_id = prediction.name[: -len(".nii.gz")]
        result = write_challenge_outputs(
            prediction,
            staged_images / f"{case_id}{IMAGE_SUFFIX}",
            output_dir / "task1" / f"{case_id}.json",
            output_dir / "task2" / f"{case_id}.nii.gz",
            max_label=max_label,
            min_component_voxels=min_component_voxels,
            min_component_mm3=min_component_mm3,
        )
        records.append(
            {"case_id": result.case_id, "locations": list(result.locations), "shape": list(result.shape)}
        )
    if not records:
        raise FileNotFoundError(f"nnU-Net did not produce masks in {raw_predictions}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


def predict_folder(
    input_dir: Path,
    raw_predictions: Path,
    *,
    dataset_id: int,
    configuration: str,
    trainer: str,
    plans: str,
    inference_folds: Sequence[str],
    device: str,
    env: dict[str, str],
    command_log: list[list[str]],
) -> list[Path]:
    inputs = sorted(input_dir.glob(f"*{IMAGE_SUFFIX}"))
    if not inputs:
        raise FileNotFoundError(f"No staged nnU-Net inputs found in {input_dir}")
    expected = {path.name[: -len(IMAGE_SUFFIX)] for path in inputs}
    raw_predictions.mkdir(parents=True, exist_ok=True)
    for stale in raw_predictions.glob("*.nii.gz"):
        if stale.name[: -len(".nii.gz")] not in expected:
            stale.unlink()
    run_command(
        [
            "nnUNetv2_predict",
            "-i",
            str(input_dir),
            "-o",
            str(raw_predictions),
            "-d",
            str(dataset_id),
            "-c",
            configuration,
            "-tr",
            trainer,
            "-p",
            plans,
            "-f",
            *inference_folds,
            "-device",
            device,
        ],
        env,
        command_log,
    )
    predictions = [raw_predictions / f"{case_id}.nii.gz" for case_id in sorted(expected)]
    missing = [str(path) for path in predictions if not path.exists()]
    if missing:
        raise FileNotFoundError(f"nnU-Net did not create {len(missing)} expected predictions: {missing[:3]}")
    return predictions


def main() -> None:
    args = parse_args()
    args.workspace = args.workspace.resolve()
    args.data_root = args.data_root.resolve()
    if args.eval_every < 1:
        raise ValueError("--eval-every must be at least 1")
    folder_name = dataset_folder_name(args.dataset_id, args.dataset_name)
    paths = nnunet_paths(args.workspace)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(nnunet_environment(args.workspace))
    tensorboard_root = args.workspace / "tensorboard"
    tensorboard_root.mkdir(parents=True, exist_ok=True)
    env["TOPANEU_TENSORBOARD_ROOT"] = str(tensorboard_root)
    env["TOPANEU_EVAL_EVERY"] = str(args.eval_every)
    if args.split_mode == "holdout":
        env["TOPANEU_MONITOR_TEST_INPUT"] = str(paths["raw"] / folder_name / "imagesTs")
        env["TOPANEU_MONITOR_TEST_TRUTH"] = str(args.data_root / "location_masks")
        env["TOPANEU_MONITOR_MAX_LABEL"] = str(max(load_label_mapping(args.data_root / "location_mapping.json").values()))
    command_log: list[list[str]] = []
    folds = args.folds or ([0] if args.split_mode == "holdout" else [0, 1, 2, 3, 4])
    if args.split_mode == "holdout" and folds != [0]:
        raise ValueError("holdout mode has one fixed nnU-Net split; use --folds 0 or omit --folds")

    if not args.skip_prepare:
        inspection = inspect_dataset(args.data_root, check_mask_values=args.verify_mask_values)
        if inspection["geometry_errors"] or inspection["annotation_errors"]:
            raise RuntimeError(f"Dataset validation failed:\n{json.dumps(inspection, indent=2)}")
        print(json.dumps({"dataset_inspection": inspection}, indent=2))
        manifest = prepare_nnunet_dataset(
            args.data_root,
            args.workspace,
            dataset_id=args.dataset_id,
            dataset_name=args.dataset_name,
            n_splits=5,
            seed=args.seed,
            link_mode=args.link_mode,
            test_image_dir=args.test_images,
            split_mode=args.split_mode,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            split_csv=args.split_csv,
        )
    else:
        manifest_path = args.workspace / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"--skip-prepare requires an existing manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split_mode") != args.split_mode:
            raise ValueError(
                f"Existing workspace uses split_mode={manifest.get('split_mode')!r}, requested {args.split_mode!r}"
            )
        if args.test_images is not None:
            manifest["external_test_case_ids"] = stage_inference_images(
                args.test_images,
                args.workspace / "inference_inputs" / "external",
                link_mode=args.link_mode,
            )
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not args.skip_preprocess:
        run_command(
            [
                "nnUNetv2_plan_and_preprocess",
                "-d",
                str(args.dataset_id),
                "-pl",
                args.planner,
                "--verify_dataset_integrity",
                "-c",
                args.configuration,
            ],
            env,
            command_log,
        )

    if not args.skip_train:
        training_folds: list[int | str] = list(folds)
        if args.train_all:
            training_folds.append("all")
        for fold in training_folds:
            command = [
                sys.executable,
                "-m",
                "topaneu_baseline.train_nnunet_tensorboard",
                str(args.dataset_id),
                args.configuration,
                str(fold),
                "-tr",
                args.trainer,
                "-p",
                args.plans,
                "-device",
                args.device,
            ]
            if args.save_validation_probabilities and fold != "all":
                command.append("--npz")
            if args.resume:
                command.append("--c")
            run_command(command, env, command_log)

    mapping = load_label_mapping(args.data_root / "location_mapping.json")
    max_label = max(mapping.values())
    if not args.skip_evaluate:
        predictions = find_crossval_predictions(
            args.workspace,
            folder_name,
            trainer=args.trainer,
            plans=args.plans,
            configuration=args.configuration,
            folds=folds,
        )
        summary, per_class, per_case = evaluate_prediction_masks(
            predictions,
            args.data_root / "location_masks",
            max_label=max_label,
            min_component_voxels=args.min_component_voxels,
            min_component_mm3=args.min_component_mm3,
        )
        evaluation_name = "validation" if args.split_mode == "holdout" else "cross_validation"
        evaluation_dir = args.workspace / "evaluation" / evaluation_name
        write_evaluation(evaluation_dir, summary, per_class, per_case)
        write_challenge_metrics_tensorboard(
            tensorboard_root / "final_evaluation", evaluation_name, summary, per_class
        )
        print(f"\n{evaluation_name} challenge metrics:\n" + json.dumps(summary, indent=2))

    if not args.skip_test:
        inference_folds = args.inference_folds or (["all"] if args.train_all else [str(v) for v in folds])
        raw_dataset = paths["raw"] / folder_name
        if args.split_mode == "holdout":
            internal_predictions = predict_folder(
                raw_dataset / "imagesTs",
                args.workspace / "predictions" / "internal_test_raw",
                dataset_id=args.dataset_id,
                configuration=args.configuration,
                trainer=args.trainer,
                plans=args.plans,
                inference_folds=[str(value) for value in inference_folds],
                device=args.device,
                env=env,
                command_log=command_log,
            )
            internal_output = args.workspace / "predictions" / "internal_test_outputs"
            export_test_outputs(
                args.workspace / "predictions" / "internal_test_raw",
                raw_dataset / "imagesTs",
                internal_output,
                max_label=max_label,
                min_component_voxels=args.min_component_voxels,
                min_component_mm3=args.min_component_mm3,
            )
            summary, per_class, per_case = evaluate_prediction_masks(
                internal_predictions,
                args.data_root / "location_masks",
                max_label=max_label,
                prediction_json_dir=internal_output / "task1",
                min_component_voxels=args.min_component_voxels,
                min_component_mm3=args.min_component_mm3,
            )
            write_evaluation(args.workspace / "evaluation" / "test", summary, per_class, per_case)
            write_challenge_metrics_tensorboard(
                tensorboard_root / "final_evaluation", "test", summary, per_class
            )
            print("\nHeld-out test challenge metrics:\n" + json.dumps(summary, indent=2))

        external_input = args.workspace / "inference_inputs" / "external"
        if args.test_images is not None or manifest.get("external_test_case_ids"):
            predict_folder(
                external_input,
                args.workspace / "predictions" / "external_raw",
                dataset_id=args.dataset_id,
                configuration=args.configuration,
                trainer=args.trainer,
                plans=args.plans,
                inference_folds=[str(value) for value in inference_folds],
                device=args.device,
                env=env,
                command_log=command_log,
            )
            export_test_outputs(
                args.workspace / "predictions" / "external_raw",
                external_input,
                args.workspace / "predictions" / "challenge_outputs",
                max_label=max_label,
                min_component_voxels=args.min_component_voxels,
                min_component_mm3=args.min_component_mm3,
            )

    (args.workspace / "commands.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"\nPipeline complete. Workspace: {args.workspace}")


if __name__ == "__main__":
    main()
