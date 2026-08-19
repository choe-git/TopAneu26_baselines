from __future__ import annotations

import hashlib
import csv
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np

from .challenge_io import IMAGE_SUFFIX, load_label_mapping
from .utils import CaseRecord, discover_cases, group_stratified_split


DEFAULT_DATASET_ID = 501
DEFAULT_DATASET_NAME = "TopAneu"
DEFAULT_PLANNER = "nnUNetPlannerResEncM"
DEFAULT_PLANS = "nnUNetResEncUNetMPlans"
DEFAULT_CONFIGURATION = "3d_fullres"
DEFAULT_TRAINER = "nnUNetTrainer"


def repository_layout(project_dir: str | Path) -> dict[str, Path]:
    """Resolve either the server tree or this repository beside an unpacked dataset."""
    project_dir = Path(project_dir).resolve()
    if (project_dir.parent / "images").is_dir() and (project_dir.parent / "location_masks").is_dir():
        repository_root = project_dir.parent
        data_root = repository_root
        workspace = project_dir / "runs"
    elif projects_dir := next((parent for parent in project_dir.parents if parent.name == "projects"), None):
        repository_root = projects_dir.parent
        data_root = repository_root / "resources" / "topaneu_release"
        workspace = repository_root / "runs" / "5_TopAneu"
    else:
        repository_root = project_dir.parent.parent
        data_root = repository_root / "resources" / "topaneu_release"
        workspace = repository_root / "runs" / "5_TopAneu"
    return {
        "project": project_dir,
        "repository_root": repository_root,
        "data_root": data_root,
        "workspace": workspace,
    }


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    if not 1 <= int(dataset_id) <= 999:
        raise ValueError("nnU-Net dataset id must be between 1 and 999")
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", dataset_name)
    if not cleaned:
        raise ValueError("dataset_name must contain letters or digits")
    return f"Dataset{int(dataset_id):03d}_{cleaned}"


def nnunet_paths(workspace: str | Path) -> dict[str, Path]:
    workspace = Path(workspace).resolve()
    return {
        "workspace": workspace,
        "raw": workspace / "nnUNet_raw",
        "preprocessed": workspace / "nnUNet_preprocessed",
        "results": workspace / "nnUNet_results",
    }


def nnunet_environment(workspace: str | Path) -> dict[str, str]:
    paths = nnunet_paths(workspace)
    return {
        "nnUNet_raw": str(paths["raw"]),
        "nnUNet_preprocessed": str(paths["preprocessed"]),
        "nnUNet_results": str(paths["results"]),
    }


def _stable_number(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def make_group_folds(cases: Iterable[CaseRecord], n_splits: int = 5, seed: int = 2026) -> list[dict[str, list[str]]]:
    """Greedy multi-label patient-group split used by nnU-Net's splits_final.json."""
    cases = list(cases)
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    grouped: dict[str, list[CaseRecord]] = defaultdict(list)
    for case in cases:
        grouped[case.patient_group].append(case)
    if len(grouped) < n_splits:
        raise ValueError(f"Need at least {n_splits} patient groups, found {len(grouped)}")

    centers = sorted({case.center for case in cases})
    modalities = sorted({case.modality for case in cases})
    feature_names = (
        [f"label_{index}" for index in range(1, 53)]
        + [f"center_{value}" for value in centers]
        + [f"modality_{value}" for value in modalities]
        + ["positive", "scan_count"]
    )
    feature_index = {name: index for index, name in enumerate(feature_names)}

    group_features: dict[str, np.ndarray] = {}
    for group, group_cases in grouped.items():
        vector = np.zeros(len(feature_names), dtype=np.float64)
        labels = {value for case in group_cases for value in case.locations}
        for label in labels:
            vector[feature_index[f"label_{label}"]] = 1.0
        vector[feature_index[f"center_{group_cases[0].center}"]] = 1.0
        vector[feature_index[f"modality_{group_cases[0].modality}"]] = 1.0
        vector[feature_index["positive"]] = float(bool(labels))
        vector[feature_index["scan_count"]] = float(len(group_cases))
        group_features[group] = vector

    totals = np.sum(list(group_features.values()), axis=0)
    target = totals / n_splits
    rarity_weight = np.zeros_like(totals)
    nonzero = totals > 0
    rarity_weight[nonzero] = 1.0 / totals[nonzero]
    ordered_groups = sorted(
        grouped,
        key=lambda group: (
            -float(np.dot(group_features[group], rarity_weight)),
            _stable_number(f"{seed}:{group}"),
        ),
    )

    fold_features = np.zeros((n_splits, len(feature_names)), dtype=np.float64)
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    scale = np.maximum(target, 1.0)
    for group in ordered_groups:
        vector = group_features[group]
        best_fold = 0
        best_score: tuple[float, int] | None = None
        for fold in range(n_splits):
            proposed = fold_features.copy()
            proposed[fold] += vector
            balance_cost = float(np.sum(((proposed - target) / scale) ** 2))
            group_count_cost = (len(fold_groups[fold]) + 1) / max(1, len(grouped) / n_splits)
            score = (balance_cost + 0.02 * group_count_cost**2, _stable_number(f"{seed}:{group}:{fold}"))
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold
        fold_groups[best_fold].append(group)
        fold_features[best_fold] += vector

    all_case_ids = {case.case_id for case in cases}
    folds: list[dict[str, list[str]]] = []
    validation_seen: set[str] = set()
    for groups in fold_groups:
        val = sorted(case.case_id for group in groups for case in grouped[group])
        train = sorted(all_case_ids - set(val))
        if set(train) & set(val):
            raise RuntimeError("Patient split created overlapping cases")
        validation_seen.update(val)
        folds.append({"train": train, "val": val})
    if validation_seen != all_case_ids:
        raise RuntimeError("Every case must occur in exactly one validation fold")
    for fold in folds:
        train_groups = {case.patient_group for case in cases if case.case_id in set(fold["train"])}
        val_groups = {case.patient_group for case in cases if case.case_id in set(fold["val"])}
        if train_groups & val_groups:
            raise RuntimeError("Patient leakage detected in nnU-Net folds")
    return folds


def make_holdout_split(
    cases: Iterable[CaseRecord],
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 2026,
) -> dict[str, list[CaseRecord]]:
    """Create leakage-safe train/validation/test partitions by patient group."""
    cases = list(cases)
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    train_validation, test = group_stratified_split(cases, val_fraction=test_fraction, seed=seed)
    adjusted_validation_fraction = validation_fraction / (1.0 - test_fraction)
    train, validation = group_stratified_split(
        train_validation,
        val_fraction=adjusted_validation_fraction,
        seed=seed + 1,
    )
    partitions = {"train": train, "validation": validation, "test": test}
    group_sets = {name: {case.patient_group for case in values} for name, values in partitions.items()}
    if (
        group_sets["train"] & group_sets["validation"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["validation"] & group_sets["test"]
    ):
        raise RuntimeError("Patient leakage detected in hold-out split")
    case_ids = [case.case_id for values in partitions.values() for case in values]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != {case.case_id for case in cases}:
        raise RuntimeError("Hold-out split must contain every case exactly once")
    return partitions


def write_split_csv(path: str | Path, partitions: dict[str, Sequence[CaseRecord]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for split_name in ("train", "validation", "test"):
        for case in partitions[split_name]:
            rows.append(
                {
                    "case_id": case.case_id,
                    "split": split_name,
                    "patient_group": case.patient_group,
                    "modality": case.modality,
                    "center": case.center,
                    "locations": " ".join(str(value) for value in case.locations),
                }
            )
    rows.sort(key=lambda row: row["case_id"])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("case_id", "split", "patient_group", "modality", "center", "locations"),
        )
        writer.writeheader()
        writer.writerows(rows)


def load_split_csv(path: str | Path, cases: Iterable[CaseRecord]) -> dict[str, list[CaseRecord]]:
    """Load an editable split CSV while enforcing completeness and patient isolation."""
    path = Path(path)
    cases = list(cases)
    by_id = {case.case_id: case for case in cases}
    partitions: dict[str, list[CaseRecord]] = {"train": [], "validation": [], "test": []}
    seen: set[str] = set()
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"case_id", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns: case_id, split")
        for line_number, row in enumerate(reader, start=2):
            case_id = (row.get("case_id") or "").strip()
            split_name = (row.get("split") or "").strip().lower()
            if case_id not in by_id:
                raise ValueError(f"Unknown case_id at {path}:{line_number}: {case_id!r}")
            if case_id in seen:
                raise ValueError(f"Duplicate case_id at {path}:{line_number}: {case_id}")
            if split_name not in partitions:
                raise ValueError(
                    f"Invalid split at {path}:{line_number}: {split_name!r}; use train, validation, or test"
                )
            seen.add(case_id)
            partitions[split_name].append(by_id[case_id])

    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} cases, including: {missing[:5]}")
    empty = [name for name, values in partitions.items() if not values]
    if empty:
        raise ValueError(f"Split partitions may not be empty: {empty}")

    group_to_split: dict[str, str] = {}
    for split_name, values in partitions.items():
        for case in values:
            previous = group_to_split.setdefault(case.patient_group, split_name)
            if previous != split_name:
                raise ValueError(
                    f"Patient group {case.patient_group!r} is split across {previous} and {split_name} in {path}"
                )
    observed = {value for case in cases for value in case.locations}
    train_labels = {value for case in partitions["train"] for value in case.locations}
    if train_labels != observed:
        raise ValueError(
            f"Training split is missing observed labels: {sorted(observed - train_labels)}. "
            "Move at least one complete patient group carrying each label back to train."
        )
    for values in partitions.values():
        values.sort(key=lambda case: case.case_id)
    return partitions


def _same_file_content_hint(source: Path, destination: Path) -> bool:
    try:
        return source.stat().st_size == destination.stat().st_size
    except OSError:
        return False


def link_or_copy(source: str | Path, destination: str | Path, mode: str = "auto") -> str:
    source = Path(source).resolve()
    destination = Path(destination)
    if mode not in {"auto", "hardlink", "symlink", "copy"}:
        raise ValueError(f"Unknown link mode: {mode}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if _same_file_content_hint(source, destination):
            return "existing"
        raise FileExistsError(f"Destination already exists with different size: {destination}")

    attempts = [mode] if mode != "auto" else ["hardlink", "symlink", "copy"]
    errors: list[str] = []
    for attempt in attempts:
        try:
            if attempt == "hardlink":
                os.link(source, destination)
            elif attempt == "symlink":
                destination.symlink_to(source)
            else:
                shutil.copy2(source, destination)
            return attempt
        except OSError as error:
            errors.append(f"{attempt}: {error}")
    raise OSError(f"Unable to stage {source} -> {destination}: {'; '.join(errors)}")


def _remove_unexpected_files(directory: Path, expected_names: set[str]) -> int:
    removed = 0
    for path in directory.glob("*.nii.gz"):
        if path.name not in expected_names:
            path.unlink()
            removed += 1
    return removed


def stage_inference_images(
    source_dir: str | Path,
    destination_dir: str | Path,
    *,
    link_mode: str = "auto",
) -> list[str]:
    source_dir = Path(source_dir).resolve()
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(source_dir.glob("*.nii.gz"))
    if not sources:
        raise FileNotFoundError(f"No .nii.gz test images found in {source_dir}")
    case_ids: list[str] = []
    expected: set[str] = set()
    for source in sources:
        case_id = source.name[: -len(IMAGE_SUFFIX)] if source.name.endswith(IMAGE_SUFFIX) else source.name[: -len(".nii.gz")]
        if case_id in case_ids:
            raise ValueError(f"Duplicate test case id: {case_id}")
        case_ids.append(case_id)
        destination_name = f"{case_id}{IMAGE_SUFFIX}"
        expected.add(destination_name)
        link_or_copy(source, destination_dir / destination_name, link_mode)
    _remove_unexpected_files(destination_dir, expected)
    return case_ids


def prepare_nnunet_dataset(
    data_root: str | Path,
    workspace: str | Path,
    *,
    dataset_id: int = DEFAULT_DATASET_ID,
    dataset_name: str = DEFAULT_DATASET_NAME,
    n_splits: int = 5,
    seed: int = 2026,
    link_mode: str = "auto",
    test_image_dir: str | Path | None = None,
    split_mode: str = "holdout",
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    split_csv: str | Path | None = None,
) -> dict[str, object]:
    data_root = Path(data_root).resolve()
    paths = nnunet_paths(workspace)
    folder_name = dataset_folder_name(dataset_id, dataset_name)
    raw_dataset = paths["raw"] / folder_name
    preprocessed_dataset = paths["preprocessed"] / folder_name
    images_tr = raw_dataset / "imagesTr"
    labels_tr = raw_dataset / "labelsTr"
    images_ts = raw_dataset / "imagesTs"
    for directory in (images_tr, labels_tr, images_ts, preprocessed_dataset, paths["results"]):
        directory.mkdir(parents=True, exist_ok=True)

    mapping_path = data_root / "location_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing location mapping: {mapping_path}")
    labels = load_label_mapping(mapping_path)
    cases = discover_cases(data_root)
    if split_mode == "holdout":
        split_csv_path = Path(split_csv).resolve() if split_csv is not None else paths["workspace"] / "split.csv"
        if split_csv_path.exists():
            partitions = load_split_csv(split_csv_path, cases)
            split_source = "existing_csv"
        else:
            partitions = make_holdout_split(
                cases,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=seed,
            )
            write_split_csv(split_csv_path, partitions)
            split_source = "generated_csv"
        training_cases = partitions["train"] + partitions["validation"]
        internal_test_cases = partitions["test"]
        folds = [
            {
                "train": sorted(case.case_id for case in partitions["train"]),
                "val": sorted(case.case_id for case in partitions["validation"]),
            }
        ]
    elif split_mode == "crossval":
        split_csv_path = None
        split_source = "generated_crossval"
        partitions = None
        training_cases = cases
        internal_test_cases = []
        folds = make_group_folds(cases, n_splits=n_splits, seed=seed)
    else:
        raise ValueError("split_mode must be 'holdout' or 'crossval'")

    expected_images_tr = {f"{case.case_id}{IMAGE_SUFFIX}" for case in training_cases}
    expected_labels_tr = {f"{case.case_id}.nii.gz" for case in training_cases}
    expected_images_ts = {f"{case.case_id}{IMAGE_SUFFIX}" for case in internal_test_cases}
    removed = {
        "imagesTr": _remove_unexpected_files(images_tr, expected_images_tr),
        "labelsTr": _remove_unexpected_files(labels_tr, expected_labels_tr),
        "imagesTs": _remove_unexpected_files(images_ts, expected_images_ts),
    }
    staging_counts: dict[str, int] = defaultdict(int)
    for case in training_cases:
        staging_counts[link_or_copy(case.image, images_tr / f"{case.case_id}{IMAGE_SUFFIX}", link_mode)] += 1
        staging_counts[link_or_copy(case.location_mask, labels_tr / f"{case.case_id}.nii.gz", link_mode)] += 1

    for case in internal_test_cases:
        staging_counts[link_or_copy(case.image, images_ts / f"{case.case_id}{IMAGE_SUFFIX}", link_mode)] += 1

    external_test_case_ids: list[str] = []
    if test_image_dir is not None:
        external_test_case_ids = stage_inference_images(
            test_image_dir,
            paths["workspace"] / "inference_inputs" / "external",
            link_mode=link_mode,
        )

    dataset_json = {
        "channel_names": {"0": "angiography"},
        "labels": labels,
        "numTraining": len(training_cases),
        "file_ending": ".nii.gz",
        "name": folder_name,
        "description": "TopAneu 2026 vessel-specific aneurysm segmentation",
        "reference": "https://topaneu-26.grand-challenge.org/",
        "licence": "See bundled Terms_of_use.txt",
    }
    (raw_dataset / "dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")

    (preprocessed_dataset / "splits_final.json").write_text(json.dumps(folds, indent=2) + "\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "dataset_id": int(dataset_id),
        "dataset_name": folder_name,
        "data_root": str(data_root),
        "workspace": str(paths["workspace"]),
        "split_mode": split_mode,
        "split_source": split_source,
        "split_csv": str(split_csv_path) if split_csv_path is not None else None,
        "num_training": len(training_cases),
        "num_patient_groups": len({case.patient_group for case in cases}),
        "num_internal_test": len(internal_test_cases),
        "num_external_test": len(external_test_case_ids),
        "max_label": max(labels.values()),
        "staging": dict(staging_counts),
        "removed_stale_staging_files": removed,
        "internal_test_case_ids": sorted(case.case_id for case in internal_test_cases),
        "external_test_case_ids": external_test_case_ids,
        "folds": [
            {"fold": index, "train": len(fold["train"]), "validation": len(fold["val"])}
            for index, fold in enumerate(folds)
        ],
    }
    if partitions is not None:
        manifest["holdout"] = {
            "train": len(partitions["train"]),
            "validation": len(partitions["validation"]),
            "test": len(partitions["test"]),
            "requested_fractions": {
                "validation": validation_fraction,
                "test": test_fraction,
            },
        }
    (paths["workspace"] / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def inspect_dataset(data_root: str | Path, *, check_mask_values: bool = False) -> dict[str, object]:
    data_root = Path(data_root).resolve()
    cases = discover_cases(data_root)
    shapes: list[tuple[int, ...]] = []
    spacings: list[tuple[float, ...]] = []
    geometry_errors: list[str] = []
    annotation_errors: list[str] = []
    observed: set[int] = set()
    for case in cases:
        image = nib.load(case.image)
        mask = nib.load(case.location_mask)
        shapes.append(tuple(int(v) for v in image.shape))
        spacings.append(tuple(float(v) for v in image.header.get_zooms()[:3]))
        if image.shape != mask.shape or not np.allclose(image.affine, mask.affine, rtol=1e-5, atol=1e-4):
            geometry_errors.append(case.case_id)
        if check_mask_values:
            mask_locations = {int(v) for v in np.unique(np.asanyarray(mask.dataobj)) if int(v) > 0}
            observed.update(mask_locations)
            if mask_locations != set(case.locations):
                annotation_errors.append(case.case_id)
        else:
            observed.update(case.locations)

    shape_array = np.asarray(shapes)
    spacing_array = np.asarray(spacings)
    return {
        "num_scans": len(cases),
        "num_patient_groups": len({case.patient_group for case in cases}),
        "modalities": {value: sum(case.modality == value for case in cases) for value in sorted({c.modality for c in cases})},
        "centers": {value: sum(case.center == value for case in cases) for value in sorted({c.center for c in cases})},
        "positive_scans": sum(bool(case.locations) for case in cases),
        "negative_scans": sum(not case.locations for case in cases),
        "shape_min": shape_array.min(axis=0).tolist(),
        "shape_median": np.median(shape_array, axis=0).tolist(),
        "shape_max": shape_array.max(axis=0).tolist(),
        "spacing_min": spacing_array.min(axis=0).tolist(),
        "spacing_median": np.median(spacing_array, axis=0).tolist(),
        "spacing_max": spacing_array.max(axis=0).tolist(),
        "observed_labels": sorted(observed),
        "missing_labels": sorted(set(range(1, 53)) - observed),
        "geometry_errors": geometry_errors,
        "annotation_errors": annotation_errors,
        "mask_values_checked": bool(check_mask_values),
    }


def find_crossval_predictions(
    workspace: str | Path,
    dataset_folder: str,
    *,
    trainer: str = DEFAULT_TRAINER,
    plans: str = DEFAULT_PLANS,
    configuration: str = DEFAULT_CONFIGURATION,
    folds: Sequence[int] = (0, 1, 2, 3, 4),
) -> list[Path]:
    paths = nnunet_paths(workspace)
    model_folder = paths["results"] / dataset_folder / f"{trainer}__{plans}__{configuration}"
    predictions: list[Path] = []
    for fold in folds:
        validation = model_folder / f"fold_{int(fold)}" / "validation"
        if not validation.exists():
            raise FileNotFoundError(f"Missing nnU-Net validation directory: {validation}")
        predictions.extend(sorted(validation.glob("*.nii.gz")))
    return predictions
