from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from topaneu_baseline.challenge_io import locations_from_mask, validate_challenge_outputs, write_challenge_outputs
from topaneu_baseline.challenge_metrics import evaluate_prediction_masks, segmentation_metrics
from topaneu_baseline.nnunet_pipeline import (
    load_split_csv,
    make_group_folds,
    make_holdout_split,
    repository_layout,
    write_split_csv,
)
from topaneu_baseline.tensorboard_logging import challenge_metric_scalars
from topaneu_baseline.utils import CaseRecord


def _case(case_id: str, patient_group: str, locations: tuple[int, ...], modality: str = "mr") -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        image=f"{case_id}_0000.nii.gz",
        location_json=f"{case_id}.json",
        location_mask=f"{case_id}.nii.gz",
        modality=modality,
        center="center1",
        patient_group=patient_group,
        locations=locations,
    )


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    affine = np.diag([0.5, 0.5, 1.0, 1.0]) if affine is None else affine
    nib.save(nib.Nifti1Image(data, affine), str(path))


def test_group_folds_prevent_patient_leakage() -> None:
    cases = [
        _case(f"topaneu_center1_mr_{index:03d}", f"patient_{index}", ((index % 4) + 1,))
        for index in range(10)
    ]
    cases += [
        _case("topaneu_center1_mr_100_1", "longitudinal", (1,)),
        _case("topaneu_center1_mr_100_2", "longitudinal", (2,)),
    ]
    folds = make_group_folds(cases, n_splits=5, seed=7)
    validation = [case for fold in folds for case in fold["val"]]
    assert sorted(validation) == sorted(case.case_id for case in cases)
    assert len(validation) == len(set(validation))
    for fold in folds:
        train_groups = {case.patient_group for case in cases if case.case_id in fold["train"]}
        val_groups = {case.patient_group for case in cases if case.case_id in fold["val"]}
        assert not train_groups & val_groups


def test_holdout_split_is_complete_and_leakage_safe() -> None:
    cases = [
        _case(f"topaneu_center1_mr_{index:03d}", f"patient_{index}", ((index % 4) + 1,))
        for index in range(30)
    ]
    cases += [
        _case("topaneu_center1_mr_100_1", "longitudinal", (1,)),
        _case("topaneu_center1_mr_100_2", "longitudinal", (2,)),
    ]
    split = make_holdout_split(cases, validation_fraction=0.15, test_fraction=0.15, seed=3)
    all_ids = [case.case_id for values in split.values() for case in values]
    assert sorted(all_ids) == sorted(case.case_id for case in cases)
    assert len(all_ids) == len(set(all_ids))
    group_sets = {name: {case.patient_group for case in values} for name, values in split.items()}
    assert not group_sets["train"] & group_sets["validation"]
    assert not group_sets["train"] & group_sets["test"]
    assert not group_sets["validation"] & group_sets["test"]


def test_editable_split_csv_roundtrip(tmp_path: Path) -> None:
    cases = [
        _case(f"topaneu_center1_mr_{index:03d}", f"patient_{index}", ((index % 4) + 1,))
        for index in range(30)
    ]
    split = make_holdout_split(cases, validation_fraction=0.15, test_fraction=0.15, seed=5)
    path = tmp_path / "split.csv"
    write_split_csv(path, split)
    restored = load_split_csv(path, cases)
    assert {
        name: [case.case_id for case in values]
        for name, values in restored.items()
    } == {
        name: sorted(case.case_id for case in values)
        for name, values in split.items()
    }
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "case_id,split,patient_group,modality,center,locations"


def test_repository_layout_matches_server_tree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    layout = repository_layout(repository / "projects" / "5_TopAneu")
    assert layout["repository_root"] == repository
    assert layout["data_root"] == repository / "resources" / "topaneu_release"
    assert layout["workspace"] == repository / "runs" / "5_TopAneu"


def test_repository_layout_detects_adjacent_local_dataset(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "location_masks").mkdir()
    project = tmp_path / "baseline_work"
    project.mkdir()
    layout = repository_layout(project)
    assert layout["data_root"] == tmp_path
    assert layout["workspace"] == project / "runs"


def test_challenge_output_format_and_geometry(tmp_path: Path) -> None:
    image_data = np.zeros((8, 8, 8), dtype=np.int16)
    prediction = np.zeros_like(image_data, dtype=np.uint8)
    prediction[1:3, 1:3, 1:3] = 7
    prediction[6, 6, 6] = 9
    image_path = tmp_path / "case_0000.nii.gz"
    prediction_path = tmp_path / "case.nii.gz"
    task1_path = tmp_path / "task1" / "case.json"
    task2_path = tmp_path / "task2" / "case.nii.gz"
    _save_nifti(image_path, image_data)
    _save_nifti(prediction_path, prediction)

    result = write_challenge_outputs(
        prediction_path,
        image_path,
        task1_path,
        task2_path,
        max_label=52,
        min_component_voxels=2,
    )
    assert result.locations == (7,)
    assert json.loads(task1_path.read_text()) == [7]
    assert nib.load(task2_path).get_data_dtype() == np.dtype(np.uint8)
    assert np.array_equal(nib.load(task2_path).affine, nib.load(image_path).affine)
    validate_challenge_outputs(image_path, task1_path, task2_path, max_label=52)


def test_locations_use_physical_component_volume() -> None:
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 4
    assert locations_from_mask(mask, max_label=52, min_component_mm3=0.9, spacing=(0.5, 0.5, 0.5)) == [4]
    assert locations_from_mask(mask, max_label=52, min_component_mm3=1.1, spacing=(0.5, 0.5, 0.5)) == []


def test_metrics_match_definitions(tmp_path: Path) -> None:
    truth_dir = tmp_path / "truth"
    pred_dir = tmp_path / "pred"
    truth_dir.mkdir()
    pred_dir.mkdir()
    truth = np.zeros((6, 6, 6), dtype=np.uint8)
    prediction = np.zeros_like(truth)
    truth[1:3, 1:3, 1:3] = 1
    prediction[1:3, 1:3, 1:3] = 1
    _save_nifti(truth_dir / "case.nii.gz", truth)
    _save_nifti(pred_dir / "case.nii.gz", prediction)

    dice, volsim, hd95 = segmentation_metrics(truth == 1, prediction == 1, (0.5, 0.5, 1.0))
    assert dice == 1.0
    assert volsim == 1.0
    assert hd95 == 0.0
    summary, per_class, per_case = evaluate_prediction_masks(
        [pred_dir / "case.nii.gz"], truth_dir, max_label=2
    )
    assert summary["macro"]["Dice"] == 1.0
    assert per_class[0].precision == 1.0
    assert per_class[0].recall == 1.0
    assert per_case[0]["Dice"] == 1.0

    scalars = challenge_metric_scalars("test", summary, per_class)
    assert scalars["test/final/macro/Dice"] == 1.0
    assert scalars["test/final/per_class/class_01/Dice"] == 1.0
    assert "test/final/per_class/class_02/Dice" not in scalars
    periodic = challenge_metric_scalars("monitor_test", summary, per_class, stage="periodic")
    assert periodic["monitor_test/periodic/macro/Dice"] == 1.0
