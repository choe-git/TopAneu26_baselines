from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from .constants import NUM_LOCATIONS


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    image: str
    location_json: str
    location_mask: str
    modality: str
    center: str
    patient_group: str
    locations: tuple[int, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_id_from_image(path: str | Path) -> str:
    name = Path(path).name
    suffix = "_0000.nii.gz"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected image filename: {name}")
    return name[: -len(suffix)]


def parse_case_id(case_id: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"topaneu_(center\d+)_(ct|mr)_(.+)", case_id)
    if match is None:
        raise ValueError(f"Unexpected case id: {case_id}")
    center, modality, patient = match.groups()
    # Center-4 includes longitudinal scans such as 008_1 and 008_2.
    if center == "center4":
        patient = re.sub(r"_(\d+)$", "", patient)
    return center, modality, f"{center}_{patient}"


def discover_cases(data_root: str | Path) -> list[CaseRecord]:
    root = Path(data_root).resolve()
    records: list[CaseRecord] = []
    for image_path in sorted((root / "images").glob("*_0000.nii.gz")):
        case_id = case_id_from_image(image_path)
        center, modality, patient_group = parse_case_id(case_id)
        json_path = root / "location_jsons" / f"{case_id}.json"
        mask_path = root / "location_masks" / f"{case_id}.nii.gz"
        if not json_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing annotation for {case_id}")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        locations = tuple(sorted({int(v) for v in payload.get("locations", [])}))
        invalid = [v for v in locations if not 1 <= v <= NUM_LOCATIONS]
        if invalid:
            raise ValueError(f"Invalid location labels for {case_id}: {invalid}")
        records.append(
            CaseRecord(
                case_id=case_id,
                image=str(image_path),
                location_json=str(json_path),
                location_mask=str(mask_path),
                modality=modality,
                center=center,
                patient_group=patient_group,
                locations=locations,
            )
        )
    if not records:
        raise FileNotFoundError(f"No TopAneu images found under {root / 'images'}")
    return records


def _stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def group_stratified_split(
    cases: Iterable[CaseRecord], val_fraction: float = 0.2, seed: int = 2026
) -> tuple[list[CaseRecord], list[CaseRecord]]:
    """Split by patient while preserving strata and every observable training class."""
    cases = list(cases)
    grouped: dict[str, list[CaseRecord]] = {}
    for case in cases:
        grouped.setdefault(case.patient_group, []).append(case)

    strata: dict[tuple[str, str], list[str]] = {}
    for group, group_cases in grouped.items():
        key = (group_cases[0].center, group_cases[0].modality)
        strata.setdefault(key, []).append(group)

    val_groups: set[str] = set()
    for key, groups in sorted(strata.items()):
        groups.sort(key=lambda g: _stable_int(f"{seed}:{key}:{g}"))
        n_val = max(1, int(round(len(groups) * val_fraction))) if len(groups) > 1 else 0
        val_groups.update(groups[:n_val])

    group_labels = {
        group: {label for case in group_cases for label in case.locations}
        for group, group_cases in grouped.items()
    }
    observed_labels = {label for labels in group_labels.values() for label in labels}

    # A class represented by only one patient must remain trainable. Move the
    # smallest required set of patient groups from validation back to training.
    for label in sorted(observed_labels):
        train_count = sum(label in labels for group, labels in group_labels.items() if group not in val_groups)
        if train_count == 0:
            candidates = [group for group in val_groups if label in group_labels[group]]
            candidates.sort(key=lambda group: (len(group_labels[group]), _stable_int(f"repair:{seed}:{group}")))
            val_groups.remove(candidates[0])

    # When possible, expose a positive validation example without removing the
    # final training example of any label carried by that same patient.
    for label in sorted(observed_labels):
        if any(label in group_labels[group] for group in val_groups):
            continue
        candidate_groups = []
        for group, labels in group_labels.items():
            if group in val_groups or label not in labels:
                continue
            safe = all(
                sum(other in values for candidate, values in group_labels.items() if candidate not in val_groups) > 1
                for other in labels
            )
            if safe:
                candidate_groups.append(group)
        if candidate_groups:
            candidate_groups.sort(key=lambda group: (len(group_labels[group]), _stable_int(f"validation:{seed}:{group}")))
            val_groups.add(candidate_groups[0])

    train = [case for case in cases if case.patient_group not in val_groups]
    val = [case for case in cases if case.patient_group in val_groups]
    if {c.patient_group for c in train} & {c.patient_group for c in val}:
        raise RuntimeError("Patient leakage detected in split")
    train_labels = {label for case in train for label in case.locations}
    if train_labels != observed_labels:
        raise RuntimeError(f"Training split lost observable labels: {sorted(observed_labels - train_labels)}")
    return train, val


def save_split(path: str | Path, train: list[CaseRecord], val: list[CaseRecord]) -> None:
    payload = {
        "train": [asdict(case) for case in train],
        "val": [asdict(case) for case in val],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def labels_to_multihot(locations: Iterable[int]) -> np.ndarray:
    target = np.zeros(NUM_LOCATIONS, dtype=np.float32)
    for location in locations:
        target[int(location) - 1] = 1.0
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return payload


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
