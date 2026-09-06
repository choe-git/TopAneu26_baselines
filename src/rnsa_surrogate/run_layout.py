"""Canonical paths for one shared RNSA surrogate baseline experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def validate_run_root(path: str | Path) -> Path:
    """Resolve RUN_DIR and reject accidental use of an nnU-Net data root."""
    root = Path(path).resolve()
    conflicts = [
        name
        for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
        if (root / name).exists()
    ]
    if conflicts:
        raise ValueError(
            f"RUN_DIR is an nnU-Net experiment ({', '.join(conflicts)}): {root}. "
            "Create a new timestamp below runs/5_TopAneu/baseline."
        )
    return root


def validate_variant_name(value: str, option: str) -> str:
    """Validate a user-selected artifact directory without allowing traversal."""
    value = str(value)
    if not value or not value.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"{option} must be a simple directory name")
    return value


@dataclass(frozen=True)
class BaselineRunLayout:
    """Resolve every pipeline artifact below one timestamped RUN_DIR."""

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> BaselineRunLayout:
        return cls(validate_run_root(root))

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def baseline(self) -> Path:
        return self.root / "baseline"

    @property
    def checkpoint(self) -> Path:
        return self.baseline / "checkpoint_best.pth"

    @property
    def fold_manifest(self) -> Path:
        return self.baseline / "folds.json"

    @property
    def vessel_pretrain(self) -> Path:
        return self.baseline / "vessel_pretrain" / "shared"

    @property
    def folds(self) -> Path:
        return self.baseline / "folds"

    @property
    def ensemble(self) -> Path:
        return self.baseline / "ensemble"

    @property
    def refiner(self) -> Path:
        return self.baseline / "refiner"

    @property
    def refiner_candidates(self) -> Path:
        return self.refiner / "candidates"

    def refiner_candidates_for(self, variant: str = "candidates") -> Path:
        return self.refiner / validate_variant_name(variant, "candidate variant")

    @property
    def refiner_folds(self) -> Path:
        return self.refiner / "folds"

    def refiner_folds_for(self, variant: str = "refiner") -> Path:
        return (
            self.baseline
            / validate_variant_name(variant, "refiner variant")
            / "folds"
        )

    @property
    def refiner_location(self) -> Path:
        return self.baseline / "refiner_location"

    @property
    def refiner_location_folds(self) -> Path:
        return self.refiner_location / "folds"

    @property
    def roi_refiner(self) -> Path:
        return self.baseline / "roi_refiner"

    @property
    def roi_refiner_folds(self) -> Path:
        return self.roi_refiner / "folds"

    @property
    def tensorboard(self) -> Path:
        return self.root / "tensorboard" / "baseline"

    @property
    def refiner_tensorboard(self) -> Path:
        return self.tensorboard / "refiner"

    def refiner_tensorboard_for(self, variant: str = "refiner") -> Path:
        return self.tensorboard / validate_variant_name(variant, "refiner variant")

    def refiner_evaluation_for(self, variant: str = "refiner") -> Path:
        variant = validate_variant_name(variant, "refiner variant")
        name = "oof_refined" if variant == "refiner" else f"oof_refined_{variant}"
        return self.ensemble / "evaluation" / name

    @property
    def refiner_location_tensorboard(self) -> Path:
        return self.tensorboard / "refiner_location"

    @property
    def roi_refiner_tensorboard(self) -> Path:
        return self.tensorboard / "roi_refiner"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"


def create_legacy_run(output_root: str | Path, name: str) -> Path:
    """Create the old name/timestamp layout when RUN_DIR is not supplied."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = Path(output_root).resolve() / name / timestamp
    root.mkdir(parents=True, exist_ok=False)
    return root
