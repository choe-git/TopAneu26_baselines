"""Deterministic multilabel-stratified folds for TopAneu development cases."""

from __future__ import annotations

from typing import Any

import numpy as np


def multilabel_folds(
    cases: list[dict[str, Any]], n_folds: int = 5, seed: int = 2026
) -> dict[str, int]:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(cases) < n_folds:
        raise ValueError("Number of cases must be at least n_folds")

    # 52 locations + aneurysm presence + MR/CT modality provide the balancing
    # targets described by the winning solution.
    features = np.zeros((len(cases), 55), dtype=np.float64)
    for row, case in enumerate(cases):
        labels = {
            int(value)
            for value in case.get("json_locations", [])
            if 1 <= int(value) <= 52
        }
        for label in labels:
            features[row, label - 1] = 1.0
        features[row, 52] = float(bool(labels))
        features[row, 53 if case["modality"] == "mr" else 54] = 1.0

    frequency = features.sum(axis=0)
    rarity = (features / np.maximum(frequency, 1.0)).sum(axis=1)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(cases)), -rarity))
    fold_features = np.zeros((n_folds, features.shape[1]), dtype=np.float64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)
    target_features = frequency / n_folds
    target_size = len(cases) / n_folds
    assignments: dict[str, int] = {}

    for index in order:
        candidate_scores = []
        for fold in range(n_folds):
            label_cost = (
                fold_features[fold]
                * features[index]
                / np.maximum(target_features, 1.0)
            ).sum()
            size_cost = fold_sizes[fold] / target_size
            candidate_scores.append(label_cost + 0.25 * size_cost)
        best_score = min(candidate_scores)
        choices = [
            fold
            for fold, score in enumerate(candidate_scores)
            if np.isclose(score, best_score)
        ]
        fold = min(choices, key=lambda value: (fold_sizes[value], value))
        fold_features[fold] += features[index]
        fold_sizes[fold] += 1
        assignments[str(cases[index]["case_id"])] = fold
    return assignments
