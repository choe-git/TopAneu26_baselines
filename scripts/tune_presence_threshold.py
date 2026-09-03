"""Tune Task 1 presence threshold from a completed validation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rnsa_surrogate.cache import atomic_json_dump
from rnsa_surrogate.official_metrics import summarize_task1, task1_case_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--per-case",
        type=Path,
        required=True,
        help="Validation per_case_metrics.json produced by evaluate.py",
    )
    parser.add_argument("--minimum", type=float, default=0.05)
    parser.add_argument("--maximum", type=float, default=0.75)
    parser.add_argument("--steps", type=int, default=71)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def selection_score(summary: dict[str, Any]) -> float:
    macro = summary["macro"]
    return float(np.mean([macro["precision"], macro["recall"], macro["mcc"]]))


def main() -> None:
    args = parse_args()
    if args.steps < 2 or not 0.0 <= args.minimum < args.maximum <= 1.0:
        raise ValueError("Use steps >= 2 and 0 <= minimum < maximum <= 1")
    cases = json.loads(args.per_case.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("per-case metrics are empty")
    required = {"task1_truth", "task1_location_scores"}
    missing = required - set(cases[0])
    if missing:
        raise ValueError(
            "Evaluation predates score logging; rerun validation first. "
            f"Missing fields: {sorted(missing)}"
        )

    results = []
    for threshold in np.linspace(args.minimum, args.maximum, args.steps):
        counts = []
        predicted_total = 0
        for case in cases:
            scores = np.asarray(case["task1_location_scores"], dtype=np.float64)
            if scores.shape != (52,):
                raise ValueError(
                    f"{case.get('case_id', '<unknown>')} has {scores.shape}, expected (52,)"
                )
            prediction = (np.flatnonzero(scores >= threshold) + 1).tolist()
            predicted_total += len(prediction)
            counts.append(task1_case_counts(case["task1_truth"], prediction))
        summary = summarize_task1(counts)
        results.append(
            {
                "threshold": float(threshold),
                "selection_score": selection_score(summary),
                "predicted_labels_total": predicted_total,
                "predicted_labels_per_case": predicted_total / len(cases),
                "official_task1": summary,
            }
        )

    best = max(results, key=lambda item: item["selection_score"])
    payload = {
        "source": str(args.per_case.resolve()),
        "cases": len(cases),
        "objective": "mean of official Task 1 macro precision, recall, and MCC",
        "best": best,
        "sweep": results,
    }
    output = args.output or args.per_case.with_name("presence_threshold_sweep.json")
    atomic_json_dump(payload, output)
    print(f"Best presence threshold: {best['threshold']:.4f}")
    print(f"Selection score: {best['selection_score']:.6f}")
    print(f"Threshold sweep: {output}")


if __name__ == "__main__":
    main()
