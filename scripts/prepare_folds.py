"""Create deterministic multilabel-stratified cross-validation folds."""

from __future__ import annotations

import argparse
from pathlib import Path

from rnsa_surrogate.cache import atomic_json_dump, load_cache_index, sha256_file
from rnsa_surrogate.folds import multilabel_folds
from rnsa_surrogate.run_layout import BaselineRunLayout


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    layout = BaselineRunLayout.from_root(args.run_dir)
    output = layout.fold_manifest
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    index = load_cache_index(layout.cache)
    development = [case for case in index["cases"] if case["split"] != "test"]
    assignments = multilabel_folds(development, args.n_folds, args.seed)
    folds = {
        str(fold): sorted(
            case_id for case_id, assigned in assignments.items() if assigned == fold
        )
        for fold in range(args.n_folds)
    }
    atomic_json_dump(
        {
            "n_folds": args.n_folds,
            "seed": args.seed,
            "pool": "original train + val; original test excluded",
            "cache_index": index["index_path"],
            "cache_index_sha256": sha256_file(index["index_path"]),
            "case_to_fold": assignments,
            "folds": folds,
        },
        output,
    )
    print(f"Fold manifest: {output}")
    print("Fold sizes: " + ", ".join(f"{key}={len(value)}" for key, value in folds.items()))


if __name__ == "__main__":
    main()
