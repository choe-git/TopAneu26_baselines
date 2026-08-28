"""Create the physical-space cache required by RNSA surrogate training."""

from __future__ import annotations

import argparse
from pathlib import Path

from rnsa_surrogate.cache import build_cache


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-csv", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--run-dir", type=Path, help="Experiment root; cache is RUN_DIR/cache")
    destination.add_argument("--output", type=Path, help="Explicit cache directory")
    parser.add_argument("--spacing", type=float, nargs=3, default=(0.6, 0.6, 0.6))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.run_dir / "cache" if args.run_dir is not None else args.output
    index = build_cache(args.source, args.split_csv, output, args.spacing, args.overwrite)
    print(f"Cache index: {index}")


if __name__ == "__main__":
    main()
