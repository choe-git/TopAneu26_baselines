"""Validate a completed RNSA surrogate cache and emit a compact report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rnsa_surrogate.cache import atomic_json_dump, validate_cache


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_cache(args.cache, args.deep)
    if args.output is not None:
        atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
