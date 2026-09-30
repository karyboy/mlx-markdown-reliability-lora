#!/usr/bin/env python3
"""Generate the deterministic pilot SFT dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.generator import generate_pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "pilot",
    )
    parser.add_argument("--train-count", type=int, default=450)
    parser.add_argument("--valid-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = generate_pilot(
        args.output_dir,
        train_count=args.train_count,
        valid_count=args.valid_count,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Pilot data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
