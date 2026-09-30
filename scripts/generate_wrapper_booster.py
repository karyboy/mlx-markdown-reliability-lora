#!/usr/bin/env python3
"""Generate the short contrastive wrapper-correction dataset for Run 2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.wrapper_booster import generate_wrapper_booster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--content-plan-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "content_plans",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "wrapper_booster",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "wrapper_booster_manifest.json",
    )
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["wrapper_booster"]
    manifest = generate_wrapper_booster(
        args.content_plan_dir, args.output_dir, args.summary, settings["seed"]
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrapper booster: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
