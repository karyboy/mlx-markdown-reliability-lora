#!/usr/bin/env python3
"""Generate the final 5,000/500 Markdown SFT train/validation corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.final_generator import generate_final_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--content-plan-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "content_plans",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "final",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "results" / "final_sft_manifest.json",
    )
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["final_sft_data"]
    manifest = generate_final_dataset(
        args.content_plan_dir, args.output_dir, args.summary, settings
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Final SFT data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
