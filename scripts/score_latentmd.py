#!/usr/bin/env python3
"""Score frozen LatentMD predictions with the local structural evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.latentmd_structural import aggregate_scores, score_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional second copy of the compact metrics JSON for publication.",
    )
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored_records = [{**record, "score": score_response(record)} for record in records]
    metrics = aggregate_scores(scored_records)

    scored_path = args.predictions.with_name(args.predictions.stem + "_scored.jsonl")
    metrics_path = args.predictions.with_name(args.predictions.stem + "_metrics.json")
    with scored_path.open("w", encoding="utf-8") as handle:
        for record in scored_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Detailed scores: {scored_path.resolve()}")
    print(f"Metrics: {metrics_path.resolve()}")
    if args.summary:
        print(f"Published summary: {args.summary.resolve()}")


if __name__ == "__main__":
    main()
