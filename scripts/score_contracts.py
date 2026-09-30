#!/usr/bin/env python3
"""Score generated responses against embedded deterministic contracts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.verifier import verify_markdown


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in args.predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    passed = 0
    check_totals: Counter[str] = Counter()
    task_totals: Counter[str] = Counter()
    task_passes: Counter[str] = Counter()
    scored = []
    for record in records:
        contract = record["metadata"].get("contract")
        if contract is None:
            raise SystemExit("Prediction metadata has no contract; use LatentMD's official evaluator.")
        result = verify_markdown(record["response"], contract)
        task_type = record["metadata"].get("task_type", "unknown")
        task_totals[task_type] += 1
        if result["passed"]:
            passed += 1
            task_passes[task_type] += 1
        for name, value in result["checks"].items():
            check_totals[name] += int(value)
        scored.append({**record, "score": result})

    count = len(records)
    metrics = {
        "count": count,
        "full_contract_pass_rate": passed / count if count else 0.0,
        "check_pass_rates": {
            name: total / count if count else 0.0 for name, total in sorted(check_totals.items())
        },
        "task_pass_rates": {
            name: task_passes[name] / total for name, total in sorted(task_totals.items())
        },
    }
    scored_path = args.predictions.with_name(args.predictions.stem + "_scored.jsonl")
    metrics_path = args.predictions.with_name(args.predictions.stem + "_metrics.json")
    with scored_path.open("w", encoding="utf-8") as handle:
        for record in scored:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Detailed scores: {scored_path.resolve()}")


if __name__ == "__main__":
    main()
