#!/usr/bin/env python3
"""Re-run the contract verifier against every assistant target in JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.verifier import verify_markdown
from markdown_reliability.latentmd_structural import score_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    failures = []
    seen_ids = set()
    seen_hashes: Counter[str] = Counter()
    families_by_split: dict[str, set[str]] = defaultdict(set)
    for path in args.paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                record_id = record["id"]
                if record_id in seen_ids:
                    failures.append(
                        {"id": record_id, "failure": "duplicate_id", "path": str(path)}
                    )
                    continue
                seen_ids.add(record_id)
                target = record["messages"][-1]["content"]
                contract = record["metadata"]["contract"]
                if contract.get("contract_type") == "latentmd_structural_v1":
                    result = score_response(
                        {"response": target, "metadata": record["metadata"]}
                    )
                else:
                    result = verify_markdown(target, contract)
                counts["records"] += 1
                counts[f"task_{record['metadata']['task_type']}"] += 1
                seen_hashes[record["metadata"]["target_sha256"]] += 1
                split = record["metadata"].get("split", "unknown")
                family_id = record["metadata"].get("task_family_id")
                if family_id:
                    families_by_split[split].add(family_id)
                if result["passed"]:
                    counts["passed"] += 1
                else:
                    failures.append(
                        {
                            "id": record_id,
                            "failure": result["failures"],
                            "path": str(path),
                            "line": line_number,
                        }
                    )

    train_valid_overlap = sorted(
        families_by_split.get("train", set()) & families_by_split.get("valid", set())
    )
    if train_valid_overlap:
        failures.append(
            {"failure": "task_family_overlap", "values": train_valid_overlap[:20]}
        )
    summary = {
        "counts": dict(counts),
        "unique_ids": len(seen_ids),
        "unique_targets": len(seen_hashes),
        "reused_target_instances": sum(value - 1 for value in seen_hashes.values()),
        "task_families_by_split": {
            split: len(values) for split, values in sorted(families_by_split.items())
        },
        "train_valid_task_family_overlap": train_valid_overlap,
        "failures": failures[:20],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
