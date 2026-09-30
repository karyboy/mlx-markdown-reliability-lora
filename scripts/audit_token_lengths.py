#!/usr/bin/env python3
"""Measure final chat-template sequence lengths with the pinned tokenizer."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "final",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "token_length_audit.json",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    args = parser.parse_args()
    model = yaml.safe_load(args.config.read_text(encoding="utf-8"))["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["id"], revision=model["revision"], local_files_only=True
    )

    summary = {
        "model": model,
        "chat_template": "tokenizer.apply_chat_template(add_generation_prompt=False)",
        "configured_max_seq_length": args.max_seq_length,
        "splits": {},
    }
    for split in ("train", "valid"):
        rows = [
            json.loads(line)
            for line in (args.data_dir / f"{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        lengths = []
        for row in rows:
            chat = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            lengths.append(len(tokenizer.encode(chat, add_special_tokens=False)))
        summary["splits"][split] = {
            "count": len(lengths),
            "min_tokens": min(lengths),
            "median_tokens": percentile(lengths, 0.50),
            "p95_tokens": percentile(lengths, 0.95),
            "p99_tokens": percentile(lengths, 0.99),
            "max_tokens": max(lengths),
            "above_configured_max": sum(
                length > args.max_seq_length for length in lengths
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if any(
        split["above_configured_max"]
        for split in summary["splits"].values()
    ):
        raise SystemExit("At least one sequence exceeds the configured maximum")


if __name__ == "__main__":
    main()
