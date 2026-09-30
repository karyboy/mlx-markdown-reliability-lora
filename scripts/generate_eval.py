#!/usr/bin/env python3
"""Generate raw Markdown responses for custom or LatentMD evaluation records."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt_for(record: dict) -> str:
    if "prompt_text" in record:
        return record["prompt_text"]
    for message in record["messages"]:
        if message["role"] == "user":
            return message["content"]
    raise ValueError(f"No user prompt in record {record.get('id', '<unknown>')}")


def metadata_for(record: dict) -> dict:
    """Preserve evaluator-relevant source fields in the response schema."""
    metadata = dict(record.get("metadata", {}))
    for key in ("category", "axis_conditions", "slots"):
        if key in record:
            metadata[key] = record[key]
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler
    import mlx.core as mx

    generation = config["generation"]
    mx.random.seed(generation["seed"])
    sampler = make_sampler(temp=generation["temperature"], top_p=generation["top_p"])
    model, tokenizer = load(
        config["model"]["id"],
        revision=config["model"]["revision"],
        adapter_path=str(args.adapter_path.resolve()) if args.adapter_path else None,
    )
    records = read_jsonl(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for record in tqdm(records, desc="Generating Markdown"):
            prompt_text = prompt_for(record)
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            started = time.perf_counter()
            response = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=generation["max_tokens"],
                sampler=sampler,
                verbose=False,
            )
            elapsed = time.perf_counter() - started
            output = {
                "prompt_id": record.get("prompt_id", record.get("id")),
                "model": config["model"]["id"],
                "prompt_text": prompt_text,
                "response": response,
                "finish_reason": "stop",
                "latency": elapsed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata_for(record),
                "generation": generation,
                "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
            }
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Saved {len(records)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
