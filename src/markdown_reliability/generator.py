"""Generate reproducible SFT records with programmatically verified targets."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .catalog import TOPICS
from .renderer import (
    corrupt_markdown,
    generation_prompt,
    make_contract,
    repair_prompt,
    render_markdown,
    transformation_prompt,
)
from .verifier import verify_markdown


def _task_types(count: int) -> list[str]:
    generation = round(count * 0.60)
    repair = round(count * 0.30)
    transform = count - generation - repair
    return ["generation"] * generation + ["repair"] * repair + ["transformation"] * transform


def build_records(
    count: int,
    split: str,
    topics: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    tasks = _task_types(count)
    rng.shuffle(tasks)
    records = []
    for index, task_type in enumerate(tasks):
        topic = topics[index % len(topics)]
        variant = seed * 10_000 + index
        contract = make_contract(topic, variant)
        target = render_markdown(topic, contract)
        verification = verify_markdown(target, contract)
        if not verification["passed"]:
            raise ValueError(
                f"Renderer produced an invalid target for {topic['id']}: "
                f"{verification['failures']}"
            )

        corruption = None
        if task_type == "generation":
            prompt = generation_prompt(contract)
        elif task_type == "transformation":
            prompt = transformation_prompt(topic, contract)
        else:
            broken, corruption = corrupt_markdown(target, contract, variant)
            prompt = repair_prompt(broken, contract)

        record_id = f"{split}-{index:04d}-{topic['id']}-{task_type}"
        records.append(
            {
                "id": record_id,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "split": split,
                    "topic_id": topic["id"],
                    "task_type": task_type,
                    "corruption": corruption,
                    "contract": contract,
                    "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
                    "target_verified": True,
                },
            }
        )
    rng.shuffle(records)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def generate_pilot(
    output_dir: Path,
    train_count: int = 450,
    valid_count: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate topic-disjoint train and validation JSONL files plus a manifest."""
    topic_ids = [topic["id"] for topic in TOPICS]
    rng = random.Random(seed)
    rng.shuffle(topic_ids)
    valid_topic_ids = set(topic_ids[:4])
    train_topics = [topic for topic in TOPICS if topic["id"] not in valid_topic_ids]
    valid_topics = [topic for topic in TOPICS if topic["id"] in valid_topic_ids]

    train = build_records(train_count, "train", train_topics, seed)
    valid = build_records(valid_count, "valid", valid_topics, seed + 1)
    _write_jsonl(output_dir / "train.jsonl", train)
    _write_jsonl(output_dir / "valid.jsonl", valid)

    train_topic_ids = {x["metadata"]["topic_id"] for x in train}
    valid_topic_ids_observed = {x["metadata"]["topic_id"] for x in valid}
    manifest = {
        "seed": seed,
        "counts": {
            "train": len(train),
            "valid": len(valid),
            "total": len(train) + len(valid),
        },
        "task_types": {
            "train": dict(Counter(x["metadata"]["task_type"] for x in train)),
            "valid": dict(Counter(x["metadata"]["task_type"] for x in valid)),
        },
        "train_topic_ids": sorted(train_topic_ids),
        "valid_topic_ids": sorted(valid_topic_ids_observed),
        "topic_overlap": sorted(train_topic_ids & valid_topic_ids_observed),
        "all_targets_verified": all(
            x["metadata"]["target_verified"] for x in train + valid
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
