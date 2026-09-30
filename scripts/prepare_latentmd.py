#!/usr/bin/env python3
"""Create a fixed 240-prompt A×B-stratified LatentMD evaluation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "latentmd" / "eval_240.jsonl",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config["latentmd"]
    revision = settings["revision"]
    source_path = hf_hub_download(
        repo_id=settings["hf_id"],
        filename=settings["data_file"],
        repo_type="dataset",
        revision=revision,
    )
    raw = yaml.safe_load(Path(source_path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        dataset = raw
    elif isinstance(raw, dict) and isinstance(raw.get("prompts"), list):
        dataset = raw["prompts"]
    else:
        raise SystemExit("Unexpected LatentMD prompt YAML structure")

    cells = defaultdict(list)
    for record in dataset:
        axes = record["axis_conditions"]
        cells[(axes["A"], axes["B"])].append(record)
    expected = settings["expected_axis_cells"]
    if len(cells) != expected:
        raise SystemExit(f"Expected {expected} A×B cells, found {len(cells)}")

    rng = random.Random(settings["sample_seed"])
    selected = []
    cell_counts = {}
    for cell, records in sorted(cells.items()):
        count = settings["samples_per_axis_cell"]
        if len(records) < count:
            raise SystemExit(f"Cell {cell} has only {len(records)} prompts")
        records.sort(key=lambda item: item["prompt_id"])
        chosen = rng.sample(records, count)
        selected.extend(chosen)
        cell_counts[f"{cell[0]}_{cell[1]}"] = len(chosen)
    selected.sort(key=lambda item: item["prompt_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    ids = "\n".join(record["prompt_id"] for record in selected).encode()
    manifest = {
        "dataset": settings["hf_id"],
        "dataset_revision": revision,
        "source_file": settings["data_file"],
        "selection_seed": settings["sample_seed"],
        "selection": "20 prompts sampled without replacement from each A×B main-grid cell",
        "count": len(selected),
        "cell_counts": cell_counts,
        "prompt_ids_sha256": hashlib.sha256(ids).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Evaluation prompts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
