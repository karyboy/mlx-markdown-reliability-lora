"""Generate the final task-disjoint Markdown SFT dataset."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .axis_renderer import (
    AXIS_CELLS,
    corrupt_axis_markdown,
    generation_prompt,
    make_axis_contract,
    render_axis_markdown,
    repair_prompt,
    transformation_prompt,
)
from .latentmd_structural import score_response


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_types(count: int, settings: dict[str, Any]) -> list[str]:
    generation = round(count * settings["generation_fraction"])
    repair = round(count * settings["repair_fraction"])
    transformation = count - generation - repair
    return (
        ["generation"] * generation
        + ["repair"] * repair
        + ["transformation"] * transformation
    )


def _flatten_plans(families: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (family, variant)
        for family in families
        for variant in family["language_variants"]
    ]


def _select_pairs(
    families: list[dict[str, Any]], count: int, seed: int
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Select unique plan/contract pairs with exact 12-cell stratification."""
    rng = random.Random(seed)
    plans = _flatten_plans(families)
    base, remainder = divmod(count, len(AXIS_CELLS))
    cell_order = list(AXIS_CELLS)
    rng.shuffle(cell_order)
    quotas = {
        cell: base + int(index < remainder) for index, cell in enumerate(cell_order)
    }
    selected = []
    direct_render_keys: set[tuple[str, str, str]] = set()
    for cell_index, cell in enumerate(AXIS_CELLS):
        axis_a, axis_b = cell.split("_", maxsplit=1)
        candidates = list(plans)
        random.Random(seed + 1009 * (cell_index + 1)).shuffle(candidates)
        if axis_a in {"A1", "A3"}:
            candidates = [
                plan
                for plan in candidates
                if (plan[0]["task_family_id"], plan[1]["source_config"], axis_b)
                not in direct_render_keys
            ]
        if quotas[cell] > len(candidates):
            raise ValueError(
                f"Cell {cell} needs {quotas[cell]} examples but only {len(candidates)} "
                "unique content plans are available"
            )
        chosen = candidates[: quotas[cell]]
        selected.extend((*plan, cell) for plan in chosen)
        if axis_a in {"A1", "A3"}:
            direct_render_keys.update(
                (family["task_family_id"], variant["source_config"], axis_b)
                for family, variant in chosen
            )
    rng.shuffle(selected)
    return selected


def _score_target(
    target: str, family: dict[str, Any], variant: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    return score_response(
        {
            "response": target,
            "metadata": {
                "axis_conditions": {
                    "A": contract["axis_a"],
                    "B": contract["axis_b"],
                },
                "slots": {
                    "LANG": variant["language"],
                    "TASK": family["description"],
                    "TASK_ID": family["task_family_id"],
                },
            },
        }
    )


def build_split(
    families: list[dict[str, Any]],
    count: int,
    split: str,
    seed: int,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pairs = _select_pairs(families, count, seed)
    task_types = _task_types(count, settings)
    rng.shuffle(task_types)
    records = []
    for index, ((family, variant, axis_cell), task_type) in enumerate(
        zip(pairs, task_types, strict=True)
    ):
        contract = make_axis_contract(axis_cell, variant)
        target = render_axis_markdown(family, variant, contract)
        verification = _score_target(target, family, variant, contract)
        if not verification["passed"]:
            raise ValueError(
                f"Invalid canonical target for {family['task_family_id']} "
                f"{variant['source_config']} {axis_cell}: {verification['failures']}"
            )

        corruption = None
        if task_type == "generation":
            prompt = generation_prompt(family, variant, contract)
        elif task_type == "transformation":
            prompt = transformation_prompt(family, variant, contract)
        else:
            broken, corruption = corrupt_axis_markdown(target, contract, index)
            broken_score = _score_target(broken, family, variant, contract)
            if broken_score["passed"]:
                raise ValueError(
                    f"Corruption {corruption} did not violate {axis_cell} for "
                    f"{family['task_family_id']}"
                )
            prompt = repair_prompt(broken, contract)

        target_hash = hashlib.sha256(target.encode()).hexdigest()
        records.append(
            {
                "id": (
                    f"{split}-{index:05d}-{family['task_family_id']}-"
                    f"{variant['source_config']}-{axis_cell}-{task_type}"
                ),
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ],
                "metadata": {
                    "split": split,
                    "task_family_id": family["task_family_id"],
                    "task_id": family["task_id"],
                    "source_config": variant["source_config"],
                    "source_name": variant["source_name"],
                    "language": variant["language"],
                    "task_type": task_type,
                    "corruption": corruption,
                    "axis_conditions": {
                        "A": contract["axis_a"],
                        "B": contract["axis_b"],
                    },
                    "slots": {
                        "LANG": variant["language"],
                        "TASK": family["description"],
                        "TASK_ID": family["task_family_id"],
                    },
                    "contract": contract,
                    "target_sha256": target_hash,
                    "target_verified": True,
                },
            }
        )
    return records


def _split_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(records),
        "task_families": len({record["metadata"]["task_family_id"] for record in records}),
        "unique_targets": len({record["metadata"]["target_sha256"] for record in records}),
        "task_types": dict(sorted(Counter(record["metadata"]["task_type"] for record in records).items())),
        "axis_cells": dict(
            sorted(
                Counter(
                    f"{record['metadata']['axis_conditions']['A']}_{record['metadata']['axis_conditions']['B']}"
                    for record in records
                ).items()
            )
        ),
        "languages": dict(sorted(Counter(record["metadata"]["language"] for record in records).items())),
        "corruptions": dict(
            sorted(
                Counter(
                    record["metadata"]["corruption"]
                    for record in records
                    if record["metadata"]["corruption"] is not None
                ).items()
            )
        ),
    }


def generate_final_dataset(
    content_plan_dir: Path,
    output_dir: Path,
    summary_path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    train_families = _read_jsonl(content_plan_dir / "train_families.jsonl")
    valid_families = _read_jsonl(content_plan_dir / "valid_families.jsonl")
    train_ids = {family["task_family_id"] for family in train_families}
    valid_ids = {family["task_family_id"] for family in valid_families}
    overlap = sorted(train_ids & valid_ids)
    if overlap:
        raise ValueError(f"Content-plan leakage detected: {overlap}")

    train = build_split(
        train_families,
        settings["train_examples"],
        "train",
        settings["seed"],
        settings,
    )
    valid = build_split(
        valid_families,
        settings["valid_examples"],
        "valid",
        settings["seed"] + 1,
        settings,
    )
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, valid)

    train_hashes = {record["metadata"]["target_sha256"] for record in train}
    valid_hashes = {record["metadata"]["target_sha256"] for record in valid}
    project_root = output_dir.parents[2]

    def portable(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(project_root.resolve()))
        except ValueError:
            return path.name

    manifest = {
        "dataset_version": "markdown_sft_v1",
        "seed": settings["seed"],
        "contract_grid": {"A": ["A1", "A2", "A3"], "B": ["B1", "B2", "B3", "B4"]},
        "train": _split_summary(train),
        "valid": _split_summary(valid),
        "leakage_audit": {
            "task_family_overlap": overlap,
            "target_hash_overlap": sorted(train_hashes & valid_hashes),
        },
        "all_targets_programmatically_verified": True,
        "files": {
            "train": {"path": portable(train_path), "sha256": _sha256(train_path)},
            "valid": {"path": portable(valid_path), "sha256": _sha256(valid_path)},
        },
        "source_content_plan_manifest": portable(content_plan_dir / "manifest.json"),
    }
    if manifest["leakage_audit"]["target_hash_overlap"]:
        raise ValueError("Exact target leakage detected")
    if manifest["train"]["unique_targets"] != len(train):
        raise ValueError("Duplicate targets detected in the training split")
    if manifest["valid"]["unique_targets"] != len(valid):
        raise ValueError("Duplicate targets detected in the validation split")
    output_manifest = output_dir / "manifest.json"
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
