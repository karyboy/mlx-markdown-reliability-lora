#!/usr/bin/env python3
"""Build fixed, task-family-disjoint content plans from MBPP and MultiPL-E."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_ID = re.compile(r"^mbpp_(\d+)_")

LANGUAGES = {
    "mbpp-cpp": {"name": "C++", "tag": "cpp", "comment": "//"},
    "mbpp-java": {"name": "Java", "tag": "java", "comment": "//"},
    "mbpp-js": {"name": "JavaScript", "tag": "javascript", "comment": "//"},
    "mbpp-go": {"name": "Go", "tag": "go", "comment": "//"},
    "mbpp-rs": {"name": "Rust", "tag": "rust", "comment": "//"},
    "mbpp-rb": {"name": "Ruby", "tag": "ruby", "comment": "#"},
    "mbpp-sh": {"name": "Shell", "tag": "bash", "comment": "#"},
    "mbpp-ts": {"name": "TypeScript", "tag": "typescript", "comment": "//"},
}

MALFORMED_LANGUAGE_WORD = re.compile(r"\b[a-z]+thon\b", re.IGNORECASE)
PYTHON_QUALIFIER = re.compile(r"\bpython\s+(function|program|method)\b", re.IGNORECASE)


def parse_task_id(name: str) -> int:
    match = TASK_ID.match(name)
    if not match:
        raise ValueError(f"Unexpected MultiPL-E task name: {name}")
    return int(match.group(1))


def normalise_text(value: str) -> str:
    return " ".join(value.lower().split())


def clean_description(value: str) -> str:
    """Remove Python-only wording from a task shared across many languages."""
    cleaned = PYTHON_QUALIFIER.sub(lambda match: match.group(1).lower(), value.strip())
    return " ".join(cleaned.split())


def clean_starter_code(config_name: str, value: str, description: str) -> str:
    """Replace known MultiPL-E translation-comment artifacts reproducibly."""
    comment = LANGUAGES[config_name]["comment"]
    replacement = f"{comment} Task: {description}"
    lines = []
    for line in value.strip().splitlines():
        if MALFORMED_LANGUAGE_WORD.search(line):
            indent = line[: len(line) - len(line.lstrip())]
            lines.append(indent + replacement)
        else:
            lines.append(line.rstrip())
    return "\n".join(lines) + "\n"


def load_original_mbpp(settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    dataset = load_dataset(
        settings["hf_id"],
        settings["config"],
        revision=settings["revision"],
    )
    records: dict[int, dict[str, Any]] = {}
    for split_name, split in dataset.items():
        for row in split:
            task_id = int(row["task_id"])
            records[task_id] = {**dict(row), "source_split": split_name}
    return records


def load_multipl_e(settings: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    by_config = {}
    for config_name in settings["configs"]:
        dataset = load_dataset(
            settings["hf_id"],
            config_name,
            revision=settings["revision"],
            split="test",
        )
        by_config[config_name] = {
            parse_task_id(row["name"]): dict(row) for row in dataset
        }
    return by_config


def make_title(name: str) -> str:
    suffix = TASK_ID.sub("", name)
    return suffix.replace("_", " ").strip().title()


def make_family(
    task_id: int,
    split: str,
    original: dict[str, Any],
    translated: dict[str, dict[int, dict[str, Any]]],
    source_settings: dict[str, Any],
) -> dict[str, Any]:
    source_description = original["text"].strip()
    description = clean_description(source_description)
    variants = []
    title = None
    for config_name in source_settings["configs"]:
        row = translated[config_name][task_id]
        title = title or make_title(row["name"])
        variants.append(
            {
                "source_config": config_name,
                "source_name": row["name"],
                "language": LANGUAGES[config_name]["name"],
                "markdown_language_tag": LANGUAGES[config_name]["tag"],
                "starter_code": clean_starter_code(
                    config_name, row["prompt"], description
                ),
                "test_harness": row["tests"].strip() + "\n",
                "stop_tokens": row["stop_tokens"],
            }
        )
    return {
        "task_family_id": f"mbpp_{task_id}",
        "task_id": task_id,
        "split": split,
        "title": title,
        "description": description,
        "source_description": source_description,
        "python_reference_code": original["code"].strip() + "\n",
        "reference_tests": list(original["test_list"]),
        "mbpp_source_split": original["source_split"],
        "language_variants": variants,
        "source": {
            "mbpp_repo": "google-research-datasets/mbpp",
            "mbpp_revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
            "multipl_e_repo": source_settings["hf_id"],
            "multipl_e_revision": source_settings["revision"],
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latentmd_descriptions(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    descriptions = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            task = record.get("slots", {}).get("TASK")
            if task:
                descriptions.add(normalise_text(task))
    return descriptions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment.yaml"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "content_plans",
    )
    parser.add_argument(
        "--latentmd",
        type=Path,
        default=PROJECT_ROOT / "data" / "latentmd" / "eval_240.jsonl",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["content_plans"]
    multipl_e_settings = config["multipl_e"]
    original = load_original_mbpp(config["mbpp"])
    translated = load_multipl_e(multipl_e_settings)

    common_ids = set(original)
    for records in translated.values():
        common_ids &= set(records)
    required = config["train_task_families"] + config["valid_task_families"]
    if len(common_ids) < required:
        raise SystemExit(f"Only {len(common_ids)} common task families; need {required}")

    selected_ids = sorted(common_ids)
    random.Random(config["seed"]).shuffle(selected_ids)
    selected_ids = selected_ids[:required]
    train_ids = selected_ids[: config["train_task_families"]]
    valid_ids = selected_ids[config["train_task_families"] :]
    train = [
        make_family(task_id, "train", original[task_id], translated, multipl_e_settings)
        for task_id in train_ids
    ]
    valid = [
        make_family(task_id, "valid", original[task_id], translated, multipl_e_settings)
        for task_id in valid_ids
    ]

    train_path = args.output_dir / "train_families.jsonl"
    valid_path = args.output_dir / "valid_families.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    overlap = set(train_ids) & set(valid_ids)
    external_descriptions = latentmd_descriptions(args.latentmd)
    description_overlap = sorted(
        family["task_family_id"]
        for family in train + valid
        if normalise_text(family["description"]) in external_descriptions
    )
    malformed_starter_artifacts = sorted(
        {
            match.group(0)
            for family in train + valid
            for variant in family["language_variants"]
            for match in MALFORMED_LANGUAGE_WORD.finditer(variant["starter_code"])
        }
    )
    manifest = {
        "seed": config["seed"],
        "counts": {
            "train_task_families": len(train),
            "valid_task_families": len(valid),
            "total_task_families": len(train) + len(valid),
            "language_variants_per_family": len(multipl_e_settings["configs"]),
            "total_language_specific_plans": sum(
                len(family["language_variants"]) for family in train + valid
            ),
            "common_source_task_families": len(common_ids),
        },
        "languages": [LANGUAGES[name] for name in multipl_e_settings["configs"]],
        "task_family_overlap": sorted(overlap),
        "latentmd_exact_description_overlap": description_overlap,
        "malformed_starter_artifacts": malformed_starter_artifacts,
        "sources": {
            "multipl_e": multipl_e_settings,
            "mbpp": config["mbpp"],
        },
        "files": {
            "train": {"path": str(train_path), "sha256": file_sha256(train_path)},
            "valid": {"path": str(valid_path), "sha256": file_sha256(valid_path)},
        },
    }
    if overlap:
        raise SystemExit(f"Task-family leakage detected: {sorted(overlap)}")
    if malformed_starter_artifacts:
        raise SystemExit(
            f"Malformed language artifacts remain: {malformed_starter_artifacts}"
        )
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Content plans: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
