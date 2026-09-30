"""Generate a short, contrastive SFT booster for outer Markdown wrappers."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .axis_renderer import make_axis_contract
from .latentmd_structural import score_response


B_AXES = ("B1", "B2", "B3", "B4")


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


def _plans(families: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (family, variant)
        for family in families
        for variant in family["language_variants"]
    ]


def _description(family: dict[str, Any]) -> str:
    text = " ".join(family["description"].split())
    return text if len(text) <= 240 else text[:237].rstrip() + "..."


def _compact_starter(variant: dict[str, Any]) -> str:
    lines = [line.rstrip() for line in variant["starter_code"].splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if not line.lstrip().startswith(("// Task:", "# Task:"))
    ]
    selected = lines[-4:] if lines else ["implementation goes here"]
    return "\n".join(selected)


def _compact_test(variant: dict[str, Any]) -> str:
    lines = [line.strip() for line in variant["test_harness"].splitlines() if line.strip()]
    preferred = [
        line
        for line in lines
        if any(marker in line for marker in ("assert", "assert_eq!", "[[", "candidate("))
    ]
    selected = preferred[:2] or lines[:3] or ["verify the result"]
    return "\n".join(selected)


def _code(tag: str, value: str) -> list[str]:
    return [f"```{tag}", value.rstrip(), "```"]


def _raw_code(tag: str, value: str) -> list[str]:
    return ["~~~~markdown", f"```{tag}", value.rstrip(), "```", "~~~~"]


def render_short_document(
    family: dict[str, Any], variant: dict[str, Any], axis_a: str, axis_b: str
) -> str:
    """Render a compact target whose wrapper tokens have meaningful loss weight."""
    tag = variant["markdown_language_tag"]
    starter = _compact_starter(variant)
    test = _compact_test(variant)
    lines = [f"# {family['title']}", "", _description(family), ""]

    if axis_b == "B4":
        lines.extend(
            [
                "1. Review the interface.",
                "2. Implement the behavior.",
                "3. Run the check.",
                "",
                "> Keep examples reproducible. — [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E)",
                "",
            ]
        )

    lines.extend(_code(tag, starter))
    lines.append("")

    if axis_b in {"B3", "B4"}:
        lines.extend(_code(tag, test))
        lines.append("")

    if axis_b in {"B2", "B3"}:
        lines.extend(_raw_code(tag, starter))
        lines.append("")
    if axis_b == "B3":
        lines.extend(_raw_code(tag, test))
        lines.append("")

    if axis_b == "B4":
        lines.extend(
            [
                "| Item | Purpose |",
                "| --- | --- |",
                "| Interface | Defines the entry point |",
                "| Check | Verifies behavior |",
                "",
            ]
        )

    inner = "\n".join(lines).rstrip() + "\n"
    return f"````markdown\n{inner}````\n" if axis_a == "A2" else inner


def generation_prompt(family: dict[str, Any], variant: dict[str, Any], axis_a: str, axis_b: str) -> str:
    """Use the benchmark's exact instruction patterns for direct generation."""
    language = variant["language"]
    requirement = {
        "B1": f"Include exactly one {language} code example.",
        "B2": (
            f"Include one {language} code example. In the document, also show the raw "
            "Markdown source for that code example, including the opening fence, the "
            "language tag, the code, and the closing fence."
        ),
        "B3": (
            f"Include {language} code examples. In at least two places in the document, "
            "show the raw Markdown source for the code examples, including the opening "
            "fence, the language tag, the code, and the closing fence."
        ),
        "B4": (
            f"Include {language} code examples, a comparison table, a blockquote with a "
            "citation, and a numbered list of steps."
        ),
    }[axis_b]
    outer = {
        "A1": (
            "Your response IS the markdown file itself — start directly with the file "
            "content. Do NOT wrap your response in an outer code fence."
        ),
        "A2": "Wrap your entire response in a fenced markdown code block.",
        "A3": "",
    }[axis_a]
    instruction = " ".join(
        part for part in ("Write a Markdown document that addresses the following task.", requirement, outer) if part
    )
    return f"{instruction}\n\nTask:\n{_description(family)}"


def repair_prompt(broken: str) -> str:
    return (
        "Repair the Markdown response. Wrap the entire corrected response in one fenced "
        "markdown code block, preserving its inner content and code fences. Return only "
        f"the corrected response.\n\nMalformed response:\n{broken}"
    )


def transformation_prompt(family: dict[str, Any], variant: dict[str, Any], axis_b: str) -> str:
    payload = {
        "title": family["title"],
        "task": _description(family),
        "language": variant["language"],
        "structure": axis_b,
    }
    return (
        "Transform this JSON outline into Markdown. Wrap your entire response in a fenced "
        "markdown code block. Return only the wrapped document.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _score(
    response: str,
    family: dict[str, Any],
    variant: dict[str, Any],
    axis_a: str,
    axis_b: str,
) -> dict[str, Any]:
    return score_response(
        {
            "response": response,
            "metadata": {
                "axis_conditions": {"A": axis_a, "B": axis_b},
                "slots": {
                    "LANG": variant["language"],
                    "TASK": _description(family),
                    "TASK_ID": family["task_family_id"],
                },
            },
        }
    )


def _record(
    split: str,
    index: int,
    family: dict[str, Any],
    variant: dict[str, Any],
    axis_a: str,
    axis_b: str,
    task_type: str,
    pair_id: str | None,
) -> dict[str, Any]:
    target = render_short_document(family, variant, axis_a, axis_b)
    result = _score(target, family, variant, axis_a, axis_b)
    if not result["passed"]:
        raise ValueError(
            f"Invalid booster target {family['task_family_id']} {axis_a}_{axis_b}: "
            f"{result['failures']}"
        )
    if task_type == "generation":
        prompt = generation_prompt(family, variant, axis_a, axis_b)
    elif task_type == "repair":
        broken = target.removeprefix("````markdown\n").removesuffix("````\n")
        if _score(broken, family, variant, axis_a, axis_b)["passed"]:
            raise ValueError("A2 repair corruption unexpectedly passed")
        prompt = repair_prompt(broken)
    else:
        prompt = transformation_prompt(family, variant, axis_b)

    contract = make_axis_contract(f"{axis_a}_{axis_b}", variant)
    target_hash = hashlib.sha256(target.encode()).hexdigest()
    return {
        "id": (
            f"wrapper-{split}-{index:04d}-{family['task_family_id']}-"
            f"{variant['source_config']}-{axis_a}_{axis_b}-{task_type}"
        ),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "metadata": {
            "split": split,
            "task_family_id": family["task_family_id"],
            "source_config": variant["source_config"],
            "language": variant["language"],
            "task_type": task_type,
            "contrast_pair_id": pair_id,
            "axis_conditions": {"A": axis_a, "B": axis_b},
            "slots": {
                "LANG": variant["language"],
                "TASK": _description(family),
                "TASK_ID": family["task_family_id"],
            },
            "contract": contract,
            "target_sha256": target_hash,
            "target_verified": True,
        },
    }


def _build_train(families: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    plans = _plans(families)
    specs: list[tuple[dict[str, Any], dict[str, Any], str, str, str, str | None]] = []
    for b_index, axis_b in enumerate(B_AXES):
        candidates = list(plans)
        random.Random(seed + 1009 * (b_index + 1)).shuffle(candidates)
        paired, a3, extra_generation, repairs, transforms = (
            candidates[:30],
            candidates[30:60],
            candidates[60:90],
            candidates[90:105],
            candidates[105:120],
        )
        for pair_index, (family, variant) in enumerate(paired):
            pair_id = f"train-{axis_b}-pair-{pair_index:02d}"
            specs.append((family, variant, "A1", axis_b, "generation", pair_id))
            specs.append((family, variant, "A2", axis_b, "generation", pair_id))
        specs.extend((*plan, "A3", axis_b, "generation", None) for plan in a3)
        specs.extend((*plan, "A2", axis_b, "generation", None) for plan in extra_generation)
        specs.extend((*plan, "A2", axis_b, "repair", None) for plan in repairs)
        specs.extend((*plan, "A2", axis_b, "transformation", None) for plan in transforms)
    random.Random(seed).shuffle(specs)
    return [_record("train", index, *spec) for index, spec in enumerate(specs)]


def _build_valid(families: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    plans = _plans(families)
    specs: list[tuple[dict[str, Any], dict[str, Any], str, str, str, str | None]] = []
    for b_index, axis_b in enumerate(B_AXES):
        candidates = list(plans)
        random.Random(seed + 2027 * (b_index + 1)).shuffle(candidates)
        paired, a3, extra_a2 = candidates[:6], candidates[6:12], candidates[12:24]
        for pair_index, (family, variant) in enumerate(paired):
            pair_id = f"valid-{axis_b}-pair-{pair_index:02d}"
            specs.append((family, variant, "A1", axis_b, "generation", pair_id))
            specs.append((family, variant, "A2", axis_b, "generation", pair_id))
        specs.extend((*plan, "A3", axis_b, "generation", None) for plan in a3)
        specs.extend((*plan, "A2", axis_b, "generation", None) for plan in extra_a2)
    random.Random(seed).shuffle(specs)
    return [_record("valid", index, *spec) for index, spec in enumerate(specs)]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "examples": len(records),
        "task_families": len({r["metadata"]["task_family_id"] for r in records}),
        "unique_targets": len({r["metadata"]["target_sha256"] for r in records}),
        "axis_a": dict(sorted(Counter(r["metadata"]["axis_conditions"]["A"] for r in records).items())),
        "axis_b": dict(sorted(Counter(r["metadata"]["axis_conditions"]["B"] for r in records).items())),
        "task_types": dict(sorted(Counter(r["metadata"]["task_type"] for r in records).items())),
        "languages": dict(sorted(Counter(r["metadata"]["language"] for r in records).items())),
        "contrast_pairs": len({r["metadata"]["contrast_pair_id"] for r in records if r["metadata"]["contrast_pair_id"]}),
        "maximum_target_characters": max(len(r["messages"][1]["content"]) for r in records),
    }


def generate_wrapper_booster(
    content_plan_dir: Path, output_dir: Path, summary_path: Path, seed: int
) -> dict[str, Any]:
    train_families = _read_jsonl(content_plan_dir / "train_families.jsonl")
    valid_families = _read_jsonl(content_plan_dir / "valid_families.jsonl")
    train = _build_train(train_families, seed)
    valid = _build_valid(valid_families, seed + 1)
    train_path, valid_path = output_dir / "train.jsonl", output_dir / "valid.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(valid_path, valid)

    train_fids = {r["metadata"]["task_family_id"] for r in train}
    valid_fids = {r["metadata"]["task_family_id"] for r in valid}
    train_hashes = {r["metadata"]["target_sha256"] for r in train}
    valid_hashes = {r["metadata"]["target_sha256"] for r in valid}
    project_root = output_dir.parents[2]
    manifest = {
        "dataset_version": "wrapper_booster_v1",
        "seed": seed,
        "design": (
            "Short exact-wording generation examples with matched A1/A2 contrast pairs; "
            "A2 is oversampled 3:1 relative to each control axis."
        ),
        "train": _summary(train),
        "valid": _summary(valid),
        "leakage_audit": {
            "task_family_overlap": sorted(train_fids & valid_fids),
            "target_hash_overlap": sorted(train_hashes & valid_hashes),
        },
        "files": {
            "train": {"path": str(train_path.relative_to(project_root)), "sha256": _sha256(train_path)},
            "valid": {"path": str(valid_path.relative_to(project_root)), "sha256": _sha256(valid_path)},
        },
        "all_targets_programmatically_verified": True,
    }
    if manifest["leakage_audit"]["task_family_overlap"]:
        raise ValueError("Task-family leakage in wrapper booster")
    if manifest["leakage_audit"]["target_hash_overlap"]:
        raise ValueError("Exact target leakage in wrapper booster")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
