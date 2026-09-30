"""Render verified SFT targets for the 3 x 4 Markdown contract grid."""

from __future__ import annotations

import json
import re
from typing import Any


AXIS_CELLS = [f"{axis_a}_{axis_b}" for axis_a in ("A1", "A2", "A3") for axis_b in ("B1", "B2", "B3", "B4")]


def make_axis_contract(axis_cell: str, variant: dict[str, Any]) -> dict[str, Any]:
    axis_a, axis_b = axis_cell.split("_", maxsplit=1)
    outer_policy = {"A1": "prohibited", "A2": "required", "A3": "unspecified"}[axis_a]
    rules = {
        "B1": {"code_mode": "exactly_one", "minimum_raw_source_blocks": 0},
        "B2": {"code_mode": "at_least_one", "minimum_raw_source_blocks": 1},
        "B3": {"code_mode": "at_least_two", "minimum_raw_source_blocks": 2},
        "B4": {"code_mode": "at_least_two", "minimum_raw_source_blocks": 0},
    }[axis_b]
    return {
        "contract_type": "latentmd_structural_v1",
        "axis_a": axis_a,
        "axis_b": axis_b,
        "axis_cell": axis_cell,
        "outer_policy": outer_policy,
        "language": variant["language"],
        "markdown_language_tag": variant["markdown_language_tag"],
        **rules,
        "require_table": axis_b == "B4",
        "require_blockquote_citation": axis_b == "B4",
        "require_numbered_list": axis_b == "B4",
    }


def describe_contract(contract: dict[str, Any]) -> str:
    language = contract["language"]
    axis_a = contract["axis_a"]
    axis_b = contract["axis_b"]
    outer = {
        "A1": (
            "Your response is the Markdown document itself: begin directly with its "
            "content and do not wrap the whole response in an outer code fence."
        ),
        "A2": (
            "Wrap the entire Markdown document in one outer four-backtick code fence "
            "labeled markdown: begin with ````markdown and end with ````."
        ),
        "A3": "",
    }[axis_a]
    structure = {
        "B1": f"Include exactly one fenced {language} code example.",
        "B2": (
            f"Include one fenced {language} code example and one separate raw Markdown "
            "source block that visibly shows the fence markup for that example."
        ),
        "B3": (
            f"Include at least two fenced {language} code examples and at least two "
            "separate raw Markdown source blocks that visibly show their fence markup."
        ),
        "B4": (
            f"Include at least two fenced {language} code examples, a comparison table, "
            "a cited blockquote, and a numbered list."
        ),
    }[axis_b]
    return " ".join(part for part in (structure, outer) if part)


def _code_fence(tag: str, code: str) -> list[str]:
    return [f"```{tag}", code.rstrip(), "```"]


def _raw_source(tag: str, code: str) -> list[str]:
    return ["~~~~markdown", f"```{tag}", code.rstrip(), "```", "~~~~"]


def _inner_document(
    family: dict[str, Any], variant: dict[str, Any], contract: dict[str, Any]
) -> str:
    tag = contract["markdown_language_tag"]
    axis_b = contract["axis_b"]
    lines = [
        f"# {family['title']}: {variant['language']} Implementation Brief",
        "",
        family["description"],
        "",
        "## Implementation outline",
        "",
    ]
    if axis_b == "B4":
        lines.extend(
            [
                "1. Review the supplied function signature and input types.",
                "2. Implement the behavior described by the task.",
                "3. Run the supplied test harness and inspect every assertion.",
                "",
                (
                    "> Reuse the interface and tests as reproducible task scaffolding. "
                    "— [MultiPL-E dataset](https://huggingface.co/datasets/nuprl/MultiPL-E)"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Use the supplied interface as the implementation starting point, then "
                "check the result against the provided assertions.",
                "",
            ]
        )

    lines.extend(["## Starter interface", ""])
    lines.extend(_code_fence(tag, variant["starter_code"]))
    lines.append("")

    if axis_b in {"B3", "B4"}:
        lines.extend(["## Test harness", ""])
        lines.extend(_code_fence(tag, variant["test_harness"]))
        lines.append("")

    if axis_b in {"B2", "B3"}:
        lines.extend(["## Raw Markdown source", ""])
        lines.extend(_raw_source(tag, variant["starter_code"]))
        lines.append("")
    if axis_b == "B3":
        lines.extend(["## Additional raw Markdown source", ""])
        lines.extend(_raw_source(tag, variant["test_harness"]))
        lines.append("")

    if axis_b == "B4":
        lines.extend(
            [
                "## Artifact comparison",
                "",
                "| Artifact | Purpose |",
                "| --- | --- |",
                "| Starter interface | Defines the implementation entry point |",
                "| Test harness | Checks the required behavior |",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_axis_markdown(
    family: dict[str, Any], variant: dict[str, Any], contract: dict[str, Any]
) -> str:
    inner = _inner_document(family, variant, contract)
    if contract["outer_policy"] == "required":
        return f"````markdown\n{inner}````\n"
    return inner


def _source_payload(family: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": family["title"],
        "task": family["description"],
        "language": variant["language"],
        "starter_code": variant["starter_code"].rstrip(),
        "test_harness": variant["test_harness"].rstrip(),
    }


def generation_prompt(
    family: dict[str, Any], variant: dict[str, Any], contract: dict[str, Any]
) -> str:
    payload = _source_payload(family, variant)
    return (
        "Create a concise Markdown implementation brief from the supplied task material. "
        "Do not solve or complete the starter code; document the supplied interface and "
        "tests. Return only the requested Markdown.\n\n"
        f"Formatting contract: {describe_contract(contract)}\n\n"
        f"Title: {payload['title']}\n"
        f"Task: {payload['task']}\n"
        f"Language: {payload['language']}\n\n"
        f"Starter code:\n{payload['starter_code']}\n\n"
        f"Test harness:\n{payload['test_harness']}"
    )


def transformation_prompt(
    family: dict[str, Any], variant: dict[str, Any], contract: dict[str, Any]
) -> str:
    return (
        "Transform the following JSON task material into a concise Markdown implementation "
        "brief. Do not solve or complete the starter code. Return only the requested "
        "Markdown.\n\n"
        f"Formatting contract: {describe_contract(contract)}\n\n"
        "Task material:\n"
        + json.dumps(_source_payload(family, variant), indent=2, ensure_ascii=False)
    )


def _replace_first_direct_fence(markdown: str, tag: str) -> str:
    return markdown.replace(f"```{tag}", "```text", 1)


def _remove_first_raw_language(markdown: str, tag: str) -> str:
    pattern = re.compile(
        rf"(~~~~markdown\n)```{re.escape(tag)}", flags=re.MULTILINE
    )
    return pattern.sub(r"\1```text", markdown, count=1)


def corrupt_axis_markdown(
    markdown: str, contract: dict[str, Any], variant_number: int
) -> tuple[str, str]:
    """Introduce one deterministic, contract-relevant defect."""
    choices = ["code_language"]
    if contract["minimum_raw_source_blocks"]:
        choices.append("raw_source")
    if contract["require_table"]:
        choices.extend(["table", "citation", "numbered_list"])
    if contract["outer_policy"] == "required":
        choices.append("outer_wrapper")
    corruption = choices[variant_number % len(choices)]
    tag = contract["markdown_language_tag"]

    if corruption == "code_language":
        broken = _replace_first_direct_fence(markdown, tag)
    elif corruption == "raw_source":
        broken = _remove_first_raw_language(markdown, tag)
    elif corruption == "table":
        broken = markdown.replace("| --- | --- |", "Artifact versus purpose", 1)
    elif corruption == "citation":
        broken = markdown.replace(
            "— [MultiPL-E dataset](https://huggingface.co/datasets/nuprl/MultiPL-E)",
            "— MultiPL-E dataset",
            1,
        )
    elif corruption == "numbered_list":
        broken = re.sub(r"(?m)^[123]\. ", "- ", markdown, count=3)
    else:
        broken = markdown.replace("````markdown\n", "", 1).rsplit("````\n", 1)[0] + "\n"
    if broken == markdown:
        raise ValueError(f"Corruption {corruption} made no change")
    return broken, corruption


def repair_prompt(broken: str, contract: dict[str, Any]) -> str:
    return (
        "Repair the malformed Markdown so it satisfies the formatting contract. Preserve "
        "the task content and code verbatim. Return only the corrected Markdown.\n\n"
        f"Formatting contract: {describe_contract(contract)}\n\n"
        f"Malformed Markdown:\n\n{broken}"
    )
