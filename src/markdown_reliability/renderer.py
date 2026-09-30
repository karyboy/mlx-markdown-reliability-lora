"""Render canonical Markdown from structured content and an explicit contract."""

from __future__ import annotations

import json
from typing import Any


def make_contract(topic: dict[str, Any], variant: int) -> dict[str, Any]:
    """Create a deterministic formatting contract for a topic and variant."""
    audiences = [
        "a new contributor",
        "a small engineering team",
        "an application maintainer",
        "a technical reviewer",
        "a project lead",
        "an operations engineer",
        "an independent developer",
    ]
    contexts = [
        "a local development environment",
        "a production-readiness review",
        "a team onboarding session",
        "a repeatable testing workflow",
        "a controlled rollout",
        "a troubleshooting exercise",
        "a documentation handoff",
        "a security review",
        "a performance investigation",
        "a release preparation cycle",
        "a maintenance window",
    ]
    emphases = [
        "reproducibility",
        "safe defaults",
        "clear verification",
        "incremental changes",
        "observable results",
        "reversible decisions",
        "concise documentation",
        "failure recovery",
        "least privilege",
        "measurable outcomes",
        "explicit ownership",
        "consistent naming",
        "small feedback loops",
    ]
    require_table = variant % 2 == 0
    require_blockquote = variant % 3 == 0
    require_code = variant % 5 != 0
    h2_texts = ["Overview", "Procedure"]
    if require_table:
        h2_texts.append("Reference")
    return {
        "title": topic["title"],
        "h1_count": 1,
        "h2_texts": h2_texts,
        "list_style": "ordered" if variant % 2 else "bullet",
        "list_depth": 1 + (variant % 3),
        "require_code": require_code,
        "code_language": topic["language"] if require_code else None,
        "require_table": require_table,
        "require_blockquote": require_blockquote,
        "audience": audiences[variant % len(audiences)],
        "context": contexts[(variant // len(audiences)) % len(contexts)],
        "emphasis": emphases[
            (variant // (len(audiences) * len(contexts))) % len(emphases)
        ],
    }


def _nested_list(topic: dict[str, Any], style: str, depth: int) -> list[str]:
    marker = "1." if style == "ordered" else "-"
    lines = [f"{marker} {topic['actions'][0]}"]
    if depth >= 2:
        lines.append(f"    {marker} {topic['details'][0]}")
    if depth >= 3:
        lines.append(f"        {marker} Verify the result before continuing")
    lines.extend(f"{marker} {item}" for item in topic["actions"][1:])
    return lines


def render_markdown(topic: dict[str, Any], contract: dict[str, Any]) -> str:
    """Return one canonical answer satisfying the supplied contract."""
    lines = [
        f"# {topic['title']}",
        "",
        topic["intro"],
        "",
        "## Overview",
        "",
        f"This guide presents a practical workflow for {topic['title'].lower()}.",
        (
            f"It is intended for {contract['audience']} working in {contract['context']} "
            f"and emphasizes {contract['emphasis']}."
        ),
        "",
    ]
    if contract["require_blockquote"]:
        lines.extend(["> Verify each change in a safe environment before wider use.", ""])
    lines.extend(["## Procedure", ""])
    lines.extend(_nested_list(topic, contract["list_style"], contract["list_depth"]))
    lines.append("")
    if contract["require_code"]:
        lines.extend(
            [
                f"```{contract['code_language']}",
                topic["code"],
                "```",
                "",
            ]
        )
    if contract["require_table"]:
        lines.extend(
            [
                "## Reference",
                "",
                "| Check | Purpose |",
                "| --- | --- |",
                "| Validate | Confirm the expected behavior |",
                "| Record | Preserve a reproducible result |",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def describe_contract(contract: dict[str, Any]) -> str:
    requirements = [
        f'an H1 titled "{contract["title"]}"',
        f'{len(contract["h2_texts"])} H2 sections named '
        + ", ".join(f'"{x}"' for x in contract["h2_texts"]),
        f'a {contract["list_style"]} list nested to exactly {contract["list_depth"]} level(s)',
    ]
    if contract["require_code"]:
        requirements.append(
            f'one fenced code block with the language tag "{contract["code_language"]}"'
        )
    else:
        requirements.append("no fenced code blocks")
    requirements.append("one Markdown table" if contract["require_table"] else "no Markdown tables")
    requirements.append(
        "one blockquote" if contract["require_blockquote"] else "no blockquotes"
    )
    requirements.append(
        f'content written for {contract["audience"]} in {contract["context"]}, '
        f'emphasizing {contract["emphasis"]}'
    )
    return "; ".join(requirements) + "."


def generation_prompt(contract: dict[str, Any]) -> str:
    return (
        "Write a concise Markdown guide using the following formatting contract. "
        "Return only the Markdown document. Contract: "
        + describe_contract(contract)
    )


def transformation_prompt(topic: dict[str, Any], contract: dict[str, Any]) -> str:
    source = {
        "title": topic["title"],
        "summary": topic["intro"],
        "steps": topic["actions"],
        "details": topic["details"],
        "code_language": topic["language"],
        "code": topic["code"],
    }
    return (
        "Transform the following JSON outline into Markdown. Return only the Markdown document.\n\n"
        f"Formatting contract: {describe_contract(contract)}\n\n"
        f"Outline:\n{json.dumps(source, indent=2, ensure_ascii=False)}"
    )


def corrupt_markdown(markdown: str, contract: dict[str, Any], variant: int) -> tuple[str, str]:
    """Introduce one deterministic structural defect for a repair example."""
    lines = markdown.rstrip("\n").splitlines()
    choices: list[str] = ["heading"]
    if contract["require_code"]:
        choices.append("fence")
    if contract["list_depth"] > 1:
        choices.append("indent")
    if contract["require_table"]:
        choices.append("table")
    corruption = choices[variant % len(choices)]

    if corruption == "heading":
        idx = lines.index("## Procedure")
        lines[idx] = "#### Procedure"
    elif corruption == "fence":
        closing = max(i for i, line in enumerate(lines) if line == "```")
        del lines[closing]
    elif corruption == "indent":
        idx = next(i for i, line in enumerate(lines) if line.startswith("    "))
        lines[idx] = lines[idx].lstrip()
    else:
        idx = lines.index("| --- | --- |")
        lines[idx] = "This row is missing the required table delimiters."
    return "\n".join(lines) + "\n", corruption


def repair_prompt(broken: str, contract: dict[str, Any]) -> str:
    return (
        "Repair the malformed Markdown so that it satisfies the formatting contract. "
        "Preserve its meaning and return only the corrected Markdown.\n\n"
        f"Contract: {describe_contract(contract)}\n\n"
        f"Malformed Markdown:\n\n{broken}"
    )
