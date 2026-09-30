"""Deterministic contract verifier for generated Markdown."""

from __future__ import annotations

import re
from typing import Any

from markdown_it import MarkdownIt


_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def fences_balanced(markdown: str) -> bool:
    """Check strict CommonMark-style opening and closing fence balance."""
    opened: tuple[str, int] | None = None
    for line in markdown.splitlines():
        match = _FENCE_OPEN.match(line)
        if opened is None:
            if not match:
                continue
            run, rest = match.groups()
            if run[0] == "`" and "`" in rest:
                continue
            opened = (run[0], len(run))
            continue
        char, minimum = opened
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3:
            continue
        close = re.fullmatch(re.escape(char) + "{" + str(minimum) + r",}\s*", stripped)
        if close:
            opened = None
    return opened is None


def verify_markdown(markdown: str, contract: dict[str, Any]) -> dict[str, Any]:
    """Parse Markdown and verify every objective requirement in the contract."""
    parser = MarkdownIt("commonmark").enable("table")
    try:
        tokens = parser.parse(markdown)
        parse_success = True
        parse_error = None
    except Exception as exc:  # pragma: no cover - defensive for parser failures
        tokens = []
        parse_success = False
        parse_error = f"{type(exc).__name__}: {exc}"

    headings: list[tuple[int, str]] = []
    fences: list[str] = []
    table_count = 0
    blockquote_count = 0
    list_depth = 0
    max_list_depth = 0
    root_list_types: list[str] = []

    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            text = tokens[index + 1].content if index + 1 < len(tokens) else ""
            headings.append((int(token.tag[1:]), text.strip()))
        elif token.type == "fence":
            fences.append(token.info.strip())
        elif token.type == "table_open":
            table_count += 1
        elif token.type == "blockquote_open":
            blockquote_count += 1
        elif token.type in {"bullet_list_open", "ordered_list_open"}:
            if list_depth == 0:
                root_list_types.append("bullet" if token.type.startswith("bullet") else "ordered")
            list_depth += 1
            max_list_depth = max(max_list_depth, list_depth)
        elif token.type in {"bullet_list_close", "ordered_list_close"}:
            list_depth -= 1

    h1_values = [text for level, text in headings if level == 1]
    h2_values = [text for level, text in headings if level == 2]
    checks = {
        "parse_success": parse_success,
        "balanced_fences": fences_balanced(markdown),
        "h1_count": len(h1_values) == contract["h1_count"],
        "h1_title": h1_values == [contract["title"]],
        "h2_structure": h2_values == contract["h2_texts"],
        "list_style": root_list_types == [contract["list_style"]],
        "list_depth": max_list_depth == contract["list_depth"],
        "code_count": len(fences) == (1 if contract["require_code"] else 0),
        "code_language": (
            fences == [contract["code_language"]] if contract["require_code"] else not fences
        ),
        "table_count": table_count == (1 if contract["require_table"] else 0),
        "blockquote_count": blockquote_count
        == (1 if contract["require_blockquote"] else 0),
        "audience_present": contract["audience"] in markdown,
        "context_present": contract["context"] in markdown,
        "emphasis_present": contract["emphasis"] in markdown,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "observed": {
            "headings": headings,
            "fence_languages": fences,
            "table_count": table_count,
            "blockquote_count": blockquote_count,
            "max_list_depth": max_list_depth,
            "root_list_types": root_list_types,
        },
        "parse_error": parse_error,
    }
