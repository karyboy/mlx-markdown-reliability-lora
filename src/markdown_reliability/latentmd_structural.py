"""Transparent, local structural evaluator for the frozen LatentMD subset.

This is intentionally not a copy of LatentMD's third-party evaluator. It
scores only requirements that can be checked deterministically from each
prompt: outer wrapper policy, parsed fences, raw Markdown-source blocks,
tables, blockquotes with a citation marker, and numbered lists.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from markdown_it import MarkdownIt

from .verifier import fences_balanced


_OPEN_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_CITATION_MARKER = re.compile(
    r"\[[^\]]+\]\([^)]+\)|https?://|\b(source|citation|reference)\b", re.I
)


def _normalise_language(value: str) -> str:
    return value.strip().lower().replace(" ", "")


def _language_matches(observed: str, expected: str) -> bool:
    observed = _normalise_language(observed)
    expected = _normalise_language(expected)
    aliases = {
        "c++": {"c++", "cpp", "cplusplus"},
        "c#": {"c#", "csharp", "cs"},
        "javascript": {"javascript", "js", "node"},
        "typescript": {"typescript", "ts"},
        "shell": {"shell", "sh", "bash", "zsh"},
    }
    return observed in aliases.get(expected, {expected})


def _outer_fence(response: str) -> dict[str, Any]:
    """Return the outer fenced wrapper, if the response begins with one."""
    lines = response.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip()), None)
    if first is None:
        return {"found": False, "proper": False, "content": ""}
    match = _OPEN_FENCE.match(lines[first])
    if not match:
        return {"found": False, "proper": False, "content": response}
    opener, info = match.groups()
    char, length = opener[0], len(opener)
    close_index = None
    for index in range(first + 1, len(lines)):
        candidate = lines[index].lstrip(" ")
        indent = len(lines[index]) - len(candidate)
        if indent > 3:
            continue
        if re.fullmatch(re.escape(char) + "{" + str(length) + r",}\s*", candidate):
            close_index = index
            break
    if close_index is None:
        return {
            "found": True,
            "proper": False,
            "info": info.strip(),
            "content": "\n".join(lines[first + 1 :]),
        }
    trailing_content = any(line.strip() for line in lines[close_index + 1 :])
    return {
        "found": True,
        "proper": not trailing_content,
        "info": info.strip(),
        "content": "\n".join(lines[first + 1 : close_index]) + "\n",
    }


def _prompt_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    axes = metadata["axis_conditions"]
    expected_language = metadata["slots"]["LANG"]
    b_axis = axes["B"]
    code_rules = {
        "B1": {"code_mode": "exactly_one", "raw_blocks": 0},
        "B2": {"code_mode": "at_least_one", "raw_blocks": 1},
        "B3": {"code_mode": "at_least_two", "raw_blocks": 2},
        "B4": {"code_mode": "at_least_two", "raw_blocks": 0},
    }
    rule = code_rules[b_axis]
    outer_policy = {"A1": "prohibited", "A2": "required", "A3": "unspecified"}[
        axes["A"]
    ]
    return {
        "outer_policy": outer_policy,
        "language": expected_language,
        "code_mode": rule["code_mode"],
        "minimum_raw_source_blocks": rule["raw_blocks"],
        "require_table": b_axis == "B4",
        "require_blockquote_citation": b_axis == "B4",
        "require_numbered_list": b_axis == "B4",
    }


def _parse_structure(markdown: str, expected_language: str) -> dict[str, Any]:
    parser = MarkdownIt("commonmark").enable("table")
    tokens = parser.parse(markdown)
    fences = [token for token in tokens if token.type == "fence"]
    target_fences = [
        token
        for token in fences
        if _language_matches(
            token.info.split(maxsplit=1)[0] if token.info else "", expected_language
        )
    ]
    raw_source_blocks = 0
    inner_fence = re.compile(r"(?m)^ {0,3}(`{3,}|~{3,})([^\s]*)\s*$")
    for token in fences:
        inner_languages = [match.group(2) for match in inner_fence.finditer(token.content)]
        if any(_language_matches(language, expected_language) for language in inner_languages):
            raw_source_blocks += 1

    quote_lines = [
        line.lstrip()[1:].strip()
        for line in markdown.splitlines()
        if line.lstrip().startswith(">")
    ]
    quote_text = "\n".join(quote_lines)
    return {
        "target_code_fences": len(target_fences),
        "raw_source_blocks": raw_source_blocks,
        "table_count": sum(token.type == "table_open" for token in tokens),
        "ordered_list_count": sum(token.type == "ordered_list_open" for token in tokens),
        "blockquote_count": sum(token.type == "blockquote_open" for token in tokens),
        "blockquote_has_citation": bool(
            quote_text and _CITATION_MARKER.search(quote_text)
        ),
        "parse_success": True,
    }


def score_response(record: dict[str, Any]) -> dict[str, Any]:
    """Score one generated response against its explicit LatentMD prompt axes."""
    metadata = record["metadata"]
    contract = _prompt_contract(metadata)
    response = record["response"]
    outer = _outer_fence(response)

    if contract["outer_policy"] == "required":
        outer_pass = (
            outer["found"]
            and outer["proper"]
            and _normalise_language(outer.get("info", "")) == "markdown"
        )
        markdown_to_parse = outer["content"] if outer_pass else response
    elif contract["outer_policy"] == "prohibited":
        outer_pass = not outer["found"]
        markdown_to_parse = response
    else:
        outer_pass = True
        markdown_to_parse = response

    try:
        structure = _parse_structure(markdown_to_parse, contract["language"])
    except Exception as exc:  # pragma: no cover - defensive parser guard
        structure = {
            "target_code_fences": 0,
            "raw_source_blocks": 0,
            "table_count": 0,
            "ordered_list_count": 0,
            "blockquote_count": 0,
            "blockquote_has_citation": False,
            "parse_success": False,
            "parse_error": f"{type(exc).__name__}: {exc}",
        }

    code_count = structure["target_code_fences"]
    if contract["code_mode"] == "exactly_one":
        code_pass = code_count == 1
    elif contract["code_mode"] == "at_least_one":
        code_pass = code_count >= 1
    else:
        code_pass = code_count >= 2

    checks = {
        "outer_wrapper": outer_pass,
        "balanced_fences": fences_balanced(response),
        "parse_success": structure["parse_success"],
        "language_code_examples": code_pass,
        "raw_markdown_source": structure["raw_source_blocks"]
        >= contract["minimum_raw_source_blocks"],
        "table": (structure["table_count"] >= 1) if contract["require_table"] else True,
        "blockquote_with_citation": (
            structure["blockquote_count"] >= 1 and structure["blockquote_has_citation"]
            if contract["require_blockquote_citation"]
            else True
        ),
        "numbered_list": (
            structure["ordered_list_count"] >= 1
            if contract["require_numbered_list"]
            else True
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "contract": contract,
        "observed": {
            **structure,
            "outer_fence_found": outer["found"],
            "outer_fence_proper": outer["proper"],
            "outer_fence_info": outer.get("info"),
        },
    }


def aggregate_scores(scored_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate and A×B-cell metrics for local structural scoring."""
    count = len(scored_records)
    check_totals: Counter[str] = Counter()
    failure_totals: Counter[str] = Counter()
    cell_totals: dict[str, dict[str, int]] = {}
    for record in scored_records:
        score = record["score"]
        cell = record["metadata"]["axis_conditions"]
        cell_id = f"{cell['A']}_{cell['B']}"
        cell_totals.setdefault(cell_id, {"count": 0, "passed": 0})
        cell_totals[cell_id]["count"] += 1
        cell_totals[cell_id]["passed"] += int(score["passed"])
        for name, value in score["checks"].items():
            check_totals[name] += int(value)
        failure_totals.update(score["failures"])

    return {
        "count": count,
        "full_structural_contract_pass_rate": sum(
            record["score"]["passed"] for record in scored_records
        )
        / count
        if count
        else 0.0,
        "check_pass_rates": {
            name: total / count for name, total in sorted(check_totals.items())
        },
        "failure_counts": dict(sorted(failure_totals.items())),
        "by_axis_cell": {
            key: {
                **values,
                "pass_rate": values["passed"] / values["count"],
            }
            for key, values in sorted(cell_totals.items())
        },
        "scoring_method": {
            "name": "local_structural_contract_v1",
            "description": (
                "Deterministic parser- and prompt-contract-based scoring written for this "
                "experiment. It does not assess arbitrary task-answer semantic correctness "
                "and is not LatentMD's authors' official evaluator."
            ),
        },
    }
