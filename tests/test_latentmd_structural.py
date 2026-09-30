from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.latentmd_structural import score_response


def record(a_axis: str, b_axis: str, response: str) -> dict:
    return {
        "response": response,
        "metadata": {
            "axis_conditions": {"A": a_axis, "B": b_axis},
            "slots": {"LANG": "C"},
        },
    }


class LatentMdStructuralTests(unittest.TestCase):
    def test_a1_b1_direct_document_passes(self) -> None:
        result = score_response(
            record("A1", "B1", "# Title\n\n```c\nint main(void) { return 0; }\n```\n")
        )
        self.assertTrue(result["passed"], result)

    def test_a1_b1_outer_fence_fails(self) -> None:
        result = score_response(
            record(
                "A1",
                "B1",
                "```markdown\n# Title\n\n```c\nint main(void) { return 0; }\n```\n",
            )
        )
        self.assertFalse(result["passed"], result)
        self.assertIn("outer_wrapper", result["failures"])

    def test_a2_b1_four_tick_outer_wrapper_passes(self) -> None:
        result = score_response(
            record(
                "A2",
                "B1",
                "````markdown\n# Title\n\n```c\nint main(void) { return 0; }\n```\n````\n",
            )
        )
        self.assertTrue(result["passed"], result)

    def test_b4_requires_table_quote_and_numbered_list(self) -> None:
        result = score_response(
            record(
                "A3",
                "B4",
                "# Title\n\n1. First step\n\n> Source: internal guide\n\n"
                "| Name | Value |\n| --- | --- |\n| A | B |\n\n"
                "```c\nint one(void) { return 1; }\n```\n\n"
                "```c\nint two(void) { return 2; }\n```\n",
            )
        )
        self.assertTrue(result["passed"], result)
