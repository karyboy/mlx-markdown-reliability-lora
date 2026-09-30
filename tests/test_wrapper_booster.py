from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.latentmd_structural import score_response
from markdown_reliability.wrapper_booster import generation_prompt, render_short_document


FAMILY = {
    "task_family_id": "mbpp_test",
    "title": "Return A Value",
    "description": "Return the supplied integer.",
}
VARIANT = {
    "language": "Python",
    "markdown_language_tag": "python",
    "source_config": "test-python",
    "starter_code": "# Task: Return the value.\ndef identity(value):\n    pass\n",
    "test_harness": "assert identity(3) == 3\n",
}


def score(response: str, axis_a: str, axis_b: str) -> dict:
    return score_response(
        {
            "response": response,
            "metadata": {
                "axis_conditions": {"A": axis_a, "B": axis_b},
                "slots": {
                    "LANG": "Python",
                    "TASK": FAMILY["description"],
                    "TASK_ID": FAMILY["task_family_id"],
                },
            },
        }
    )


class WrapperBoosterTests(unittest.TestCase):
    def test_all_twelve_short_targets_pass(self) -> None:
        for axis_a in ("A1", "A2", "A3"):
            for axis_b in ("B1", "B2", "B3", "B4"):
                target = render_short_document(FAMILY, VARIANT, axis_a, axis_b)
                self.assertTrue(score(target, axis_a, axis_b)["passed"], (axis_a, axis_b))

    def test_a2_exact_benchmark_wording_and_wrapper(self) -> None:
        prompt = generation_prompt(FAMILY, VARIANT, "A2", "B1")
        target = render_short_document(FAMILY, VARIANT, "A2", "B1")
        self.assertIn("Wrap your entire response in a fenced markdown code block.", prompt)
        self.assertTrue(target.startswith("````markdown\n"))
        self.assertTrue(target.endswith("````\n"))

    def test_removing_a2_wrapper_fails(self) -> None:
        target = render_short_document(FAMILY, VARIANT, "A2", "B2")
        broken = target.removeprefix("````markdown\n").removesuffix("````\n")
        self.assertFalse(score(broken, "A2", "B2")["passed"])


if __name__ == "__main__":
    unittest.main()
