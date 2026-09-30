from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.axis_renderer import (
    AXIS_CELLS,
    corrupt_axis_markdown,
    make_axis_contract,
    render_axis_markdown,
)
from markdown_reliability.latentmd_structural import score_response


FAMILY = {
    "task_family_id": "mbpp_test",
    "title": "Add Two Values",
    "description": "Return the sum of two values.",
}
VARIANT = {
    "language": "Python",
    "markdown_language_tag": "python",
    "starter_code": "def add(a, b):\n    pass\n",
    "test_harness": "assert add(2, 3) == 5\n",
}


def score(markdown: str, contract: dict) -> dict:
    return score_response(
        {
            "response": markdown,
            "metadata": {
                "axis_conditions": {
                    "A": contract["axis_a"],
                    "B": contract["axis_b"],
                },
                "slots": {
                    "LANG": VARIANT["language"],
                    "TASK": FAMILY["description"],
                    "TASK_ID": FAMILY["task_family_id"],
                },
            },
        }
    )


class AxisRendererTests(unittest.TestCase):
    def test_all_twelve_targets_pass(self) -> None:
        for cell in AXIS_CELLS:
            contract = make_axis_contract(cell, VARIANT)
            target = render_axis_markdown(FAMILY, VARIANT, contract)
            self.assertTrue(score(target, contract)["passed"], cell)

    def test_every_repair_corruption_fails(self) -> None:
        observed = set()
        for cell in AXIS_CELLS:
            contract = make_axis_contract(cell, VARIANT)
            target = render_axis_markdown(FAMILY, VARIANT, contract)
            for variant_number in range(12):
                broken, corruption = corrupt_axis_markdown(
                    target, contract, variant_number
                )
                self.assertFalse(
                    score(broken, contract)["passed"], (cell, corruption)
                )
                observed.add(corruption)
        self.assertEqual(
            observed,
            {
                "code_language",
                "raw_source",
                "table",
                "citation",
                "numbered_list",
                "outer_wrapper",
            },
        )


if __name__ == "__main__":
    unittest.main()
