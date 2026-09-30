from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from markdown_reliability.catalog import TOPICS
from markdown_reliability.renderer import corrupt_markdown, make_contract, render_markdown
from markdown_reliability.verifier import fences_balanced, verify_markdown


class RendererVerifierTests(unittest.TestCase):
    def test_all_contract_variants_pass(self) -> None:
        for topic in TOPICS:
            for variant in range(30):
                contract = make_contract(topic, variant)
                markdown = render_markdown(topic, contract)
                result = verify_markdown(markdown, contract)
                self.assertTrue(result["passed"], (topic["id"], variant, result))

    def test_each_corruption_is_detected(self) -> None:
        detected = set()
        for variant in range(60):
            topic = TOPICS[variant % len(TOPICS)]
            contract = make_contract(topic, variant)
            target = render_markdown(topic, contract)
            broken, corruption = corrupt_markdown(target, contract, variant)
            result = verify_markdown(broken, contract)
            self.assertFalse(result["passed"], (corruption, result))
            detected.add(corruption)
        self.assertEqual(detected, {"heading", "fence", "indent", "table"})

    def test_unclosed_fence_fails_balance_check(self) -> None:
        self.assertFalse(fences_balanced("```python\nprint('hello')\n"))


if __name__ == "__main__":
    unittest.main()
