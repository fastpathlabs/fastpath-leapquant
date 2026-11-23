"""Fixed-budget rule (campaign.json quality_qualification.answer_budget_rule): a capped answer counts as wrong even when
the bench's parser found the right number in it, in lqh.quality and in the independent checker alike."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lqh import checker, quality  # noqa: E402


def item(arm, question, correct, capped=False):
    return {"schema": "lqh.quality_item.v1", "dataset": "aime2026", "sample_id": f"aime2026:{question}",
            "question": question, "sample": 0, "seed": 1234, "arm": arm, "prompt_sha256": f"p{question}",
            "correct": correct, "capped": capped, "error": None}


class CappedAnswers(unittest.TestCase):
    def setUp(self):
        self.reference = [item("fp32", f"q{i}", True) for i in range(10)]
        # the arm's answer to q0 is "correct" by the parser but was cut at the budget
        self.arm = [item("leapquant", "q0", True, capped=True)] + [item("leapquant", f"q{i}", True) for i in range(1, 10)]

    def test_quality_counts_capped_as_wrong(self):
        result = quality.compare_task(self.reference, self.arm, 1.0)
        self.assertAlmostEqual(result["accuracy_arm"], 0.9)
        self.assertAlmostEqual(result["accuracy_reference"], 1.0)
        self.assertAlmostEqual(result["cap_rate_arm"], 0.1)

    def test_checker_agrees(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, rows in (("fp32", self.reference), ("leapquant", self.arm)):
                Path(folder, f"quality_items_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = checker._quality(folder, "leapquant", "aime2026", "fp32")
        self.assertAlmostEqual(out["accuracy_arm"], 0.9)
        self.assertAlmostEqual(out["difference"], -0.1)


if __name__ == "__main__":
    unittest.main()
