"""Statistics and quality-decision fixtures: the zero-discordance interval, noninferiority, t quantiles, paired
ratios (mean of ratios vs ratio of means), disjoint or duplicated cohorts, and the frozen decision rule."""
import json
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

from lqh import quality, stats


class Stats(unittest.TestCase):
    def test_t_quantiles(self):
        self.assertAlmostEqual(stats.t_quantile(0.975, 4), 2.7764, places=3)
        self.assertAlmostEqual(stats.t_quantile(0.975, 9), 2.2622, places=3)
        self.assertAlmostEqual(stats.t_quantile(0.975, 10000), 1.9602, places=3)

    def test_zero_discordance_interval_is_not_degenerate(self):
        result = stats.newcombe_paired(30, 0, 0, 0)
        self.assertLess(result["low"], -0.05)
        self.assertEqual(result["discordant"], 0)
        self.assertEqual(stats.noninferiority(result["low"], result["high"], 0.01), "INCONCLUSIVE")

    def test_noninferiority_verdicts(self):
        self.assertEqual(stats.noninferiority(-0.005, 0.02, 0.01), "PASS")
        self.assertEqual(stats.noninferiority(-0.08, -0.03, 0.01), "FAIL")
        self.assertEqual(stats.noninferiority(-0.05, 0.03, 0.01), "INCONCLUSIVE")
        self.assertEqual(stats.noninferiority(None, None, 0.01), "INCONCLUSIVE")

    def test_mean_of_ratios_is_not_ratio_of_means(self):
        result = stats.paired_ratio([100.0, 1000.0], [200.0, 1000.0])
        self.assertAlmostEqual(result["mean_of_ratios"], 1.5)
        self.assertAlmostEqual(result["ratio_of_means"], 1200 / 1100)
        self.assertAlmostEqual(result["geometric_mean"], 2 ** 0.5)
        self.assertEqual(result["value"], result["geometric_mean"])

    def test_one_block_has_no_interval(self):
        result = stats.paired_ratio([100.0], [110.0])
        self.assertIsNone(result["low"])
        self.assertEqual(result["interval_null_reason"], "fewer_than_two_blocks")

    def test_required_blocks(self):
        self.assertEqual(stats.required_blocks(0.001, 0.015), 5)            # declared minimum
        self.assertGreater(stats.required_blocks(0.03, 0.015), 5)
        self.assertEqual(stats.required_blocks(1.0, 0.001, maximum=12), 12)  # budget cap, then INCONCLUSIVE


class QualityDecision(unittest.TestCase):
    CAMPAIGN = {"quality_qualification": {"margin_points": 1.0, "decision_rule_frozen": "rule",
                                          "tasks": {"aime2026": "x", "beyond_aime": "y"}}}

    def write(self, folder, arm, items):
        with open(Path(folder) / f"quality_items_{arm}.jsonl", "w") as handle:
            for item in items:
                handle.write(json.dumps(item) + "\n")

    def items(self, task, correct, samples=3, capped=()):
        out = []
        for q, flags in enumerate(correct):
            for s in range(samples):
                out.append({"dataset": task, "sample_id": f"{task}:{q}#s{s}", "question": f"{task}-{q}", "sample": s,
                            "prompt_sha256": f"p{q}", "correct": bool(flags[s]), "capped": (q, s) in capped,
                            "error": None})
        return out

    def test_identical_arms_on_small_tasks_are_inconclusive_not_pass(self):
        folder = tempfile.mkdtemp()
        same = [[1, 1, 0]] * 30
        for arm in ("fp32", "bf16_window", "leapquant"):
            self.write(folder, arm, self.items("aime2026", same) + self.items("beyond_aime", same))
        doc = quality.decide(Path(folder), self.CAMPAIGN)
        self.assertEqual(doc["arms"]["bf16_window"]["aime2026"]["verdict"], "INCONCLUSIVE")
        self.assertFalse(doc["decision"]["bf16_window_primary_product_baseline"])
        self.assertTrue(doc["decision"]["comparator_for_cost_claims"].startswith("fp32"))

    def test_task_not_run_blocks_the_rule(self):
        folder = tempfile.mkdtemp()
        for arm in ("fp32", "bf16_window"):
            self.write(folder, arm, self.items("aime2026", [[1, 1, 1]] * 30))
        doc = quality.decide(Path(folder), self.CAMPAIGN)
        self.assertIn("beyond_aime: not run", doc["decision"]["bf16_window_reasons"])

    def test_disjoint_and_duplicated_cohorts_are_invalid(self):
        folder = tempfile.mkdtemp()
        self.write(folder, "fp32", self.items("aime2026", [[1, 1, 1]] * 30))
        other = self.items("aime2026", [[1, 1, 1]] * 30)
        for item in other:
            item["sample_id"] = "other-" + item["sample_id"]
        self.write(folder, "leapquant", other)
        dup = self.items("aime2026", [[1, 1, 1]] * 30)
        self.write(folder, "bf16_window", dup + dup[:1])
        doc = quality.decide(Path(folder), self.CAMPAIGN)
        self.assertEqual(doc["arms"]["leapquant"]["aime2026"]["verdict"], "INVALID")
        self.assertEqual(doc["arms"]["bf16_window"]["aime2026"]["verdict"], "INVALID")

    def test_changed_prompt_is_invalid_and_large_loss_fails(self):
        folder = tempfile.mkdtemp()
        self.write(folder, "fp32", self.items("aime2026", [[1, 1, 1]] * 200, samples=3))
        worse = self.items("aime2026", [[0, 0, 0]] * 60 + [[1, 1, 1]] * 140, samples=3,
                           capped={(0, 0), (1, 0)})
        self.write(folder, "bf16_rounded", worse)
        changed = self.items("aime2026", [[1, 1, 1]] * 200)
        changed[0]["prompt_sha256"] = "different"
        self.write(folder, "leapquant", changed)
        doc = quality.decide(Path(folder), self.CAMPAIGN)
        self.assertEqual(doc["arms"]["bf16_rounded"]["aime2026"]["verdict"], "FAIL")
        self.assertAlmostEqual(doc["arms"]["bf16_rounded"]["aime2026"]["cap_rate_arm"], 2 / 600)
        self.assertEqual(doc["arms"]["leapquant"]["aime2026"]["verdict"], "INVALID")


if __name__ == "__main__":
    unittest.main()
