"""Evidence sections of the campaign report: kernel full cycle, sanitizer, quality decision and NLL summaries are
produced by their own tools from synthetic raw files, published, and re-derived by the independent checker."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import helpers  # noqa: F401

from lqh import checker, kernel_cycle, nll, quality, report, sanitizer
from lqh.util import harness_root, read_json, write_json, write_jsonl


def kernel_fixture(folder: Path) -> None:
    rng = random.Random(1)
    folder.mkdir(parents=True)
    for rep in range(1, 6):
        lines = []
        for batch, lq, bf in ((512, 0.300, 0.382), (32, 0.030, 0.031)):
            lq_v = lq * (1 + 0.01 * rng.random())
            bf_v = bf * (1 + 0.01 * rng.random())
            lines += [f"  batch {batch}",
                      f"    full cycle   LeapQuant: {lq_v * 16:.4f} ms per window of 16 tokens = {lq_v:.4f} ms per token "
                      f"(one CUDA graph: 16 state-advancing decode steps + fold; start state restored outside the timing; "
                      f"median of 10 replays; boundary state advanced by the cycle: True; fold backend cuda, decode backend cuda; order lq-first)",
                      f"    full cycle   sglang window kernel, fp32 state: {bf_v * 1.8:.4f} ms per token (same method)",
                      f"    full cycle   sglang window kernel, bf16 state: {bf_v:.4f} ms per token (same method)"]
        (folder / f"rep_{rep}.txt").write_text("\n".join(lines) + "\n")
    write_json(folder / "kernel_cycle.json", kernel_cycle.summarise(folder))


def sanitizer_fixture(folder: Path) -> None:
    folder.mkdir(parents=True)
    (folder / "memcheck_ok.txt").write_text("========= COMPUTE-SANITIZER\n========= ERROR SUMMARY: 0 errors\n")
    (folder / "memcheck_api.txt").write_text("========= COMPUTE-SANITIZER\n" + "========= Program hit CUDA_ERROR_INVALID_VALUE "
                                             "(error 1) due to \"invalid argument\" on CUDA API call to cuFuncSetAttribute.\n" * 2
                                             + "========= ERROR SUMMARY: 2 errors\n")
    (folder / "racecheck_cut.txt").write_text("========= COMPUTE-SANITIZER\n========= Potential WAR hazard detected\n")
    doc = {"schema": "lqh.sanitizer_results.v1", "records": [
        sanitizer.record(folder / "memcheck_ok.txt", "memcheck", "cmd", 0, None, "fixture", "measured_raw"),
        sanitizer.record(folder / "memcheck_api.txt", "memcheck", "cmd", 1, None, "fixture", "measured_raw"),
        sanitizer.record(folder / "racecheck_cut.txt", "racecheck", "cmd", None, None, "fixture", "measured_raw")]}
    write_json(folder / "sanitizer_results.json", doc)


def quality_fixture(folder: Path) -> None:
    folder.mkdir(parents=True)
    rng = random.Random(5)
    for arm, p in (("fp32", 0.8), ("bf16_window", 0.78), ("leapquant", 0.8)):
        rows = []
        for q in range(30):
            for s in range(3):
                rows.append({"dataset": "aime2026", "sample_id": f"aime2026:{q}#s{s}", "question": f"q{q}", "sample": s,
                             "prompt_sha256": f"p{q}", "correct": rng.random() < p, "capped": rng.random() < 0.05,
                             "error": None})
        write_jsonl(folder / f"quality_items_{arm}.jsonl", rows)
    campaign = read_json(harness_root() / "config" / "campaign.json")
    write_json(folder / "quality_decision.json", quality.decide(folder, campaign))


def nll_fixture(folder: Path) -> None:
    rng = random.Random(9)
    for name, shift in (("nll_fp32_c16", 0.0), ("nll_leapquant_c16", 0.01)):
        sub = folder / name
        sub.mkdir(parents=True)
        rows = [{"doc_id": f"d{d}", "length": 4096, "nll": [None] + [2.0 + shift + rng.random() * 0.1 for _ in range(4095)]}
                for d in range(4)]
        write_jsonl(sub / "nll_by_position.jsonl", rows)
        write_json(sub / "nll_summary.json", {"buckets": nll.bucket_means(rows)})
    write_json(folder / "nll_compare_c16.json", nll.compare(folder / "nll_fp32_c16", [folder / "nll_leapquant_c16"]))


class EvidenceSections(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "harness"
        kernel_fixture(self.root / "kernel-cycle")
        sanitizer_fixture(self.root / "sanitizers")
        quality_fixture(self.root / "quality-aime2026")
        nll_fixture(self.root / "longcontext")
        self.bundle = self.root.parent / "bundle"
        report.build([self.root], self.bundle)

    def test_sections_are_rendered_and_checked(self):
        page = (self.bundle / "index.html").read_text()
        for title in ("complete state-advancing layer cycle", "compute-sanitizer", "Quality qualification", "Long-context NLL"):
            self.assertIn(title, page)
        result = checker.check_bundle(self.bundle)
        self.assertTrue(result["ok"], result["failed"][:5])
        kinds = {d.get("evidence") for d in json.loads((self.bundle / "displayed_values.json").read_text())}
        self.assertEqual(kinds - {None}, {"kernel_cycle", "sanitizer", "quality", "nll"})

    def test_classifications(self):
        records = read_json(self.root / "sanitizers" / "sanitizer_results.json")["records"]
        self.assertEqual([r["classification"] for r in records], ["clean", "api_errors_only", "interrupted"])

    def test_tampered_evidence_value_fails(self):
        displayed = json.loads((self.bundle / "displayed_values.json").read_text())
        for item in displayed:
            if item.get("evidence") == "quality" and item["field"] == "accuracy_arm":
                item["value"] += 0.05
                break
        (self.bundle / "displayed_values.json").write_text(json.dumps(displayed))
        self.assertFalse(checker.check_bundle(self.bundle)["ok"])

    def test_checker_t_quantile_matches_exact(self):
        from lqh import stats
        for df in (1, 2, 3, 4, 9, 30):
            self.assertAlmostEqual(checker._t_q(0.975, df), stats.t_quantile(0.975, df), delta=2e-3 * stats.t_quantile(0.975, df))


if __name__ == "__main__":
    unittest.main()
