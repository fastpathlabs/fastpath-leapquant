"""Regression fixtures for the archive's report failures (section 16): invented concurrency rows, wrong source-round
captions, mixed capacities under one label, stale gate ids, missing error-norm definitions, chart/table disagreement,
and wording that needs separate evidence. Plus the pinned InferenceX adapter's conversion and command."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import make_run, simple_finite

from lqh import analyze, checker, claims, gates, report, sidecars
from lqh.adapters import infx


def bundle_from(runs_root):
    """A bundle whose campaign names the fixture's arms: plan validation_pair, baseline 'base'."""
    from lqh.util import harness_root
    campaign = json.loads((harness_root() / "config" / "campaign.json").read_text())
    campaign["plans"] = {"validation_pair": {"candidates": ["base", "cand"], "comparison_class": "same_capacity"}}
    folder = Path(tempfile.mkdtemp())
    (folder / "campaign.json").write_text(json.dumps(campaign))
    out = folder / "bundle"
    report.build([runs_root], out, folder / "campaign.json")
    return out


def fixture_runs(root, blocks=2, slots=(1100, 1100)):
    """Two candidates x blocks on one cell, through analyze."""
    for block in range(1, blocks + 1):
        for cand, tokens, slot in (("base", 8, slots[0]), ("cand", 10, slots[1])):
            reqs, phases = simple_finite(n=4, tokens_per_request=tokens)
            cfg = {"launch": "sglang", "configured_slots": slot}
            from lqh.util import config_hash
            run = make_run(reqs, phases, root=Path(root) / f"b{block}" / cand,
                           extra_manifest={"block_id": block, "plan_id": "validation_pair",
                                           "run_id": f"fx.b{block}.{cand}", "experiment_id": f"fx.exp{block}",
                                           "candidate": {"candidate_id": cand, "config": cfg,
                                                         "config_hash": config_hash(cfg)}})
            analyze.compute(run)
    return Path(root)


class Bundle(unittest.TestCase):
    def setUp(self):
        self.root = fixture_runs(tempfile.mkdtemp())
        self.bundle = bundle_from(self.root)

    def test_checker_reproduces_every_displayed_value(self):
        result = checker.check_bundle(self.bundle)
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["checked"], 100)

    def test_no_invented_concurrency_rows(self):
        summary = json.loads((self.bundle / "summary.json").read_text())
        self.assertEqual({r["client_concurrency"] for r in summary["rows"]}, {4})
        self.assertEqual(len(summary["rows"]), 2)

    def test_figure_points_name_their_own_experiment(self):
        for path in (self.bundle / "figures").glob("*.json"):
            sidecar = json.loads(path.read_text())
            self.assertEqual(sidecars.sidecar_problems(sidecar), [])
            for point in sidecar["points"]:
                block = point["run_id"].split(".")[1][1:]
                self.assertEqual(point["originating_experiment"], f"fx.exp{block}")

    def test_table_tampering_is_detected(self):
        displayed = json.loads((self.bundle / "displayed_values.json").read_text())
        target = next(d for d in displayed if d.get("metric") == "finite_cohort_output_tps")
        target["value"] *= 1.03
        (self.bundle / "displayed_values.json").write_text(json.dumps(displayed))
        result = checker.check_bundle(self.bundle)
        self.assertFalse(result["ok"])
        self.assertTrue(result["failed"] or result["hash_failures"])

    def test_chart_tampering_is_detected(self):
        path = next((self.bundle / "figures").glob("*.json"))
        sidecar = json.loads(path.read_text())
        sidecar["points"][0]["y_output_tps_per_gpu"] *= 1.10
        path.write_text(json.dumps(sidecar))
        result = checker.check_bundle(self.bundle)
        self.assertFalse(result["ok"])

    def test_paired_comparison_and_claim(self):
        summary = json.loads((self.bundle / "summary.json").read_text())
        comp = summary["paired_comparisons"][0]
        self.assertEqual(comp["n_blocks"], 2)
        # base: 4 x 8 tokens over 0.421 s; cand: 4 x 10 tokens over 0.441 s (longer answers lengthen the cohort).
        self.assertAlmostEqual(comp["geometric_mean"], (40 / 0.441) / (32 / 0.421), places=6)
        ledger = json.loads((self.bundle / "claims.json").read_text())
        self.assertEqual(len(ledger["claims"]), 1)
        self.assertIn("experimental", ledger["claims"][0]["eligibility"])

    def test_uncomputable_comparison_is_not_a_displayed_number(self):
        # Job 114, 15 Nov 2026: a comparison with a zero-throughput arm has no ratio; it was listed as a displayed
        # value of None, and the checker failed the whole report on it.
        comparison = {"plan_id": "p", "cell_id": "c", "candidate": "a", "baseline": "b", "blocks": [1],
                      "geometric_mean": None, "null_reason": "no_positive_pairs"}
        self.assertEqual(report.displayed_values([], [], [comparison]), [])


class ServiceEligibility(unittest.TestCase):
    """Multi-GPU benchmark plan (15 Nov 2026), section 4: service eligibility is shown, so the checker re-derives it."""

    def setUp(self):
        root = Path(tempfile.mkdtemp())
        for block in (1, 2):
            for cand, tokens in (("base", 8), ("cand", 10)):
                reqs, phases = simple_finite(n=4, tokens_per_request=tokens)
                cfg = {"launch": "sglang", "configured_slots": 1100}
                from lqh.util import config_hash
                run = make_run(reqs, phases, root=root / f"b{block}" / cand, slo_id="slo-interactive-v1",
                               extra_manifest={"block_id": block, "plan_id": "validation_pair",
                                               "run_id": f"se.b{block}.{cand}", "experiment_id": f"se.exp{block}",
                                               "candidate": {"candidate_id": cand, "config": cfg,
                                                             "config_hash": config_hash(cfg)}})
                analyze.compute(run)
        self.bundle = bundle_from(root)

    def test_shown_and_reproduced(self):
        displayed = json.loads((self.bundle / "displayed_values.json").read_text())
        shown = [d for d in displayed if d.get("metric") == "service_eligible"]
        self.assertEqual(len(shown), 4)
        result = checker.check_bundle(self.bundle)
        self.assertTrue(result["ok"], result)

    def test_flipped_verdict_is_detected(self):
        path = self.bundle / "displayed_values.json"
        displayed = json.loads(path.read_text())
        target = next(d for d in displayed if d.get("metric") == "service_eligible")
        target["value"] = 1.0 - target["value"]
        path.write_text(json.dumps(displayed))
        result = checker.check_bundle(self.bundle)
        self.assertFalse(result["ok"])


class MixedConfigurations(unittest.TestCase):
    def test_one_label_two_configurations_is_flagged_not_merged(self):
        root = Path(tempfile.mkdtemp())
        fixture_runs(root / "a", blocks=1, slots=(1100, 1100))
        fixture_runs(root / "b", blocks=1, slots=(1100, 1560))      # same label 'cand', different slots
        for manifest_path in (root / "b").rglob("run_manifest.json"):
            manifest = json.loads(manifest_path.read_text())
            manifest["run_id"] += ".second"
            manifest["block_id"] = 2
            manifest_path.write_text(json.dumps(manifest))
        rows, comparisons = report.summarise([report.load_run(d) for d in report.find_runs([root])],
                                             {"plans": {"validation_pair": {"candidates": ["base", "cand"]}}})
        cand = next(r for r in rows if r["candidate_id"] == "cand")
        self.assertIn("MIXED_CONFIGURATIONS", cand["statuses"])
        self.assertEqual(comparisons, [])


class GatesAndWording(unittest.TestCase):
    def test_stale_gate_ids_are_refused(self):
        for gate_id in ("p5.G3", "plan5.G4", "review2.G6"):
            with self.assertRaises(gates.GateError):
                gates.record(gate_id, "PASS", ["x"])
        with self.assertRaises(gates.GateError):
            gates.record("v6.G9", "PASS", ["x"])                      # not in the registry
        with self.assertRaises(gates.GateError):
            gates.record("v6.G2", "PASS", [])                         # a pass needs evidence
        with self.assertRaises(gates.GateError):
            gates.record("v6.G2", "PASS", ["kernel_cycle.json"])      # prerequisite v6.G1 not passed

    def test_banned_wording(self):
        for text in ("LeapQuant is production-ready", "lossless compression", "official InferenceX result",
                     "Official AA score", "universally faster", "reproduced the paper"):
            with self.assertRaises(claims.WordingError):
                claims.check_wording(text)
        claims.check_wording("Observed successful output throughput 12.0% higher (95% interval 1.08 to 1.16)")

    def test_error_metric_needs_a_norm(self):
        sidecar = {"generator": "g", "data_sha256": "h", "points": [{
            "figure_id": "f", "candidate_id": "c", "originating_experiment": "e", "hardware": "H100",
            "measurement_boundary": "single-layer replay", "comparison_class": "diagnostic",
            "evidence_kind": "measured_raw", "derivation": "d", "source_row_id": "file:1",
            "metric": "output error", "y_value": 0.7, "y_unit": "%", "plotted": True}]}
        self.assertIn("point 0: error metric without a stated norm", sidecars.sidecar_problems(sidecar))
        sidecar["points"][0]["error_norm"] = "pooled relative L2 x 100 over tokens, heads, values"
        self.assertEqual(sidecars.sidecar_problems(sidecar), [])


class InferenceXAdapter(unittest.TestCase):
    def test_argv_matches_the_pinned_client_argv(self):
        argv = infx.client_argv("python3", "http://h:1", "m", "t", 8192, 1024, 1.0, 32, 320, Path("/r/x.json"))
        expected = ["python3", "-m", "infx.bench_serving.benchmark_serving", "--model", "m", "--backend", "vllm",
                    "--base-url", "http://h:1", "--dataset-name", "random", "--random-input-len", "8192",
                    "--random-output-len", "1024", "--random-range-ratio", "1.0", "--num-prompts", "320",
                    "--max-concurrency", "32", "--request-rate", "inf", "--ignore-eos", "--save-result",
                    "--num-warmups", "64", "--percentile-metrics", "ttft,tpot,itl,e2el", "--result-dir", "/r",
                    "--result-filename", "x.json", "--tokenizer", "t"]
        self.assertEqual(argv[:len(expected)], expected)
        self.assertEqual(argv[len(expected):], ["--save-detailed", "--seed", "0"])   # D-IX-1 and the default seed

    def test_conversion_of_detailed_output(self):
        result = {"input_lens": [128, 128, 128], "output_lens": [4, 0, 4], "ttfts": [0.1, 0.0, 0.2],
                  "itls": [[0.01, 0.01, 0.01], [], [0.02, 0.05]], "errors": ["", "HTTP 500", ""]}
        out = Path(tempfile.mkdtemp())
        counts = infx.convert(result, out, osl=4, isl=128)
        records = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
        self.assertEqual(counts["inferred_single_token_chunks"], 1)        # 4 chunks for 4 tokens
        self.assertEqual([r["status"] for r in records], ["success", "failed", "success"])
        self.assertEqual(records[0]["clock_domain"], "request_relative")
        self.assertIsNone(records[0]["t_terminal"])                          # unknown, never invented
        self.assertAlmostEqual(records[0]["t_last_token"], 0.13)
        self.assertFalse(records[2]["exact_chunk_tokens"])                   # 3 chunks for 4 tokens: unknown

    def test_unequal_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            infx.convert({"input_lens": [1], "output_lens": [1, 2], "ttfts": [0.1], "itls": [[]], "errors": [""]},
                         Path(tempfile.mkdtemp()), 1, 1)


if __name__ == "__main__":
    unittest.main()
