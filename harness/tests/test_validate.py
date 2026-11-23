"""Validation fixtures for the gaps found in the archive (section 12). None may become a successful case."""
import gzip
import json
import shutil
import unittest

from helpers import GPU, make_run, request, simple_finite

from lqh import analyze, validate


def status(run):
    return validate.validate_run(run, write=False)


class RejectsBadEvidence(unittest.TestCase):
    def test_valid_fixture_is_measured(self):
        reqs, phases = simple_finite()
        result = status(make_run(reqs, phases))
        self.assertEqual(result["status"], "measured", result["errors"])

    def test_missing_requests_file(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        (run / "requests.jsonl").unlink()
        self.assertEqual(status(run)["status"], "invalid")

    def test_missing_driver_summary_is_crashed_not_measured(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        (run / "driver_summary.json").unlink()
        self.assertEqual(status(run)["status"], "crashed")

    def test_malformed_json(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        with open(run / "requests.jsonl", "a") as handle:
            handle.write('{"request_id": "broken"\n')
        self.assertEqual(status(run)["status"], "invalid")

    def test_nan_is_rejected(self):
        reqs, phases = simple_finite()
        reqs[0][0]["t_first_token"] = float("nan")
        run = make_run(reqs, phases)
        text = (run / "requests.jsonl").read_text()
        self.assertIn("NaN", text)
        self.assertEqual(status(run)["status"], "invalid")

    def test_timestamp_reversal_and_negative_duration(self):
        reqs, phases = simple_finite()
        reqs[1][0]["t_last_token"] = reqs[1][0]["t_first_token"] - 0.5
        self.assertEqual(status(make_run(reqs, phases))["status"], "invalid")
        reqs, phases = simple_finite()
        phases["main"]["end"] = phases["main"]["start"] - 1
        self.assertEqual(status(make_run(reqs, phases))["status"], "invalid")

    def test_duplicate_ids(self):
        reqs, phases = simple_finite()
        reqs.append(reqs[0])
        self.assertIn("V011_duplicate_request_ids", status(make_run(reqs, phases, num_prompts=4))["errors"])

    def test_mismatched_hash(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        with open(run / "requests.jsonl", "a") as handle:
            handle.write("\n")                 # same records, different bytes
        self.assertEqual(status(run)["status"], "invalid")

    def test_impossible_token_count(self):
        reqs, phases = simple_finite()
        reqs[0][0]["output_tokens"] = 99
        reqs[0][0]["requested_output_tokens"] = 8
        self.assertEqual(status(make_run(reqs, phases))["status"], "invalid")

    def test_chunk_tokens_disagree_with_usage(self):
        reqs, phases = simple_finite()
        reqs[0][0]["output_tokens"] = 7
        self.assertIn("V018_chunk_token_sum", status(make_run(reqs, phases))["errors"])

    def test_missing_records(self):
        reqs, phases = simple_finite(n=4)
        self.assertIn("V019_expected_requests", status(make_run(reqs[:3], phases, num_prompts=4))["errors"])

    def test_inconsistent_device_allocation(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases, extra_manifest={"device": {"gpus": [GPU], "allocated_physical_gpus": 2}})
        self.assertIn("V022_device_count_consistent", status(run)["errors"])

    def test_unknown_gpu_fields(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases, gpus=[dict(GPU, name="unknown")])
        self.assertEqual(status(run)["status"], "invalid")

    def test_samples_from_another_gpu(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        (run / "memory_and_power.jsonl").write_text(json.dumps({"kind": "gpu_sample", "gpu_uuid": "GPU-other",
                                                                "t_unix": 1.0}) + "\n")
        self.assertIn("V023_samples_from_allocated_devices", status(run)["errors"])

    def test_candidate_hash_mismatch(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        manifest = json.loads((run / "run_manifest.json").read_text())
        manifest["candidate"]["config"]["flags"] = ["changed after the run"]
        (run / "run_manifest.json").write_text(json.dumps(manifest))
        self.assertIn("V004_candidate_hash", status(run)["errors"])

    def test_invalid_run_metrics_are_null_not_zero(self):
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        (run / "requests.jsonl").unlink()
        doc = analyze.compute(run)
        self.assertIsNone(doc["metrics"]["finite_cohort_output_tps"]["value"])
        self.assertEqual(doc["metrics"]["finite_cohort_output_tps"]["null_reason"], "run_invalid")

    def test_mock_run_is_never_measured(self):
        reqs, phases = simple_finite()
        self.assertEqual(status(make_run(reqs, phases, mock=True))["status"], "smoke_only")

    def test_dead_server_is_crashed_not_zero_throughput(self):
        # Job 114, 15 Nov 2026: the server died in its warmup (CUDA out of memory) and every measured request was
        # refused; the run was published as 0.0 tok/s "measured".
        refused = "ConnectionRefusedError: [Errno 111] Connect call failed ('127.0.0.1', 30000)"
        reqs = [request(f"m{i}", 1.0 + i * 0.1, [], status="failed", error=refused) for i in range(4)]
        run = make_run(reqs, {"main": {"start": 1.0, "end": 1.5}})
        result = status(run)
        self.assertEqual(result["status"], "crashed")
        self.assertEqual(result["execution_validity"], "invalid")
        self.assertIn("V052_server_reachable", result["errors"])
        self.assertIsNone(analyze.compute(run)["metrics"]["finite_cohort_output_tps"]["value"])

    def test_one_refused_request_also_means_no_server(self):
        reqs, phases = simple_finite()
        refused = request("late", 1.2, [], status="failed", error="ConnectionRefusedError: [Errno 111] Connect call failed")
        result = status(make_run(reqs + [refused], phases, num_prompts=5))
        self.assertIn("V052_server_reachable", result["errors"])

    def test_client_file_limit_is_invalid_not_load(self):
        # Job 117, 15 Nov 2026: at 1,024 users the client hit its open-file limit; every attempt failed at once.
        reqs, phases = simple_finite()
        emfile = request("late", 1.2, [], status="failed", error="OSError: [Errno 24] Too many open files")
        result = status(make_run(reqs + [emfile], phases, num_prompts=5))
        self.assertIn("V054_client_resources", result["errors"])
        self.assertEqual(result["status"], "invalid")

    def test_failure_storm_is_invalid(self):
        reqs, phases = simple_finite(n=2)
        storm = [request(f"x{i}", 1.0 + i * 1e-4, [], status="failed", error="HTTP 503: overloaded") for i in range(60)]
        result = status(make_run(reqs + storm, phases, num_prompts=62))
        self.assertIn("V053_failure_storm", result["errors"])
        self.assertEqual(result["status"], "invalid")

    def test_a_few_failures_are_not_a_storm(self):
        reqs, phases = simple_finite()
        failed = request("bad", 1.2, [1.25], [1], status="failed", error="HTTP 400: prompt too long")
        result = status(make_run(reqs + [failed], phases, num_prompts=5))
        self.assertNotIn("V053_failure_storm", result["errors"])
        self.assertNotIn("V054_client_resources", result["errors"])

    def test_native_warmups_read_from_the_client_log(self):
        # The pinned InferenceX client exports only the measured cohort: its log is the warmup evidence.
        reqs, phases = simple_finite()
        run = make_run(reqs, phases)
        (run / "native").mkdir()
        (run / "native" / "client.log").write_text(
            "Namespace(max_concurrency=4, num_prompts=4, num_warmups=8)\nStarting initial single prompt test run...\n"
            "Warming up with 8 requests...\nWarmup completed.\nTraffic request rate: inf\nStarting main benchmark run...\n")
        result = status(run)
        self.assertEqual(result["native_warmups"]["requested"], 8)
        self.assertTrue(result["native_warmups"]["before_measured_cohort"])
        self.assertNotIn("V035_native_warmups", result["warnings"])
        (run / "native" / "client.log").write_text("Namespace(num_warmups=0)\nStarting main benchmark run...\n")
        result = status(run)
        self.assertEqual(result["native_warmups"]["requested"], 0)
        self.assertEqual(result["status"], "measured")            # a protocol warning, not an execution failure
        self.assertIn("V035_native_warmups", result["warnings"])

    def test_other_failures_do_not_trip_the_server_check(self):
        reqs, phases = simple_finite()
        failed = request("bad", 1.2, [1.25], [1], status="failed", error="HTTP 400: prompt too long")
        result = status(make_run(reqs + [failed], phases, num_prompts=5))
        self.assertNotIn("V052_server_reachable", result["errors"])


class StatusSemantics(unittest.TestCase):
    def test_earlier_pass_then_later_fail(self):
        report = validate.Report()
        verdict = validate.check_status_lines(["S1 PASS 3 min first", "S2 PASS 1 min", "S1 FAIL 2 min rerun"], report)
        self.assertEqual(verdict, "FAIL")

    def test_skipped_phase_is_not_a_pass(self):
        report = validate.Report()
        verdict = validate.check_status_lines(["P1 PASS 193 min", "P2 PASS 0 min  SKIPPED, not run"], report)
        self.assertEqual(verdict, "SKIPPED")

    def test_no_verdicts(self):
        self.assertIsNone(validate.check_status_lines([], validate.Report()))


class CampaignLedger(unittest.TestCase):
    def test_duplicate_repetitions_are_rejected_and_counts_distinct(self):
        cells = [{"cell_id": "c", "candidate_id": "a", "block_id": 1}, {"cell_id": "c", "candidate_id": "a", "block_id": 2},
                 {"cell_id": "c", "candidate_id": "b", "block_id": 1}]
        m1 = {"cell_id": "c", "candidate": {"candidate_id": "a"}, "block_id": 1, "run_id": "r1"}
        m2 = {"cell_id": "c", "candidate": {"candidate_id": "a"}, "block_id": 1, "run_id": "r1-copy"}
        m3 = {"cell_id": "c", "candidate": {"candidate_id": "b"}, "block_id": 1, "run_id": "r3"}
        m4 = {"cell_id": "x", "candidate": {"candidate_id": "b"}, "block_id": 9, "run_id": "unplanned"}
        ledger = validate.validate_campaign(cells, [(m1, {"status": "measured"}), (m2, {"status": "measured"}),
                                                    (m3, {"status": "invalid", "errors": ["V013"]}), (m4, {"status": "measured"})])
        statuses = {(c["candidate_id"], c["block_id"]): c["status"] for c in ledger["cells"]}
        self.assertEqual(statuses[("a", 1)], "invalid")          # duplicate repetition
        self.assertEqual(statuses[("a", 2)], "not_run")
        self.assertEqual(statuses[("b", 1)], "invalid")
        self.assertEqual(ledger["counts"]["expected"], 3)
        self.assertEqual(ledger["counts"]["accepted"], 0)
        self.assertEqual(ledger["unexpected_runs"], ["unplanned"])

    def test_copied_rows_across_source_folders(self):
        rows = [{"source_artifact_sha256": "abc", "origin_experiment_id": "round-v2", "source_row_id": "v2/serving_fp32.log"},
                {"source_artifact_sha256": "abc", "origin_experiment_id": "round-v3", "source_row_id": "v3/serving_fp32.log"},
                {"source_artifact_sha256": "def", "origin_experiment_id": "round-v3", "source_row_id": "v3/serving_lq.log"}]
        copies = validate.detect_copied_rows(rows)
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0]["experiments"], ["round-v2", "round-v3"])


if __name__ == "__main__":
    unittest.main()
