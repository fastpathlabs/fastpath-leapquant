"""Token and timing accounting fixtures (prompt rev. 3, section 12): tokens spanning profile boundaries, single-token
responses, multi-token chunks, failures after partial output, cancellation, timeouts and drain cutoffs."""
import random
import unittest

from helpers import make_run, request, simple_finite

from lqh import analyze, metrics
from lqh.records import load_views


class FiniteCohort(unittest.TestCase):
    def test_throughput_counts_successful_tokens_over_main_phase(self):
        reqs, phases = simple_finite(n=4, tokens_per_request=8)
        run = make_run(reqs, phases)
        doc = analyze.compute(run)
        m = doc["metrics"]
        duration = phases["main"]["end"] - phases["main"]["start"]
        self.assertEqual(doc["status"], "measured")
        self.assertEqual(m["output_tokens_succeeded"]["value"], 32)
        self.assertAlmostEqual(m["finite_cohort_output_tps"]["value"], 32 / duration)
        self.assertAlmostEqual(m["finite_cohort_input_tps"]["value"], 400 / duration)
        self.assertAlmostEqual(m["finite_cohort_total_tps"]["value"], 432 / duration)

    def test_warmup_is_excluded(self):
        reqs, phases = simple_finite(n=3)
        warm = request("w0", 0.1, [0.15, 0.16], [1, 1], phase="warmup")
        phases["warmup"] = {"start": 0.1, "end": 0.2}
        run = make_run(reqs + [warm], phases, num_prompts=3)
        m = analyze.compute(run)["metrics"]
        self.assertEqual(m["requests_attempted"]["value"], 3)
        self.assertEqual(m["warmup_requests"]["value"], 1)
        self.assertEqual(m["output_tokens_succeeded"]["value"], 24)

    def test_failure_after_partial_output_is_never_credited(self):
        reqs, phases = simple_finite(n=3)
        bad = request("m9", 1.0, [1.05, 1.06, 1.07], [1, 1, 1], status="failed", error="stream error")
        run = make_run(reqs + [bad], phases, num_prompts=4)
        m = analyze.compute(run)["metrics"]
        self.assertEqual(m["output_tokens_succeeded"]["value"], 24)
        self.assertEqual(m["output_tokens_unsuccessful"]["value"], 3)
        self.assertEqual(m["requests_failed"]["value"], 1)
        self.assertAlmostEqual(m["error_rate"]["value"], 0.25)
        self.assertAlmostEqual(m["first_token_response_rate"]["value"], 1.0)   # a first token is not a success
        self.assertAlmostEqual(m["completion_rate"]["value"], 0.75)

    def test_timeout_and_cancellation_counts(self):
        reqs, phases = simple_finite(n=2)
        timeout = request("m7", 1.0, [1.05], [1], status="timeout")
        cancel = request("m8", 1.0, [], None, status="cancelled")
        m = analyze.compute(make_run(reqs + [timeout, cancel], phases, num_prompts=4))["metrics"]
        self.assertEqual(m["requests_timeout"]["value"], 1)
        self.assertEqual(m["requests_cancelled"]["value"], 1)
        self.assertEqual(m["output_tokens_unsuccessful"]["value"], 1)


class LatencyDefinitions(unittest.TestCase):
    def test_single_token_response(self):
        record = request("m0", 1.0, [1.2], [1])
        run = make_run([record], {"main": {"start": 1.0, "end": 1.3}})
        m = analyze.compute(run)["metrics"]
        self.assertEqual(m["tpot_native_ms"]["n"], 0)          # N <= 1: TPOT undefined, excluded
        self.assertEqual(m["span_last_first_ms"]["n"], 0)
        self.assertAlmostEqual(m["ttft_ms"]["p50"], 200.0)
        self.assertAlmostEqual(m["user_speed_e2e_tps"]["p50"], 1 / 0.2)

    def test_native_tpot_and_terminal_variant(self):
        chunks = [1.1, 1.2, 1.3, 1.4]
        record = request("m0", 1.0, chunks, [1, 1, 1, 1], t_terminal=1.5)
        m = analyze.compute(make_run([record], {"main": {"start": 1.0, "end": 1.5}}))["metrics"]
        self.assertAlmostEqual(m["tpot_native_ms"]["p50"], (0.4 - 0.1) / 3 * 1000)     # last content chunk
        self.assertAlmostEqual(m["tpot_terminal_ms"]["p50"], (0.5 - 0.1) / 3 * 1000)   # terminal response
        self.assertAlmostEqual(m["span_last_first_ms"]["p50"], m["tpot_native_ms"]["p50"])
        self.assertAlmostEqual(m["e2e_terminal_ms"]["p50"], 500.0)
        self.assertAlmostEqual(m["e2e_last_token_ms"]["p50"], 400.0)

    def test_multi_token_chunks_are_not_zero_gaps(self):
        chunks = [1.1, 1.2, 1.3]
        record = request("m0", 1.0, chunks, [1, 3, 2])
        m = analyze.compute(make_run([record], {"main": {"start": 1.0, "end": 1.4}}))["metrics"]
        self.assertEqual(m["itl_token_ms"]["n"], 0)                 # not exact: excluded and counted
        self.assertEqual(m["itl_token_ms"]["requests_excluded_multi_token_chunks"], 1)
        self.assertEqual(m["chunk_gap_ms"]["n"], 2)                 # one gap per chunk, never zero-gap tokens
        self.assertTrue(m["itl_estimated_uniform_ms"].get("estimate"))
        self.assertEqual(m["itl_estimated_uniform_ms"]["n"], 5)
        self.assertGreater(m["itl_estimated_uniform_ms"]["min"], 0)

    def test_percentile_matches_numpy_linear(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("numpy not installed")
        rng = random.Random(3)
        values = sorted(rng.random() for _ in range(1001))
        for p in (50, 90, 95, 99):
            self.assertAlmostEqual(metrics.percentile(values, p), float(numpy.percentile(values, p)), places=12)


class SteadyWindow(unittest.TestCase):
    def build(self):
        w0, w1 = 10.0, 20.0
        reqs = [
            request("s0", 8.0, [9.0, 9.5, 10.5, 11.0], [1, 1, 1, 1], phase="sustained", t_terminal=11.1),   # crosses start
            request("s1", 12.0, [12.5, 13.0], [1, 2], phase="sustained", t_terminal=13.1),                  # inside, 2-token chunk
            request("s2", 18.0, [19.0, 19.5, 20.5], [1, 1, 1], phase="sustained", t_terminal=21.0),          # crosses end
            request("s3", 9.0, [9.5, 15.0, 25.0], [1, 1, 1], phase="sustained", t_terminal=26.0),            # spans the window
            request("s4", 14.0, [14.5, 15.5], [1, 1], phase="sustained", status="failed", t_terminal=15.6),
            request("s5", 19.0, [19.5, 25.0], [1, 1], phase="sustained", status="incomplete"),               # drain cutoff
        ]
        phases = {"ramp": {"start": 0.0, "end": w0}, "steady": {"start": w0, "end": w1},
                  "drain": {"start": w1, "end": 30.0}}
        return make_run(reqs, phases, lane="sustained"), (w0, w1)

    def test_only_in_window_tokens_count(self):
        run, (w0, w1) = self.build()
        m = analyze.compute(run)["metrics"]
        # in-window tokens: s0 2, s1 3, s2 2, s3 1, s4 2 (failed), s5 1 (incomplete) = 11
        self.assertAlmostEqual(m["steady_output_event_tps"]["value"], 11 / 10)
        self.assertAlmostEqual(m["successful_steady_output_tps"]["value"], 8 / 10)
        self.assertEqual(m["steady_failed_request_tokens"]["value"], 2)
        self.assertEqual(m["steady_unresolved_tokens"]["value"], 1)
        self.assertEqual(m["steady_unresolved_requests"]["value"], 1)
        self.assertEqual(m["steady_boundary_started_before"]["value"], 1)    # s0
        self.assertEqual(m["steady_boundary_ended_after"]["value"], 1)       # s2
        self.assertEqual(m["steady_boundary_spanning"]["value"], 1)          # s3
        self.assertEqual(m["steady_completions"]["value"], 2)                # s0, s1 succeeded inside
        self.assertIsNone(m["finite_cohort_output_tps"]["value"])            # no finite cohort in this lane

    def test_relative_clock_has_no_steady_window(self):
        views = load_views(self.build()[0])
        for v in views:
            v.clock_domain = "request_relative"
        out, _ = metrics.steady_window(views, {"steady": {"start": 10.0, "end": 20.0}}, 1, None)
        self.assertEqual(out["steady_output_event_tps"]["null_reason"], "absolute_timestamps_unavailable")


if __name__ == "__main__":
    unittest.main()
