"""GPU energy (15 Nov 2026): the independent checker re-derives gpu_energy_j and both per-token energies from the raw
power samples as lqh.analyze computes them. The first real runs with power samples (job 113) published energy that the
checker could not recompute, so every report after it failed its check as 'uncovered'."""
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import GPU, make_run, request, simple_finite  # noqa: E402
from lqh import analyze, checker  # noqa: E402

REGISTRIES = Path(__file__).resolve().parents[1] / "config"
ORIGIN = 1_000_000.0          # helpers.make_run: clock origin_unix
NAMES = ("gpu_energy_j", "gpu_energy_j_per_output_token", "gpu_energy_j_per_qualified_output_token")


def write_power(run, t0, t1, step=0.5, watts=lambda t: 200.0 + 50.0 * t, uuid=GPU["uuid"], hole=None):
    """Samples every `step` s on the run clock from t0 to t1, power linear in t (so the trapezoid rule is exact);
    `hole` = (a, b) leaves out the samples strictly inside it."""
    rows, k = [], 0
    while t0 + k * step <= t1 + 1e-9:
        t = t0 + k * step
        if hole is None or not (hole[0] < t < hole[1]):
            rows.append({"kind": "gpu_sample", "t_unix": ORIGIN + t, "gpu_index": 0, "gpu_uuid": uuid,
                         "power_w": watts(t), "memory_used_mib": 1000.0, "memory_total_mib": 81559.0})
        k += 1
    (run / "memory_and_power.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def exact_joules(a, b):
    """Integral of 200 + 50 t over [a, b]."""
    return 200.0 * (b - a) + 25.0 * (b * b - a * a)


class Energy(unittest.TestCase):
    def both(self, run):
        shown = analyze.compute(run)["metrics"]
        again = checker.recompute(run, REGISTRIES)
        for name in NAMES:
            self.assertIn(name, again, f"the checker does not cover {name}")
            a, b = shown[name]["value"], again[name]
            if a is None or b is None:
                self.assertIsNone(a, name)
                self.assertIsNone(b, name)
            else:
                self.assertTrue(math.isclose(a, b, rel_tol=1e-12), f"{name}: analyze {a}, checker {b}")
        return shown

    def test_finite_run(self):
        reqs, phases = simple_finite(n=6, tokens_per_request=40)
        run = make_run(reqs, phases)
        write_power(run, 0.0, phases["main"]["end"] + 1.0)
        m = self.both(run)
        start, end = phases["main"]["start"], phases["main"]["end"]
        # Unix times near 1e6 s carry ~1e-10 s of rounding: exact to about 1e-9 relative, not to 9 decimal places.
        self.assertTrue(math.isclose(m["gpu_energy_j"]["value"], exact_joules(start, end), rel_tol=1e-9))
        self.assertTrue(math.isclose(m["gpu_energy_j_per_output_token"]["value"], exact_joules(start, end) / 240,
                                     rel_tol=1e-9))

    def test_finite_run_with_slo(self):
        for slo_id in ("slo-batch-v1", "slo-interactive-v1"):
            reqs, phases = simple_finite(n=6, tokens_per_request=40)
            run = make_run(reqs, phases, slo_id=slo_id)
            write_power(run, 0.0, phases["main"]["end"] + 1.0)
            self.both(run)

    def test_sustained_run_uses_the_steady_window(self):
        w0, w1 = 10.0, 20.0
        reqs = [request("s0", 8.0, [9.0, 9.5, 10.5, 11.0], [1, 1, 1, 1], phase="sustained", t_terminal=11.1),
                request("s1", 12.0, [12.5, 13.0], [1, 2], phase="sustained", t_terminal=13.1),
                request("s2", 18.0, [19.0, 19.5, 20.5], [1, 1, 1], phase="sustained", t_terminal=21.0),
                request("s3", 9.0, [9.5, 15.0, 25.0], [1, 1, 1], phase="sustained", t_terminal=26.0)]
        phases = {"ramp": {"start": 0.0, "end": w0}, "steady": {"start": w0, "end": w1}, "drain": {"start": w1, "end": 30.0}}
        run = make_run(reqs, phases, lane="sustained")
        write_power(run, 0.0, 30.0)
        m = self.both(run)
        self.assertTrue(math.isclose(m["gpu_energy_j"]["value"], exact_joules(w0, w1), rel_tol=1e-9))
        in_window = round(m["successful_steady_output_tps"]["value"] * (w1 - w0))
        self.assertTrue(math.isclose(m["gpu_energy_j_per_output_token"]["value"], exact_joules(w0, w1) / in_window,
                                     rel_tol=1e-9))

    def test_gap_in_the_samples_nulls_energy(self):
        reqs, phases = simple_finite(n=4)
        phases = {"main": {"start": 1.0, "end": 21.0}}       # a 20 s cohort; a 4 s hole leaves < 95% covered
        run = make_run(reqs, phases)
        write_power(run, 0.0, 22.0, hole=(5.0, 9.0))
        m = self.both(run)
        self.assertIsNone(m["gpu_energy_j"]["value"])

    def test_samples_of_another_device_do_not_count(self):
        reqs, phases = simple_finite(n=4)
        run = make_run(reqs, phases)
        write_power(run, 0.0, phases["main"]["end"] + 1.0, uuid="GPU-someone-else")
        m = self.both(run)
        self.assertIsNone(m["gpu_energy_j"]["value"])


if __name__ == "__main__":
    unittest.main()
