"""Costing fixtures (section 12): two GPUs, whole-instance pricing, shared prefill/decode GPUs, unequal allocations,
zero qualified output, missing price, unit conversion and optional energy coverage."""
import unittest

import helpers  # noqa: F401  (puts the harness on sys.path)

from lqh import arith, cost


class Cost(unittest.TestCase):
    PER_GPU = {"pricing_id": "p", "mode": "per_gpu", "currency": "USD", "usd_per_gpu_hour": 2.0}
    INSTANCE = {"pricing_id": "i", "mode": "whole_instance", "currency": "USD", "usd_per_instance_hour": 20.0}

    def test_unit_conversion(self):
        hourly, reason, _ = cost.deployment_usd_per_hour(self.PER_GPU, {"allocated_physical_gpus": 1})
        value, why = cost.usd_per_million(hourly, 1000.0, reason)
        self.assertAlmostEqual(value, 2.0 * 1e6 / (3600 * 1000.0))
        self.assertIsNone(why)

    def test_two_gpus_per_gpu_rate(self):
        hourly, _, details = cost.deployment_usd_per_hour(self.PER_GPU, {"allocated_physical_gpus": 2})
        self.assertEqual(hourly, 4.0)
        self.assertEqual(details["allocated_gpus"], 2)

    def test_whole_instance_is_not_multiplied_by_gpus(self):
        hourly, _, _ = cost.deployment_usd_per_hour(self.INSTANCE, {"allocated_physical_gpus": 8, "instances": 1})
        self.assertEqual(hourly, 20.0)

    def test_shared_prefill_decode_gpus_charge_the_union(self):
        # 2 prefill + 2 decode GPUs, one of them shared: 3 physical GPUs allocated, all charged.
        physical = {"prefill": {"g0", "g1"}, "decode": {"g1", "g2"}}
        allocated = len(physical["prefill"] | physical["decode"])
        hourly, _, _ = cost.deployment_usd_per_hour(self.PER_GPU, {"allocated_physical_gpus": allocated})
        self.assertEqual(hourly, 6.0)

    def test_unequal_allocations_use_the_full_cost_ratio(self):
        # candidate: 2 GPUs at 1,500 tok/s; baseline: 1 GPU at 1,000 tok/s. Speedup 1.5x, but cost ratio 1.333.
        ratio = cost.full_cost_ratio(4.0, 1500.0, 2.0, 1000.0)
        self.assertAlmostEqual(ratio, (4.0 / 1500.0) / (2.0 / 1000.0))
        self.assertGreater(ratio, 1.0)

    def test_zero_qualified_output_is_null_not_zero(self):
        value, why = cost.usd_per_million(2.0, 0.0, None)
        self.assertIsNone(value)
        self.assertEqual(why, "zero_denominator")

    def test_missing_price(self):
        block = cost.cost_block({"pricing_id": "x", "mode": "per_gpu", "currency": "USD", "usd_per_gpu_hour": None},
                                {"allocated_physical_gpus": 1}, 1000.0, 900.0, 5000.0, 6000.0)
        for key in ("usd_per_hour", "usd_per_million_output_tokens", "usd_per_million_qualified_output_tokens"):
            self.assertIsNone(block[key]["value"])
            self.assertEqual(block[key]["null_reason"], "price_missing")
        self.assertIn("alternative_view", block["usd_per_million_total_tokens"])

    def test_api_price_is_not_infrastructure_cost(self):
        hourly, reason, _ = cost.deployment_usd_per_hour({"pricing_id": "a", "mode": "api_price", "currency": "USD"},
                                                         {"allocated_physical_gpus": 1})
        self.assertIsNone(hourly)
        self.assertEqual(reason, "api_price_is_not_infrastructure_cost")

    def test_speedup_and_cost(self):
        out = cost.speedup_to_cost(1.25)
        self.assertAlmostEqual(out["cost_ratio"], 0.8)
        self.assertAlmostEqual(out["cost_reduction"], 0.2)
        self.assertAlmostEqual(out["throughput_improvement_pct"], 25.0)

    def test_energy_coverage(self):
        uuid = "GPU-0"
        samples = [{"t_unix": 100.0 + 0.5 * i, "power_w": 300.0, "gpu_uuid": uuid} for i in range(21)]
        joules, reason, _ = cost.energy(samples, 100.0, 110.0, {uuid})
        self.assertAlmostEqual(joules, 3000.0)
        self.assertIsNone(reason)
        sparse = [s for i, s in enumerate(samples) if i < 5]            # covers only the first 2 s
        joules, reason, _ = cost.energy(sparse, 100.0, 110.0, {uuid})
        self.assertIsNone(joules)
        self.assertEqual(reason, "power_coverage_insufficient")
        foreign = [dict(s, gpu_uuid="GPU-other") for s in samples]
        joules, reason, _ = cost.energy(foreign, 100.0, 110.0, {uuid})
        self.assertIsNone(joules)
        self.assertEqual(reason, "power_samples_missing")


class Arithmetic(unittest.TestCase):
    def test_time_reduction_is_not_speed_increase(self):
        out = arith.time_and_speed(0.3821, arith.conditional_cycle(0.2524, 1.27))
        self.assertAlmostEqual(arith.conditional_cycle(0.2524, 1.27), 0.331775)
        self.assertAlmostEqual(out["speedup_x"], 1.15168, places=5)
        self.assertAlmostEqual(out["time_reduction"], 0.1317, places=4)
        self.assertNotAlmostEqual(out["time_reduction"], out["speed_increase"], places=3)

    def test_byte_reduction_is_not_capacity_increase(self):
        out = arith.bytes_and_capacity(2.0, 1.19)
        self.assertAlmostEqual(out["byte_reduction"], 0.405)
        self.assertAlmostEqual(out["ideal_capacity_x"], 1.6807, places=4)
        self.assertAlmostEqual(1310 / 1100 - 1, 0.1909, places=4)

    def test_fold_budget(self):
        self.assertAlmostEqual(arith.fold_budget(0.3821 * 0.8, 0.2524), 0.85248, places=5)


if __name__ == "__main__":
    unittest.main()
