"""Serving cost on the same resource and time boundary as throughput.

USD per million successful output tokens = deployment USD/hour x 1,000,000 / (3,600 x successful output tok/s).
A missing price gives null dollar metrics (reason "price_missing"); a zero denominator gives null with reason
"zero_denominator" (never 0, never JSON infinity). Cost modes (pricing.json):
  whole_instance  instances x full hourly instance rate (+ separately billed extras). Never multiplied by GPU count.
  per_gpu         allocated GPUs x per-GPU rate + separately billed shared resources.
  ownership       a MODELED scenario (amortisation, power, cooling, utilisation, maintenance); labelled modeled.
  api_price       a customer tariff (input, cached input, output); not an infrastructure serving cost.
"""
from __future__ import annotations

from typing import Any

from .metrics import scalar
from .util import harness_root, read_json

DEFAULT_PRICING_FILE = harness_root() / "config" / "pricing.json"


def load_pricing(pricing_id: str | None, path=None) -> dict | None:
    registry = read_json(path or DEFAULT_PRICING_FILE)
    if pricing_id is None:
        pricing_id = registry.get("default_pricing_id")
    for entry in registry["pricing"]:
        if entry["pricing_id"] == pricing_id:
            return entry
    return None


def deployment_usd_per_hour(pricing: dict | None, allocation: dict) -> tuple[float | None, str | None, dict]:
    """(USD per hour of the whole allocated deployment, null reason, details)."""
    if pricing is None:
        return None, "price_missing", {}
    mode = pricing.get("mode")
    details: dict[str, Any] = {"mode": mode, "pricing_id": pricing.get("pricing_id"),
                               "currency": pricing.get("currency"), "source": pricing.get("source"),
                               "date": pricing.get("date"), "region": pricing.get("region")}
    if pricing.get("currency") not in (None, "USD"):
        return None, "currency_not_usd", details
    gpus = allocation.get("allocated_physical_gpus")
    if mode == "whole_instance":
        rate, count = pricing.get("usd_per_instance_hour"), allocation.get("instances", 1)
        if rate is None:
            return None, "price_missing", details
        extras = sum(item.get("usd_per_hour", 0.0) for item in pricing.get("separately_billed", []))
        details.update(instances=count, usd_per_instance_hour=rate, extras_usd_per_hour=extras)
        return rate * count + extras, None, details
    if mode == "per_gpu":
        rate = pricing.get("usd_per_gpu_hour")
        if rate is None:
            return None, "price_missing", details
        if not gpus:
            return None, "gpu_count_unknown", details
        shared = sum(item.get("usd_per_hour", 0.0) for item in pricing.get("separately_billed", []))
        details.update(allocated_gpus=gpus, usd_per_gpu_hour=rate, shared_usd_per_hour=shared)
        return rate * gpus + shared, None, details
    if mode == "ownership":
        model = pricing.get("ownership_model") or {}
        needed = ("hardware_usd_per_gpu", "amortization_hours", "utilization")
        if any(model.get(key) is None for key in needed) or not gpus:
            return None, "ownership_inputs_missing", details
        hourly = model["hardware_usd_per_gpu"] * gpus / model["amortization_hours"]
        hourly += (model.get("power_kw_per_gpu", 0.0) * gpus * model.get("pue", 1.0)
                   * model.get("usd_per_kwh", 0.0))
        hourly += model.get("maintenance_usd_per_gpu_hour", 0.0) * gpus
        hourly /= model["utilization"]
        details.update(modeled=True, ownership_model=model)
        return hourly, None, details
    if mode == "api_price":
        return None, "api_price_is_not_infrastructure_cost", details
    return None, "unknown_cost_mode", details


def usd_per_million(usd_per_hour: float | None, tokens_per_second: float | None,
                    price_reason: str | None) -> tuple[float | None, str | None]:
    if usd_per_hour is None:
        return None, price_reason or "price_missing"
    if tokens_per_second is None:
        return None, "throughput_missing"
    if tokens_per_second <= 0:
        return None, "zero_denominator"
    return usd_per_hour * 1_000_000.0 / (3600.0 * tokens_per_second), None


def cost_block(pricing: dict | None, allocation: dict, output_tps: float | None, qualified_tps: float | None,
               input_tps: float | None, total_tps: float | None) -> dict[str, dict]:
    hourly, reason, details = deployment_usd_per_hour(pricing, allocation)
    out = {"usd_per_hour": scalar(hourly, "USD/h", reason or "price_missing", pricing=details)}
    for key, tps in (("usd_per_million_output_tokens", output_tps),
                     ("usd_per_million_qualified_output_tokens", qualified_tps),
                     ("usd_per_million_input_tokens", input_tps),
                     ("usd_per_million_total_tokens", total_tps)):
        value, why = usd_per_million(hourly, tps, reason)
        entry = scalar(value, "USD/1M tok", why or "")
        if key in ("usd_per_million_input_tokens", "usd_per_million_total_tokens"):
            entry["alternative_view"] = "whole deployment cost attributed to this denominator; not additive"
        if details.get("modeled"):
            entry["modeled"] = True
        out[key] = entry
    return out


def api_price_cost(tariff: dict, usage: dict) -> tuple[float | None, str | None]:
    """Customer price of measured usage under an API tariff (USD per million tokens of each kind)."""
    total = 0.0
    for kind in ("input", "cached_input", "output"):
        tokens = usage.get(f"{kind}_tokens", 0)
        if tokens and tariff.get(f"usd_per_million_{kind}") is None:
            return None, f"tariff_missing_{kind}"
        total += tokens * (tariff.get(f"usd_per_million_{kind}") or 0.0) / 1_000_000.0
    return total, None


def speedup_to_cost(speedup: float | None) -> dict:
    """At the same hourly price: cost ratio 1/s, reduction 1 - 1/s; also the percent throughput improvement."""
    if speedup is None or speedup <= 0:
        return {"speedup_x": speedup, "cost_ratio": None, "cost_reduction": None,
                "throughput_improvement_pct": None, "null_reason": "speedup_missing"}
    return {"speedup_x": speedup, "cost_ratio": 1.0 / speedup, "cost_reduction": 1.0 - 1.0 / speedup,
            "throughput_improvement_pct": (speedup - 1.0) * 100.0}


def full_cost_ratio(candidate_usd_per_hour: float, candidate_tps: float,
                    baseline_usd_per_hour: float, baseline_tps: float) -> float | None:
    """Cost ratio when GPU count or hourly cost differ between the arms (not just 1/speedup)."""
    if min(candidate_tps, baseline_tps) <= 0:
        return None
    return (candidate_usd_per_hour / candidate_tps) / (baseline_usd_per_hour / baseline_tps)


def energy(samples: list[dict], start_utc: float, end_utc: float, device_uuids: set[str],
           max_gap_s: float = 2.0, min_coverage: float = 0.95) -> tuple[float | None, str | None, dict]:
    """GPU energy (J) over [start, end] (Unix seconds) by the trapezoid rule, from power samples of the allocated
    devices only. Null when samples are missing, from another device, too sparse, or cover too little."""
    info: dict[str, Any] = {"devices": sorted(device_uuids), "max_gap_s": max_gap_s, "min_coverage": min_coverage}
    if end_utc <= start_utc:
        return None, "empty_interval", info
    total = 0.0
    for uuid in sorted(device_uuids):
        points = sorted((s["t_unix"], s["power_w"]) for s in samples
                        if s.get("gpu_uuid") == uuid and s.get("power_w") is not None
                        and start_utc - max_gap_s <= s["t_unix"] <= end_utc + max_gap_s)
        foreign = {s.get("gpu_uuid") for s in samples if s.get("power_w") is not None} - device_uuids
        if foreign:
            info["foreign_devices_ignored"] = sorted(str(f) for f in foreign)
        inside = [p for p in points if start_utc <= p[0] <= end_utc]
        if len(points) < 2 or not inside:
            return None, "power_samples_missing", info
        covered = 0.0
        joules = 0.0
        for (t0, p0), (t1, p1) in zip(points, points[1:]):
            a, b = max(t0, start_utc), min(t1, end_utc)
            if b <= a:
                continue
            if t1 - t0 > max_gap_s:
                continue
            covered += b - a
            # Linear power between the samples, integrated over the clipped part.
            pa = p0 + (p1 - p0) * (a - t0) / (t1 - t0)
            pb = p0 + (p1 - p0) * (b - t0) / (t1 - t0)
            joules += (pa + pb) / 2.0 * (b - a)
        coverage = covered / (end_utc - start_utc)
        info.setdefault("coverage", {})[uuid] = coverage
        if coverage < min_coverage:
            return None, "power_coverage_insufficient", info
        total += joules
    if not device_uuids:
        return None, "device_identity_unknown", info
    info["gpu_only"] = True
    return total, None, info
