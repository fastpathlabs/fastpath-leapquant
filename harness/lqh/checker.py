"""Independent aggregate checker.

  python -m lqh check BUNDLE_DIR [--tolerance 1e-7]

Reads the bundle's displayed_values.json and, for every value shown, recomputes it from the run's immutable raw
artifacts (requests.jsonl, stream_events.jsonl.gz, driver_summary.json, run_manifest.json, server_events.jsonl,
memory_and_power.jsonl) and the SLO / pricing registries. It deliberately shares no code with lqh.metrics,
lqh.records or lqh.analyze (a second implementation of the definitions). A displayed value that cannot be
recomputed is a failure, not a pass. Also verifies the bundle's MANIFEST.sha256 and the raw artifact hashes.
Exit status 0 only when every displayed value is reproduced.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path


def _strict(text):
    def bad(name):
        raise ValueError(f"non-finite JSON constant {name}")
    return json.loads(text, parse_constant=bad)


def _jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [_strict(line) for line in handle if line.strip()]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _quantile(values, q):
    """Linear interpolation between order statistics at rank q*(n-1) (the stated definition)."""
    data = sorted(values)
    if not data:
        return None
    position = q * (len(data) - 1)
    below = math.floor(position)
    above = math.ceil(position)
    if below == above:
        return data[int(position)]
    weight = position - below
    return data[below] * (1 - weight) + data[above] * weight


def _dist(values):
    data = list(values)
    if not data:
        return {"n": 0}
    n = len(data)
    mean = math.fsum(data) / n
    return {"n": n, "mean": mean, "std": math.sqrt(math.fsum((x - mean) ** 2 for x in data) / n),
            "min": min(data), "max": max(data), "median": _quantile(data, 0.5), "p50": _quantile(data, 0.5),
            "p90": _quantile(data, 0.9), "p95": _quantile(data, 0.95), "p99": _quantile(data, 0.99)}


def _div(a, b):
    return None if a is None or b in (None, 0) else a / b


def recompute(run_dir: Path, registries: Path) -> dict:
    """Every metric of one run, recomputed from raw files. Returns {metric: value or {field: value}}."""
    man = _strict((run_dir / "run_manifest.json").read_text())
    summary = _strict((run_dir / "driver_summary.json").read_text())
    reqs = _jsonl(run_dir / "requests.jsonl")
    series = {}
    if (run_dir / "stream_events.jsonl.gz").exists():
        for s in _jsonl(run_dir / "stream_events.jsonl.gz"):
            series[(s["request_id"], s.get("attempt", 1))] = s
    lane = man["workload"]["lane"]
    phases = summary.get("phases") or {}
    gpus = man["device"]["allocated_physical_gpus"]
    native = man.get("native") or {}
    out: dict = {}
    meas = [r for r in reqs if r["phase"] in ("measured", "sustained")]
    ok = [r for r in meas if r["status"] == "success"]

    def chunks(r):
        s = series.get((r["request_id"], r.get("attempt", 1))) or {}
        times = [r["t_send"] + d / 1e6 for d in (s.get("dt_us") or [])]
        toks = s.get("tokens")
        return times, (toks if toks is not None and len(toks) == len(times) else None)

    out["requests_expected"] = man["workload"].get("num_prompts") if lane == "finite" else None
    out["requests_attempted"] = len(meas)
    for status, key in (("success", "requests_succeeded"), ("failed", "requests_failed"), ("timeout", "requests_timeout"),
                        ("cancelled", "requests_cancelled"), ("incomplete", "requests_incomplete")):
        out[key] = sum(1 for r in meas if r["status"] == status)
    out["requests_with_first_token"] = sum(1 for r in meas if r.get("t_first_token") is not None)
    out["retries"] = sum(1 for r in meas if r.get("retry_of"))
    out["completion_rate"] = _div(out["requests_succeeded"], len(meas))
    out["error_rate"] = _div(len(meas) - out["requests_succeeded"], len(meas))
    out["first_token_response_rate"] = _div(out["requests_with_first_token"], len(meas))
    out["input_tokens_succeeded"] = sum(r.get("input_tokens") or 0 for r in ok)
    out["output_tokens_succeeded"] = sum(r.get("output_tokens") or 0 for r in ok)
    unsuccessful = 0
    for r in meas:
        if r["status"] != "success":
            times, toks = chunks(r)
            unsuccessful += sum(toks) if toks is not None else len(times)
    out["output_tokens_unsuccessful"] = unsuccessful
    out["warmup_requests"] = sum(1 for r in reqs if r["phase"] == "warmup")
    exact_eos = man["workload"].get("ignore_eos")
    out["output_length_mismatches"] = sum(1 for r in ok if exact_eos and r.get("requested_output_tokens") is not None
                                          and r.get("output_tokens") != r.get("requested_output_tokens"))
    out["requests_exact_chunk_tokens"] = sum(1 for r in ok if chunks(r)[1] is not None)
    duration = None
    if lane in ("finite", "open_loop"):
        if native.get("duration") is not None:
            duration = native["duration"]
        elif phases.get("main"):
            duration = phases["main"]["end"] - phases["main"]["start"]
        out["cohort_duration_s"] = duration
        out["finite_cohort_output_tps"] = _div(out["output_tokens_succeeded"], duration)
        out["finite_cohort_input_tps"] = _div(out["input_tokens_succeeded"], duration)
        out["finite_cohort_total_tps"] = _div(out["input_tokens_succeeded"] + out["output_tokens_succeeded"], duration)
        out["request_throughput_rps"] = _div(out["requests_succeeded"], duration)
        out["output_tps_per_gpu"] = _div(out["finite_cohort_output_tps"], gpus)
        out["input_tps_per_gpu"] = _div(out["finite_cohort_input_tps"], gpus)
        out["total_tps_per_gpu"] = _div(out["finite_cohort_total_tps"], gpus)
        main = phases.get("main")
        if main and all(r.get("clock_domain", "run") == "run" for r in meas) and main["end"] - main["start"] >= 10:
            events = []
            for r in meas:
                times, toks = chunks(r)
                events += list(zip(times, toks if toks is not None else [1] * len(times)))
            best, start = 0.0, main["start"]
            while start + 10 <= main["end"] + 1e-9:
                best = max(best, sum(n for t, n in events if start <= t < start + 10) / 10)
                start += 1
            out["output_event_tps_max_10s"] = best
    if lane == "open_loop":
        arrivals = phases.get("arrivals") or {}
        out["offered_rate_rps"] = arrivals.get("offered_rate_rps")
        out["attempted_rate_rps"] = _div(len(meas), arrivals["end"] - arrivals["start"])
        out["achieved_completion_rps"] = _div(out["requests_succeeded"], duration)
    # Per-request latency values.
    cohort = ok
    steady = phases.get("steady") if lane == "sustained" else None
    if steady:
        cohort = [r for r in ok if steady["start"] <= r["t_send"] < steady["end"]]
    vals = {k: [] for k in ("ttft", "e2e_t", "e2e_l", "tpot", "tpot_t", "span", "gap", "itl", "qd", "sp1", "sp2", "est")}
    excluded = 0
    for r in cohort:
        n = r.get("output_tokens")
        send, first, last, term = r["t_send"], r.get("t_first_token"), r.get("t_last_token"), r.get("t_terminal")
        if first is not None:
            vals["ttft"].append((first - send) * 1000)
        if term is not None:
            vals["e2e_t"].append((term - send) * 1000)
        if last is not None:
            vals["e2e_l"].append((last - send) * 1000)
        if r.get("t_scheduled") is not None:
            vals["qd"].append((send - r["t_scheduled"]) * 1000)
        if n and n > 1 and first is not None:
            if last is not None:
                vals["tpot"].append(((last - send) - (first - send)) / (n - 1) * 1000)
                vals["span"].append((last - first) / (n - 1) * 1000)
                if last > first:
                    vals["sp1"].append((n - 1) / (last - first))
            if term is not None:
                vals["tpot_t"].append(((term - send) - (first - send)) / (n - 1) * 1000)
        if n and n >= 1 and last is not None and last > send:
            vals["sp2"].append(n / (last - send))
        times, toks = chunks(r)
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        vals["gap"] += gaps
        if len(times) >= 2:
            if toks is not None and all(t == 1 for t in toks):
                vals["itl"] += gaps
            else:
                excluded += 1
                if toks is not None:
                    for g, k in zip(gaps, toks[1:]):
                        if k > 0:
                            vals["est"] += [g / k] * k
    for key, name in (("ttft", "ttft_ms"), ("e2e_t", "e2e_terminal_ms"), ("e2e_l", "e2e_last_token_ms"),
                      ("tpot", "tpot_native_ms"), ("tpot_t", "tpot_terminal_ms"), ("span", "span_last_first_ms"),
                      ("gap", "chunk_gap_ms"), ("itl", "itl_token_ms"), ("qd", "client_queue_delay_ms"),
                      ("sp1", "user_speed_post_first_token_tps"), ("sp2", "user_speed_e2e_tps")):
        out[name] = _dist(vals[key])
    if vals["est"]:
        out["itl_estimated_uniform_ms"] = _dist(vals["est"])
    med = out["tpot_native_ms"].get("median")
    out["interactivity_native_median_tps"] = 1000 / med if med else None
    # SLO.
    slo = None
    if man.get("slo_id"):
        for entry in _strict((registries / "slo.json").read_text())["slos"]:
            if entry["slo_id"] == man["slo_id"]:
                slo = entry
    good = set()
    if slo:
        rule = slo["request_predicate"]
        for r in meas:
            n = r.get("output_tokens")
            send, first, last, term = r["t_send"], r.get("t_first_token"), r.get("t_last_token"), r.get("t_terminal")
            fine = r["status"] == "success" or not rule.get("require_success", True)
            if rule.get("ttft_ms_max") is not None:
                fine = fine and first is not None and (first - send) * 1000 <= rule["ttft_ms_max"]
            if rule.get("tpot_ms_max") is not None:
                if n and n > 1 and first is not None and last is not None:
                    tp = ((last - send) - (first - send)) / (n - 1)
                elif n == 1:
                    tp = 0.0
                else:
                    tp = None
                fine = fine and tp is not None and tp * 1000 <= rule["tpot_ms_max"]
            if rule.get("e2e_ms_max") is not None:
                end = term if rule.get("e2e_metric", "e2e_terminal_ms") == "e2e_terminal_ms" else last
                fine = fine and end is not None and (end - send) * 1000 <= rule["e2e_ms_max"]
            if fine:
                good.add((r["request_id"], r.get("attempt", 1)))
        out["slo_qualifying_requests"] = len(good)
        out["slo_qualifying_fraction"] = _div(len(good), len(meas))
    if lane == "sustained" and steady:
        w0, w1 = steady["start"], steady["end"]
        window = w1 - w0
        tot = succ = failed = unres_t = unres = comp = firsts = before = after = span = q_tok = q_comp = 0
        for r in meas:
            times, toks = chunks(r)
            inside = sum((toks[i] if toks is not None else 1) for i, t in enumerate(times) if w0 <= t < w1)
            tot += inside
            key = (r["request_id"], r.get("attempt", 1))
            if r["status"] == "success":
                succ += inside
                if key in good:
                    q_tok += inside
            elif r["status"] == "incomplete":
                unres_t += inside
                unres += 1 if inside else 0
            else:
                failed += inside
            if r.get("t_first_token") is not None and w0 <= r["t_first_token"] < w1:
                firsts += 1
            end = r.get("t_terminal")
            if r["status"] == "success" and end is not None and w0 <= end < w1:
                comp += 1
                q_comp += 1 if key in good else 0
            if end is not None:
                before += r["t_send"] < w0 <= end < w1
                after += w0 <= r["t_send"] < w1 <= end
                span += r["t_send"] < w0 and end >= w1
        out.update({"steady_window_s": window, "steady_output_event_tps": tot / window,
                    "successful_steady_output_tps": succ / window, "steady_failed_request_tokens": failed,
                    "steady_unresolved_requests": unres, "steady_unresolved_tokens": unres_t,
                    "steady_completions": comp, "steady_request_rate_rps": comp / window,
                    "steady_first_token_responses": firsts, "steady_boundary_started_before": before,
                    "steady_boundary_ended_after": after, "steady_boundary_spanning": span,
                    "steady_cohort_requests": sum(1 for r in meas if w0 <= r["t_send"] < w1),
                    "output_tps_per_gpu": _div(succ / window, gpus)})
        if slo:
            out["request_goodput_rps"] = q_comp / window
            out["output_goodput_tps"] = q_tok / window
    elif slo:
        q_tokens = sum(r.get("output_tokens") or 0 for r in meas if (r["request_id"], r.get("attempt", 1)) in good)
        out["request_goodput_rps"] = _div(len(good), duration)
        out["output_goodput_tps"] = _div(q_tokens, duration)
    if slo:
        out["output_goodput_tps_per_gpu"] = _div(out.get("output_goodput_tps"), gpus)
    # Telemetry over the measurement interval (Unix time).
    origin = (man.get("clock") or {}).get("origin_unix")
    phase = steady if lane == "sustained" else phases.get("main")
    if origin is not None and phase:
        start, end = origin + phase["start"], origin + phase["end"]
    else:
        start, end = native.get("benchmark_start_time_unix"), native.get("benchmark_end_time_unix")
    events = _jsonl(run_dir / "server_events.jsonl") if (run_dir / "server_events.jsonl").exists() else []
    if start is not None:
        sel = [e for e in events if e.get("kind") in ("decode", "prefill", "retract") and start - 1 <= e["t_unix"] <= end + 1]
        decode = [e for e in sel if e["kind"] == "decode"]
        out["server_running_requests"] = _dist([e["running"] for e in decode if "running" in e])
        out["server_queue_depth"] = _dist([e["queue"] for e in sel if "queue" in e])
        out["server_token_usage"] = _dist([e["token_usage"] for e in sel if "token_usage" in e])
        out["server_retractions"] = sum(e.get("retracted", 0) for e in sel)
        out["server_cuda_graph_fraction"] = _div(sum(1 for e in decode if e.get("cuda_graph")), len(decode))
        out["server_reported_gen_tps"] = _dist([e["gen_tps"] for e in decode if "gen_tps" in e])
        samples = _jsonl(run_dir / "memory_and_power.jsonl") if (run_dir / "memory_and_power.jsonl").exists() else []
        gpu = [s for s in samples if s.get("kind") == "gpu_sample" and start <= s["t_unix"] <= end]
        for key, name in (("memory_used_mib", "gpu_memory_used_mib"), ("power_w", "gpu_power_w"),
                          ("sm_clock_mhz", "gpu_sm_clock_mhz"), ("temperature_c", "gpu_temperature_c")):
            out[name] = _dist([s[key] for s in gpu if s.get(key) is not None])
        totals = [s["memory_total_mib"] for s in samples if s.get("kind") == "gpu_sample" and s.get("memory_total_mib")]
        out["gpu_memory_total_mib"] = totals[0] if totals else None
    # GPU energy over the same interval (second implementation of the definition): power samples of the run's own
    # devices only, power linear between consecutive samples, no bridging of a gap over 2 s between samples, and at
    # least 95% of the interval covered on every device; otherwise null. Per token: the same interval's successful
    # output tokens (the steady window's in a sustained run) and SLO-qualifying output tokens.
    joules = None
    devices = sorted({g["uuid"] for g in (man.get("device") or {}).get("gpus", []) if g.get("uuid")})
    if start is not None and end is not None and end > start and devices:
        power = _jsonl(run_dir / "memory_and_power.jsonl") if (run_dir / "memory_and_power.jsonl").exists() else []
        joules = 0.0
        for dev in devices:
            pts = sorted((s["t_unix"], s["power_w"]) for s in power if s.get("gpu_uuid") == dev
                         and s.get("power_w") is not None and start - 2.0 <= s["t_unix"] <= end + 2.0)
            if len(pts) < 2 or not any(start <= t <= end for t, _ in pts):
                joules = None
                break
            seconds = dev_joules = 0.0
            for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
                lo, hi = max(ta, start), min(tb, end)
                if hi <= lo or tb - ta > 2.0:
                    continue
                slope = (wb - wa) / (tb - ta)
                dev_joules += (2 * wa + slope * ((lo - ta) + (hi - ta))) / 2 * (hi - lo)
                seconds += hi - lo
            if seconds < 0.95 * (end - start):
                joules = None
                break
            joules += dev_joules
    out["gpu_energy_j"] = joules
    if lane == "sustained" and steady:
        energy_tokens = round(out["successful_steady_output_tps"] * out["steady_window_s"])
    else:
        energy_tokens = out["output_tokens_succeeded"]
    out["gpu_energy_j_per_output_token"] = _div(joules, energy_tokens)
    qualified = None
    if slo:
        window = out.get("steady_window_s") if (lane == "sustained" and steady) else duration
        rate = out.get("output_goodput_tps")
        qualified = rate * window if rate is not None and window else None
    out["gpu_energy_j_per_qualified_output_token"] = _div(joules, qualified)
    # Cost.
    pricing = None
    reg = _strict((registries / "pricing.json").read_text())
    for entry in reg["pricing"]:
        if entry["pricing_id"] == (man.get("pricing_id") or reg.get("default_pricing_id")):
            pricing = entry
    hourly = None
    if pricing and pricing.get("currency") in (None, "USD"):
        if pricing.get("mode") == "per_gpu" and pricing.get("usd_per_gpu_hour") is not None and gpus:
            hourly = pricing["usd_per_gpu_hour"] * gpus + sum(x.get("usd_per_hour", 0) for x in pricing.get("separately_billed", []))
        elif pricing.get("mode") == "whole_instance" and pricing.get("usd_per_instance_hour") is not None:
            hourly = pricing["usd_per_instance_hour"] * man["device"].get("instances", 1) + \
                sum(x.get("usd_per_hour", 0) for x in pricing.get("separately_billed", []))
    out["usd_per_hour"] = hourly
    tps = out.get("successful_steady_output_tps") if lane == "sustained" else out.get("finite_cohort_output_tps")
    for name, rate in (("usd_per_million_output_tokens", tps), ("usd_per_million_qualified_output_tokens", out.get("output_goodput_tps")),
                       ("usd_per_million_input_tokens", None if lane == "sustained" else out.get("finite_cohort_input_tps")),
                       ("usd_per_million_total_tokens", None if lane == "sustained" else out.get("finite_cohort_total_tps"))):
        out[name] = None if hourly is None or not rate else hourly * 1e6 / (3600 * rate)
    # Service eligibility, re-derived here from this module's own numbers (not lqh.eligibility): attainment >= 0.95,
    # completion >= 0.99, the SLO's error-rate and p99 token-gap limits, at least one qualifying request.
    if slo and (slo.get("global_constraints") or {}).get("denominator") == "offered_logical_requests":
        # slo-interactive-v2: attainment and complete success over OFFERED logical requests; p99 gap not gated.
        limits = slo["global_constraints"]
        logical_ids = {r["request_id"] for r in meas}
        offered = max(int(out.get("requests_expected") or 0), len(logical_ids)) or None
        good_ids = {rid for rid, _ in good}
        done_ids = {r["request_id"] for r in ok}
        att = _div(len(good_ids), offered) if offered else None
        comp = _div(len(done_ids), offered) if offered else None
        passes = (limits.get("attainment_min") is None or (att is not None and att >= limits["attainment_min"])) \
            and (limits.get("complete_success_min") is None
                 or (comp is not None and comp >= limits["complete_success_min"])) \
            and (limits.get("error_rate_max") is None
                 or (out.get("error_rate") is not None and out["error_rate"] <= limits["error_rate_max"])) \
            and not out.get("retries") and len(good_ids) > 0
        out["service_eligible"] = 1.0 if passes else 0.0
    elif slo:
        limits = slo.get("global_constraints") or {}
        gap_name = limits.get("p99_token_gap_metric", "itl_token_ms")
        gap = (out.get(gap_name) or {}).get("p99") if isinstance(out.get(gap_name), dict) else None
        att, comp, err = out.get("slo_qualifying_fraction"), out.get("completion_rate"), out.get("error_rate")
        passes = (att is not None and att >= 0.95) and (comp is not None and comp >= 0.99) \
            and (limits.get("error_rate_max") is None or (err is not None and err <= limits["error_rate_max"])) \
            and (limits.get("p99_token_gap_ms_max") is None or (gap is not None and gap <= limits["p99_token_gap_ms_max"])) \
            and (out.get("slo_qualifying_requests") or 0) > 0
        out["service_eligible"] = 1.0 if passes else 0.0
    return out


# ---- evidence sections (kernel cycle, sanitizer, quality, NLL): independent re-derivation ----------------------
def _norm_q(p):
    """Standard normal quantile (Wichura AS241-style rational form, written separately from lqh.stats)."""
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r
                   + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r
                 + 133.14166789178437745) * r + 3.387132872796366608)
        den = (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r
                   + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r
                 + 42.313330701600911252) * r + 1.0)
        return q * num / den
    r = math.sqrt(-math.log(min(p, 1 - p)))
    if r <= 5:
        r -= 1.6
        num = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r
                   + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r
                 + 4.6303378461565452959) * r + 1.42343711074968357734)
        den = (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r
                   + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r
                 + 2.05319162663775882187) * r + 1.0)
    else:
        r -= 5
        num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r
                   + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r
                 + 5.4637849111641143699) * r + 6.6579046435011037772)
        den = (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r
                   + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r
                 + 0.59983220655588793769) * r + 1.0)
    value = num / den
    return -value if q < 0 else value


def _t_q(p, df):
    """Student t quantile: exact for df 1 and 2, Cornish-Fisher expansion otherwise (|error| < 1e-4 for df >= 3)."""
    if df == 1:
        return math.tan(math.pi * (p - 0.5))
    if df == 2:
        return (2 * p - 1) / math.sqrt(2 * p * (1 - p))
    z = _norm_q(p)
    g1 = (z ** 3 + z) / 4
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4


def _kernel_cycle(folder):
    batch_re, lq_re = re.compile(r"^\s*batch (\d+)\s*$"), re.compile(r"full cycle\s+LeapQuant: [0-9.]+ ms per window of \d+ tokens = ([0-9.]+) ms per token")
    bf_re = re.compile(r"full cycle\s+sglang window kernel, bf16 state: ([0-9.]+) ms per token")
    per_batch = {}
    for rep in sorted(Path(folder).glob("rep_*.txt")):
        batch = None
        seen = {}
        for line in rep.read_text(errors="replace").splitlines():
            m = batch_re.match(line)
            if m:
                batch = m.group(1)
                continue
            if batch is None:
                continue
            a, b = lq_re.search(line), bf_re.search(line)
            if a:
                seen.setdefault(batch, {})["lq"] = float(a.group(1))
            if b:
                seen.setdefault(batch, {})["bf"] = float(b.group(1))
        for batch, values in seen.items():
            if "lq" in values and "bf" in values:
                per_batch.setdefault(batch, []).append(math.log(values["bf"] / values["lq"]))
    out = {}
    for batch, logs in per_batch.items():
        n = len(logs)
        mean = math.fsum(logs) / n
        entry = {"geometric_mean": math.exp(mean)}
        if n >= 2:
            sd = math.sqrt(math.fsum((x - mean) ** 2 for x in logs) / (n - 1))
            half = _t_q(0.975, n - 1) * sd / math.sqrt(n)
            entry["low"], entry["high"] = math.exp(mean - half), math.exp(mean + half)
        out[batch] = entry
    return out


def _classify_sanitizer(log, tool):
    if not Path(log).exists():
        return "missing_log"
    text = Path(log).read_text(errors="replace")
    if tool == "racecheck":
        m = re.search(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)", text)
        if not m:
            return "interrupted"
        return "clean" if m.group(1) == m.group(2) == m.group(3) == "0" else "violations"
    m = re.search(r"ERROR SUMMARY: (\d+) error", text)
    if not m:
        return "interrupted"
    errors = int(m.group(1))
    if errors == 0:
        return "clean"
    api = len(re.findall(r"Program hit CUDA_ERROR_\w+ \(error \d+\) due to .* on CUDA API call to \w+", text))
    access = len(re.findall(r"Invalid (__global__|__shared__|__local__) (read|write)|Uninitialized __global__ memory read|out of bounds|misaligned", text))
    return "api_errors_only" if access == 0 and api == errors else "violations"


def _wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def _quality(folder, arm, task, reference="fp32"):
    def items(name):
        return [_strict(l) for l in (Path(folder) / f"quality_items_{name}.jsonl").read_text().splitlines() if l.strip()]
    def scored(i):     # the fixed-budget primary outcome: a capped answer is wrong, whatever it contains
        return {**i, "correct": bool(i["correct"]) and not i.get("capped")}
    mine = {i["sample_id"]: scored(i) for i in items(arm) if i["dataset"] == task}
    ref = {i["sample_id"]: scored(i) for i in items(reference) if i["dataset"] == task}
    ids = sorted(set(mine) & set(ref))
    n = len(ids)
    out = {"accuracy_arm": sum(mine[i]["correct"] for i in ids) / n, "accuracy_reference": sum(ref[i]["correct"] for i in ids) / n,
           "cap_rate_arm": sum(bool(mine[i].get("capped")) for i in ids) / n}
    out["difference"] = out["accuracy_arm"] - out["accuracy_reference"]
    groups = {}
    for i in ids:
        groups.setdefault(mine[i]["question"], []).append((mine[i]["correct"], ref[i]["correct"]))
    k = max(len(v) for v in groups.values())
    a = sum(1 for i in ids if mine[i]["correct"] and ref[i]["correct"])
    b = sum(1 for i in ids if mine[i]["correct"] and not ref[i]["correct"])
    c = sum(1 for i in ids if ref[i]["correct"] and not mine[i]["correct"])
    d = n - a - b - c
    diffs = [sum(x for x, _ in v) / len(v) - sum(y for _, y in v) / len(v) for v in groups.values()]
    q = len(diffs)
    m = math.fsum(diffs) / q
    if k > 1:
        var = math.fsum((x - m) ** 2 for x in diffs) / max(q - 1, 1)
        floor = math.fsum(((sum(x for x, _ in v) + sum(y for _, y in v) + 1) / (2 * len(v) + 2))
                          * (1 - (sum(x for x, _ in v) + sum(y for _, y in v) + 1) / (2 * len(v) + 2)) * (2 / len(v))
                          for v in groups.values()) / q
        half = _t_q(0.975, max(q - 1, 1)) * math.sqrt(max(var, floor) / q)
        out["low"], out["high"] = m - half, m + half
    else:
        p1, p2 = (a + b) / n, (a + c) / n
        l1, u1 = _wilson(a + b, n)
        l2, u2 = _wilson(a + c, n)
        den = (a + b) * (c + d) * (a + c) * (b + d)
        num = a * d - b * c
        if num > 0:
            num = max(num - n / 2, 0)
        phi = num / math.sqrt(den) if den else 0.0
        delta = p1 - p2
        out["low"] = max(-1.0, delta - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0)))
        out["high"] = min(1.0, delta + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0)))
    return out


def _nll_delta(reference_dir, candidate_dir, bucket):
    length, span = bucket[1:].split(":")
    lo, hi = (int(x) for x in span.split("-"))

    def per_doc(folder):
        out = {}
        for line in (Path(folder) / "nll_by_position.jsonl").read_text().splitlines():
            row = _strict(line)
            if str(row.get("length")) != length or "nll" not in row:
                continue
            values = [v for i, v in enumerate(row["nll"], start=1) if lo <= i < hi and v is not None]
            if values:
                out[row["doc_id"]] = math.fsum(values) / len(values)
        return out
    ref, mine = per_doc(reference_dir), per_doc(candidate_dir)
    docs = sorted(set(ref) & set(mine))
    return math.fsum(mine[d] - ref[d] for d in docs) / len(docs) if docs else None


def _carried_block(path):
    """'Name: number' lines of the last 'Serving Benchmark Result' block of a client log (re-parsed here)."""
    text = Path(path).read_text(errors="replace")
    block = text[text.rfind("Serving Benchmark Result"):]
    out = {}
    for line in block.splitlines()[1:]:
        line = line.strip()
        if out and line and set(line) == {"="}:
            break
        m = re.match(r"^([A-Za-z][A-Za-z0-9 /()%._-]{1,80}?):\s+(-?[0-9][0-9,]*\.?[0-9]*(?:[eE][-+]?\d+)?)\s*$", line)
        if m:
            out[m.group(1).strip()] = float(m.group(2).replace(",", ""))
    return out


def _carried_server(path, end, duration):
    """Max '#running-req' over decode lines and max '#queue-req' over decode/prefill lines inside
    [end - duration - 120, end + 5], with the host's UTC timestamps (one-second resolution, +-1 s)."""
    import calendar
    start, stop = end - duration - 120, end + 5
    running, waiting = [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[^\]]*\]\s*(Decode batch|Prefill batch)(.*)$", line)
        if not m:
            continue
        t = calendar.timegm(_time.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"))
        if t < start - 1 or t > stop + 1:
            continue
        r = re.search(r"#running-req: (\d+)", m.group(3))
        q = re.search(r"#queue-req: (\d+)", m.group(3))
        if r and m.group(2) == "Decode batch":
            running.append(int(r.group(1)))
        if q:
            waiting.append(int(q.group(1)))
    return {"admitted_running_max": max(running) if running else None, "waiting_max": max(waiting) if waiting else None}


def check_evidence(items, tolerance=1e-7):
    failed, checked = [], 0
    cache = {}
    for item in items:
        checked += 1
        kind = item["evidence"]
        folder = str(Path(item["file"]).parent)
        try:
            if kind == "kernel_cycle":
                values = cache.setdefault(("k", folder), _kernel_cycle(folder))
                expected = values.get(item["batch"], {}).get(item["field"])
                ok = expected is not None and math.isclose(item["value"], expected, rel_tol=1e-3 if item["field"] != "geometric_mean" else 1e-9)
            elif kind == "sanitizer":
                expected = _classify_sanitizer(item["log"], item["tool"])
                ok = expected == item["value"]
            elif kind == "quality":
                reference = _strict(Path(item["file"]).read_text()).get("reference_arm", "fp32")
                values = cache.setdefault(("q", folder, item["arm"], item["task"]),
                                          _quality(folder, item["arm"], item["task"], reference))
                expected = values.get(item["field"])
                tol = 1e-3 if item["field"] in ("low", "high") else 1e-9
                ok = expected is not None and math.isclose(item["value"], expected, rel_tol=tol, abs_tol=1e-9)
            elif kind == "carried_metric":
                values = cache.setdefault(("c", item["file"]), _carried_block(item["file"]))
                expected = values.get(item["field"])
                ok = expected is not None and math.isclose(float(item["value"]), expected, rel_tol=1e-12, abs_tol=1e-12)
            elif kind == "carried_server":
                expected = _carried_server(item["file"], item["log_end_unix"], item["duration"]).get(item["field"])
                ok = expected is not None and float(item["value"]) == float(expected)
            elif kind == "nll":
                doc = _strict(Path(item["file"]).read_text())
                expected = _nll_delta(doc["reference"], item["candidate"], item["bucket"])
                ok = expected is not None and math.isclose(item["value"], expected, rel_tol=1e-9, abs_tol=1e-12)
            else:
                expected, ok = None, False
        except (OSError, ValueError, KeyError, ZeroDivisionError) as exc:
            expected, ok = f"error: {exc}", False
        if not ok:
            failed.append({"evidence": kind, "file": item["file"], "field": item["field"], "shown": item["value"],
                           "recomputed": expected})
    return checked, failed


def check_bundle(bundle: Path, tolerance: float = 1e-7) -> dict:
    registries = Path(__file__).resolve().parents[1] / "config"
    report = {"bundle": str(bundle), "checked": 0, "failed": [], "uncovered": [], "hash_failures": []}
    manifest = bundle / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest, name = line.split("  ", 1)
            path = bundle / name
            if not path.exists() or _sha(path) != digest:
                report["hash_failures"].append(name)
    else:
        report["hash_failures"].append("MANIFEST.sha256 missing")
    displayed = _strict((bundle / "displayed_values.json").read_text())
    cache: dict = {}
    headline: dict = {}
    for item in displayed:
        if "run_dir" not in item:
            continue
        run_dir = Path(item["run_dir"])
        if run_dir not in cache:
            man = _strict((run_dir / "run_manifest.json").read_text())
            for name, meta in (man.get("artifacts") or {}).items():
                if _sha(run_dir / name) != meta["sha256"]:
                    report["hash_failures"].append(f"{run_dir}/{name}")
            cache[run_dir] = recompute(run_dir, registries)
        values = cache[run_dir]
        expected = values.get(item["metric"])
        if isinstance(expected, dict):
            expected = expected.get(item["field"])
        report["checked"] += 1
        if item["metric"] not in values:
            report["uncovered"].append(f"{item['run_id']}:{item['metric']}.{item['field']}")
            continue
        shown = item["value"]
        if expected is None or not math.isclose(float(shown), float(expected), rel_tol=tolerance, abs_tol=1e-9):
            report["failed"].append({"run": item["run_id"], "metric": item["metric"], "field": item["field"],
                                     "shown": shown, "recomputed": expected})
    evidence_items = [item for item in displayed if "evidence" in item]
    if evidence_items:
        count, problems = check_evidence(evidence_items, tolerance)
        report["checked"] += count
        report["failed"] += problems
    # Paired comparisons: recompute from the recomputed per-run headline throughput.
    for item in displayed:
        if "comparison" not in item:
            continue
        report["checked"] += 1
        plan, cell, pair = item["comparison"].split("|")
        cand, base = pair.split("/")
        by = {}
        for run_dir, values in cache.items():
            man = _strict((run_dir / "run_manifest.json").read_text())
            if man.get("plan_id") == plan and man.get("cell_id") == cell:
                lane = man["workload"]["lane"]
                tps = values.get("successful_steady_output_tps") if lane == "sustained" else values.get("finite_cohort_output_tps")
                by[(man["candidate"]["candidate_id"], man["block_id"])] = tps
        logs = [math.log(by[(cand, b)] / by[(base, b)]) for b in item.get("blocks") or []
                if by.get((cand, b)) and by.get((base, b))]
        expected = math.exp(math.fsum(logs) / len(logs)) if logs else None
        if expected is None or not math.isclose(item["value"], expected, rel_tol=tolerance):
            report["failed"].append({"comparison": item["comparison"], "shown": item["value"], "recomputed": expected})
    # Charts against tables: every plotted sidecar point must equal the recomputed run values.
    for sidecar_path in sorted((bundle / "figures").glob("*.json")) if (bundle / "figures").exists() else []:
        sidecar = _strict(sidecar_path.read_text())
        for point in sidecar.get("points", []):
            if not point.get("plotted"):
                continue
            run_dir = next((d for d in cache if _strict((d / "run_manifest.json").read_text()).get("run_id")
                            == point["run_id"]), None)
            report["checked"] += 1
            if run_dir is None:
                report["uncovered"].append(f"{sidecar_path.name}:{point['run_id']} (run not among displayed values)")
                continue
            values = cache[run_dir]
            y = values.get("output_tps_per_gpu")
            x = (values.get("user_speed_post_first_token_tps") or {}).get("median")
            for shown, expected, axis in ((point.get("y_output_tps_per_gpu"), y, "y"),
                                          (point.get("x_per_user_tps_median"), x, "x")):
                if expected is None or shown is None or not math.isclose(shown, expected, rel_tol=tolerance):
                    report["failed"].append({"figure": sidecar_path.name, "run": point["run_id"], "axis": axis,
                                             "shown": shown, "recomputed": expected})
    report["ok"] = (not report["failed"] and not report["uncovered"] and not report["hash_failures"]
                    and report["checked"] > 0)
    if report["checked"] == 0:
        report["note"] = "nothing was checked: a bundle without reproduced values is not a pass"
    report["checked_utc"] = datetime.fromtimestamp(_time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh check")
    parser.add_argument("bundle")
    parser.add_argument("--tolerance", type=float, default=1e-7)
    args = parser.parse_args(argv)
    result = check_bundle(Path(args.bundle), args.tolerance)
    (Path(args.bundle) / "check_report.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in result.items()}))
    for failure in result["failed"][:20]:
        print("MISMATCH", failure)
    for missing in result["uncovered"][:20]:
        print("UNCOVERED", missing)
    for bad in result["hash_failures"][:20]:
        print("HASH", bad)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
