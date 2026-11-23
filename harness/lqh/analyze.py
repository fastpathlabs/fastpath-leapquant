"""Turns one validated run directory into metrics.json: every metric, with units, aggregation, provenance and null
reasons, plus the native upstream aggregates and the site-defined metrics where a profile has them.

Order: validation first (lqh.validate). An invalid or crashed run still gets a metrics file, whose metrics are all
null with reason "run_<status>", so the report can show it as a failed point instead of omitting it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import cost as cost_mod
from . import metrics as m
from . import slo as slo_mod
from .records import load_views
from .telemetry import parse_server_log, summarise_gpu, summarise_server
from .util import read_json, sha256_file, utc_now, write_json
from .validate import validate_run

HARNESS_VERSION = "lqh 1.0.0 (2026-11-15)"
INTERACTIVITY_THRESHOLDS = [30.0, 50.0, 75.0, 100.0]


def _interval(manifest: dict, summary: dict) -> tuple[float | None, float | None, str]:
    """The measurement interval on the Unix clock: finite/open-loop main phase, or the steady window."""
    origin = (manifest.get("clock") or {}).get("origin_unix")
    phases = summary.get("phases") or {}
    lane = manifest["workload"]["lane"]
    phase = phases.get("steady") if lane == "sustained" else phases.get("main")
    if origin is None or not phase or phase.get("start") is None or phase.get("end") is None:
        native = manifest.get("native") or {}
        if native.get("benchmark_start_time_unix") and native.get("benchmark_end_time_unix"):
            return native["benchmark_start_time_unix"], native["benchmark_end_time_unix"], "client_reported_unix"
        return None, None, "interval_unknown"
    return origin + phase["start"], origin + phase["end"], "steady" if lane == "sustained" else "main"


def compute(run_dir: str | Path, slo_registry=None, pricing_registry=None, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    validation = validate_run(run_dir, write=write)
    status = validation["status"]
    manifest = read_json(run_dir / "run_manifest.json") if (run_dir / "run_manifest.json").exists() else {}
    doc: dict[str, Any] = {
        "schema": "lqh.metrics.v1", "harness_version": HARNESS_VERSION, "computed_utc": utc_now(),
        "run_id": manifest.get("run_id"), "campaign_id": manifest.get("campaign_id"),
        "experiment_id": manifest.get("experiment_id"), "cell_id": manifest.get("cell_id"),
        "plan_id": manifest.get("plan_id"), "block_id": manifest.get("block_id"),
        "pair_id": manifest.get("pair_id"), "profile_id": manifest.get("profile_id"),
        "profile_version": manifest.get("profile_version"),
        "candidate_id": (manifest.get("candidate") or {}).get("candidate_id"),
        "candidate_config_hash": (manifest.get("candidate") or {}).get("config_hash"),
        "comparison_class": manifest.get("comparison_class"), "evidence_kind": manifest.get("evidence_kind"),
        "is_carried": manifest.get("is_carried"), "status": status, "validation_errors": validation["errors"],
        "validation_warnings": validation["warnings"], "workload": manifest.get("workload"),
        "mock": bool((manifest.get("engine") or {}).get("mock")),
    }
    if status not in ("measured", "smoke_only"):
        doc["metrics"] = {name: {"value": None, "null_reason": f"run_{status}", "unit": d.unit}
                          for name, d in m.METRIC_DEFS.items()}
        doc["capacity"] = _capacity(manifest, [], {}, None)
        if write:
            write_json(run_dir / "metrics.json", doc)
        return doc
    summary = read_json(run_dir / "driver_summary.json")
    lane = manifest["workload"]["lane"]
    phases = summary.get("phases") or {}
    gpus = (manifest.get("device") or {}).get("allocated_physical_gpus")
    views = load_views(run_dir)
    measured = [v for v in views if v.phase in ("measured", "sustained")]
    per = {v.key: m.per_request(v) for v in measured}
    out: dict[str, Any] = {}
    expected = manifest["workload"].get("num_prompts") if lane == "finite" else None
    out.update(m.outcome_counts(measured, expected))
    succeeded = [v for v in measured if v.success]
    out["input_tokens_succeeded"] = m.scalar(sum(v.input_tokens or 0 for v in succeeded), "tokens")
    out["output_tokens_succeeded"] = m.scalar(sum(v.output_tokens or 0 for v in succeeded), "tokens")
    out["output_tokens_unsuccessful"] = m.scalar(sum(m._tokens_unsuccessful(v) for v in measured), "tokens")
    out["warmup_requests"] = m.scalar(sum(1 for v in views if v.phase == "warmup"), "requests")
    mismatches = sum(1 for v in succeeded if manifest["workload"].get("ignore_eos")
                     and v.requested_output_tokens is not None and v.output_tokens != v.requested_output_tokens)
    out["output_length_mismatches"] = m.scalar(mismatches, "requests")
    out["requests_exact_chunk_tokens"] = m.scalar(sum(1 for v in succeeded if v.tokens_known_per_chunk), "requests")
    native = manifest.get("native") or {}
    duration = None
    if lane in ("finite", "open_loop"):
        finite, duration = m.finite_cohort(measured, phases, gpus, native.get("duration"))
        out.update(finite)
    if lane == "open_loop":
        arrivals = phases.get("arrivals") or {}
        span = (arrivals.get("end") or 0) - (arrivals.get("start") or 0)
        out["offered_rate_rps"] = m.scalar(arrivals.get("offered_rate_rps"), "req/s")
        out["attempted_rate_rps"] = m.scalar(m.safe_div(len(measured), span), "req/s", "zero_denominator")
        out["achieved_completion_rps"] = m.scalar(m.safe_div(len(succeeded), duration), "req/s", "zero_denominator")
    latency_cohort = succeeded
    if lane == "sustained":
        steady = phases.get("steady") or {}
        if steady.get("start") is not None:
            latency_cohort = [v for v in succeeded if steady["start"] <= v.t_send < steady["end"]]
    out.update(m.latency_block(latency_cohort, per))
    slo_id = manifest.get("slo_id")
    slo = slo_mod.load_slo(slo_id, slo_registry) if slo_id else None
    quality = (manifest.get("candidate") or {}).get("quality_status")
    steady_extra = None
    ok: dict = {}
    if slo is not None:
        _, ok = slo_mod.evaluate(measured, per, slo, lane, duration, None, gpus, quality, out)
    if lane == "sustained":
        steady_metrics, steady_extra = m.steady_window(measured, phases, gpus, ok if slo else None)
        out.update(steady_metrics)
    slo_metrics, _ = slo_mod.evaluate(measured, per, slo, lane, duration, steady_extra, gpus, quality, out)
    out.update(slo_metrics)
    # Telemetry over the measurement interval.
    start_unix, end_unix, basis = _interval(manifest, summary)
    events: list[dict] = []
    allocations: list[dict] = []
    capacity_facts: dict = {}
    if (run_dir / "server_events.jsonl").exists():
        from .util import read_jsonl
        records = read_jsonl(run_dir / "server_events.jsonl")
        events = [e for e in records if e.get("kind") in ("decode", "prefill", "retract")
                  and (start_unix is None or start_unix - 1 <= e["t_unix"] <= end_unix + 1)]
        capacity_facts = next((e for e in records if e.get("kind") == "capacity"), {})
    out.update(summarise_server(events))
    samples: list[dict] = []
    if (run_dir / "memory_and_power.jsonl").exists():
        from .util import read_jsonl
        samples = read_jsonl(run_dir / "memory_and_power.jsonl")
        allocations = [s for s in samples if s.get("kind") == "allocation"]
    if start_unix is not None:
        out.update(summarise_gpu(samples, start_unix, end_unix))
        uuids = {g.get("uuid") for g in (manifest.get("device") or {}).get("gpus", []) if g.get("uuid")}
        joules, reason, info = cost_mod.energy(samples, start_unix, end_unix, uuids)
    else:
        joules, reason, info = None, "interval_unknown", {}
    out["gpu_energy_j"] = m.scalar(joules, "J", reason or "", coverage=info.get("coverage"), gpu_only=True)
    tokens_out = out["output_tokens_succeeded"]["value"]
    if lane == "sustained" and steady_extra:
        # Energy is over the steady window, so its denominator is the in-window successful tokens.
        tokens_out = round((out.get("successful_steady_output_tps", {}).get("value") or 0) * steady_extra["window"])
    out["gpu_energy_j_per_output_token"] = m.scalar(m.safe_div(joules, tokens_out), "J/token",
                                                    reason or "zero_denominator")
    qualified_tokens = None
    if slo is not None:
        q_tps = out["output_goodput_tps"].get("value")
        window = steady_extra["window"] if steady_extra else duration
        qualified_tokens = q_tps * window if (q_tps is not None and window) else None
    out["gpu_energy_j_per_qualified_output_token"] = m.scalar(
        m.safe_div(joules, qualified_tokens), "J/token", reason or ("slo_not_configured" if slo is None else "zero_denominator"))
    # Cost on the same boundary as throughput.
    pricing = cost_mod.load_pricing(manifest.get("pricing_id"), pricing_registry)
    allocation = {"allocated_physical_gpus": gpus, "instances": (manifest.get("device") or {}).get("instances", 1)}
    if lane == "sustained":
        output_tps = out.get("successful_steady_output_tps", {}).get("value")
        input_tps = total_tps = None
    else:
        output_tps = out.get("finite_cohort_output_tps", {}).get("value")
        input_tps = out.get("finite_cohort_input_tps", {}).get("value")
        total_tps = out.get("finite_cohort_total_tps", {}).get("value")
    out.update(cost_mod.cost_block(pricing, allocation, output_tps,
                                   out.get("output_goodput_tps", {}).get("value") if slo else None,
                                   input_tps, total_tps))
    # Every metric of the glossary appears in every run: what does not apply to this lane is null with its reason.
    for name, definition in m.METRIC_DEFS.items():
        if name not in out:
            reason = (f"not_applicable_to_lane:{lane}" if definition.lane not in ("all", lane)
                      else "not_computed_for_this_run")
            out[name] = {"value": None, "null_reason": reason, "unit": definition.unit}
    doc["metrics"] = out
    doc["headline"] = {
        "deployment_output_tps": output_tps,
        "output_tps_per_gpu": out.get("output_tps_per_gpu", {}).get("value"),
        "usd_per_million_output_tokens": out["usd_per_million_output_tokens"].get("value"),
        "qualified_output_tps": out.get("output_goodput_tps", {}).get("value") if slo else None,
        "qualified_output_tps_per_gpu": out.get("output_goodput_tps_per_gpu", {}).get("value") if slo else None,
        "usd_per_million_qualified_output_tokens": out["usd_per_million_qualified_output_tokens"].get("value"),
        "basis": "steady window (successful in-window tokens)" if lane == "sustained"
        else "finite cohort (drain included)" if lane == "finite" else "open loop (arrivals to drain end)",
    }
    doc["interval"] = {"start_unix": start_unix, "end_unix": end_unix, "basis": basis}
    doc["capacity"] = _capacity(manifest, events, capacity_facts, out)
    doc["slo"] = {"slo_id": slo_id, "definition": slo}
    doc["interactivity"] = slo_mod.interactivity_slices(out, INTERACTIVITY_THRESHOLDS, "median")
    doc["native"] = native.get("aggregates")
    doc["parity"] = _parity(native.get("aggregates"), out, duration) if native.get("aggregates") else None
    if manifest.get("profile_id") == "aa_slt_local":
        doc["site_metrics"] = _aa_slt(out, measured, phases)
    if manifest.get("profile_id") == "infx_fixed_seq":
        doc["site_metrics"] = _infx_site(out, gpus)
    doc["api_v3_equivalents"] = {
        "output_speed_full_duration_median_tps": out.get("user_speed_e2e_tps", {}).get("median"),
        "definition": "per request: output tokens / (last token - send), TTFT included; median over the cohort. "
                      "Local metric equivalent, not an AA score (no 5-day history, private prompts not reproduced).",
    }
    doc["allocations"] = allocations
    doc["inputs_sha256"] = {name: sha256_file(run_dir / name) for name in
                            ("requests.jsonl", "stream_events.jsonl.gz", "driver_summary.json", "run_manifest.json")
                            if (run_dir / name).exists()}
    if write:
        write_json(run_dir / "metrics.json", doc)
    return doc


def _capacity(manifest: dict, events: list[dict], facts: dict, out: dict | None) -> dict:
    """Configured slots, admitted running requests and waiting requests: printed for every arm in every run."""
    server = manifest.get("server") or {}
    configured = facts.get("leapquant_slots") or facts.get("max_mamba_cache_size") or server.get("configured_slots")
    running = (out or {}).get("server_running_requests", {})
    queue = (out or {}).get("server_queue_depth", {})
    return {
        "configured_state_slots": configured,
        "configured_state_slots_source": "server log" if (facts.get("leapquant_slots") or facts.get("max_mamba_cache_size"))
        else ("server flags" if server.get("configured_slots") else None),
        "server_max_running_requests": facts.get("max_running_requests") or server.get("max_running_requests"),
        "kv_tokens": facts.get("kv_tokens") or facts.get("max_total_num_tokens"),
        "client_concurrency": (manifest.get("workload") or {}).get("concurrency"),
        "admitted_running_median": running.get("median"), "admitted_running_max": running.get("max"),
        "waiting_median": queue.get("median"), "waiting_max": queue.get("max"),
        "note": "client concurrency is not the decode batch: the admitted running requests are",
    }


def _parity(native: dict, out: dict, duration: float | None) -> dict:
    """Upstream-native aggregates against the harness's recomputation from the same raw per-request data."""
    pairs = [("output_throughput", out.get("finite_cohort_output_tps", {}).get("value")),
             ("total_token_throughput", out.get("finite_cohort_total_tps", {}).get("value")),
             ("request_throughput", out.get("request_throughput_rps", {}).get("value")),
             ("median_ttft_ms", out.get("ttft_ms", {}).get("median")),
             ("median_tpot_ms", out.get("tpot_native_ms", {}).get("median")),
             ("median_e2el_ms", out.get("e2e_last_token_ms", {}).get("median")),
             ("median_itl_ms", out.get("chunk_gap_ms", {}).get("median")),
             ("completed", out.get("requests_succeeded", {}).get("value"))]
    rows = []
    for key, ours in pairs:
        theirs = native.get(key)
        if theirs is None or ours is None:
            rows.append({"metric": key, "native": theirs, "harness": ours, "ok": None})
            continue
        # Chunk times are stored as integer microseconds, so millisecond statistics may differ by < 0.002 ms.
        tolerance = max(1e-6 * abs(theirs), 0.002 if key.endswith("_ms") else 1e-9)
        rows.append({"metric": key, "native": theirs, "harness": ours, "abs_diff": abs(theirs - ours),
                     "tolerance": tolerance, "ok": abs(theirs - ours) <= tolerance})
    return {"rows": rows, "ok": all(r["ok"] for r in rows if r["ok"] is not None),
            "rule": "harness recomputation from the client's own per-request arrays must equal its aggregates"}


def _aa_slt(out: dict, measured: list, phases: dict) -> dict:
    """The four AA-SLT site metrics (methodology page, read 2026-11-15), computed on the steady phase."""
    steady = phases.get("steady") or {}
    cohort = [v for v in measured if steady.get("start") is not None and steady["start"] <= v.t_send < steady["end"]]
    with_token = sum(1 for v in cohort if v.t_first is not None)
    return {
        "system_output_throughput_tps": out.get("steady_output_event_tps", {}).get("value"),
        "response_rate": m.safe_div(with_token, len(cohort)),
        "median_e2e_latency_s": (out.get("e2e_terminal_ms", {}).get("median") or 0) / 1000.0
        if out.get("e2e_terminal_ms", {}).get("median") is not None else None,
        "median_output_speed_after_first_token_tps": out.get("user_speed_post_first_token_tps", {}).get("median"),
        "definitions": {
            "system_output_throughput_tps": "average aggregate output tokens/s received during the 180 s phase",
            "response_rate": "share of queries sent in the phase that received at least one output token "
                             "(a first token is not a successful completion; see completion_rate)",
            "median_e2e_latency_s": "median over queries sent in the phase, terminal response",
            "median_output_speed_after_first_token_tps": "median over queries sent in the phase",
        },
        "label": "Independent AA-SLT-method-inspired result (not an Artificial Analysis measurement)",
    }


def _infx_site(out: dict, gpus: int | None) -> dict:
    """InferenceX-style fields (InferenceX-app axis definitions, e713065): throughput per GPU uses input + output."""
    total = out.get("finite_cohort_total_tps", {}).get("value")
    median_tpot = out.get("tpot_native_ms", {}).get("median")
    mean_tpot = out.get("tpot_native_ms", {}).get("mean")
    return {
        "tput_per_gpu_total": m.safe_div(total, gpus),
        "output_tput_per_gpu": out.get("output_tps_per_gpu", {}).get("value"),
        "input_tput_per_gpu": out.get("input_tps_per_gpu", {}).get("value"),
        "median_intvty": 1000.0 / median_tpot if median_tpot else None,
        "mean_intvty": 1000.0 / mean_tpot if mean_tpot else None,
        "label": "Unofficial independent InferenceX harness result",
    }


def server_log_to_events(log_text: str, start_unix: float | None, end_unix: float | None,
                         origin_unix: float | None) -> tuple[list[dict], list[dict], dict]:
    events, allocations, capacity = parse_server_log(log_text, start_unix, end_unix, origin_unix)
    return events, allocations, capacity
