"""Metric definitions (one place) and their computation from raw request records.

Every metric has a definition in METRIC_DEFS: unit, aggregation unit (request, token gap, chunk gap, run), the
numerator and denominator it is built from, and the cohort it covers. compute_run_metrics() returns every metric
of one run; a metric that cannot be computed is null with a reason, never zero.

Conventions verified against the pinned InferenceX client (infx/bench_serving, commit fabf7ba):
  - the client's "latency" (E2EL) ends at the LAST CONTENT CHUNK, not at the terminal response; its TPOT is
    (latency - TTFT) / (N - 1) for N > 1, so it equals the last-token span per token. Both are kept, and the
    terminal-response variant is a separate, separately named metric.
  - its ITL list holds gaps between content chunks (chunk gaps). A chunk gap is a token gap only when every chunk
    carries one token; the harness records exact per-chunk token counts where the server reports them.
  - percentiles: linear interpolation between closest ranks (numpy's default), population standard deviation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .records import RequestView
from .util import null, safe_div

PERCENTILES = (50, 90, 95, 99)


@dataclass(frozen=True)
class MetricDef:
    unit: str
    aggregation: str
    definition: str
    numerator: str = ""
    denominator: str = ""
    lane: str = "all"


METRIC_DEFS: dict[str, MetricDef] = {
    # --- counts and rates of outcomes -------------------------------------------------------------------------
    "requests_expected": MetricDef("requests", "run", "Measured requests the profile asked for (warmup excluded)."),
    "requests_attempted": MetricDef("requests", "run", "Measured request attempts sent (retries count separately)."),
    "requests_succeeded": MetricDef("requests", "run", "Measured attempts that ended with a complete response."),
    "requests_failed": MetricDef("requests", "run", "Measured attempts that ended with an error (HTTP, stream or server error)."),
    "requests_timeout": MetricDef("requests", "run", "Measured attempts stopped at their deadline."),
    "requests_cancelled": MetricDef("requests", "run", "Measured attempts cancelled by the client."),
    "requests_incomplete": MetricDef("requests", "run", "Measured attempts with no final status when the run ended (unresolved)."),
    "requests_with_first_token": MetricDef("requests", "run", "Measured attempts that received at least one output token."),
    "retries": MetricDef("requests", "run", "Attempts that retry an earlier attempt (credited once if they succeed)."),
    "warmup_requests": MetricDef("requests", "run", "Warmup requests (never part of any metric)."),
    "completion_rate": MetricDef("fraction", "run", "Succeeded / attempted measured requests.", "requests_succeeded", "requests_attempted"),
    "error_rate": MetricDef("fraction", "run", "(failed + timeout + cancelled + incomplete) / attempted measured requests.", "unsuccessful attempts", "requests_attempted"),
    "first_token_response_rate": MetricDef("fraction", "run", "Attempts with at least one output token / attempted. A first token is not a successful completion.", "requests_with_first_token", "requests_attempted"),
    "output_length_mismatches": MetricDef("requests", "run", "Succeeded requests whose output token count differs from the requested exact length (profiles that ignore EOS)."),
    "requests_exact_chunk_tokens": MetricDef("requests", "run", "Succeeded requests whose every streamed chunk carried an exact server-reported token count."),
    # --- tokens -------------------------------------------------------------------------------------------------
    "input_tokens_succeeded": MetricDef("tokens", "run", "Prompt tokens of succeeded measured requests."),
    "output_tokens_succeeded": MetricDef("tokens", "run", "Output tokens of succeeded measured requests (server usage count)."),
    "output_tokens_unsuccessful": MetricDef("tokens", "run", "Output tokens streamed to measured requests that did not succeed (partial, failed, superseded). Never credited."),
    # --- finite-cohort throughput (InferenceX-style, drain included) --------------------------------------------
    "cohort_duration_s": MetricDef("s", "run", "From the start of main scheduling (warmup excluded) to the last measured completion or deadline. Drain included.", lane="finite"),
    "finite_cohort_output_tps": MetricDef("tok/s", "run", "Successful cohort output tokens / cohort duration.", "output_tokens_succeeded", "cohort_duration_s", "finite"),
    "finite_cohort_input_tps": MetricDef("tok/s", "run", "Successful cohort input tokens / cohort duration.", "input_tokens_succeeded", "cohort_duration_s", "finite"),
    "finite_cohort_total_tps": MetricDef("tok/s", "run", "Successful cohort input + output tokens / cohort duration (same cohort).", "input + output tokens of succeeded requests", "cohort_duration_s", "finite"),
    "request_throughput_rps": MetricDef("req/s", "run", "Succeeded measured requests / cohort duration.", "requests_succeeded", "cohort_duration_s", "finite"),
    "output_tps_per_gpu": MetricDef("tok/s/GPU", "run", "Deployment output tok/s (finite cohort or steady successful) / all allocated physical GPUs, idle and prefill GPUs included.", "deployment output tok/s", "allocated physical GPUs"),
    "input_tps_per_gpu": MetricDef("tok/s/GPU", "run", "Deployment input tok/s / all allocated physical GPUs.", "deployment input tok/s", "allocated physical GPUs"),
    "total_tps_per_gpu": MetricDef("tok/s/GPU", "run", "Deployment total tok/s / all allocated physical GPUs.", "deployment total tok/s", "allocated physical GPUs"),
    "output_event_tps_max_10s": MetricDef("tok/s", "10 s window", "Diagnostic: the largest output-token arrival rate in any 10 s window of the main phase (all requests). Not a headline.", "tokens arriving in the window", "10 s"),
    # --- steady window (AA-SLT-style) ---------------------------------------------------------------------------
    "steady_window_s": MetricDef("s", "run", "Length of the declared steady phase (ramp and drain excluded).", lane="sustained"),
    "steady_output_event_tps": MetricDef("tok/s", "run", "Output tokens received inside the steady window (any request) / window seconds.", "tokens received in window", "steady_window_s", "sustained"),
    "successful_steady_output_tps": MetricDef("tok/s", "run", "In-window output tokens of requests that eventually succeeded / window seconds (status known after drain).", "in-window tokens of succeeded requests", "steady_window_s", "sustained"),
    "steady_failed_request_tokens": MetricDef("tokens", "run", "In-window output tokens of requests that did not succeed.", lane="sustained"),
    "steady_unresolved_requests": MetricDef("requests", "run", "Requests with in-window tokens whose final status was still unknown after drain.", lane="sustained"),
    "steady_unresolved_tokens": MetricDef("tokens", "run", "In-window tokens of unresolved requests (neither successful nor failed; never credited).", lane="sustained"),
    "steady_completions": MetricDef("requests", "run", "Succeeded requests whose terminal response arrived inside the window.", lane="sustained"),
    "steady_request_rate_rps": MetricDef("req/s", "run", "steady_completions / window seconds.", "steady_completions", "steady_window_s", "sustained"),
    "steady_first_token_responses": MetricDef("requests", "run", "Requests whose first output token arrived inside the window.", lane="sustained"),
    "steady_boundary_started_before": MetricDef("requests", "run", "Requests sent before the window that completed inside it.", lane="sustained"),
    "steady_boundary_ended_after": MetricDef("requests", "run", "Requests sent inside the window that completed after it.", lane="sustained"),
    "steady_boundary_spanning": MetricDef("requests", "run", "Requests sent before the window and completed after it.", lane="sustained"),
    "steady_cohort_requests": MetricDef("requests", "run", "Requests SENT inside the window (cohort of the per-query medians).", lane="sustained"),
    # --- open loop --------------------------------------------------------------------------------------------
    "offered_rate_rps": MetricDef("req/s", "run", "Declared Poisson arrival rate.", lane="open_loop"),
    "attempted_rate_rps": MetricDef("req/s", "run", "Requests sent / arrival span.", "requests sent", "arrival span", "open_loop"),
    "achieved_completion_rps": MetricDef("req/s", "run", "Succeeded requests / (arrival start to end of drain).", "requests_succeeded", "arrival start to drain end", "open_loop"),
    # --- latency distributions ----------------------------------------------------------------------------------
    "ttft_ms": MetricDef("ms", "request", "First content chunk arrival minus send. Succeeded measured requests."),
    "e2e_terminal_ms": MetricDef("ms", "request", "Terminal response (stream end) minus send. Succeeded measured requests."),
    "e2e_last_token_ms": MetricDef("ms", "request", "Last content chunk arrival minus send: the pinned InferenceX client's E2EL. Succeeded measured requests."),
    "tpot_native_ms": MetricDef("ms", "request", "Pinned InferenceX TPOT: (E2EL - TTFT) / (N - 1), N > 1, E2EL ending at the last content chunk. Its reciprocal is the client's 'interactivity'."),
    "tpot_terminal_ms": MetricDef("ms", "request", "(terminal response latency - TTFT) / (N - 1), N > 1. Differs from the native TPOT by the stream's closing delay."),
    "span_last_first_ms": MetricDef("ms", "request", "(last token - first token) / (N - 1), N > 1. Equal to the native TPOT by construction of the pinned client; named separately as the specification asks."),
    "itl_token_ms": MetricDef("ms", "token gap", "Gaps between consecutive output tokens, from requests whose every chunk carried exactly one token (exact). Requests with multi-token chunks are excluded and counted."),
    "chunk_gap_ms": MetricDef("ms", "chunk gap", "Gaps between consecutive content chunks (all succeeded requests): the pinned client's ITL. A multi-token chunk is ONE gap, never several zero gaps."),
    "itl_estimated_uniform_ms": MetricDef("ms", "token gap (estimate)", "ESTIMATE, only when multi-token chunks exist: each chunk gap divided evenly among its tokens."),
    "client_queue_delay_ms": MetricDef("ms", "request", "Send minus scheduled time (client-side wait for a concurrency slot or a late arrival)."),
    # --- per-user speed ---------------------------------------------------------------------------------------
    "user_speed_post_first_token_tps": MetricDef("tok/s/user", "request", "(N - 1) / (last token - first token), N > 1: generation speed one user sees after the first token."),
    "user_speed_e2e_tps": MetricDef("tok/s/user", "request", "N / (last token - send): speed one user sees including the wait for the first token."),
    "interactivity_native_median_tps": MetricDef("tok/s/user", "run", "1000 / median native TPOT (ms): the pinned client's 'median_intvty'.", "1000", "median tpot_native_ms"),
    # --- SLO goodput --------------------------------------------------------------------------------------------
    "slo_qualifying_requests": MetricDef("requests", "run", "Succeeded measured requests that meet every request-level SLO condition together."),
    "slo_qualifying_fraction": MetricDef("fraction", "run", "Qualifying requests / attempted measured requests.", "slo_qualifying_requests", "requests_attempted"),
    "request_goodput_rps": MetricDef("req/s", "run", "Qualifying requests / the lane's duration (finite: cohort duration; steady: qualifying completions inside the window / window).", "qualifying requests", "duration"),
    "output_goodput_tps": MetricDef("tok/s", "run", "Output tokens of qualifying requests / the lane's duration (steady: their in-window tokens / window).", "qualifying output tokens", "duration"),
    "output_goodput_tps_per_gpu": MetricDef("tok/s/GPU", "run", "output_goodput_tps / allocated physical GPUs."),
    # --- server telemetry ---------------------------------------------------------------------------------------
    "server_running_requests": MetricDef("requests", "decode log line", "Running requests in the server's decode batches during the main phase (the actual kernel batch, not client concurrency)."),
    "server_queue_depth": MetricDef("requests", "log line", "Requests waiting in the server queue during the main phase."),
    "server_token_usage": MetricDef("fraction", "log line", "KV token pool usage reported by the server."),
    "server_retractions": MetricDef("events", "run", "Decode retractions (a request's state discarded and rebuilt by prefill) during the run."),
    "server_cuda_graph_fraction": MetricDef("fraction", "decode log line", "Share of decode log lines that ran a CUDA graph."),
    "server_reported_gen_tps": MetricDef("tok/s", "decode log line", "The server's own generation throughput readings (diagnostic, not a headline)."),
    # --- memory, power, energy ----------------------------------------------------------------------------------
    "gpu_memory_used_mib": MetricDef("MiB", "sample", "Device memory in use (nvidia-smi), whole device, during the main phase."),
    "gpu_memory_total_mib": MetricDef("MiB", "run", "Device memory size."),
    "gpu_power_w": MetricDef("W", "sample", "GPU board power (nvidia-smi power.draw), during the main phase. GPU only, not host or facility."),
    "gpu_sm_clock_mhz": MetricDef("MHz", "sample", "SM clock during the main phase (thermal/clock telemetry)."),
    "gpu_temperature_c": MetricDef("C", "sample", "GPU temperature during the main phase."),
    "gpu_energy_j": MetricDef("J", "run", "Integral of GPU power over the measurement interval (trapezoid), only with adequate sample coverage."),
    "gpu_energy_j_per_output_token": MetricDef("J/token", "run", "GPU energy / successful output tokens of the same interval.", "gpu_energy_j", "output_tokens_succeeded"),
    "gpu_energy_j_per_qualified_output_token": MetricDef("J/token", "run", "GPU energy / output tokens of SLO-qualifying requests.", "gpu_energy_j", "qualifying output tokens"),
    # --- cost ---------------------------------------------------------------------------------------------------
    "usd_per_hour": MetricDef("USD/h", "run", "Deployment cost per hour from pricing.json under its cost mode (null when no price is recorded)."),
    "usd_per_million_output_tokens": MetricDef("USD/1M tok", "run", "USD/h x 1e6 / (3600 x successful deployment output tok/s).", "USD/h x 1e6", "3600 x output tok/s"),
    "usd_per_million_qualified_output_tokens": MetricDef("USD/1M tok", "run", "Same with SLO-qualified output tok/s.", "USD/h x 1e6", "3600 x output_goodput_tps"),
    "usd_per_million_input_tokens": MetricDef("USD/1M tok", "run", "Alternative view: the WHOLE deployment cost attributed to input tokens. Not additive with the output view.", "USD/h x 1e6", "3600 x input tok/s"),
    "usd_per_million_total_tokens": MetricDef("USD/1M tok", "run", "Alternative view: the whole deployment cost attributed to input + output tokens. Never an output-token cost.", "USD/h x 1e6", "3600 x total tok/s"),
}


# ---------------------------------------------------------------------------------------------------------------
def percentile(sorted_values: list[float], p: float) -> float | None:
    """Linear interpolation between closest ranks (numpy's default method)."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (n - 1)
    lower = int(math.floor(rank))
    upper = min(lower + 1, n - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def distribution(values: Iterable[float], unit: str, aggregation: str, scale: float = 1.0) -> dict:
    data = sorted(v * scale for v in values)
    out: dict[str, Any] = {"unit": unit, "aggregation": aggregation, "n": len(data)}
    if not data:
        out.update(null("no_observations"))
        return out
    mean = sum(data) / len(data)
    out.update({
        "mean": mean,
        "std": math.sqrt(sum((v - mean) ** 2 for v in data) / len(data)),
        "min": data[0],
        "max": data[-1],
        "median": percentile(data, 50),
    })
    for p in PERCENTILES:
        out[f"p{p}"] = percentile(data, p)
    return out


def scalar(value: float | int | None, unit: str, reason: str = "not_computable", **extra: Any) -> dict:
    if value is None:
        out = null(reason)
        out["unit"] = unit
    else:
        out = {"value": value, "unit": unit}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------------------------------------------
@dataclass
class PerRequest:
    ttft: float | None = None
    e2e_terminal: float | None = None
    e2e_last: float | None = None
    tpot_native: float | None = None
    tpot_terminal: float | None = None
    span: float | None = None
    speed_post_first: float | None = None
    speed_e2e: float | None = None
    queue_delay: float | None = None
    chunk_gaps: list[float] | None = None
    token_gaps_exact: list[float] | None = None
    token_gaps_estimated: list[float] | None = None


def per_request(view: RequestView) -> PerRequest:
    out = PerRequest()
    n = view.output_tokens
    if view.t_first is not None:
        out.ttft = view.t_first - view.t_send
    if view.t_last is not None:
        out.e2e_last = view.t_last - view.t_send
    if view.t_terminal is not None:
        out.e2e_terminal = view.t_terminal - view.t_send
    if view.t_scheduled is not None:
        out.queue_delay = view.t_send - view.t_scheduled
    if n is not None and n > 1 and out.ttft is not None:
        if out.e2e_last is not None:
            out.tpot_native = (out.e2e_last - out.ttft) / (n - 1)
        if out.e2e_terminal is not None:
            out.tpot_terminal = (out.e2e_terminal - out.ttft) / (n - 1)
        if view.t_last is not None and view.t_first is not None:
            out.span = (view.t_last - view.t_first) / (n - 1)
            if view.t_last > view.t_first:
                out.speed_post_first = (n - 1) / (view.t_last - view.t_first)
    if n is not None and n >= 1 and view.t_last is not None and view.t_last > view.t_send:
        out.speed_e2e = n / (view.t_last - view.t_send)
    times = view.chunk_times
    if len(times) >= 2:
        gaps = [b - a for a, b in zip(times, times[1:])]
        out.chunk_gaps = gaps
        if view.token_exact_gaps:
            out.token_gaps_exact = gaps
        elif view.tokens_known_per_chunk:
            estimated: list[float] = []
            for gap, tokens in zip(gaps, view.chunk_tokens[1:]):  # type: ignore[index]
                if tokens and tokens > 0:
                    estimated.extend([gap / tokens] * tokens)
            out.token_gaps_estimated = estimated
    return out


# ---------------------------------------------------------------------------------------------------------------
def _tokens_unsuccessful(view: RequestView) -> int:
    if view.success:
        return 0
    if view.tokens_known_per_chunk:
        return sum(view.chunk_tokens)  # type: ignore[arg-type]
    return len(view.chunk_times)


def outcome_counts(measured: list[RequestView], expected: int | None) -> dict[str, dict]:
    attempted = len(measured)
    count = {status: sum(1 for v in measured if v.status == status) for status in
             ("success", "failed", "timeout", "cancelled", "incomplete")}
    first = sum(1 for v in measured if v.t_first is not None)
    unsuccessful = attempted - count["success"]
    out = {
        "requests_expected": scalar(expected, "requests", "not_declared"),
        "requests_attempted": scalar(attempted, "requests"),
        "requests_succeeded": scalar(count["success"], "requests"),
        "requests_failed": scalar(count["failed"], "requests"),
        "requests_timeout": scalar(count["timeout"], "requests"),
        "requests_cancelled": scalar(count["cancelled"], "requests"),
        "requests_incomplete": scalar(count["incomplete"], "requests"),
        "requests_with_first_token": scalar(first, "requests"),
        "retries": scalar(sum(1 for v in measured if v.retry_of), "requests"),
        "completion_rate": scalar(safe_div(count["success"], attempted), "fraction", "zero_denominator"),
        "error_rate": scalar(safe_div(unsuccessful, attempted), "fraction", "zero_denominator"),
        "first_token_response_rate": scalar(safe_div(first, attempted), "fraction", "zero_denominator"),
    }
    return out


def latency_block(succeeded: list[RequestView], per: dict[tuple[str, int], PerRequest]) -> dict[str, dict]:
    rows = [per[v.key] for v in succeeded]

    def pick(name: str) -> list[float]:
        return [getattr(r, name) for r in rows if getattr(r, name) is not None]

    chunk_gaps = [g for r in rows if r.chunk_gaps for g in r.chunk_gaps]
    exact_gaps = [g for r in rows if r.token_gaps_exact for g in r.token_gaps_exact]
    estimated = [g for r in rows if r.token_gaps_estimated for g in r.token_gaps_estimated]
    excluded = sum(1 for v in succeeded if len(v.chunk_times) >= 2 and not v.token_exact_gaps)
    out = {
        "ttft_ms": distribution(pick("ttft"), "ms", "request", 1000.0),
        "e2e_terminal_ms": distribution(pick("e2e_terminal"), "ms", "request", 1000.0),
        "e2e_last_token_ms": distribution(pick("e2e_last"), "ms", "request", 1000.0),
        "tpot_native_ms": distribution(pick("tpot_native"), "ms", "request (N > 1)", 1000.0),
        "tpot_terminal_ms": distribution(pick("tpot_terminal"), "ms", "request (N > 1)", 1000.0),
        "span_last_first_ms": distribution(pick("span"), "ms", "request (N > 1)", 1000.0),
        "chunk_gap_ms": distribution(chunk_gaps, "ms", "chunk gap", 1000.0),
        "itl_token_ms": distribution(exact_gaps, "ms", "token gap (exact)", 1000.0),
        "client_queue_delay_ms": distribution(pick("queue_delay"), "ms", "request", 1000.0),
        "user_speed_post_first_token_tps": distribution(pick("speed_post_first"), "tok/s/user", "request (N > 1)"),
        "user_speed_e2e_tps": distribution(pick("speed_e2e"), "tok/s/user", "request"),
    }
    out["itl_token_ms"]["requests_excluded_multi_token_chunks"] = excluded
    if estimated:
        out["itl_estimated_uniform_ms"] = distribution(estimated, "ms", "token gap (ESTIMATE)", 1000.0)
        out["itl_estimated_uniform_ms"]["estimate"] = True
    else:
        out["itl_estimated_uniform_ms"] = scalar(None, "ms", "no_multi_token_chunks")
    median_tpot = out["tpot_native_ms"].get("median")
    out["interactivity_native_median_tps"] = scalar(
        1000.0 / median_tpot if median_tpot else None, "tok/s/user", "no_tpot")
    return out


def max_window_rate(views: list[RequestView], start: float, end: float, width: float = 10.0) -> float | None:
    """Largest token arrival rate in any `width`-second window inside [start, end] (1 s steps)."""
    if end - start < width:
        return None
    arrivals: list[tuple[float, int]] = []
    for v in views:
        if v.clock_domain != "run":
            return None
        if v.tokens_known_per_chunk:
            arrivals.extend(zip(v.chunk_times, v.chunk_tokens))  # type: ignore[arg-type]
        else:
            arrivals.extend((t, 1) for t in v.chunk_times)
    arrivals.sort()
    best = 0.0
    times = [a[0] for a in arrivals]
    prefix = [0]
    for _, n in arrivals:
        prefix.append(prefix[-1] + n)
    import bisect
    t = start
    while t + width <= end + 1e-9:
        lo = bisect.bisect_left(times, t)
        hi = bisect.bisect_left(times, t + width)
        best = max(best, (prefix[hi] - prefix[lo]) / width)
        t += 1.0
    return best


# ---------------------------------------------------------------------------------------------------------------
def finite_cohort(measured: list[RequestView], phases: dict, gpus: int | None,
                  native_duration: float | None = None) -> tuple[dict[str, dict], float | None]:
    succeeded = [v for v in measured if v.success]
    out: dict[str, dict] = {}
    main = phases.get("main") or {}
    duration = None
    source = None
    if native_duration is not None:
        duration, source = native_duration, "client_reported_duration"
    elif main.get("start") is not None and main.get("end") is not None:
        duration, source = main["end"] - main["start"], "driver_phase_main"
    if duration is not None and duration <= 0:
        duration = None
        source = "non_positive_duration"
    out["cohort_duration_s"] = scalar(duration, "s", source or "phase_missing", source=source)
    tokens_out = sum(v.output_tokens or 0 for v in succeeded)
    tokens_in = sum(v.input_tokens or 0 for v in succeeded)
    out["finite_cohort_output_tps"] = scalar(safe_div(tokens_out, duration), "tok/s", "zero_denominator")
    out["finite_cohort_input_tps"] = scalar(safe_div(tokens_in, duration), "tok/s", "zero_denominator")
    out["finite_cohort_total_tps"] = scalar(safe_div(tokens_in + tokens_out, duration), "tok/s", "zero_denominator")
    out["request_throughput_rps"] = scalar(safe_div(len(succeeded), duration), "req/s", "zero_denominator")
    for name, key in (("output", "finite_cohort_output_tps"), ("input", "finite_cohort_input_tps"),
                      ("total", "finite_cohort_total_tps")):
        value = out[key].get("value")
        out[f"{name}_tps_per_gpu"] = scalar(safe_div(value, gpus), "tok/s/GPU",
                                            "gpu_count_unknown" if not gpus else "zero_denominator")
    if main.get("start") is not None and main.get("end") is not None:
        out["output_event_tps_max_10s"] = scalar(
            max_window_rate(measured, main["start"], main["end"]), "tok/s", "shorter_than_window_or_relative_clock",
            diagnostic=True)
    else:
        out["output_event_tps_max_10s"] = scalar(None, "tok/s", "phase_missing", diagnostic=True)
    return out, duration


def steady_window(views: list[RequestView], phases: dict, gpus: int | None,
                  slo_ok: dict[tuple[str, int], bool] | None) -> tuple[dict[str, dict], dict]:
    steady = phases.get("steady") or {}
    w0, w1 = steady.get("start"), steady.get("end")
    out: dict[str, dict] = {}
    extra: dict[str, Any] = {}
    if w0 is None or w1 is None or w1 <= w0:
        reason = "steady_phase_missing"
        for key in ("steady_window_s", "steady_output_event_tps", "successful_steady_output_tps"):
            out[key] = scalar(None, "tok/s" if "tps" in key else "s", reason)
        return out, extra
    if any(v.clock_domain != "run" for v in views):
        for key in ("steady_window_s", "steady_output_event_tps", "successful_steady_output_tps"):
            out[key] = scalar(None, "tok/s" if "tps" in key else "s", "absolute_timestamps_unavailable")
        return out, extra
    window = w1 - w0
    total = success_tokens = failed_tokens = unresolved_tokens = 0
    exact = True
    unresolved = completions = first_tokens = before = after = spanning = 0
    qualifying_tokens = qualifying_completions = 0
    for v in views:
        tokens, is_exact = v.tokens_between(w0, w1)
        tokens = tokens or 0
        exact = exact and is_exact
        total += tokens
        if v.success:
            success_tokens += tokens
            if slo_ok is not None and slo_ok.get(v.key):
                qualifying_tokens += tokens
        elif v.status == "incomplete":
            if tokens:
                unresolved += 1
            unresolved_tokens += tokens
        else:
            failed_tokens += tokens
        if v.t_first is not None and w0 <= v.t_first < w1:
            first_tokens += 1
        end = v.t_terminal
        if v.success and end is not None and w0 <= end < w1:
            completions += 1
            if slo_ok is not None and slo_ok.get(v.key):
                qualifying_completions += 1
        if end is not None:
            if v.t_send < w0 and w0 <= end < w1:
                before += 1
            if w0 <= v.t_send < w1 and end >= w1:
                after += 1
            if v.t_send < w0 and end >= w1:
                spanning += 1
    out["steady_window_s"] = scalar(window, "s")
    out["steady_output_event_tps"] = scalar(total / window, "tok/s", exact_token_counts=exact)
    out["successful_steady_output_tps"] = scalar(success_tokens / window, "tok/s", exact_token_counts=exact)
    out["steady_failed_request_tokens"] = scalar(failed_tokens, "tokens")
    out["steady_unresolved_requests"] = scalar(unresolved, "requests")
    out["steady_unresolved_tokens"] = scalar(unresolved_tokens, "tokens")
    out["steady_completions"] = scalar(completions, "requests")
    out["steady_request_rate_rps"] = scalar(completions / window, "req/s")
    out["steady_first_token_responses"] = scalar(first_tokens, "requests")
    out["steady_boundary_started_before"] = scalar(before, "requests")
    out["steady_boundary_ended_after"] = scalar(after, "requests")
    out["steady_boundary_spanning"] = scalar(spanning, "requests")
    cohort = [v for v in views if w0 <= v.t_send < w1]
    out["steady_cohort_requests"] = scalar(len(cohort), "requests")
    value = out["successful_steady_output_tps"]["value"]
    out["output_tps_per_gpu"] = scalar(safe_div(value, gpus), "tok/s/GPU", "gpu_count_unknown")
    extra.update(window=window, qualifying_tokens=qualifying_tokens,
                 qualifying_completions=qualifying_completions, cohort=cohort)
    return out, extra
