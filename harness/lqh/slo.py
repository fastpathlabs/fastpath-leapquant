"""Versioned SLO objects and goodput.

An SLO object (harness/config/slo.json) holds every threshold with its unit, the request eligibility rule and the
global constraints. Request-level conditions are evaluated TOGETHER per request; global constraints (p99 token gap,
error rate, quality status) are evaluated separately and decide whether an operating point is eligible. A missing SLO
means goodput is "not configured" (null), never inferred from a log or a chart label.
"""
from __future__ import annotations

from typing import Any

from .metrics import PerRequest, distribution, scalar
from .records import RequestView
from .util import harness_root, read_json, safe_div

DEFAULT_SLO_FILE = harness_root() / "config" / "slo.json"


def load_slo(slo_id: str | None, path=None) -> dict | None:
    if not slo_id:
        return None
    registry = read_json(path or DEFAULT_SLO_FILE)
    for slo in registry["slos"]:
        if slo["slo_id"] == slo_id:
            return slo
    raise KeyError(f"SLO {slo_id!r} is not in {path or DEFAULT_SLO_FILE}")


def request_ok(view: RequestView, per: PerRequest, slo: dict) -> tuple[bool, list[str]]:
    """All request-level conditions together. Returns (qualifies, failed conditions)."""
    rule = slo["request_predicate"]
    failed: list[str] = []
    if rule.get("require_success", True) and not view.success:
        failed.append("not_successful")
    limit = rule.get("ttft_ms_max")
    if limit is not None and (per.ttft is None or per.ttft * 1000.0 > limit):
        failed.append("ttft")
    limit = rule.get("tpot_ms_max")
    if limit is not None:
        # Pinned InferenceX convention: TPOT counts as 0 for responses of one token (no gap to measure).
        tpot = per.tpot_native if per.tpot_native is not None else (0.0 if view.output_tokens == 1 else None)
        if tpot is None or tpot * 1000.0 > limit:
            failed.append("tpot")
    limit = rule.get("e2e_ms_max")
    if limit is not None:
        e2e = per.e2e_terminal if rule.get("e2e_metric", "e2e_terminal_ms") == "e2e_terminal_ms" else per.e2e_last
        if e2e is None or e2e * 1000.0 > limit:
            failed.append("e2e")
    gap_rule = rule.get("token_gap_rule")
    if gap_rule:
        gaps = per.token_gaps_exact if gap_rule.get("basis", "token") == "token" else per.chunk_gaps
        if gaps is None:
            failed.append("token_gap_unmeasurable")
        else:
            worst = max(gaps) if gaps else 0.0
            if worst * 1000.0 > gap_rule["max_gap_ms"]:
                failed.append("token_gap")
    return (not failed), failed


def evaluate(measured: list[RequestView], per: dict[tuple[str, int], PerRequest], slo: dict | None,
             lane: str, duration: float | None, steady_extra: dict | None, gpus: int | None,
             quality_status: str | None, metrics: dict[str, Any]) -> tuple[dict[str, dict], dict[tuple[str, int], bool]]:
    """Goodput, qualifying counts, global constraints and the eligibility of the operating point."""
    out: dict[str, dict] = {}
    if slo is None:
        for key in ("slo_qualifying_requests", "slo_qualifying_fraction", "request_goodput_rps",
                    "output_goodput_tps", "output_goodput_tps_per_gpu"):
            out[key] = scalar(None, "", "slo_not_configured")
        out["operating_point_eligible"] = {"value": None, "null_reason": "slo_not_configured"}
        return out, {}
    ok: dict[tuple[str, int], bool] = {}
    reasons: dict[str, int] = {}
    for v in measured:
        good, failed = request_ok(v, per[v.key], slo)
        ok[v.key] = good
        for name in failed:
            reasons[name] = reasons.get(name, 0) + 1
    qualifying = [v for v in measured if ok[v.key]]
    attempted = len(measured)
    out["slo_qualifying_requests"] = scalar(len(qualifying), "requests", failed_conditions=reasons)
    out["slo_qualifying_fraction"] = scalar(safe_div(len(qualifying), attempted), "fraction", "zero_denominator")
    if lane == "sustained" and steady_extra:
        window = steady_extra["window"]
        out["request_goodput_rps"] = scalar(steady_extra["qualifying_completions"] / window, "req/s",
                                            basis="qualifying completions inside the steady window")
        out["output_goodput_tps"] = scalar(steady_extra["qualifying_tokens"] / window, "tok/s",
                                           basis="in-window tokens of qualifying requests")
    else:
        tokens = sum(v.output_tokens or 0 for v in qualifying)
        out["request_goodput_rps"] = scalar(safe_div(len(qualifying), duration), "req/s", "zero_denominator",
                                            basis="qualifying requests / cohort duration")
        out["output_goodput_tps"] = scalar(safe_div(tokens, duration), "tok/s", "zero_denominator",
                                           basis="qualifying output tokens / cohort duration")
    out["output_goodput_tps_per_gpu"] = scalar(safe_div(out["output_goodput_tps"].get("value"), gpus),
                                                "tok/s/GPU", "gpu_count_unknown")
    # Global constraints, separately from the request predicate.
    constraints = slo.get("global_constraints", {})
    checks: list[dict] = []
    limit = constraints.get("error_rate_max")
    if limit is not None:
        rate = metrics.get("error_rate", {}).get("value")
        checks.append({"constraint": "error_rate_max", "limit": limit, "observed": rate,
                       "ok": rate is not None and rate <= limit})
    limit = constraints.get("p99_token_gap_ms_max")
    if limit is not None:
        basis = constraints.get("p99_token_gap_metric", "itl_token_ms")
        observed = metrics.get(basis, {}).get("p99")
        checks.append({"constraint": "p99_token_gap_ms_max", "basis": basis, "limit": limit, "observed": observed,
                       "ok": observed is not None and observed <= limit})
    # slo-interactive-v2 (16 Nov 2026): joint attainment and complete success over OFFERED logical requests (the
    # finite cohort's declared size, or the logical requests started in the steady window; a retry is the same
    # logical request), so a run that stopped early cannot pass on the requests it happened to send.
    if constraints.get("attainment_min") is not None or constraints.get("complete_success_min") is not None:
        logical = {v.key[0] for v in measured}
        expected = (metrics.get("requests_expected") or {}).get("value")
        offered = max(int(expected or 0), len(logical)) or None
        observed = {"attainment_min": safe_div(len({v.key[0] for v in qualifying}), offered),
                    "complete_success_min": safe_div(len({v.key[0] for v in measured if v.success}), offered)}
        for name in ("attainment_min", "complete_success_min"):
            limit = constraints.get(name)
            if limit is not None:
                checks.append({"constraint": name, "basis": "logical requests / offered logical requests",
                               "offered": offered, "limit": limit, "observed": observed[name],
                               "ok": observed[name] is not None and observed[name] >= limit})
    reported = []
    if constraints.get("p99_token_gap_report_only"):
        basis = constraints.get("p99_token_gap_metric", "itl_token_ms")
        reported.append({"quantity": "p99_token_gap_ms", "basis": basis, "observed": metrics.get(basis, {}).get("p99"),
                         "chunk_gap_p99_ms": metrics.get("chunk_gap_ms", {}).get("p99"),
                         "note": "recorded separately, not part of the service gate"})
    required = constraints.get("quality_status_required")
    if required is not None:
        checks.append({"constraint": "quality_status_required", "required": required, "observed": quality_status,
                       "ok": quality_status == required})
    eligible = all(c["ok"] for c in checks) and len(qualifying) > 0
    out["global_constraints"] = {"checks": checks, **({"reported": reported} if reported else {})}
    out["operating_point_eligible"] = {"value": eligible,
                                       "reasons": [c["constraint"] for c in checks if not c["ok"]]
                                       + ([] if qualifying else ["no_qualifying_requests"])}
    return out, ok


def interactivity_slices(metrics: dict[str, Any], thresholds: list[float], statistic: str) -> dict:
    """Which declared per-user speed thresholds the operating point meets (local reporting slices)."""
    source = metrics.get("user_speed_post_first_token_tps", {})
    observed = source.get(statistic)
    return {
        "statistic": f"user_speed_post_first_token_tps.{statistic}",
        "observed": observed,
        "slices": [{"threshold_tps_per_user": t, "meets": observed is not None and observed >= t}
                   for t in thresholds],
    }


def qualifying_distribution(measured: list[RequestView], per: dict, ok: dict) -> dict:
    """TTFT of qualifying requests only (diagnostic: what a user inside the SLO sees)."""
    values = [per[v.key].ttft for v in measured if ok.get(v.key) and per[v.key].ttft is not None]
    return distribution(values, "ms", "request (qualifying)", 1000.0)
