"""Claim ledger: publication text is generated from evidence, never written free-hand.

A claim needs: a paired comparison of measured (non-mock) runs with at least two blocks, so that it carries an
interval; its metric, cells, named baseline, scope, uncertainty, eligibility and limitation. Wording that needs
separate evidence is refused: production-ready, lossless, universally faster, official InferenceX, official AA,
reproduced paper result, quality-equivalent.
"""
from __future__ import annotations

import re

BANNED = [
    (r"production[- ]ready", "production readiness is a release decision (gate G5), not a benchmark result"),
    (r"\blossless\b", "the state is compressed; quality is qualified by noninferiority at a margin"),
    (r"universally faster|always faster|faster everywhere", "results hold for the measured cells only"),
    (r"official\s+inferencex", "InferenceX-derived results are Unofficial"),
    (r"official\s+(aa|artificial analysis)", "AA-SLT results are method-inspired, not AA measurements"),
    (r"reproduc(ed|es) the paper|paper reproduced", "an H100/SGLang port is not a reproduction of B200/vLLM results"),
    (r"quality[- ]equivalent", "quality is noninferiority at a declared margin, not equivalence"),
]


class WordingError(ValueError):
    pass


def check_wording(text: str) -> None:
    for pattern, reason in BANNED:
        if re.search(pattern, text, flags=re.IGNORECASE):
            raise WordingError(f"refused wording {pattern!r} in claim: {reason}")


def _fmt(x: float | None, digits: int = 3) -> str:
    return "null" if x is None else f"{x:.{digits}f}"


def build_claims(comparisons: list[dict], summary: list[dict], runs: list[dict], campaign: dict) -> dict:
    mock_cells = {(r["manifest"].get("plan_id"), r["manifest"].get("cell_id")) for r in runs
                  if (r["manifest"].get("engine") or {}).get("mock")}
    quality = campaign.get("quality_qualification", {})
    claims = []
    refused = []
    for comp in comparisons:
        key = (comp["plan_id"], comp["cell_id"])
        if key in mock_cells:
            refused.append({"comparison": key, "reason": "mock server: no performance meaning"})
            continue
        if comp.get("n_blocks", 0) < 2 or comp.get("low") is None:
            refused.append({"comparison": key, "reason": "fewer than two paired blocks: no interval"})
            continue
        g, lo, hi = comp.get("geometric_mean"), comp.get("low"), comp.get("high")
        change = (g - 1.0) * 100.0
        word = "higher" if change >= 0 else "lower"
        text = (f"Observed successful output throughput of {comp['candidate']} {abs(change):.1f}% {word} than "
                f"{comp['baseline']} on {comp['cell_id']} (ratio {_fmt(g)}, 95% interval {_fmt(lo)} to {_fmt(hi)}, "
                f"{comp['n_blocks']} paired blocks)")
        if lo is not None and hi is not None and lo <= 1.0 <= hi:
            text += "; the interval includes no change"
        check_wording(text)
        rows = [s for s in summary if s["plan_id"] == comp["plan_id"] and s["cell_id"] == comp["cell_id"]]
        latency = "; ".join(f"{s['candidate_id']} TTFT p99 {_fmt(s.get('ttft_p99_ms'), 1)} ms, ITL p99 "
                            f"{_fmt(s.get('itl_p99_ms'), 2)} ms" for s in rows)
        claims.append({
            "text": text, "metric": "deployment successful output tok/s (geometric mean of per-block ratios)",
            "cells": [comp["cell_id"]], "plan": comp["plan_id"], "baseline": comp["baseline"],
            "scope": "1 x H100 PCIe, SGLang c8ea707, Qwen3.5-9B; this cell only",
            "uncertainty": f"95% t interval on log ratios, {comp['n_blocks']} blocks; per-block ratios "
                           f"{', '.join(_fmt(r) for r in comp.get('ratios') or [])}",
            "latency_cost": latency,
            "eligibility": "experimental: quality not yet qualified (campaign quality_qualification pending)"
            if not quality.get("result") else quality["result"],
            "limitation": "H100 port evaluated with paper-inspired methodology; not a paper reproduction; "
                          "screening results are not confirmation",
        })
    note = "" if claims else ("No claim: " + ("; ".join(sorted({r['reason'] for r in refused})) if refused
                                              else "no paired comparison yet"))
    return {"claims": claims, "refused": refused, "note": note,
            "banned_wording": [{"pattern": p, "reason": r} for p, r in BANNED]}
