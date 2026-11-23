"""Publication bundle: one HTML report plus portable CSV/JSON, figure sidecars, the claim ledger, the unmeasured-cell
ledger, release notes and captions, and the list of displayed values that the independent checker must reproduce.

  python -m lqh report RESULTS_DIR [RESULTS_DIR ...] --out reports/harness_<name>_<UTC stamp> [--plan-ledger]

Reads only run directories (metrics.json, run_manifest.json, validation.json). Every run is published with ALL its
metrics; failed, invalid and unsupported points stay visible; a missing value is shown as "null (reason)", never 0.
Nothing is submitted or published: the bundle is for review.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import stats
from .analyze import HARNESS_VERSION
from .claims import build_claims
from .eligibility import service_rule
from .metrics import METRIC_DEFS
from .util import canonical_json, harness_root, read_json, sha256_file, sha256_text, utc_now, write_json

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIST_FIELDS = ("n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max")
LATENCY_KEYS = ("ttft_ms", "tpot_native_ms", "tpot_terminal_ms", "span_last_first_ms", "itl_token_ms", "chunk_gap_ms",
                "e2e_terminal_ms", "e2e_last_token_ms", "user_speed_post_first_token_tps", "user_speed_e2e_tps")


# ---- loading --------------------------------------------------------------------------------------------------
def find_runs(roots: list[Path]) -> list[Path]:
    runs = []
    for root in roots:
        for manifest in sorted(Path(root).rglob("run_manifest.json")):
            runs.append(manifest.parent)
    return runs


def load_run(run_dir: Path) -> dict:
    metrics = read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else None
    manifest = read_json(run_dir / "run_manifest.json")
    validation = read_json(run_dir / "validation.json") if (run_dir / "validation.json").exists() else {}
    return {"dir": run_dir, "metrics": metrics, "manifest": manifest, "validation": validation}


def value(metric: dict | None, field: str = "value") -> Any:
    if not isinstance(metric, dict):
        return None
    if field == "value":
        return metric.get("value", metric.get("median") if "median" in metric else None)
    return metric.get(field)


def fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        if not math.isfinite(x):
            return "invalid"
        if x == 0:
            return "0"
        magnitude = abs(x)
        if magnitude >= 1000:
            return f"{x:,.1f}"
        if magnitude >= 1:
            return f"{x:.{max(1, digits - 1)}f}".rstrip("0").rstrip(".") if digits > 2 else f"{x:.2f}"
        return f"{x:.{digits}g}"
    return str(x)


def null_text(metric: dict | None) -> str:
    if not isinstance(metric, dict):
        return "null (not computed)"
    return f"null ({metric.get('null_reason') or 'not computed'})"


def cell_text(metric: dict | None, field: str = "value") -> str:
    v = value(metric, field)
    return fmt(v) if v is not None else null_text(metric)


# ---- flattening (long-form table: every metric of every run) ---------------------------------------------------
def flatten(run: dict) -> dict[str, Any]:
    m = run["metrics"] or {}
    manifest = run["manifest"]
    row: dict[str, Any] = {
        "run_id": manifest.get("run_id"), "status": m.get("status") or run["validation"].get("status"),
        "evidence_kind": manifest.get("evidence_kind"), "is_carried": manifest.get("is_carried"),
        "profile_id": manifest.get("profile_id"), "profile_version": manifest.get("profile_version"),
        "cell_id": manifest.get("cell_id"), "plan_id": manifest.get("plan_id"), "block_id": manifest.get("block_id"),
        "block_order_position": manifest.get("block_order_position"), "pair_id": manifest.get("pair_id"),
        "candidate_id": (manifest.get("candidate") or {}).get("candidate_id"),
        "candidate_config_hash": (manifest.get("candidate") or {}).get("config_hash"),
        "comparison_class": manifest.get("comparison_class"), "mock": (manifest.get("engine") or {}).get("mock"),
        "lane": (manifest.get("workload") or {}).get("lane"), "isl": (manifest.get("workload") or {}).get("isl"),
        "osl": (manifest.get("workload") or {}).get("osl"),
        "client_concurrency": (manifest.get("workload") or {}).get("concurrency"),
        "offered_rate_rps": (manifest.get("workload") or {}).get("rate_rps"),
        "gpus": (manifest.get("device") or {}).get("allocated_physical_gpus"),
        "gpu_name": ";".join(sorted({str(g.get("name")) for g in (manifest.get("device") or {}).get("gpus", [])})),
        "slo_id": manifest.get("slo_id"), "pricing_id": manifest.get("pricing_id"),
        "started_utc": manifest.get("started_utc"), "execution_status": (manifest.get("execution") or {}).get("status"),
        "execution_reason": (manifest.get("execution") or {}).get("reason"),
        "validation_errors": ";".join(run["validation"].get("errors", [])),
    }
    for name, metric in sorted((m.get("metrics") or {}).items()):
        if not isinstance(metric, dict):
            continue
        if "n" in metric and "unit" in metric and ("mean" in metric or metric.get("null_reason") == "no_observations"):
            for field in DIST_FIELDS:
                row[f"{name}.{field}"] = metric.get(field)
        elif "value" in metric:
            row[name] = metric.get("value")
            if metric.get("value") is None and metric.get("null_reason"):
                row[f"{name}.null_reason"] = metric["null_reason"]
    for key, val in (m.get("capacity") or {}).items():
        if key != "note":
            row[f"capacity.{key}"] = val
    for key, val in (m.get("site_metrics") or {}).items():
        if not isinstance(val, dict):
            row[f"site.{key}"] = val
    for alloc in (m.get("allocations") or []):
        row[f"allocation_gib.{alloc['category']}"] = alloc.get("gib")
    parity = m.get("parity")
    if parity:
        row["native_parity_ok"] = parity.get("ok")
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in keys})


# ---- summary: one row per profile / workload / operating point / candidate -------------------------------------
def headline_tps(run: dict) -> float | None:
    m = run["metrics"]
    if not m or m.get("status") not in ("measured", "smoke_only"):
        return None
    return (m.get("headline") or {}).get("deployment_output_tps")


def summarise(runs: list[dict], campaign: dict) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in runs:
        man = run["manifest"]
        key = (man.get("plan_id"), man.get("cell_id"), (man.get("candidate") or {}).get("candidate_id"))
        groups[key].append(run)
    rows = []
    for (plan, cell, cand), items in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        measured = [r for r in items if r["metrics"] and r["metrics"].get("status") in ("measured", "smoke_only")]
        tps = [headline_tps(r) for r in measured if headline_tps(r) is not None]
        qual = [(r["metrics"].get("headline") or {}).get("qualified_output_tps") for r in measured]
        qual = [q for q in qual if q is not None]
        eligible = [((r["metrics"].get("metrics") or {}).get("operating_point_eligible") or {}).get("value") for r in measured]
        first = (measured or items)[0]
        mm = (first["metrics"] or {}).get("metrics") or {}
        hashes = {(r["manifest"].get("candidate") or {}).get("config_hash") for r in items}
        rows.append({
            "plan_id": plan, "cell_id": cell, "candidate_id": cand, "profile_id": first["manifest"].get("profile_id"),
            "comparison_class": first["manifest"].get("comparison_class"),
            "runs": len(items), "measured_runs": len(measured),
            "statuses": ";".join(sorted({(r["metrics"] or {}).get("status") or "missing" for r in items}
                                        | ({"MIXED_CONFIGURATIONS"} if len(hashes) > 1 else set()))),
            "config_hashes": len(hashes),
            "output_tps_median": median(tps), "output_tps_min": min(tps) if tps else None,
            "output_tps_max": max(tps) if tps else None,
            "qualified_output_tps_median": median(qual),
            "eligible_runs": sum(1 for e in eligible if e), "mock": bool(first["manifest"].get("engine", {}).get("mock")),
            "client_concurrency": (first["manifest"].get("workload") or {}).get("concurrency"),
            "ttft_p50_ms": value(mm.get("ttft_ms"), "p50"), "ttft_p99_ms": value(mm.get("ttft_ms"), "p99"),
            "tpot_p50_ms": value(mm.get("tpot_native_ms"), "p50"),
            "itl_p99_ms": value(mm.get("itl_token_ms"), "p99") if value(mm.get("itl_token_ms"), "p99") is not None
            else value(mm.get("chunk_gap_ms"), "p99"),
            "error_rate": value(mm.get("error_rate")),
        })
    comparisons = []
    for plan_id, plan in (campaign.get("plans") or {}).items():
        baseline = plan.get("baseline") or (plan.get("candidates") or [None])[0]
        cells = {r["cell_id"] for r in rows if r["plan_id"] == plan_id}
        for cell in sorted(cells):
            base_runs = {r["manifest"].get("block_id"): headline_tps(r) for r in runs
                         if r["manifest"].get("plan_id") == plan_id and r["manifest"].get("cell_id") == cell
                         and (r["manifest"].get("candidate") or {}).get("candidate_id") == baseline}
            for cand in plan.get("candidates", []):
                if cand == baseline:
                    continue
                cand_runs = {r["manifest"].get("block_id"): headline_tps(r) for r in runs
                             if r["manifest"].get("plan_id") == plan_id and r["manifest"].get("cell_id") == cell
                             and (r["manifest"].get("candidate") or {}).get("candidate_id") == cand}
                blocks = sorted(b for b in set(base_runs) & set(cand_runs)
                                if base_runs[b] is not None and cand_runs[b] is not None)
                mixed = [r for r in rows if r["plan_id"] == plan_id and r["cell_id"] == cell
                         and r["candidate_id"] in (cand, baseline) and r.get("config_hashes", 1) > 1]
                if not blocks or mixed:
                    continue
                result = stats.paired_ratio([base_runs[b] for b in blocks], [cand_runs[b] for b in blocks])
                comparisons.append({"plan_id": plan_id, "cell_id": cell, "baseline": baseline, "candidate": cand,
                                    "blocks": blocks, **{k: result.get(k) for k in (
                                        "n_blocks", "estimator", "value", "low", "high", "geometric_mean",
                                        "mean_of_ratios", "ratio_of_means", "median_ratio", "interval_method",
                                        "interval_null_reason", "ratios")}})
    return rows, comparisons


def median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    return ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


# ---- figures with sidecars --------------------------------------------------------------------------------------
def candidate_colors(candidate_ids: list[str]) -> dict[str, str]:
    order = [c["candidate_id"] for c in read_json(harness_root() / "config" / "candidates.json")["candidates"]]
    present = [c for c in order if c in candidate_ids] + sorted(c for c in candidate_ids if c not in order)
    return {c: (PALETTE[i] if i < len(PALETTE) else "#8a8a8a") for i, c in enumerate(present)}


def frontier_figure(runs: list[dict], out: Path, figure_id: str, title: str, profile_filter=None) -> dict | None:
    """Output tok/s per GPU against median per-user speed after the first token; every concurrency labelled;
    failed points listed in the sidecar (never plotted as zero)."""
    points = []
    for run in runs:
        man, m = run["manifest"], run["metrics"] or {}
        if profile_filter and not profile_filter(man):
            continue
        mm = m.get("metrics") or {}
        x = value(mm.get("user_speed_post_first_token_tps"), "median")
        y = (m.get("headline") or {}).get("output_tps_per_gpu")
        points.append({"figure_id": figure_id, "run_id": man.get("run_id"),
                       "candidate_id": (man.get("candidate") or {}).get("candidate_id"),
                       "cell_id": man.get("cell_id"), "client_concurrency": (man.get("workload") or {}).get("concurrency"),
                       "x_per_user_tps_median": x, "x_unit": "tok/s/user (median, post first token)",
                       "y_output_tps_per_gpu": y, "y_unit": "tok/s/GPU (successful output)",
                       "status": m.get("status"), "plotted": x is not None and y is not None and m.get("status") == "measured",
                       "originating_experiment": man.get("experiment_id"), "block_id": man.get("block_id"),
                       "hardware": ";".join(sorted({str(g.get("name")) for g in (man.get("device") or {}).get("gpus", [])})),
                       "measurement_boundary": (m.get("headline") or {}).get("basis"),
                       "comparison_class": man.get("comparison_class"), "evidence_kind": man.get("evidence_kind"),
                       "derivation": "y = deployment successful output tok/s / allocated GPUs; x = median over succeeded requests of (N-1)/(last-first token)"})
    if not points:
        return None
    sidecar = {"figure_id": figure_id, "title": title, "generator": HARNESS_VERSION, "points": points}
    sidecar["data_sha256"] = sha256_text(canonical_json(points))
    write_json(out / f"{figure_id}.json", sidecar)
    write_csv(out / f"{figure_id}.csv", points)
    plotted = [p for p in points if p["plotted"]]
    if not plotted:
        return sidecar
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sidecar["image"] = None
        return sidecar
    colors = candidate_colors(sorted({p["candidate_id"] for p in plotted}))
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    for cand, color in colors.items():
        series = sorted((p for p in plotted if p["candidate_id"] == cand), key=lambda p: p["client_concurrency"] or 0)
        if not series:
            continue
        ax.plot([p["x_per_user_tps_median"] for p in series], [p["y_output_tps_per_gpu"] for p in series],
                color=color, linewidth=2, marker="o", markersize=5, label=cand)
        for p in series:
            ax.annotate(str(p["client_concurrency"]), (p["x_per_user_tps_median"], p["y_output_tps_per_gpu"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7, color="#555555")
    ax.set_xlabel("per-user speed after the first token, median (tok/s/user)")
    ax.set_ylabel("successful output tok/s per GPU")
    ax.set_title(title, fontsize=10)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(loc="lower center", ncol=min(4, len(colors)), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / f"{figure_id}.png")
    plt.close(fig)
    sidecar["image"] = f"{figure_id}.png"
    write_json(out / f"{figure_id}.json", sidecar)
    return sidecar


# ---- HTML ---------------------------------------------------------------------------------------------------------
CSS = """
:root{--bg:#ffffff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e8;--card:#f7f7f9;--bad:#b42318;--good:#067647;--warn:#b54708}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;--line:#2c2c31;--card:#1c1c20;--bad:#f97066;--good:#47cd89;--warn:#fdb022}}
:root[data-theme="dark"]{--bg:#141416;--fg:#ececf0;--muted:#a0a0a8;--line:#2c2c31;--card:#1c1c20;--bad:#f97066;--good:#47cd89;--warn:#fdb022}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;margin:0;padding:24px 16px;max-width:1400px;margin:auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px} h3{font-size:14px;margin:16px 0 6px}
.muted{color:var(--muted)} .bad{color:var(--bad)} .good{color:var(--good)} .warn{color:var(--warn)}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;margin:10px 0}
.scroll{overflow-x:auto} table{border-collapse:collapse;font-size:12px;margin:6px 0} th,td{border-bottom:1px solid var(--line);padding:3px 7px;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--card);font-weight:600} td.l,th.l{text-align:left}
details{border:1px solid var(--line);border-radius:6px;margin:6px 0;padding:4px 8px} summary{cursor:pointer;font-weight:600}
img{max-width:100%;height:auto} code{font-size:12px}
"""


def table(headers: list[str], rows: list[list[str]], left: int = 1) -> str:
    head = "".join(f"<th class='{'l' if i < left else ''}'>{html.escape(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td class='{'l' if i < left else ''}'>{c}</td>" for i, c in enumerate(r)) + "</tr>"
                   for r in rows)
    return f"<div class='scroll'><table><tr>{head}</tr>{body}</table></div>"


def run_section(run: dict) -> str:
    m = run["metrics"] or {}
    man = run["manifest"]
    status = m.get("status") or run["validation"].get("status")
    cls = "good" if status == "measured" else ("warn" if status == "smoke_only" else "bad")
    parts = [f"<details><summary>{html.escape(str(man.get('run_id')))} — <span class='{cls}'>{status}</span></summary>"]
    ident = [["run_id", man.get("run_id")], ["candidate", f"{(man.get('candidate') or {}).get('candidate_id')} "
                                               f"(config sha256 {str((man.get('candidate') or {}).get('config_hash'))[:16]}…)"],
             ["profile", f"{man.get('profile_id')} {man.get('profile_version')}"], ["cell", man.get("cell_id")],
             ["block / order position", f"{man.get('block_id')} / {man.get('block_order_position')}"],
             ["comparison class", man.get("comparison_class")], ["evidence kind", man.get("evidence_kind")],
             ["carried", man.get("is_carried")], ["started (UTC)", man.get("started_utc")],
             ["execution", f"{(man.get('execution') or {}).get('status')} {(man.get('execution') or {}).get('reason') or ''}"],
             ["GPU", ";".join(f"{g.get('name')} {g.get('uuid')}" for g in (man.get("device") or {}).get("gpus", []))],
             ["workload", html.escape(json.dumps(man.get("workload"), sort_keys=True))],
             ["SLO", man.get("slo_id")], ["pricing", man.get("pricing_id")]]
    parts.append(table(["field", "value"], [[html.escape(str(a)), html.escape(str(b))] for a, b in ident], left=2))
    cap = m.get("capacity") or {}
    if cap:
        parts.append("<h3>Capacity: configured slots, admitted running and waiting requests</h3>")
        parts.append(table(list(cap.keys()), [[html.escape(fmt(v)) for v in cap.values()]], left=0))
    metrics = m.get("metrics") or {}
    scalars, dists = [], []
    for name in sorted(metrics):
        metric = metrics[name]
        if not isinstance(metric, dict):
            continue
        definition = METRIC_DEFS.get(name)
        unit = metric.get("unit") or (definition.unit if definition else "")
        if "mean" in metric or metric.get("null_reason") == "no_observations" and "n" in metric:
            dists.append([name, unit, metric.get("aggregation", "")] +
                         [fmt(metric.get(f)) if metric.get(f) is not None else ("null" if f != "n" else "0")
                          for f in DIST_FIELDS])
        elif "value" in metric:
            scalars.append([name, unit, cell_text(metric)])
        elif name == "operating_point_eligible":
            scalars.append([name, "", f"{fmt(metric.get('value'))} {';'.join(metric.get('reasons', []))}"])
    eligible, reasons = service_rule(metrics, (m.get("slo") or {}).get("definition"))
    if eligible is not None:
        scalars.append(["service_eligible", "", ("yes" if eligible else "no: " + "; ".join(reasons))
                        + " (attainment >= 0.95, completion >= 0.99 and the SLO's global limits; quality is a "
                          "separate gate)"])
    parts.append(f"<h3>All scalar metrics ({len(scalars)})</h3>")
    parts.append(table(["metric", "unit", "value"], [[html.escape(str(c)) for c in r] for r in scalars], left=2))
    parts.append(f"<h3>All distributions ({len(dists)}) — aggregation unit stated per row</h3>")
    parts.append(table(["metric", "unit", "aggregation"] + list(DIST_FIELDS),
                       [[html.escape(str(c)) for c in r] for r in dists], left=3))
    for key, label in (("site_metrics", "Site-defined metrics"), ("api_v3_equivalents", "AA API v3 metric equivalents"),
                       ("interactivity", "Interactivity slices"), ("headline", "Headline")):
        if m.get(key):
            parts.append(f"<h3>{label}</h3><pre class='box'>{html.escape(json.dumps(m[key], indent=1))}</pre>")
    if m.get("parity"):
        rows = [[r["metric"], fmt(r.get("native")), fmt(r.get("harness")), fmt(r.get("abs_diff")), fmt(r.get("ok"))]
                for r in m["parity"]["rows"]]
        parts.append("<h3>Native upstream aggregates against the harness recomputation (parity)</h3>")
        parts.append(table(["metric", "native", "harness", "|diff|", "ok"], rows))
    checks = run["validation"].get("checks") or []
    failed = [c for c in checks if not c["ok"]]
    parts.append(f"<h3>Validation: {len(checks)} checks, {len(failed)} not passed</h3>")
    if failed:
        parts.append(table(["check", "severity", "detail"], [[html.escape(c["id"]), c["severity"],
                                                              html.escape(c["detail"][:300])] for c in failed], left=3))
    if m.get("allocations"):
        parts.append("<h3>Memory allocations from the server log</h3>")
        parts.append(table(["category", "GiB", "source", "note"], [[a["category"], fmt(a.get("gib")), a.get("source", ""),
                                                                    a.get("note", "")] for a in m["allocations"]], left=1))
    parts.append("</details>")
    return "\n".join(parts)


def build(roots: list[Path], out: Path, campaign_path: Path | None = None, title: str | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    campaign = read_json(campaign_path or harness_root() / "config" / "campaign.json")
    runs = [load_run(d) for d in find_runs(roots)]
    long_rows = [flatten(r) for r in runs]
    write_csv(out / "runs.csv", long_rows)
    write_json(out / "runs.json", long_rows)
    summary, comparisons = summarise(runs, campaign)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", {"rows": summary, "paired_comparisons": comparisons})
    figures_dir = out / "figures"
    figures_dir.mkdir()
    figures = []
    for profile in sorted({r["manifest"].get("profile_id") for r in runs}):
        for cell_prefix in sorted({(r["manifest"].get("workload") or {}).get("variant") for r in runs
                                   if r["manifest"].get("profile_id") == profile}):
            fig = frontier_figure(runs, figures_dir, f"frontier_{profile}_{cell_prefix}",
                                  f"{profile} {cell_prefix}: throughput per GPU vs per-user speed",
                                  lambda man, p=profile, c=cell_prefix: man.get("profile_id") == p
                                  and (man.get("workload") or {}).get("variant") == c)
            if fig:
                figures.append(fig)
    claims = build_claims(comparisons, summary, runs, campaign)
    write_json(out / "claims.json", claims)
    statuses = defaultdict(int)
    for r in runs:
        statuses[(r["metrics"] or {}).get("status") or r["validation"].get("status") or "missing"] += 1
    unmeasured = [{"run_id": r["manifest"].get("run_id"), "cell_id": r["manifest"].get("cell_id"),
                   "candidate": (r["manifest"].get("candidate") or {}).get("candidate_id"),
                   "status": (r["metrics"] or {}).get("status"),
                   "reason": (r["manifest"].get("execution") or {}).get("reason") or ";".join(r["validation"].get("errors", []))}
                  for r in runs if (r["metrics"] or {}).get("status") not in ("measured", "smoke_only")]
    write_csv(out / "unmeasured.csv", unmeasured or [{"run_id": "", "status": "", "reason": "none"}])
    write_json(out / "glossary.json", {k: vars(v) for k, v in METRIC_DEFS.items()})
    write_json(out / "sources.json", {"harness_version": HARNESS_VERSION, "campaign_id": campaign.get("campaign_id"),
                                      "campaign_sha256": sha256_file(campaign_path or harness_root() / "config" / "campaign.json"),
                                      "profiles_sha256": sha256_file(harness_root() / "config" / "profiles.json"),
                                      "candidates_sha256": sha256_file(harness_root() / "config" / "candidates.json"),
                                      "result_roots": [str(r) for r in roots]})
    from . import evidence_sections
    import shutil
    found = evidence_sections.discover(roots)
    evidence_html = evidence_sections.render(found)
    carried = evidence_sections.carried_page(found)
    if carried:
        (out / "carried.html").write_text(carried, encoding="utf-8")
        for path in found["carried_serving"]:
            shutil.copy(path, out / "carried_serving.json")
            if path.with_suffix(".csv").exists():
                shutil.copy(path.with_suffix(".csv"), out / "carried_serving.csv")
    displayed = displayed_values(runs, summary, comparisons) + evidence_sections.displayed(found)
    write_json(out / "displayed_values.json", displayed)
    mock = any(r["manifest"].get("engine", {}).get("mock") for r in runs)
    page = render(runs, summary, comparisons, figures, claims, statuses, unmeasured, campaign, mock,
                  title or f"Harness report {utc_now()[:16]}", evidence_html)
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "RELEASE_NOTES.md").write_text(release_notes(claims, statuses, mock), encoding="utf-8")
    (out / "CAPTIONS.md").write_text("\n\n".join(f"**{f['figure_id']}** — {f['title']}. Points: "
                                                 f"{sum(1 for p in f['points'] if p['plotted'])} plotted, "
                                                 f"{sum(1 for p in f['points'] if not p['plotted'])} listed in the sidecar "
                                                 f"but not plotted (failed, invalid or unmeasured). Data sha256 "
                                                 f"{f['data_sha256'][:16]}…; new raw measurements"
                                                 f"{' FROM A MOCK SERVER (no performance meaning)' if mock else ''}."
                                                 for f in figures) + "\n", encoding="utf-8")
    hashes = {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "MANIFEST.sha256").write_text("".join(f"{h}  {n}\n" for n, h in hashes.items()), encoding="utf-8")
    return {"out": str(out), "runs": len(runs), "statuses": dict(statuses), "figures": len(figures)}


def displayed_values(runs: list[dict], summary: list[dict], comparisons: list[dict]) -> list[dict]:
    """Every number the report shows from a run, with where it came from: the checker recomputes each one."""
    shown = []
    for run in runs:
        m = run["metrics"] or {}
        if m.get("status") not in ("measured", "smoke_only"):
            continue
        for name, metric in (m.get("metrics") or {}).items():
            if not isinstance(metric, dict):
                continue
            if "mean" in metric:
                for field in DIST_FIELDS:
                    if metric.get(field) is not None:
                        shown.append({"run_dir": str(run["dir"]), "run_id": m.get("run_id"), "metric": name,
                                      "field": field, "value": metric[field]})
            elif "value" in metric and isinstance(metric.get("value"), (int, float)) and not isinstance(metric.get("value"), bool):
                shown.append({"run_dir": str(run["dir"]), "run_id": m.get("run_id"), "metric": name, "field": "value",
                              "value": metric["value"]})
        # Service eligibility (multi-GPU benchmark plan, section 4) is shown, so the checker re-derives it.
        eligible, _ = service_rule(m.get("metrics") or {}, (m.get("slo") or {}).get("definition"))
        if eligible is not None:
            shown.append({"run_dir": str(run["dir"]), "run_id": m.get("run_id"), "metric": "service_eligible",
                          "field": "value", "value": 1.0 if eligible else 0.0})
    for comp in comparisons:
        if not isinstance(comp.get("geometric_mean"), (int, float)):
            continue    # not computable (no positive pair): the table shows null with its reason, not a number
        shown.append({"comparison": f"{comp['plan_id']}|{comp['cell_id']}|{comp['candidate']}/{comp['baseline']}",
                      "metric": "paired_geometric_mean_ratio", "field": "value", "value": comp.get("geometric_mean"),
                      "blocks": comp.get("blocks")})
    return shown


def render(runs, summary, comparisons, figures, claims, statuses, unmeasured, campaign, mock, title,
           evidence_html: str = "") -> str:
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"]
    parts.append(f"<h1>{html.escape(title)}</h1><div class='muted'>{HARNESS_VERSION} · campaign {campaign.get('campaign_id')} · "
                 f"{campaign.get('hardware', {}).get('label', '')} · generated {utc_now()}</div>")
    if mock:
        parts.append("<div class='box warn'><b>Mock server runs present.</b> They establish harness behaviour only and "
                     "have no performance meaning; they are never used in performance claims.</div>")
    parts.append("<div class='box'>Evidence classes: <b>measured_raw</b> (new runs of this harness), <b>reported_aggregate</b>, "
                 "<b>derived_from_reported</b>, <b>unverified_assertion</b>, <b>planned</b>. Independent results: InferenceX-derived rows are "
                 "<b>Unofficial</b>; AA rows are <b>AA-SLT-method-inspired</b>, not Artificial Analysis measurements. "
                 "Missing values are shown as <code>null (reason)</code>, never as 0.</div>")
    parts.append("<h2>Run status</h2>" + table(["status", "runs"], [[k, str(v)] for k, v in sorted(statuses.items())]))
    if evidence_html:
        parts.append(evidence_html)
    parts.append("<h2>Main table: one row per plan, cell and candidate</h2>")
    parts.append(table(["plan", "cell", "candidate", "class", "runs", "measured", "statuses", "client conc.",
                        "output tok/s (median)", "min", "max", "SLO-qualified tok/s", "eligible runs", "error rate",
                        "TTFT p50 ms", "TTFT p99 ms", "TPOT p50 ms", "ITL p99 ms"],
                       [[html.escape(str(r[k])) if not isinstance(r[k], float) else fmt(r[k]) for k in
                         ("plan_id", "cell_id", "candidate_id", "comparison_class", "runs", "measured_runs", "statuses",
                          "client_concurrency", "output_tps_median", "output_tps_min", "output_tps_max",
                          "qualified_output_tps_median", "eligible_runs", "error_rate", "ttft_p50_ms", "ttft_p99_ms",
                          "tpot_p50_ms", "itl_p99_ms")] for r in summary], left=4))
    if comparisons:
        parts.append("<h2>Paired comparisons (successful output tok/s, candidate / baseline, per block)</h2>")
        parts.append("<div class='muted'>Headline estimator: geometric mean of per-block ratios with a t interval on log "
                     "ratios. Mean of ratios and ratio of means are shown so they are never confused. Speed ratio s means "
                     f"time ratio 1/s and, at the same hourly price, cost ratio 1/s.</div>")
        parts.append(table(["plan", "cell", "candidate", "baseline", "blocks", "geo-mean ratio", "95% low", "95% high",
                            "mean of ratios", "ratio of means", "per-block ratios"],
                           [[c["plan_id"], c["cell_id"], c["candidate"], c["baseline"], str(c["n_blocks"]),
                             fmt(c.get("geometric_mean")), fmt(c.get("low")) if c.get("low") else f"null ({c.get('interval_null_reason')})",
                             fmt(c.get("high")) if c.get("high") else "null", fmt(c.get("mean_of_ratios")),
                             fmt(c.get("ratio_of_means")), ", ".join(fmt(x) for x in c.get("ratios") or [])]
                            for c in comparisons], left=4))
    parts.append("<h2>Capacity: configured slots, admitted running requests and waiting requests, every run</h2>")
    cap_rows = []
    for r in runs:
        cap = (r["metrics"] or {}).get("capacity") or {}
        cap_rows.append([html.escape(str(r["manifest"].get("run_id")))] + [fmt(cap.get(k)) for k in (
            "configured_state_slots", "server_max_running_requests", "kv_tokens", "client_concurrency",
            "admitted_running_median", "admitted_running_max", "waiting_median", "waiting_max")])
    parts.append(table(["run", "configured slots", "server max running", "KV tokens", "client concurrency",
                        "running median", "running max", "waiting median", "waiting max"], cap_rows))
    parts.append("<h2>Latency and per-user speed, every measured run (p50 / p90 / p95 / p99)</h2>")
    lat_rows = []
    for r in runs:
        mm = (r["metrics"] or {}).get("metrics") or {}
        if (r["metrics"] or {}).get("status") not in ("measured", "smoke_only"):
            continue
        row = [html.escape(str(r["manifest"].get("run_id")))]
        for key in LATENCY_KEYS:
            metric = mm.get(key) or {}
            row.append(" / ".join(fmt(metric.get(f)) for f in ("p50", "p90", "p95", "p99")) if metric.get("n")
                       else null_text(metric))
        lat_rows.append(row)
    parts.append(table(["run"] + [f"{k} ({(METRIC_DEFS.get(k).aggregation if METRIC_DEFS.get(k) else '')})" for k in LATENCY_KEYS],
                       lat_rows))
    parts.append("<h2>Cost</h2>")
    priced = [r for r in runs if (((r["metrics"] or {}).get("metrics") or {}).get("usd_per_hour") or {}).get("value") is not None]
    if not priced:
        reason = next(((((r["metrics"] or {}).get("metrics") or {}).get("usd_per_hour") or {}).get("null_reason")
                       for r in runs if r["metrics"]), "price_missing")
        parts.append(f"<div class='box'>No dollar metric is shown: <code>{html.escape(str(reason))}</code>. "
                     "harness/config/pricing.json has no recorded price for this host; USD per million tokens stays null.</div>")
    parts.append("<h2>Figures</h2>")
    for fig in figures:
        parts.append(f"<h3>{html.escape(fig['title'])}</h3>")
        if fig.get("image"):
            parts.append(f"<img src='figures/{fig['image']}' alt='{html.escape(fig['title'])}'>")
        parts.append(f"<div class='muted'>Sidecar: figures/{fig['figure_id']}.json / .csv (every point, run id, unit, "
                     f"boundary, class, derivation); data sha256 {fig['data_sha256'][:16]}…</div>")
    parts.append("<h2>Claim ledger</h2>")
    if claims.get("claims"):
        parts.append(table(["claim", "metric", "cells", "baseline", "uncertainty", "eligibility", "limitation"],
                           [[html.escape(c["text"]), c["metric"], html.escape(";".join(c["cells"])), c["baseline"],
                             html.escape(c["uncertainty"]), html.escape(c["eligibility"]), html.escape(c["limitation"])]
                            for c in claims["claims"]], left=7))
    else:
        parts.append(f"<div class='box'>{html.escape(claims.get('note', 'no claims'))}</div>")
    parts.append("<h2>Planned or attempted points not measured</h2>")
    parts.append(table(["run", "cell", "candidate", "status", "reason"],
                       [[html.escape(str(u.get(k))) for k in ("run_id", "cell_id", "candidate", "status", "reason")]
                        for u in unmeasured], left=5) if unmeasured else "<div class='muted'>none</div>")
    parts.append(f"<h2>Every run, every metric ({len(runs)} runs)</h2>")
    for run in sorted(runs, key=lambda r: str(r["manifest"].get("run_id"))):
        parts.append(run_section(run))
    parts.append("<h2>Glossary</h2>")
    parts.append(table(["metric", "unit", "aggregation", "definition", "numerator", "denominator"],
                       [[k, d.unit, d.aggregation, html.escape(d.definition), html.escape(d.numerator),
                         html.escape(d.denominator)] for k, d in METRIC_DEFS.items()], left=6))
    parts.append("<p class='muted'>Exports: runs.csv / runs.json (every metric of every run), summary.csv / summary.json, "
                 "claims.json, unmeasured.csv, glossary.json, sources.json, displayed_values.json (checked by "
                 "<code>python -m lqh check</code>), figures/, MANIFEST.sha256. Nothing here has been submitted or published.</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def release_notes(claims: dict, statuses: dict, mock: bool) -> str:
    lines = ["# Release-note draft (not published)", "",
             f"Runs by status: {', '.join(f'{k} {v}' for k, v in sorted(statuses.items()))}.", ""]
    if mock:
        lines.append("This bundle includes mock-server smoke runs: harness behaviour only, no performance meaning.\n")
    for claim in claims.get("claims", []):
        lines.append(f"- {claim['text']} ({claim['uncertainty']}; {claim['eligibility']}; {claim['limitation']})")
    if not claims.get("claims"):
        lines.append(f"- {claims.get('note', 'No claim is supported yet.')}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh report")
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    result = build([Path(r) for r in args.roots], Path(args.out), Path(args.campaign) if args.campaign else None,
                   args.title)
    print(json.dumps(result))
    return 0
