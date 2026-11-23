"""Campaign-report sections for the evidence that is not a serving run: the full-cycle kernel timing, compute-sanitizer
records, lifecycle tests, quality decisions, long-context and compounding NLL, and plateau sweeps.

discover(roots) finds the summary files; render(found) gives HTML; displayed(found) lists every number shown, with
what the independent checker must recompute it from (lqh.checker.check_evidence re-derives each from raw files).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

PATTERNS = {
    "kernel_cycle": "kernel_cycle.json",
    "sanitizer": "sanitizer_results.json",
    "lifecycle": "lifecycle_results.json",
    "quality": "quality_decision.json",
    "nll_compare": "nll_compare_c*.json",
    "compounding": "compounding_compare.json",
    "plateau": "plateau_*.json",
    "carried_serving": "carried_serving.json",
}

KEY_METRICS = ("Output token throughput (tok/s)", "Request throughput (req/s)", "Median TTFT (ms)", "P99 TTFT (ms)",
               "Median TPOT (ms)", "P99 ITL (ms)", "Median E2E Latency (ms)", "Request goodput (req/s)")


def discover(roots: list[Path]) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {k: [] for k in PATTERNS}
    for root in roots:
        for kind, pattern in PATTERNS.items():
            found[kind] += sorted(Path(root).rglob(pattern))
    return found


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(x, digits=4) -> str:
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}g}" if abs(x) < 1000 else f"{x:,.1f}"
    return html.escape(str(x))


def _table(headers, rows) -> str:
    head = "".join(f"<th class='l'>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td class='l'>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><tr>{head}</tr>{body}</table></div>"


def render(found: dict[str, list[Path]]) -> str:
    parts = []
    for path in found["kernel_cycle"]:
        doc = _load(path)
        rows = []
        for batch, entry in sorted(doc["batches"].items(), key=lambda kv: -int(kv[0])):
            ratio = entry["speed_vs_bf16_window"]
            lq = sorted(entry["lq_ms_per_token"])
            bf = sorted(entry["bf16_window_ms_per_token"])
            rows.append([batch, str(entry["repetitions"]), _f(lq[len(lq) // 2] if lq else None),
                         _f(bf[len(bf) // 2] if bf else None), _f(ratio.get("geometric_mean")),
                         f"{_f(ratio.get('low'))} .. {_f(ratio.get('high'))}",
                         _f(entry.get("speed_vs_fp32_window", {}).get("geometric_mean")),
                         _f(entry["state_advanced_every_rep"]),
                         html.escape(entry.get("gate_p5_G3", {}).get("status", ""))])
        parts.append(f"<h2>Kernel: complete state-advancing layer cycle (order of work 1)</h2><div class='muted'>"
                     f"{html.escape(doc['method'])}. Source: {html.escape(str(path))}</div>")
        parts.append(_table(["batch", "reps", "LeapQuant ms/token (median)", "bf16 window ms/token (median)",
                             "bf16 window / LeapQuant (geo-mean)", "95% interval", "fp32 window / LeapQuant",
                             "state advanced", "gate (v6 G2, threshold 1.25 on the lower bound)"], rows))
    for path in found["sanitizer"]:
        doc = _load(path)
        parts.append(f"<h2>compute-sanitizer on the final kernels</h2><div class='muted'>{html.escape(str(path))}</div>")
        parts.append(_table(["log", "tool", "exit", "classification", "summary", "coverage"],
                            [[html.escape(Path(r["log"]).name), r["tool"], _f(r.get("exit_status")),
                              r["classification"], html.escape(str(r.get("summary"))),
                              html.escape(str(r.get("expected_coverage")))] for r in doc["records"]]))
    for path in found["lifecycle"]:
        doc = _load(path)
        parts.append(f"<h2>Lifecycle and numerical tests of the final binary</h2><div class='muted'>"
                     f"{html.escape(doc.get('environment', ''))}; {html.escape(str(path))}</div>")
        parts.append(_table(["test", "status", "exit", "log sha256"],
                            [[r["test"], r["status"], str(r["exit_status"]), r["log_sha256"][:16] + "…"] for r in doc["tests"]]))
    for path in found["quality"]:
        doc = _load(path)
        rows = []
        for arm, tasks in doc.get("arms", {}).items():
            for task, r in tasks.items():
                rows.append([arm, task, r.get("verdict", ""), _f(r.get("accuracy_arm")), _f(r.get("accuracy_reference")),
                             _f(r.get("difference")), f"{_f(r.get('low'))} .. {_f(r.get('high'))}",
                             _f(r.get("cap_rate_arm")), _f(r.get("samples")), html.escape(r.get("reason", ""))])
        decision = doc.get("decision", {})
        parts.append(f"<h2>Quality qualification against {html.escape(doc.get('reference_arm', 'fp32'))} "
                     f"(margin {doc.get('margin_points')} point)</h2><div class='muted'>{html.escape(str(path))}</div>")
        parts.append(_table(["arm", "task", "verdict", "accuracy", "reference", "difference", "95% interval",
                             "cap rate", "samples", "reason"], rows))
        parts.append("<div class='box'>" + "<br>".join(f"<b>{html.escape(k)}</b>: {html.escape(str(v))}"
                                                     for k, v in decision.items()) + "</div>")
    for path in found["nll_compare"] + found["compounding"]:
        doc = _load(path)
        rows = []
        for cand, buckets in doc.get("candidates", {}).items():
            for key, r in sorted(buckets.items()):
                rows.append([html.escape(Path(cand).name), key, _f(r["delta_nll_mean"]),
                             f"{_f(r['low'])} .. {_f(r['high'])}", str(r["documents"]),
                             _f(r["relative_perplexity_change"])])
        title = "Compounding test (one layer at a time vs all layers)" if "compounding" in path.name else \
            f"Long-context NLL by position ({path.stem})"
        parts.append(f"<h2>{title}</h2><div class='muted'>NLL difference against {html.escape(str(doc.get('reference')))}; "
                     f"unit of uncertainty: the document (bootstrap). {html.escape(str(path))}</div>")
        parts.append(_table(["configuration", "length:bucket", "ΔNLL (mean)", "95% interval", "documents",
                             "relative perplexity change"], rows))
        if doc.get("compounding"):
            parts.append(_table(["length:bucket", "sum of single-layer ΔNLL", "all-layer ΔNLL"],
                                [[k, _f(v["sum_of_single_layer_deltas"]), _f(v["all_layer_delta"])]
                                 for k, v in sorted(doc["compounding"].items())]))
    for path in found["carried_serving"]:
        doc = _load(path)
        rows = doc["rows"]
        flagged = [r for r in rows if len(r["flags"]) > 1]
        parts.append(f"<h2>Carried historical serving evidence ({len(rows)} client logs, "
                     f"{len({r['origin_experiment_id'] for r in rows})} experiments)</h2>")
        parts.append("<div class='box'>Carried history, not new measurements: each row is one earlier client log with every "
                     "metric that client reported (evidence kind <b>reported_aggregate</b>; per-request events were not kept), "
                     "its original identifier and source hash. Rows are single runs, never repetitions of new runs, and never "
                     f"enter new paired comparisons. {len(flagged)} rows carry flags (copied file content, one arm label over "
                     "several binaries, no build record, or a server that admitted fewer requests than the client sent). "
                     "Full tables: <a href='carried.html'>carried.html</a>; every metric: carried_serving.csv.</div>")
    for path in found["plateau"]:
        doc = _load(path)
        parts.append(f"<h2>AA-SLT plateau sweep: {html.escape(doc['candidate'])}</h2><div class='muted'>"
                     f"{html.escape(doc['label'])}; stop: {html.escape(doc['stop_reason'])}</div>")
        parts.append(_table(["concurrency", "steady output tok/s", "error rate"],
                            [[str(p["concurrency"]), _f(p["steady_output_tps"]), _f(p["error_rate"])] for p in doc["points"]]))
    return "\n".join(parts)


def displayed(found: dict[str, list[Path]]) -> list[dict]:
    """Every number the evidence sections show, keyed so the checker can recompute it from the raw files."""
    shown = []
    for path in found["kernel_cycle"]:
        for batch, entry in _load(path)["batches"].items():
            ratio = entry["speed_vs_bf16_window"]
            for field in ("geometric_mean", "low", "high"):
                if ratio.get(field) is not None:
                    shown.append({"evidence": "kernel_cycle", "file": str(path), "batch": batch, "field": field,
                                  "value": ratio[field]})
    for path in found["sanitizer"]:
        for index, r in enumerate(_load(path)["records"]):
            shown.append({"evidence": "sanitizer", "file": str(path), "index": index, "field": "classification",
                          "value": r["classification"], "log": r["log"], "tool": r["tool"]})
    for path in found["quality"]:
        for arm, tasks in _load(path).get("arms", {}).items():
            for task, r in tasks.items():
                for field in ("accuracy_arm", "accuracy_reference", "difference", "cap_rate_arm", "low", "high"):
                    if r.get(field) is not None:
                        shown.append({"evidence": "quality", "file": str(path), "arm": arm, "task": task,
                                      "field": field, "value": r[field]})
    for path in found["carried_serving"]:
        for row in _load(path)["rows"]:
            for name, value in row["metrics"].items():
                shown.append({"evidence": "carried_metric", "file": row["source_file"], "row_id": row["row_id"],
                              "field": name, "value": value})
            server = row.get("server") or {}
            for field in ("admitted_running_max", "waiting_max"):
                if server.get(field) is not None:
                    shown.append({"evidence": "carried_server", "file": server["log"], "row_id": row["row_id"],
                                  "field": field, "value": server[field], "log_end_unix": server["log_end_unix"],
                                  "duration": row["metrics"].get("Benchmark duration (s)") or 0.0})
    for path in found["nll_compare"] + found["compounding"]:
        for cand, buckets in _load(path).get("candidates", {}).items():
            for key, r in buckets.items():
                shown.append({"evidence": "nll", "file": str(path), "candidate": cand, "bucket": key,
                              "field": "delta_nll_mean", "value": r["delta_nll_mean"]})
    return shown


def carried_page(found: dict[str, list[Path]]) -> str | None:
    """carried.html: per experiment, every carried row with capacity facts, key metrics and flags; all metrics in a
    details block per row."""
    if not found["carried_serving"]:
        return None
    rows = []
    for path in found["carried_serving"]:
        rows += _load(path)["rows"]
    by_exp: dict[str, list[dict]] = {}
    for row in rows:
        by_exp.setdefault(row["origin_experiment_id"], []).append(row)
    css = ("body{font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1500px;margin:auto;padding:20px 16px;"
           "background:#fff;color:#1d1d1f}table{border-collapse:collapse;font-size:11.5px}td,th{border-bottom:1px solid #e3e3e8;"
           "padding:2px 6px;text-align:right;white-space:nowrap}td.l,th.l{text-align:left}.scroll{overflow-x:auto}"
           "details{margin:2px 0}.flag{color:#b54708}@media (prefers-color-scheme: dark){body{background:#141416;color:#ececf0}"
           "td,th{border-color:#2c2c31}}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
             f"initial-scale=1'><title>Carried serving evidence</title><style>{css}</style></head><body>",
             "<h1>Carried historical serving evidence</h1><p>Client-reported aggregates of earlier runs (evidence kind "
             "reported_aggregate), single runs, with the server's configured slots and the requests it actually admitted "
             "and kept waiting over each point's interval (approximate interval from the client log's time and duration). "
             "Not new measurements; never repetitions.</p>"]
    headers = ["row", "arm", "radix", "shape", "clients / rate", "configured slots", "running max", "waiting max",
               "peak GPU MiB"] + list(KEY_METRICS) + ["flags"]
    for exp in sorted(by_exp):
        parts.append(f"<h2>{html.escape(exp)} ({len(by_exp[exp])} rows)</h2><div class='scroll'><table><tr>"
                     + "".join(f"<th class='{'l' if i < 4 else ''}'>{html.escape(h)}</th>" for i, h in enumerate(headers)) + "</tr>")
        for row in sorted(by_exp[exp], key=lambda r: (r["arm"], r.get("shape") or "", r.get("client_concurrency") or 0,
                                                      r.get("offered_rate_rps") or 0)):
            server = row.get("server") or {}
            load = row.get("client_concurrency") if row.get("client_concurrency") is not None else f"{row.get('offered_rate_rps')} req/s"
            cells = [html.escape(Path(row["row_id"]).name), html.escape(row["arm"]), html.escape(str(row.get("radix") or "")),
                     html.escape(str(row.get("shape") or "")), html.escape(str(load)), _f(server.get("configured_slots")),
                     _f(server.get("admitted_running_max")), _f(server.get("waiting_max")), _f(row.get("peak_gpu_memory_mib"))]
            cells += [_f(row["metrics"].get(k)) for k in KEY_METRICS]
            flags = [f for f in row["flags"] if f != "single_run"]
            cells.append(f"<span class='flag'>{html.escape('; '.join(flags))}</span>")
            details = "".join(f"{html.escape(k)}: {_f(v)}<br>" for k, v in row["metrics"].items())
            parts.append("<tr>" + "".join(f"<td class='{'l' if i < 4 or i == len(cells) - 1 else ''}'>{c}</td>"
                                          for i, c in enumerate(cells)) + "</tr>")
            parts.append(f"<tr><td class='l' colspan='{len(cells)}'><details><summary>all {len(row['metrics'])} metrics, "
                         f"client settings, server facts</summary>{details}<br>client: {html.escape(str(row.get('client')))}"
                         f"<br>server: {html.escape(str({k: v for k, v in server.items() if k != 'allocations_gib'}))}"
                         f"<br>allocations (GiB): {html.escape(str(server.get('allocations_gib')))}<br>source sha256 "
                         f"{row['source_artifact_sha256'][:16]}…</details></td></tr>")
        parts.append("</table></div>")
    parts.append("</body></html>")
    return "\n".join(parts)
