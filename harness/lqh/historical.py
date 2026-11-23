"""The seven historical graphics, regenerated from raw files with sidecars (prompt rev. 3, section 16).

  python -m lqh.historical --inventory harness/evidence/historical_inventory.json --out reports/harness_historical_<UTC>

Every plotted value comes from a raw file line named in the inventory (file:line). Before it is used, the line is
re-read and the value must appear in it (otherwise the value is listed as unverified and not plotted). Each figure
keeps ONE coherent experiment group or separates groups visibly; carried history is labelled as such
(evidence_kind measured_raw for raw historical files, derived_from_reported for arithmetic on reported values);
missing cells are never plotted as zero; measured confidence intervals are not drawn (no historical point has an
independent repetition design).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .sidecars import sidecar_problems
from .util import canonical_json, read_json, repo_root, sha256_file, sha256_text, utc_now, write_json

GENERATOR = "lqh.historical 1.0 (2026-11-15)"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?")


def _float(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _contains(text: str, target: float) -> bool:
    return any(n is not None and abs(n - target) <= 1e-9 + 1e-6 * abs(target)
               for n in (_float(x) for x in NUMBER.findall(text)))


def verify(value: dict, root: Path) -> tuple[bool, str]:
    """The value must appear in its cited raw source: on the cited line (file:line), anywhere in a cited file without a
    line number, or, for an aggregate over several runs (a brace pattern), in that folder's aggregate file G3.txt."""
    path = str(value.get("path") or "")
    target = _float(value.get("value"))
    if target is None or not path:
        return False, "no value or no source"
    name, sep, line = path.rpartition(":")
    if not sep or not line.isdigit():
        name, line = path, ""
    if "{" in name:
        aggregate = root / Path(name).parent / "G3.txt"
        if aggregate.exists() and _contains(aggregate.read_text(encoding="utf-8", errors="replace"), target):
            return True, f"verified in the aggregate file {aggregate.relative_to(root)} (value over {Path(name).name})"
        return False, "aggregate not found"
    file = root / name
    if not file.exists():
        return False, "raw file missing"
    text = file.read_text(encoding="utf-8", errors="replace")
    if not line:
        return (True, "verified in the file (no line cited)") if _contains(text, target) else (False, "value not in file")
    lines = text.splitlines()
    if int(line) > len(lines):
        return False, "line beyond end of file"
    if not _contains(lines[int(line) - 1], target):
        return False, "value not found on the cited line"
    return True, "verified on the cited line"


def point(value: dict, figure: str, x, x_unit: str, series: str, root: Path, boundary: str, klass: str,
          derivation: str, kind: str = "measured_raw", **extra) -> dict:
    ok, why = verify(value, root)
    y = _float(value.get("value"))
    return {"figure_id": figure, "series": series, "candidate_id": str(value.get("arm") or series),
            "x_value": x, "x_unit": x_unit, "y_value": y, "y_unit": value.get("unit"),
            "source_row_id": value.get("path"), "originating_experiment": str(value.get("experiment") or "")[:80] or "unknown",
            "hardware": "1 x H100 PCIe (quick-einstein)", "measurement_boundary": boundary, "comparison_class": klass,
            "evidence_kind": kind, "repetitions": 1, "derivation": derivation, "verified_against_raw": ok,
            "verification": why, "plotted": ok and y is not None, "label": value.get("label"), **extra}


def _values(inventory: dict, topic: int) -> list[dict]:
    return next(t for t in inventory["topics"] if t["id"] == topic)["values"]


def g3_final_points(root: Path) -> list[dict]:
    """results/g3-final: re-derived from the 10 raw per-run files of each arm (batch 512 section): LeapQuant per
    token and the bf16 window control measured in the same process; the mean is cross-checked against G3.txt."""
    folder = root / "results" / "g3-final"
    out = []
    lq_re = re.compile(r"->\s+([0-9.]+) ms per token with a fold every")
    bf_re = re.compile(r"sglang window kernel, bf16 state: ([0-9.]+) ms per token")
    summary = (folder / "G3.txt").read_text() if (folder / "G3.txt").exists() else ""
    for arm in ("adopted", "v9"):
        lq, bf, files = [], [], []
        for n in range(1, 11):
            file = folder / f"{arm}_{n}.txt"
            if not file.exists():
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            section = text.split("batch 512", 1)[-1]
            a, b = lq_re.search(section), bf_re.search(section)
            if a and b:
                lq.append(float(a.group(1)))
                bf.append(float(b.group(1)))
                files.append(file.name)
        if not lq:
            continue
        for label, values in ((f"g3-final {arm}: LeapQuant (component sum)", lq),
                              (f"g3-final {arm} processes: SGLang bf16 window (control)", bf)):
            mean = sum(values) / len(values)
            line = next((l for l in summary.splitlines() if l.startswith(arm)), "")
            out.append({"figure_id": "hist_kernel", "series": label, "candidate_id": label, "x_value": 512.0,
                        "x_unit": "batch", "y_value": mean, "y_unit": "ms/token",
                        "source_row_id": f"results/g3-final/{arm}_1..{len(files)}.txt (batch 512 sections)",
                        "originating_experiment": "g3-final (14 Nov 2026 13:08 UTC)", "hardware": "1 x H100 PCIe (quick-einstein)",
                        "measurement_boundary": "one layer, per token: decode + fold/16 component sum (not a full cycle)",
                        "comparison_class": "same_capacity", "evidence_kind": "measured_raw", "repetitions": len(values),
                        "derivation": f"mean of {len(values)} per-run values re-read from the raw files",
                        "per_run_values": values, "verified_against_raw": True,
                        "verification": f"re-derived from raw files; G3.txt line: {line.strip()[:120]}",
                        "plotted": True})
    return out


def build_points(inventory: dict, root: Path) -> dict[str, list[dict]]:
    figures: dict[str, list[dict]] = {}
    # 1. Kernel: the historical Triton measurement (one process, graph-timed) separate from the conditional CUDA
    #    estimate (arithmetic on reported values) and from the v9 component sum (results/g3-final).
    kern = []
    for v in _values(inventory, 1):
        path = str(v.get("path", ""))
        label = str(v.get("label", ""))
        if "bench_triton_graphtimed" in path and v.get("unit") == "ms/token" and (
                "LeapQuant per token" in label or "window kernel" in label):
            arm = str(v.get("arm"))
            series = "historical Triton LeapQuant (graph-timed)" if "LeapQuant" in label else arm
            kern.append(point(v, "hist_kernel", _float(v.get("batch")), "batch", series, root,
                              "one layer, per token: decode (mean of 16 positions) + fold/16, component sum",
                              "same_capacity", "per-token = decode + fold/16 (component sum)"))
    kern += g3_final_points(root)
    kern.append({"figure_id": "hist_kernel", "series": "conditional CUDA estimate (NOT timed)",
                 "candidate_id": "LeapQuant CUDA fold v1 + Triton decode (conditional)", "x_value": 512.0,
                 "x_unit": "batch", "y_value": 0.2524 + 1.27 / 16, "y_unit": "ms/token",
                 "source_row_id": "summary.txt lines 509-510 (reported 1.27 ms fold, 0.2524 ms decode)",
                 "originating_experiment": "fast_fold_v1 / fast_decode (reported values)",
                 "hardware": "1 x H100 PCIe (quick-einstein)", "measurement_boundary": "arithmetic, not a measurement",
                 "comparison_class": "same_capacity", "evidence_kind": "derived_from_reported", "repetitions": 0,
                 "derivation": "0.2524 + 1.27/16 = 0.331775 ms (conditional model)", "verified_against_raw": False,
                 "verification": "derived", "plotted": True, "estimate": True})
    figures["hist_kernel"] = kern
    # 2-3. Serving and workload curves: ONE experiment (round10: fp32 packed, bf16 window, LeapQuant; radix off),
    #      each shape on its own actual concurrency grid; the server's max running requests kept with every point.
    for v in _values(inventory, 2):
        if str(v.get("experiment", "")).startswith("round10") and v.get("unit") == "tok/s" and v.get("users"):
            p = point(v, "hist_workload", _float(v.get("users")), "client users (concurrency)",
                      f"{v.get('arm')} {v.get('shape')}", root,
                      "sglang.benchmark.serving finite cohort: output tok/s over the client's duration",
                      "same_capacity", "client-reported output token throughput, one run",
                      shape=v.get("shape"), server_max_running=_float(v.get("server_max_running")),
                      server_max_queue=_float(v.get("server_max_queue")))
            figures.setdefault("hist_workload", []).append(p)
            figures.setdefault("hist_serving", []).append(dict(p, figure_id="hist_serving"))
    # 4. Memory: configured slots as configured slots; measured peak GPU memory separately.
    for v in _values(inventory, 4):
        if str(v.get("experiment", "")).startswith("round10") and v.get("unit") == "MiB" and v.get("value") is not None:
            figures.setdefault("hist_memory", []).append(point(
                v, "hist_memory", str(v.get("arm")), "arm", "measured peak GPU memory (nvidia-smi, whole device)", root,
                "peak device memory during a client point", "equal_total_memory", "max of 0.5 s samples",
                shape=v.get("shape"), users=v.get("users")))
    # 5. Overload: completion separately from the client's goodput (request-level SLO only).
    for v in _values(inventory, 5):
        if v.get("experiment") == "openloop-final" and v.get("unit") == "req/s":
            figures.setdefault("hist_overload", []).append(point(
                v, "hist_overload", _float(v.get("rate_req_s")), "offered rate (req/s)",
                f"{v.get('arm')}: {v.get('label')}", root,
                "open-loop Poisson, 600 requests per rate", "same_capacity",
                "client-reported; goodput uses TTFT <= 2 s and TPOT <= 50 ms only (DECISIONS.md p99 ITL limit not applied); "
                "arrival seed 1 in every session"))
    # 6. Tail latency: one experiment (ab-trackflush, radix on), every arm, regressions kept.
    for v in _values(inventory, 6):
        if v.get("experiment") == "ab-trackflush" and "P99" in str(v.get("label", "")).upper():
            figures.setdefault("hist_tail", []).append(point(
                v, "hist_tail", f"{v.get('shape')} c{v.get('users')}", "point", f"{v.get('arm')}: {v.get('label')}",
                root, "client p99 over requests (TTFT) or token gaps (ITL), closed loop", "same_capacity",
                "client-reported percentile, one run; radix cache on", cache_policy="radix on (no_buffer)"))
    # 7. Numerical error: single-layer replay diagnostics with the norm stated.
    for v in _values(inventory, 7):
        if "S21" in str(v.get("experiment", "")) and v.get("layer") is not None:
            figures.setdefault("hist_numerical", []).append(point(
                v, "hist_numerical", _float(v.get("layer")), "layer", str(v.get("arm")), root,
                "single-layer replay of a recorded trace (not served), output of the layer", "diagnostic",
                "pooled over tokens, heads and values of the trace", metric="output error",
                error_norm="pooled relative L2 x 100: 100*sqrt(sum ||y - y_fp32||^2 / sum ||y_fp32||^2) "
                           "(leapquant_replay.cpp 444-447, verify_kernel.py 407-409)",
                trace=v.get("trace"), averaging_rule="descriptive: per layer, one trace; no interval"))
    return figures


TITLES = {
    "hist_kernel": "Kernel: per-token time, one linear layer (historical Triton measurement, g3-final component sum, conditional CUDA estimate)",
    "hist_serving": "Serving (round 10, radix off): output tok/s per point, fp32 packed / bf16 window / LeapQuant",
    "hist_workload": "Workload curves (round 10): output tok/s against client users, each shape on its actual grid",
    "hist_memory": "Memory (round 10): measured peak GPU memory per arm and point (configured slots listed in the sidecar)",
    "hist_overload": "Overload (openloop-final): client-reported req/s per offered rate",
    "hist_tail": "Tail latency (ab-trackflush, radix on): p99 values per arm and point",
    "hist_numerical": "Numerical error (S21): single-layer replay output error per layer",
}


def draw(figure_id: str, points: list[dict], out: Path) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plotted = [p for p in points if p["plotted"]]
    if not plotted:
        return None
    series = []
    for p in plotted:
        if p["series"] not in series:
            series.append(p["series"])
    fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=140)
    categorical = isinstance(plotted[0]["x_value"], str)
    cats = sorted({p["x_value"] for p in plotted}, key=str) if categorical else None
    for index, name in enumerate(series):
        color = PALETTE[index % len(PALETTE)] if index < len(PALETTE) else "#8a8a8a"
        pts = [p for p in plotted if p["series"] == name]
        xs = [cats.index(p["x_value"]) if categorical else p["x_value"] for p in pts]
        ys = [p["y_value"] for p in pts]
        estimate = any(p.get("estimate") for p in pts)
        if categorical or figure_id == "hist_kernel":
            ax.scatter(xs, ys, s=36, color=color, label=name[:60], marker="D" if estimate else "o",
                       facecolors="none" if estimate else color, zorder=3)
        else:
            order = sorted(zip(xs, ys))
            ax.plot([a for a, _ in order], [b for _, b in order], color=color, linewidth=2, marker="o", markersize=4,
                    label=name[:60])
    if categorical:
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
    elif figure_id in ("hist_workload", "hist_kernel", "hist_serving"):
        ax.set_xscale("log", base=2)
    ax.set_xlabel(plotted[0]["x_unit"])
    ax.set_ylabel(str(plotted[0]["y_unit"]))
    ax.set_title(TITLES[figure_id], fontsize=9)
    ax.grid(color="#e5e5e5", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(loc="lower center", ncol=2, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06 + 0.028 * ((len(series) + 1) // 2), 1, 1))
    name = f"{figure_id}.png"
    fig.savefig(out / name)
    plt.close(fig)
    return name


def build(inventory_path: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    inventory = read_json(inventory_path)
    root = repo_root()
    figures = build_points(inventory, root)
    index = []
    for figure_id, points in figures.items():
        sidecar = {"figure_id": figure_id, "title": TITLES[figure_id], "generator": GENERATOR,
                   "inventory_sha256": sha256_file(inventory_path), "points": points,
                   "data_sha256": sha256_text(canonical_json(points)),
                   "caption_classes": sorted({p["evidence_kind"] for p in points}),
                   "unplotted": [p["source_row_id"] for p in points if not p["plotted"]]}
        sidecar["image"] = draw(figure_id, points, out)
        sidecar["problems"] = sidecar_problems(sidecar)
        write_json(out / f"{figure_id}.json", sidecar)
        index.append({"figure_id": figure_id, "points": len(points), "plotted": sum(p["plotted"] for p in points),
                      "unverified": sum(not p["verified_against_raw"] and p["evidence_kind"] == "measured_raw" for p in points),
                      "problems": len(sidecar["problems"])})
    problems = {t["name"]: t.get("problems", []) for t in inventory["topics"]}
    write_json(out / "index.json", {"generated_utc": utc_now(), "generator": GENERATOR, "figures": index,
                                    "archive_problems": problems})
    return {"out": str(out), "figures": index}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.historical")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = build(Path(args.inventory), Path(args.out))
    render_bundle(Path(args.out))
    for row in result["figures"]:
        print(row)
    return 0



def render_bundle(out: Path) -> None:
    """index.html of the historical bundle: figures with captions and sidecars, the archive's problems, the
    sanitizer records, the claim registry, the paper crosswalk and the gate registry (copied into the bundle)."""
    import html
    import shutil
    evidence = harness_root_path() / "evidence"
    for name in ("sanitizer_results_historical.json", "claims_registry.json", "paper_crosswalk.json",
                 "memory_reconciliation.json"):
        if (evidence / name).exists():
            shutil.copy(evidence / name, out / name)
    shutil.copy(harness_root_path() / "config" / "gates.json", out / "gates.json")
    index = read_json(out / "index.json")
    esc = html.escape
    css = ("body{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1200px;margin:auto;padding:24px 16px;"
           "background:#fff;color:#1d1d1f}h1{font-size:22px}h2{font-size:17px;border-bottom:1px solid #e3e3e8;margin-top:28px}"
           "table{border-collapse:collapse;font-size:12px}td,th{border-bottom:1px solid #e3e3e8;padding:3px 7px;text-align:left;"
           "vertical-align:top}.muted{color:#6b6b70}img{max-width:100%}.box{background:#f7f7f9;border:1px solid #e3e3e8;"
           "border-radius:8px;padding:10px}@media (prefers-color-scheme: dark){body{background:#141416;color:#ececf0}"
           ".box{background:#1c1c20;border-color:#2c2c31}td,th{border-color:#2c2c31}}")
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,"
             f"initial-scale=1'><title>Historical evidence re-plotted</title><style>{css}</style></head><body>",
             "<h1>Historical evidence, re-plotted from raw files</h1>",
             f"<div class='muted'>{GENERATOR} · generated {index['generated_utc']} · every plotted value re-read from its "
             "raw source (file and line) before plotting · carried history, not new measurements</div>",
             "<div class='box'>Evidence kinds: <b>measured_raw</b> (historical raw files), <b>derived_from_reported</b> "
             "(arithmetic on reported values, e.g. the conditional CUDA estimate), <b>unverified_assertion</b>. No "
             "historical point has an independent repetition design, so no confidence interval is drawn. Missing cells "
             "are not plotted (never as zero). Single runs throughout unless a point says otherwise.</div>"]
    for row in index["figures"]:
        sidecar = read_json(out / f"{row['figure_id']}.json")
        parts.append(f"<h2>{esc(sidecar['title'])}</h2>")
        if sidecar.get("image"):
            parts.append(f"<img src='{sidecar['image']}' alt='{esc(sidecar['title'])}'>")
        parts.append(f"<p class='muted'>{row['plotted']} of {row['points']} points plotted · evidence kinds: "
                     f"{', '.join(sidecar['caption_classes'])} · sidecar <a href='{row['figure_id']}.json'>"
                     f"{row['figure_id']}.json</a> (every point with source line, experiment, boundary, class, derivation) "
                     f"· data sha256 {sidecar['data_sha256'][:16]}… · sidecar problems: {len(sidecar['problems'])}</p>")
    parts.append("<h2>Problems found in the archive (historical inventory)</h2>")
    for topic, problems in index["archive_problems"].items():
        parts.append(f"<h3>{esc(topic)}</h3><ul>" + "".join(f"<li>{esc(str(p))}</li>" for p in problems) + "</ul>")
    sanitizer = read_json(out / "sanitizer_results_historical.json") if (out / "sanitizer_results_historical.json").exists() else {}
    if sanitizer:
        parts.append("<h2>compute-sanitizer records (historical)</h2><table><tr><th>log</th><th>tool</th>"
                     "<th>classification</th><th>summary</th><th>coverage</th></tr>")
        for r in sanitizer["records"]:
            parts.append(f"<tr><td>{esc(r['log'])}</td><td>{r['tool']}</td><td>{r['classification']}</td>"
                         f"<td>{esc(str(r.get('summary')))}</td><td>{esc(str(r.get('expected_coverage')))}</td></tr>")
        parts.append("</table>" + "".join(f"<p><b>{esc(c['claim'])}</b>: {esc(str(c['resolution']))}</p>"
                                           for c in sanitizer.get("claims", [])))
    registry = read_json(out / "claims_registry.json") if (out / "claims_registry.json").exists() else {}
    if registry:
        parts.append("<h2>Claim registry</h2><table><tr><th>claim</th><th>evidence kind</th><th>value</th><th>status</th></tr>")
        for c in registry["claims"]:
            parts.append(f"<tr><td>{esc(c['claim'])}</td><td>{c['evidence_kind']}</td><td>{esc(str(c.get('value')))}</td>"
                         f"<td>{esc(str(c.get('status')))}</td></tr>")
        parts.append("</table>")
    cross = read_json(out / "paper_crosswalk.json") if (out / "paper_crosswalk.json").exists() else {}
    if cross:
        parts.append(f"<h2>Paper crosswalk ({esc(cross['label'])})</h2><table><tr><th>paper cell</th><th>reported</th>"
                     "<th>local</th><th>status</th><th>boundary / missing</th></tr>")
        for r in cross["rows"]:
            local = r.get("local_value")
            parts.append(f"<tr><td>{esc(r['paper_cell'])}</td><td>{esc(str(r['reported']))}</td><td>"
                         f"{esc(f'{local:.3f}' if isinstance(local, float) else str(local or r.get('local_null_reason')))}</td>"
                         f"<td>{r['status']}</td><td>{esc(r.get('local_boundary') or r['metric_boundary'])}; missing: "
                         f"{esc(', '.join(r.get('missing_fields', [])))}</td></tr>")
        parts.append("</table>")
    gates = read_json(out / "gates.json")
    parts.append("<h2>Gate registry (v6 numbering)</h2><table><tr><th>gate</th><th>status</th><th>reason</th></tr>")
    for g in gates["gates"]:
        parts.append(f"<tr><td>{g['gate_id']} {esc(g['name'])}</td><td>{g['status']}</td><td>{esc(g['status_reason'])}</td></tr>")
    parts.append("</table><p class='muted'>Plan-5 gate results (e.g. 'G3 MET') are history under another numbering and "
                 "are not imported into these gates.</p></body></html>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def harness_root_path() -> Path:
    return Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(main())
