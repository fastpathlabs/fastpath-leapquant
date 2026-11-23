#!/usr/bin/env python3
"""
generate_reports.py

Turns the logs of a run (results/run-<host>/) into HTML reports with charts and
tables, one page per kind of run:

    python3 reports/generate_reports.py results/run-creative-franklin

Each invocation writes a NEW folder, reports/<run>_<date>-<time>/, and never
touches an earlier one. reports/all_reports.html lists them all.

A run's folder holds only what that run measured. To show the latest of every kind of
measurement in one report, name the other folders too (the run's own folder wins on a
file name both have); tabs with no measurement are left out:

    python3 reports/generate_reports.py results/run-quick-einstein-v7 \
        --with results/accuracy-v7 results/fast_checks_v2 results/s20_ablations results/gpu-suite

In the folder:

    index.html          what was run, headline numbers, links
    kernel.html         kernel time per decode step, against SGLang's kernels
    accuracy.html       error on real model states (gate 2), per prompt and layer
    verification.html   GPU kernel against the reference core
    tuning.html         the build-option sweep
    profile.html        where the kernel's time goes
    serving.html        throughput, time to first token, speed per user, per concurrency
    img/*.png           the charts

A page whose logs do not exist yet says so and shows no numbers. Nothing in a
report is typed in by hand: every number is parsed from a log file named on the page.

Needs matplotlib.
"""

import glob
import html
import json
import os
import re
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = os.path.dirname(os.path.abspath(__file__))
# Where this invocation writes. Set in main(): a new folder per run and time, so that
# no earlier report is ever overwritten.
HERE = REPORTS
IMG = os.path.join(HERE, "img")

# One colour per thing, the same on every page. (First slots of the dataviz reference palette.)
COLOR = {
    "leapquant": "#2a78d6",   # blue
    "packed": "#eb6834",      # orange: SGLang packed kernel / fp32, what runs today
    "window": "#1baf7a",      # aqua: SGLang 16-token window
    "bf16": "#eda100",        # yellow: bf16 with the window
    "slot5": "#e87ba4",       # magenta: further arms (LeapQuant variants), in this fixed order
    "slot6": "#008300",       # green
    "slot7": "#6250d6",       # violet
    "slot8": "#e34948",       # red
}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ---- small helpers ----------------------------------------------------------------------

# The results folders a report draws on: the run's own folder first, then any given with --with
# (for example the kernel checks and the accuracy run, which are measured into folders of their own).
SOURCES = []


def find(pattern):
    """Measurement files matching `pattern` in every source folder, the run's own folder first.
    A file name found in an earlier folder hides the same name in a later one."""
    found, seen = [], set()
    for folder in SOURCES:
        for path in sorted(glob.glob(os.path.join(folder, pattern))):
            name = os.path.basename(path)
            if name not in seen:
                seen.add(name)
                found.append(path)
    return found


def read(path):
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def legend_layout(labels, budget):
    """Columns and rows of a legend under a chart: as many entries per row as fit in about `budget` characters."""
    if not labels:
        return 1, 0
    longest = max(len(label) for label in labels) + 6
    columns = max(1, min(len(labels), budget // longest))
    return columns, -(-len(labels) // columns)


def grouped_bars(filename, title, subtitle, groups, series, ylabel, annotate=None, value_format="{:.2f}"):
    """groups: x labels. series: list of (label, colour key, values). annotate: text above each group."""
    count = len(series)
    columns, rows = legend_layout([label for label, _, _ in series], budget=110)
    legend_height = 0.26 * rows + 0.14 if count > 1 else 0.0       # inches: the legend sits under the chart
    height = 3.9 + legend_height
    figure, axis = plt.subplots(figsize=(7.6, height), dpi=160)
    width = 0.78 / count
    tallest = max(max(values) for _, _, values in series)
    for index, (label, color, values) in enumerate(series):
        positions = [g + (index - (count - 1) / 2) * width for g in range(len(groups))]
        axis.bar(positions, values, width * 0.9, color=COLOR[color], label=label, linewidth=0)
        if len(groups) * count <= 12:
            for x, value in zip(positions, values):
                axis.text(x, value + tallest * 0.012, value_format.format(value), ha="center", va="bottom",
                          fontsize=7.5, color=MUTED)
    if annotate:
        for g, text in enumerate(annotate):
            top = max(values[g] for _, _, values in series)
            axis.text(g, top + tallest * 0.09, text, ha="center", va="bottom", fontsize=9.5, color=INK, fontweight="bold")
    axis.set_xticks(range(len(groups)))
    axis.set_xticklabels(groups)
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, tallest * 1.25)
    axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    if count > 1:
        figure.legend(*axis.get_legend_handles_labels(), frameon=False, loc="lower left", ncol=columns, fontsize=8.5,
                      bbox_to_anchor=(0.01, 0.005))
    figure.suptitle(title, x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.995)
    figure.text(0.02, 1 - 0.37 / height, subtitle, ha="left", fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, legend_height / height, 1, 1 - 0.45 / height))
    os.makedirs(IMG, exist_ok=True)
    figure.savefig(os.path.join(IMG, filename))
    plt.close(figure)
    return f"img/{filename}"


def horizontal_bars(filename, title, subtitle, labels, values, xlabel, color="leapquant", highlight=None):
    figure, axis = plt.subplots(figsize=(7.6, 0.42 * len(labels) + 1.5), dpi=160)
    colors = [COLOR[color] if highlight is None or index == highlight else "#9fbfe8" for index in range(len(labels))]
    positions = list(range(len(labels)))[::-1]
    axis.barh(positions, values, 0.62, color=colors, linewidth=0)
    widest = max(values)
    for y, value in zip(positions, values):
        axis.text(value + widest * 0.01, y, f"{value:.3f}", va="center", fontsize=8, color=MUTED)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels, fontsize=8.5)
    axis.set_xlabel(xlabel)
    axis.set_xlim(0, widest * 1.14)
    axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(length=0)
    figure.suptitle(title, x=0.02, ha="left", fontsize=12.5, fontweight="bold", color=INK, y=0.995)
    figure.text(0.02, 1 - 0.62 / (0.42 * len(labels) + 1.5), subtitle, ha="left", fontsize=9, color=MUTED)
    figure.tight_layout(rect=(0, 0, 1, 1 - 0.75 / (0.42 * len(labels) + 1.5)))
    os.makedirs(IMG, exist_ok=True)
    figure.savefig(os.path.join(IMG, filename))
    plt.close(figure)
    return f"img/{filename}"


class Html(str):
    """A table cell that is already HTML (a link): not escaped."""


def table(headers, rows, numeric_from=1):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td class=\"{'num' if index >= numeric_from else ''}\">"
                        f"{cell if isinstance(cell, Html) else html.escape(str(cell))}</td>"
                        for index, cell in enumerate(row))
        body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def figure_block(image, alt, caption=""):
    note = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
    return f"<figure><img src=\"{image}\" alt=\"{html.escape(alt)}\">{note}</figure>"


STYLE = """
:root { --surface:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --line:#e4e3df; --accent:#2a78d6; --warn:#a15c00; }
* { box-sizing: border-box; }
body { margin:0; background:var(--surface); color:var(--ink); font:16px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 900px; margin: 0 auto; padding: 28px 16px 64px; }
nav { font-size:14px; color:var(--muted); margin-bottom:18px; } nav a { color:var(--accent); text-decoration:none; margin-right:14px; }
h1 { font-size: 30px; line-height:1.2; margin: 8px 0 10px; } h2 { font-size: 21px; margin: 40px 0 8px; } h3 { font-size:16px; margin:26px 0 6px; }
p, li { color: var(--ink); } .lede { font-size:18px; color:var(--muted); } .muted { color:var(--muted); font-size:14px; }
figure { margin: 18px 0 6px; } img { max-width:100%; height:auto; border:1px solid var(--line); border-radius:8px; }
figcaption { font-size:13px; color:var(--muted); margin-top:4px; }
.scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:14px; margin:8px 0 18px; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }
th { color:var(--muted); font-weight:600; } td.num { text-align:right; font-variant-numeric: tabular-nums; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap:12px; margin:18px 0; }
.tile { border:1px solid var(--line); border-radius:10px; padding:12px 14px; } .tile b { display:block; font-size:24px; }
.tile span { color:var(--muted); font-size:13px; }
.note { border-left:3px solid var(--warn); padding:6px 12px; margin:14px 0; color:var(--ink); background:#fff8ec; }
.pending { border:1px dashed var(--line); border-radius:10px; padding:16px; color:var(--muted); }
pre { font-size: 12px; background:#f2f1ee; padding:10px; border-radius:8px; }
code { font-size: 13px; background:#f2f1ee; padding:1px 4px; border-radius:4px; }
details { margin: 6px 0; border-bottom:1px solid var(--line); } summary { cursor:pointer; padding:8px 0; }
"""

PAGES = [("index.html", "Overview"), ("kernel.html", "Kernel speed"), ("accuracy.html", "Accuracy on real states"),
         ("tasks.html", "Task accuracy"), ("verification.html", "Verification"), ("tuning.html", "Tuning"),
         ("profile.html", "Profile"), ("serving.html", "Serving")]


WRITTEN = {}   # filename -> (title, body, run_name), written by write_pages() at the end


def has_data(body):
    """A page whose only content is the 'not measured' note is left out of the report."""
    return 'class="pending"' not in body or "<img" in body or "<table" in body


def write_page(filename, title, body, run_name):
    WRITTEN[filename] = (title, body, run_name)


def write_pages():
    shown = [(name, label) for name, label in PAGES if name in WRITTEN and (name == "index.html" or has_data(WRITTEN[name][1]))]
    for filename, (title, body, run_name) in WRITTEN.items():
        if (filename, title) not in shown and filename not in dict(shown):
            continue
        _write_page(filename, title, body, run_name, shown)


def _write_page(filename, title, body, run_name, shown):
    links = "".join(f"<a href=\"{name}\">{label}</a>" for name, label in shown)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{html.escape(title)}</title>"
            f"<style>{STYLE}</style></head><body><main><nav>{links}</nav>{body}"
            f"<p class=\"muted\">Generated {stamp} from {', '.join(f'<code>{html.escape(os.path.relpath(s, os.path.dirname(REPORTS)))}</code>' for s in SOURCES)} by "
            f"<code>reports/generate_reports.py</code>. <a href=\"../all_reports.html\">All reports</a>.</p></main></body></html>")
    with open(os.path.join(HERE, filename), "w", encoding="utf-8") as handle:
        handle.write(page)


def pending(text):
    return f"<div class=\"pending\">Not measured yet. {html.escape(text)}</div>"


# ---- parsers ------------------------------------------------------------------------------

def parse_kernel_speed(results):
    """e_kernel_speed_<dtype>.log -> {dtype: {batch: {kernel: (per_step, append, fold)}}}"""
    data = {}
    for path in find("e_kernel_speed_*.log"):
        dtype = re.search(r"e_kernel_speed_(\w+)\.log", path).group(1)
        rows = {}
        for match in re.finditer(r"^\s*(\d+)\s+(sglang packed|sglang window L=\d+|leapquant, kernel only|leapquant, with gates)"
                                 r"\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", read(path), flags=re.M):
            rows.setdefault(int(match.group(1)), {})[match.group(2)] = tuple(float(match.group(i)) for i in (3, 4, 5))
        if rows:
            data[dtype] = rows
    return data


def parse_triton_speed(results):
    """e_triton_speed*.txt (kernel/bench_triton.py --with-sglang) -> {batch: {name: ms per token}}"""
    data = {}
    for path in find("e_triton_speed*.txt"):
        batch = None
        for line in read(path).splitlines():
            match = re.match(r"\s*batch (\d+)\s*$", line)
            if match:
                batch = int(match.group(1))
                data.setdefault(batch, {})
                continue
            if batch is None:
                continue
            match = re.search(r"fast decode step ([\d.]+) ms .*fast fold ([\d.]+) ms\s+->\s+([\d.]+) ms per token", line)
            if match:
                data[batch]["leapquant decode"] = float(match.group(1))
                data[batch]["leapquant fold"] = float(match.group(2))
                data[batch]["leapquant"] = float(match.group(3))
                continue
            match = re.search(r"exact decode step [\d.]+ ms\s+exact fold [\d.]+ ms\s+->\s+([\d.]+) ms per token", line)
            if match:
                data[batch]["leapquant exact"] = float(match.group(1))
                continue
            match = re.search(r"sglang (window|packed) kernel, (fp32|bf16) state: ([\d.]+) ms per token", line)
            if match:
                data[batch][f"sglang {match.group(1)} {match.group(2)}"] = float(match.group(3))
    return {batch: values for batch, values in data.items() if "leapquant" in values}


def parse_replays(results):
    """c_replay_<prompt>_trace_layer_NN.txt -> list of dicts"""
    traces = []
    for path in find("c_replay_*_trace_layer_*.txt") + find("base_*_trace_layer_*.txt"):
        match = re.search(r"(?:c_replay|base)_(\w+?)_trace_layer_(\d+)\.txt", os.path.basename(path))
        text = read(path)
        check = re.search(r"self-check:.*?difference ([\d.e+-]+)\s*(\(ok\))?", text)
        tokens = re.search(r"(\d+) tokens replayed", text)
        methods = {}
        for line in text.splitlines():
            parts = line.split("|")
            if len(parts) >= 3 and "%" in parts[-1]:
                name = re.sub(r"\s+[\d.]+\s*$", "", parts[0]).strip()
                bytes_per_entry = float(re.search(r"([\d.]+)\s*$", parts[0]).group(1))
                state = re.findall(r"([\d.]+)%", parts[-2])
                methods[name] = {"bytes": bytes_per_entry, "output": float(parts[-1].strip().rstrip("%")),
                                 "state_median": float(state[0]), "state_worst": float(state[1])}
        traces.append({"prompt": match.group(1), "layer": int(match.group(2)), "methods": methods,
                       "self_check": check.group(1) if check else "?", "ok": bool(check and check.group(2)),
                       "tokens": int(tokens.group(1)) if tokens else 0, "file": os.path.basename(path)})
    return traces


def parse_verify(results):
    reports = []
    for path in find("d_verify_*.txt") + find("verify_*.txt"):
        text = read(path)
        fast = "kernel/leapquant_triton.py" in text
        inside = re.search(r"median ([\d.e+-]+), 99% below ([\d.e+-]+), worst ([\d.e+-]+)", text)
        folds = re.search(r"(\d+) folds of (\d+) heads; bit-identical to the reference: (\d+)", text)
        differ = re.search(r"int8 entries that differ: ([\d,]+) of ([\d,]+)", text)
        free = re.search(r"output error: kernel ([\d.]+)%, reference ([\d.]+)%", text)
        drift = re.search(r"largest difference between the two free runs: ([\d.e+-]+)", text)
        if not (inside and free):
            continue
        name = os.path.basename(path)[:-4]
        name = name[len("d_verify_"):] if name.startswith("d_verify_") else name[len("verify_"):]
        reports.append({"trace": name + (" (fast kernels)" if fast else ""), "median": inside.group(1), "p99": inside.group(2),
                        "worst": inside.group(3), "folds": folds.group(1) if folds else "0",
                        "identical": folds.group(3) if folds else "0", "int8_differ": differ.group(1) if differ else "?",
                        "int8_total": differ.group(2) if differ else "?", "kernel": float(free.group(1)),
                        "reference": float(free.group(2)), "drift": drift.group(1) if drift else "?",
                        "result": "PASS" if "RESULT: PASS" in text else "FAIL"})
    return reports


def parse_tune(results):
    path = os.path.join(results, "steps", "tune.log")
    if not os.path.exists(path):
        return [], None
    text = read(path)
    builds = []
    for match in re.finditer(r"^(t(\d+)_[ws](\d)_fmad(\w+))\s+verify (PASS|FAIL)\s+ms per step at batch \[([\d, ]+)\]:\s+([\d. ]+)$",
                             text, flags=re.M):
        builds.append({"tag": match.group(1), "threads": int(match.group(2)), "word_loads": match.group(3) == "1", "second_is_shared": "_s" in match.group(1),
                       "fmad": match.group(4) == "true", "verify": match.group(5),
                       "batches": [int(b) for b in match.group(6).split(",")],
                       "times": [float(t) for t in match.group(7).split()]})
    chosen = re.search(r"Fastest correct build.*?: (\S+)", text)
    return builds, chosen.group(1) if chosen else None


def parse_profiles(results):
    profiles = []
    for path in find("profile_*.txt"):
        text = read(path)
        head = re.search(r"batch (\d+),.*?(\d+) compensators, (\d+) sweeps\s*(.*)", text)
        if not head:
            continue
        sections = {}
        current = None
        for line in text.splitlines():
            if line.strip().startswith("decode step"):
                current = "decode"
            elif line.strip().startswith("fold of a full window"):
                current = "fold"
            row = re.match(r"^\s{4}(.+?)\s+(-?[\d.]+)\s+(-?[\d.]+)%\s+([\d.]+) \(([\d.]+) \.\. ([\d.]+)\)", line)
            if row and current:
                sections.setdefault(current, []).append((row.group(1).strip(), float(row.group(2)), float(row.group(4)),
                                                         float(row.group(5)), float(row.group(6))))
        per_step = re.search(r"= ([\d.]+) ms\s*$", text, flags=re.M)
        profiles.append({"file": os.path.basename(path), "batch": int(head.group(1)), "sweeps": int(head.group(3)),
                         "label": head.group(4).strip(), "sections": sections, "clean": "WARNING: other processes" not in text,
                         "per_step": float(per_step.group(1)) if per_step else None})
    return profiles


SERVING_KEYS = ["Output token throughput (tok/s)", "Total token throughput (tok/s)", "Request throughput (req/s)",
                "Median TTFT (ms)", "P90 TTFT (ms)", "P99 TTFT (ms)", "Median TPOT (ms)", "P90 TPOT (ms)",
                "Median ITL (ms)", "P90 ITL (ms)", "P99 ITL (ms)", "Median E2E Latency (ms)", "Successful requests",
                "Benchmark duration (s)"]


def parse_result_block(text):
    """Every number of sglang.benchmark.serving's result block ("Name (unit):   value"), the last block of the log."""
    body = text.rsplit("Serving Benchmark Result", 1)[-1]
    metrics = {}
    for line in body.splitlines():
        found = re.match(r"^\s*([A-Za-z][^:]*?):\s+(-?\d[\d.]*)\s*$", line)
        if found:
            metrics.setdefault(found.group(1).strip(), float(found.group(2)))
    return metrics


def parse_client(text):
    """The client's settings (benchmark_args=Namespace(...), first line of a serving log): what a public benchmark
    states next to its numbers."""
    found = re.search(r"benchmark_args=Namespace\((.*)\)\s*$", text, flags=re.M)
    if not found:
        return {}
    args = found.group(1)
    client = {}
    for key in ("dataset_name", "num_prompts", "random_input_len", "random_output_len", "random_range_ratio",
                "request_rate", "max_concurrency", "warmup_requests", "seed", "temperature", "disable_ignore_eos",
                "flush_cache"):
        value = re.search(r"\b" + key + r"=('?)([^,']*)\1", args)
        if value:
            client[key] = value.group(2)
    goodput = re.search(r"goodput=(\[.*?\])", args)
    if goodput and goodput.group(1) != "[]":
        client["goodput"] = ", ".join(f"{name} <= {float(limit):g} ms" for name, limit in
                                      re.findall(r"\('(\w+)', ([\d.]+)\)", goodput.group(1)))
    return client


def client_rows(clients):
    """Rows of client settings, one column per server (clients: list of dicts, in column order)."""
    def lengths(c):
        if "random_input_len" not in c:
            return "-"
        exact = c.get("random_range_ratio") in ("1.0", "1")
        return (f"{int(c['random_input_len']):,} / {int(c['random_output_len']):,}"
                + (" exact" if exact else f" (random, from {float(c.get('random_range_ratio', 0)):g} of these)"))
    fields = [("Prompts", lambda c: c.get("num_prompts", "-")), ("Input / output tokens", lengths),
              ("Arrival rate, requests/s", lambda c: c.get("request_rate", "-")),
              ("Client concurrency limit", lambda c: c.get("max_concurrency", "-")),
              ("Output length forced (ignore EOS)", lambda c: {"False": "yes", "True": "no"}.get(c.get("disable_ignore_eos"), "-")),
              ("Warm-up requests", lambda c: c.get("warmup_requests", "-")),
              ("Cache flushed before the point", lambda c: {"True": "yes", "False": "no"}.get(c.get("flush_cache"), "-")),
              ("Temperature", lambda c: c.get("temperature", "-")), ("Seed", lambda c: c.get("seed", "-")),
              ("Goodput SLO", lambda c: c.get("goodput", "-"))]
    return [[label] + [str(get(c)) if c else "-" for c in clients] for label, get in fields]


def every_metric_table(columns, base_index=0, extra_rows=None):
    """columns: [(name, metrics)] (metrics may be None), the baseline at base_index. Every metric of
    SERVING_FAMILIES, one row each, and the ratio of each other column to the baseline."""
    base = columns[base_index][1] if base_index is not None else None
    others = [i for i in range(len(columns)) if i != base_index and base is not None]
    rows = []
    for family, metrics in SERVING_FAMILIES:
        rows.append([Html(f"<b>{html.escape(family)}</b>")] + [""] * (len(columns) + len(others)))
        for metric in metrics:
            values = [(m or {}).get(metric) for _, m in columns]
            if all(v is None for v in values):
                continue
            rows.append([metric] + [metric_text(metric, v) for v in values]
                        + [ratio_text(metric, values[i], base.get(metric)) for i in others])
    clients = [(m or {}).get("_client") for _, m in columns]
    if any(clients):
        rows.append([Html("<b>Client settings</b>")] + [""] * (len(columns) + len(others)))
        rows += [row + [""] * len(others) for row in client_rows(clients)]
    for row in extra_rows or []:
        rows.append(row + [""] * len(others))
    headers = ["Metric"] + [name for name, _ in columns] + [f"{columns[i][0]} ÷ {columns[base_index][0]}" for i in others]
    return "<div class=\"scroll\">" + table(headers, rows, numeric_from=1) + "</div>"


def add_derived(metrics, mem_path=None):
    """Speed per user (interactivity, as public benchmarks plot it): 1000 / time per output token; the p90 and p99
    figures are the speeds that 90% and 99% of requests reach or beat. Peak GPU memory from the .mem file beside the log."""
    for source, name in (("Median TPOT (ms)", "Speed per user, median (tok/s)"),
                         ("P90 TPOT (ms)", "Speed per user, 90% of requests at least (tok/s)"),
                         ("P99 TPOT (ms)", "Speed per user, 99% of requests at least (tok/s)")):
        if metrics.get(source, 0) > 0:
            metrics[name] = 1000.0 / metrics[source]
    if mem_path and os.path.exists(mem_path):
        found = re.search(r"peak_memory_used_mib\s+(\d+)", read(mem_path))
        if found and int(found.group(1)) > 0:
            metrics["Peak GPU memory (MiB)"] = float(found.group(1))
    return metrics


def parse_serving(results):
    """serving_<config>_<shape>_c<N>.log -> {(config, shape, concurrency): {metric: value}}, every metric of the log."""
    runs = {}
    for path in find("serving_*_c*.log"):
        match = re.search(r"serving_(.+)_(long|short|decode2k|decode|aa1k|ix8k1k|[a-z]+\d+k\d+k)_c(\d+)\.log",
                          os.path.basename(path))
        if not match:
            continue
        text = read(path)
        metrics = parse_result_block(text)
        if metrics.get("Output token throughput (tok/s)"):
            metrics = add_derived(metrics, path[:-4] + ".mem")
            metrics["_client"] = parse_client(text)
            runs[(match.group(1), match.group(2), int(match.group(3)))] = metrics
    return runs


OPENLOOP_ARMS = {"bf16_window": ("bf16 + window", "bf16"), "leapquant_aligned": ("LeapQuant, folds aligned", "leapquant"),
                  "leapquant_own_step": ("LeapQuant, folds on each request's step", "window")}


def parse_openloop():
    """openloop_<arm>_rate<R>.log (runbook/serving_openloop.sh) -> {(arm, rate): {metric: value}}, goodput included."""
    runs = {}
    for path in find("openloop_*_rate*.log"):
        match = re.search(r"openloop_(.+)_rate([\d.]+)\.log", os.path.basename(path))
        if not match:
            continue
        text = read(path)
        metrics = add_derived(parse_result_block(text), path[:-4] + ".mem")
        if metrics:
            metrics["_client"] = parse_client(text)
            runs[(match.group(1), float(match.group(2)))] = metrics
    return runs


def openloop_section():
    """Open-loop load (plan v6 E02): requests arrive at fixed Poisson rates; goodput counts the requests that met the
    SLO. Charts against the offered rate, and every metric per rate."""
    runs = parse_openloop()
    if not runs:
        return ""
    rates = sorted({rate for _, rate in runs})
    present = sorted({arm for arm, _ in runs})
    arms = [arm for arm in OPENLOOP_ARMS if arm in present] + [arm for arm in present if arm not in OPENLOOP_ARMS]
    free = arm_colors([arm for arm in arms if arm not in OPENLOOP_ARMS])
    label = lambda arm: OPENLOOP_ARMS[arm][0] if arm in OPENLOOP_ARMS else ARM_NAMES.get(arm, arm)
    color = lambda arm: OPENLOOP_ARMS[arm][1] if arm in OPENLOOP_ARMS else free[arm]
    labels = [f"{rate:g}/s" for rate in rates]
    clients = [m.get("_client") or {} for m in runs.values()]
    slo = next((c["goodput"] for c in clients if c.get("goodput")), None)
    shape = next(((c.get("random_input_len"), c.get("random_output_len")) for c in clients if c.get("random_input_len")),
                 None)
    prompts = sorted({c.get("num_prompts") for c in clients if c.get("num_prompts")})
    body = ("<h1>Open-loop load: goodput, throughput and latency against the offered rate</h1><p class=\"lede\">"
            "Requests arrive at a fixed Poisson rate (a slower server does not get a lighter load) with no client-side "
            "cap, so waiting in the server's queue is part of the time to first token. Goodput counts the requests that "
            f"met the SLO{(': ' + html.escape(slo)) if slo else ''}."
            + (f" Shape: {int(shape[0]):,} in / {int(shape[1]):,} out." if shape else "")
            + (f" {', '.join(prompts)} requests per rate." if prompts else "") + "</p>")
    for metric, title, direction, unit, scale, fmt in [
            ("Request goodput (req/s)", "Goodput: requests per second within the SLO", "Higher is better",
             "requests/s", 1.0, "{:.2f}"),
            ("Output token throughput (tok/s)", "Output token throughput", "Higher is better", "tokens/s", 1.0, "{:,.0f}"),
            ("P99 TTFT (ms)", "Time to first token, p99", "Lower is better", "seconds", 1000.0, "{:.1f}"),
            ("Median TPOT (ms)", "Time per output token, median", "Lower is better", "milliseconds", 1.0, "{:.1f}"),
            ("P99 ITL (ms)", "Time between tokens, p99", "Lower is better", "milliseconds", 1.0, "{:.0f}")]:
        series = [(label(arm), color(arm), [runs.get((arm, rate), {}).get(metric, 0.0) / scale for rate in rates])
                  for arm in arms]
        if not any(any(values) for _, _, values in series):
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
        image = grouped_bars("openloop_goodput.png" if metric.startswith("Request goodput") else f"openloop_{slug}.png",
                             title, f"{direction} · offered load (requests per second)", labels, series, unit,
                             value_format=fmt)
        body += f"<h3>{html.escape(title)}</h3>" + figure_block(image, title)
    rows = []
    for rate in rates:
        for arm in arms:
            m = runs.get((arm, rate))
            if not m:
                continue
            rows.append([f"{rate:g}", label(arm)] + [metric_text(k, m.get(k)) if k != "Request goodput (req/s)" else
                                                     (f"{m[k]:.2f}" if k in m else "-") for k in (
                "Request goodput (req/s)", "Request throughput (req/s)", "Output token throughput (tok/s)",
                "Median TTFT (ms)", "P99 TTFT (ms)", "Median TPOT (ms)", "P99 TPOT (ms)", "Median ITL (ms)",
                "P99 ITL (ms)", "Median E2E Latency (ms)", "P99 E2E Latency (ms)")])
    body += "<div class=\"scroll\">" + table(
        ["Offered, req/s", "Server", "Goodput, req/s", "Completed, req/s", "Output tokens/s", "TTFT median, ms",
         "TTFT p99, ms", "TPOT median, ms", "TPOT p99, ms", "ITL median, ms", "ITL p99, ms", "E2E median, ms",
         "E2E p99, ms"], rows, numeric_from=2) + "</div>"
    body += "<h3>Every metric, per offered rate</h3>"
    for rate in rates:
        columns = [(label(arm), runs.get((arm, rate))) for arm in arms if (arm, rate) in runs]
        goodput = [["Request goodput (req/s)"] + [f"{m['Request goodput (req/s)']:.2f}" if "Request goodput (req/s)" in m
                                                  else "-" for _, m in columns]]
        body += (f"<details><summary><b>{rate:g} requests/s offered</b></summary>"
                 + every_metric_table(columns, 0, goodput) + "</details>")
    return body


# ---- pages --------------------------------------------------------------------------------

def triton_kernel_section(results):
    """The fast (Triton) kernels against SGLang's: kernel/bench_triton.py, timed through CUDA graph replays."""
    data = parse_triton_speed(results)
    if not data:
        return "", {}
    batches = sorted(data)
    labels = [f"batch {b}" for b in batches]
    get = lambda name: [data[b].get(name, float("nan")) for b in batches]  # noqa: E731
    ours, packed, window, bf16 = get("leapquant"), get("sglang packed fp32"), get("sglang window fp32"), get("sglang window bf16")
    body = ("<h2>Fast kernels (Triton, tensor cores): time per token</h2><p>Decode step plus one sixteenth of a fold "
            "(the window folds every 16 tokens). Every kernel timed the same way: ten calls captured in a CUDA graph, the "
            "median of 20 replays, so no host launch time is counted. Source: <code>e_triton_speed*.txt</code> "
            "(<code>kernel/bench_triton.py --with-sglang</code>).</p>")
    image = grouped_bars("kernel_triton_per_token.png", "Time per token, one layer, milliseconds",
                         "Lower is better · labels: LeapQuant against SGLang's packed fp32 kernel",
                         labels, [("SGLang packed, fp32 state", "packed", packed), ("SGLang window, fp32 state", "window", window),
                                  ("SGLang window, bf16 state", "bf16", bf16), ("LeapQuant, fast kernels", "leapquant", ours)],
                         "ms per token", annotate=[f"{p / o:.2f}×" for p, o in zip(packed, ours)], value_format="{:.3f}")
    body += figure_block(image, "Time per token by batch size")
    body += "<div class=\"scroll\">" + table(
        ["Batch", "LeapQuant", "decode step", "fold (every 16)", "SGLang packed fp32", "SGLang window fp32",
         "SGLang window bf16", "SGLang packed bf16", "LeapQuant exact kernels", "vs packed fp32", "vs window fp32", "vs window bf16"],
        [[b, f"{data[b]['leapquant']:.4f}", f"{data[b]['leapquant decode']:.4f}", f"{data[b]['leapquant fold']:.3f}",
          f"{data[b].get('sglang packed fp32', float('nan')):.4f}", f"{data[b].get('sglang window fp32', float('nan')):.4f}",
          f"{data[b].get('sglang window bf16', float('nan')):.4f}", f"{data[b].get('sglang packed bf16', float('nan')):.4f}",
          f"{data[b].get('leapquant exact', float('nan')):.4f}",
          f"{data[b].get('sglang packed fp32', float('nan')) / data[b]['leapquant']:.2f}×",
          f"{data[b].get('sglang window fp32', float('nan')) / data[b]['leapquant']:.2f}×",
          f"{data[b].get('sglang window bf16', float('nan')) / data[b]['leapquant']:.2f}×"] for b in batches]) + "</div>"
    body += ("<p>Above 1× LeapQuant is faster, below 1× slower. The paper reports 2.52× against fp32 for this "
             "configuration.</p>")
    last = batches[-1]
    return body, {"triton": {"batch": last, "ours": data[last]["leapquant"],
                             "packed": data[last].get("sglang packed fp32"), "bf16": data[last].get("sglang window bf16")}}


def page_kernel(results, run_name):
    data = parse_kernel_speed(results)
    body = ("<h1>Kernel time per decode step</h1><p class=\"lede\">One decode token for every request of the batch, "
            "for one linear-attention layer, on one GPU. LeapQuant next to the two kernels SGLang ships.</p>")
    summary = {}
    triton_body, triton_summary = triton_kernel_section(results)
    body += triton_body
    summary.update(triton_summary)
    if not data and triton_body:
        write_page("kernel.html", "Kernel speed", body, run_name)
        return summary
    if not data:
        body += pending("Run stage E: runbook/stage_d_e_kernel.sh e")
        write_page("kernel.html", "Kernel speed", body, run_name)
        return summary
    for dtype, rows in data.items():
        batches = sorted(rows)
        window_name = next(name for name in rows[batches[0]] if name.startswith("sglang window"))
        get = lambda name, column: [rows[b][name][column] for b in batches]
        packed, window, ours = get("sglang packed", 0), get(window_name, 0), get("leapquant, kernel only", 0)
        gated = get("leapquant, with gates", 0)
        labels = [f"batch {b}" for b in batches]
        body += f"<h2>Projections in {dtype}</h2>"
        image = grouped_bars(f"kernel_per_step_{dtype}.png", "Time per decode step, milliseconds",
                             "Lower is better · averaged over a 16-token window · labels: LeapQuant against the packed kernel",
                             labels, [("SGLang packed (today)", "packed", packed), ("SGLang window, L=16", "window", window),
                                      ("LeapQuant, kernel only", "leapquant", ours)], "ms per step",
                             annotate=[f"{p / o:.2f}×" for p, o in zip(packed, ours)], value_format="{:.2f}")
        body += figure_block(image, "Time per decode step by batch size")
        body += "<div class=\"scroll\">" + table(
            ["Batch", "SGLang packed", "SGLang window", "LeapQuant, kernel only", "LeapQuant, with gates",
             "LeapQuant against packed", "LeapQuant against window"],
            [[b, f"{p:.4f}", f"{w:.4f}", f"{o:.4f}", f"{g:.4f}", f"{p / o:.2f}×", f"{w / o:.2f}×"]
             for b, p, w, o, g in zip(batches, packed, window, ours, gated)]) + "</div>"

        append_ours, append_window = get("leapquant, kernel only", 1), get(window_name, 1)
        image = grouped_bars(f"kernel_append_{dtype}.png", "A step that only appends a record, milliseconds",
                             "Lower is better · 15 of every 16 steps · labels: LeapQuant against the packed kernel",
                             labels, [("SGLang packed (today)", "packed", packed), ("SGLang window, L=16", "window", append_window),
                                      ("LeapQuant, kernel only", "leapquant", append_ours)], "ms per step",
                             annotate=[f"{p / o:.2f}×" for p, o in zip(packed, append_ours)])
        body += figure_block(image, "Append-only step by batch size")
        fold_ours, fold_window = get("leapquant, kernel only", 2), get(window_name, 2)
        image = grouped_bars(f"kernel_fold_{dtype}.png", "A step that also folds the window, milliseconds",
                             "Lower is better · 1 of every 16 steps · labels: how many times slower LeapQuant's fold step is",
                             labels, [("SGLang window, L=16 (flush)", "window", fold_window),
                                      ("LeapQuant, kernel only (fold)", "leapquant", fold_ours)], "ms per step",
                             annotate=[f"{o / w:.0f}× slower" for w, o in zip(fold_window, fold_ours)])
        body += figure_block(image, "Fold step by batch size")
        body += "<div class=\"scroll\">" + table(
            ["Batch", "Append: packed", "Append: window", "Append: LeapQuant", "Fold step: window", "Fold step: LeapQuant"],
            [[b, f"{p:.4f}", f"{aw:.4f}", f"{ao:.4f}", f"{fw:.4f}", f"{fo:.4f}"]
             for b, p, aw, ao, fw, fo in zip(batches, packed, append_window, append_ours, fold_window, fold_ours)]) + "</div>"
        summary[dtype] = {"batch": batches[-1], "packed": packed[-1], "window": window[-1], "ours": ours[-1],
                          "append": append_ours[-1], "fold": fold_ours[-1]}
    for path in find("e_full_cycles_*.log"):
        body += ("<h2>Complete windows, with the state format of every kernel</h2><p>Positions 0 to 15 in order, every "
                 "kernel from the same starting state with empty record buffers (plan 5, step S07). Source: "
                 f"<code>{os.path.basename(path)}</code>.</p><pre class=\"scroll\">{html.escape(read(path))}</pre>")
    k = summary.get("fp32") or next(value for key, value in summary.items() if key != "triton")
    ratio_packed, ratio_window = k["packed"] / k["ours"], k["window"] / k["ours"]
    body += (f"<h3>What the charts show</h3><ul><li>At batch {k['batch']}, LeapQuant (kernel only) takes "
             f"<b>{k['ours']:.2f} ms</b> per step: {ratio_packed:.2f} times the speed of SGLang's packed kernel and "
             f"{ratio_window:.2f} times the speed of its window kernel (above 1 is faster, below 1 is slower).</li>"
             f"<li>A step that only appends takes {k['append']:.2f} ms; a step that also folds takes {k['fold']:.1f} ms. "
             "One step in sixteen folds.</li>"
             "<li>These averages assume all requests fold on the same step.</li></ul>"
             "<h3>Setup</h3><p>32 value heads, 16 key heads, state 128 × 128, window 16, one layer. Timed in one harness "
             "(<code>kernel/bench_kernel.py</code>): wall clock around 100 calls after 25 warm-up calls. "
             "\"With gates\" adds the gate and normalisation arithmetic that SGLang's kernels do internally.</p>"
             "<h3>What it does not show</h3><p>End-to-end serving speed, and any other GPU. Source: "
             "<code>e_kernel_speed_*.log</code>.</p>")
    write_page("kernel.html", "Kernel speed", body, run_name)
    return summary


def page_accuracy(results, run_name):
    traces = parse_replays(results)
    body = ("<h1>Error on real model states</h1><p class=\"lede\">What Qwen3.5-9B fed its Gated DeltaNet kernel while it "
            "generated text was recorded and replayed on a CPU through each way of storing the state. Error is the distance "
            "from the fp32 result, as a share of its size.</p>")
    summary = {}
    if not traces:
        body += pending("Run stage C: runbook/stage_c_traces.sh")
        write_page("accuracy.html", "Accuracy on real states", body, run_name)
        return summary
    lq_name = next(name for name in traces[0]["methods"] if "LeapQuant" in name)
    wins = 0
    for prompt in sorted({t["prompt"] for t in traces}):
        chosen = sorted((t for t in traces if t["prompt"] == prompt), key=lambda t: t["layer"])
        lq = [t["methods"][lq_name]["output"] for t in chosen]
        bf = [t["methods"]["bf16, rounded every window"]["output"] for t in chosen]
        wins += sum(a <= b for a, b in zip(lq, bf))
        body += f"<h2>Prompt: {html.escape(prompt)}</h2>"
        image = grouped_bars(f"accuracy_{prompt}.png", "Output error against fp32, percent",
                             f"Lower is better · {chosen[0]['tokens']:,} generated tokens · labels: bf16 error ÷ LeapQuant error",
                             [f"layer {t['layer']}" for t in chosen],
                             [("bf16, rounded every window (2.00 bytes/entry)", "bf16", bf),
                              ("LeapQuant (1.19 bytes/entry)", "leapquant", lq)], "output error, %",
                             annotate=[f"{b / a:.1f}×" if a > 0 else "" for a, b in zip(lq, bf)], value_format="{:.3f}")
        body += figure_block(image, f"Output error per layer, prompt {prompt}",
                             "A label above 1× means LeapQuant is the more accurate of the two; below 1× means bf16 is.")
        rows = []
        for t in chosen:
            for name, m in t["methods"].items():
                if name.startswith("fp32"):
                    continue
                rows.append([f"layer {t['layer']}", name, f"{m['bytes']:.2f}", f"{m['state_median']:.3f}%",
                             f"{m['state_worst']:.3f}%", f"{m['output']:.3f}%"])
        body += "<div class=\"scroll\">" + table(["Layer", "How the state is stored", "Bytes per entry",
                                                  "State error, median head", "State error, worst head", "Output error"],
                                                 rows, numeric_from=2) + "</div>"
    body += "<h2>Is each trace valid?</h2><p>The replay's fp32 arithmetic must reproduce the model's own kernel output.</p>"
    body += "<div class=\"scroll\">" + table(["Prompt", "Layer", "Tokens", "Self-check: worst difference", "Verdict"],
                                             [[t["prompt"], t["layer"], f"{t['tokens']:,}", t["self_check"],
                                               "ok" if t["ok"] else "NOT ok"] for t in traces], numeric_from=1) + "</div>"
    body += (f"<h3>What the charts show</h3><ul><li><b>LeapQuant is at least as accurate as bf16 with the window on "
             f"{wins} of {len(traces)} traces</b>, at 1.19 bytes per entry instead of 2.</li>"
             "<li><b>Rounding int8 after every token does not work</b>: see the tables.</li>"
             "<li><b>The result depends on the layer.</b> Look at the deepest layer before deciding.</li></ul>"
             "<h3>What it does not show</h3><p>Answer quality. These are arithmetic errors of one layer's state with the "
             "model's inputs held fixed; the model is not in the loop. Three layers of 24, three prompts. Source: "
             "<code>c_replay_*.txt</code>.</p>")
    for path in find("s19_fit_quality_*.txt"):
        body += ("<h2>Could better compensators help? (plan 5, step S19)</h2><p>For sampled fp32 states of each trace: the "
                 "share of the state's energy in its best four patterns (from an exact decomposition), and how much of "
                 "that our fit captures. A low share means there is little to capture; a low captured value means the fit "
                 f"misses what is there. Source: <code>{os.path.basename(path)}</code>.</p>"
                 f"<pre class=\"scroll\">{html.escape(read(path))}</pre>")
    rank8 = find("s20_rank8_*.txt")
    if rank8:
        rows = []
        for path in rank8:
            text = read(path)
            match = re.search(r"= LeapQuant\)\s+([\d.]+) \|.*\|\s*([\d.]+)%\s*$", text, flags=re.M)
            bf = re.search(r"bf16, rounded every window\s+[\d.]+ \|.*\|\s*([\d.]+)%\s*$", text, flags=re.M)
            if match and bf:
                rows.append([os.path.basename(path)[len("s20_rank8_"):-4], match.group(1), f"{match.group(2)}%", f"{bf.group(1)}%"])
        if rows:
            body += ("<h2>Eight compensators on layer 30 (plan 5, step S20)</h2>" +
                     table(["Trace", "Bytes per entry", "LeapQuant with 8 compensators: output error", "bf16 with the window"], rows))
    summary = {"wins": wins, "traces": len(traces), "all_ok": all(t["ok"] for t in traces)}
    write_page("accuracy.html", "Accuracy on real states", body, run_name)
    return summary


# Task-accuracy runs (eval/accuracy_bench.py --out accuracy_<name>.json) and how they are shown.
TASK_RUNS = [("fp32_run1", "fp32 state (run 1)", "packed"), ("fp32_run2", "fp32 state (run 2)", None),
             ("leapquant", "LeapQuant", "leapquant"), ("bf16_window", "bf16 + window", "bf16"),
             ("leapquant_r8", "LeapQuant, 8 compensators on layers 20-31", "window")]
TASK_NAMES = {"gsm8k": "GSM8K", "mmlu_pro": "MMLU-Pro", "aime2025": "AIME 2025"}


def page_tasks(results, run_name):
    body = ("<h1>Task accuracy</h1><p class=\"lede\">Qwen3.5-9B answers benchmark questions at temperature 0 through "
            "each server configuration; the answers are scored by rule. Runs of the same questions are compared in pairs: "
            "only the questions on which two runs disagree carry information.</p>")
    runs = {}
    for key, label, color in TASK_RUNS:
        paths = find(f"accuracy_{key}.json")
        path = paths[0] if paths else ""
        if path:
            with open(path, encoding="utf-8") as handle:
                runs[key] = json.load(handle)
    if not runs:
        body += pending("Run runbook/accuracy_fast.sh (or stage F, accuracy).")
        write_page("tasks.html", "Task accuracy", body, run_name)
        return {}
    tasks = [task for task in ("gsm8k", "mmlu_pro", "aime2025") if any(task in run["tasks"] for run in runs.values())]

    def score(run, task):
        items = run["tasks"].get(task, [])
        if not items:
            return None
        return (100.0 * sum(bool(item["correct"]) for item in items) / len(items), len(items),
                sum(bool(item.get("truncated")) for item in items),
                sum(item["completion_tokens"] for item in items) / len(items))

    series = [(label, color, [(score(runs[key], task) or (0.0,))[0] for task in tasks])
              for key, label, color in TASK_RUNS if key in runs and color]
    image = grouped_bars("tasks_accuracy.png", "Accuracy, percent of questions answered right",
                         "Higher is better · temperature 0 · labels: LeapQuant minus fp32 (run 1), points",
                         [TASK_NAMES.get(task, task) for task in tasks], series, "percent",
                         annotate=[f"{l - f:+.1f}" for l, f in zip(series[[s[0] for s in series].index('LeapQuant')][2],
                                                                    series[0][2])] if any(s[0] == 'LeapQuant' for s in series) else None,
                         value_format="{:.1f}")
    body += figure_block(image, "Accuracy per task and configuration")
    rows = []
    for key, label, _ in TASK_RUNS:
        if key not in runs:
            continue
        for task in tasks:
            result = score(runs[key], task)
            if result:
                accuracy, count, truncated, tokens = result
                rows.append([label, TASK_NAMES.get(task, task), count, f"{accuracy:.2f}%", truncated, f"{tokens:.0f}"])
    body += "<div class=\"scroll\">" + table(["Configuration", "Task", "Questions", "Accuracy", "Answers cut at the token limit",
                                               "Mean answer tokens"], rows, numeric_from=2) + "</div>"
    comparisons = []
    for path in find("accuracy_*vs*.txt") + find("accuracy_fp32_noise.txt"):
        lines = read(path).splitlines()
        for line in lines:
            match = re.match(r"\s+(gsm8k|mmlu_pro|aime2025)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+(\d+)\s+([+-][\d.]+)%\s+"
                             r"([+-][\d.]+ \.\. [+-][\d.]+)\s+(\w+)", line)
            if match:
                name = os.path.basename(path)[len("accuracy_"):-4].replace("_vs_", " against ").replace("fp32_noise", "fp32 run 2 against fp32 run 1")
                comparisons.append([name.replace("_", " "), TASK_NAMES.get(match.group(1), match.group(1)), match.group(2),
                                    f"{match.group(3)}%", f"{match.group(4)}%", match.group(5), match.group(6),
                                    f"{match.group(7)} pts", match.group(8) + " pts", match.group(9)])
    if comparisons:
        body += ("<h2>Paired comparisons</h2><p>Noninferiority with a margin of 1 point: PASS when the lower end of the 95% "
                 "interval of the difference is above -1 point; FAIL when its upper end is below -1; otherwise "
                 "INCONCLUSIVE (too few questions to tell). The fp32 server against itself shows the run-to-run noise.</p>")
        body += "<div class=\"scroll\">" + table(["Comparison", "Task", "Paired", "This run", "Baseline", "Only this right",
                                                   "Only baseline right", "Difference", "95% interval", "Verdict"],
                                                  comparisons, numeric_from=2) + "</div>"
    body += ("<h3>What it does not show</h3><p>Tasks other than these three; answers longer than the token limit are scored "
             "as they stand (see the column of cut answers). Source: <code>accuracy_*.json</code>, "
             "<code>accuracy_*_vs_*.txt</code>.</p>")
    write_page("tasks.html", "Task accuracy", body, run_name)
    return {"runs": list(runs)}


def page_verification(results, run_name):
    reports = parse_verify(results)
    body = ("<h1>GPU kernel against the reference core</h1><p class=\"lede\">The same recorded traces go through the GPU "
            "kernel and through the C++ reference. Three things are compared; <code>kernel/verify_kernel.py</code> explains why.</p>")
    summary = {}
    gpu_check = os.path.join(results, "steps", "gpu-check.log")
    if not reports and os.path.exists(gpu_check):
        text = read(gpu_check)
        if "RESULT: PASS" in text:
            body += ("<div class=\"note\">Only the synthetic trace has been verified so far (step <code>gpu-check</code>): "
                     "RESULT: PASS.</div><pre class=\"scroll\">" + html.escape(text[text.find("  A. inside"):text.find("RESULT: PASS") + 12]) + "</pre>")
    if not reports:
        body += pending("Run stage D on the recorded traces: runbook/stage_d_e_kernel.sh d")
        write_page("verification.html", "Verification", body, run_name)
        return summary
    labels = [r["trace"].replace("_trace_layer_", " L") for r in reports]
    image = grouped_bars("verify_free_run.png", "Free run: output error against fp32, percent",
                         "The kernel must be as close to fp32 as the reference is · equal bars are the goal",
                         labels, [("Reference core (CPU)", "packed", [r["reference"] for r in reports]),
                                  ("GPU kernel", "leapquant", [r["kernel"] for r in reports])], "output error, %",
                         value_format="{:.3f}")
    body += figure_block(image, "Free-run output error, kernel and reference")
    body += "<div class=\"scroll\">" + table(
        ["Trace", "Result", "In a window: median", "99% below", "worst", "Folds", "Bit-identical folds",
         "int8 entries that differ", "Free run: kernel", "reference", "Drift between the two free runs"],
        [[r["trace"], r["result"], r["median"], r["p99"], r["worst"], r["folds"], r["identical"],
          f"{r['int8_differ']} of {r['int8_total']}", f"{r['kernel']:.4f}%", f"{r['reference']:.4f}%", r["drift"]]
         for r in reports]) + "</div>"
    passed = sum(r["result"] == "PASS" for r in reports)
    body += (f"<h3>What it shows</h3><ul><li><b>{passed} of {len(reports)} traces pass.</b></li>"
             "<li>The drift column is why a plain output comparison cannot be the test: two correct runs end this far apart.</li></ul>"
             "<p class=\"muted\">Source: <code>d_verify_*.txt</code>.</p>")
    summary = {"passed": passed, "total": len(reports)}
    write_page("verification.html", "Verification", body, run_name)
    return summary


def page_tuning(results, run_name):
    builds, chosen = parse_tune(results)
    body = ("<h1>Build options of the GPU kernel</h1><p class=\"lede\">Every combination was built, verified against the "
            "reference and timed. The fastest correct build is installed.</p>")
    if not builds:
        body += pending("Run: python3 kernel/tune_kernel.py")
        write_page("tuning.html", "Tuning", body, run_name)
        return {}
    ordered = sorted(builds, key=lambda b: b["times"][-1])
    second = (lambda b: ("matrix in shared memory" if b["word_loads"] else "matrix in scratch memory")
              if b["second_is_shared"] else ("8-byte loads" if b["word_loads"] else "byte loads"))
    labels = [f"{b['threads']} threads · {second(b)} · "
              f"{'fused multiply-add' if b['fmad'] else 'no fused multiply-add'}" for b in ordered]
    image = horizontal_bars("tuning.png", f"Time per decode step at batch {ordered[0]['batches'][-1]}, milliseconds",
                            "Lower is better · LeapQuant with gates · the dark bar is the fastest", labels,
                            [b["times"][-1] for b in ordered], "ms per step", highlight=0)
    body += figure_block(image, "Time per step of every build")
    body += "<div class=\"scroll\">" + table(
        ["Threads per block", "Second option", "Fused multiply-add", "Verification"] + [f"Batch {b}" for b in ordered[0]["batches"]],
        [[b["threads"], second(b), "yes" if b["fmad"] else "no", b["verify"]] +
         [f"{t:.4f}" for t in b["times"]] for b in ordered], numeric_from=4) + "</div>"
    word = [b["times"][-1] for b in builds if b["word_loads"]] or [1.0]
    byte = [b["times"][-1] for b in builds if not b["word_loads"]] or [1.0]
    gain = 100 * (sum(byte) / len(byte) / (sum(word) / len(word)) - 1)
    what = "The fold's matrix in shared memory" if builds[0]["second_is_shared"] else "Reading the int8 matrix 8 bytes at a time"
    body += (f"<h3>What it shows</h3><ul><li><b>{what} is {gain:.0f}% faster</b> on average at the largest batch "
             "(a negative number means slower).</li>"
             f"<li>The sweep's fastest correct build was <code>{html.escape(chosen or '?')}</code>.</li></ul>"
             "<p class=\"muted\">Source: <code>steps/tune.log</code>.</p>")
    write_page("tuning.html", "Tuning", body, run_name)
    return {"gain": gain}


def page_profile(results, run_name):
    profiles = parse_profiles(results)
    body = ("<h1>Where the kernel's time goes</h1><p class=\"lede\">The kernel is stopped at named points of its work; a "
            "phase's time is the difference between two stops. Medians of 30 calls, state restored before every call.</p>")
    if not profiles:
        body += pending("Run: python3 kernel/profile_kernel.py --device cuda --batch 128")
        write_page("profile.html", "Profile", body, run_name)
        return {}
    for profile in profiles:
        body += f"<h2>Batch {profile['batch']}, {profile['sweeps']} sweeps <span class=\"muted\">{html.escape(profile['label'])}</span></h2>"
        if not profile["clean"]:
            body += "<div class=\"note\">Another job was using the GPU during this profile. Treat the numbers as rough.</div>"
        for key, title in (("decode", "Decode step (append only)"), ("fold", "Fold of a full window")):
            rows = profile["sections"].get(key, [])
            if not rows:
                continue
            tag = f"{profile['file'][:-4]}_{key}.png"
            image = horizontal_bars(tag, f"{title}: milliseconds per phase", f"batch {profile['batch']} · all rows",
                                    [r[0] for r in rows], [max(r[1], 0.0) for r in rows], "ms")
            body += figure_block(image, f"{title}, time per phase")
            total = rows[-1][2]
            body += "<div class=\"scroll\">" + table(["Phase", "ms", "Share", "Cumulative ms", "10% .. 90% of calls"],
                                                     [[r[0], f"{r[1]:.3f}", f"{100 * r[1] / total:.1f}%", f"{r[2]:.3f}",
                                                       f"{r[3]:.3f} .. {r[4]:.3f}"] for r in rows]) + "</div>"
        if profile["per_step"]:
            body += f"<p>Per decode step, averaged over a window: <b>{profile['per_step']:.3f} ms</b>.</p>"
    body += "<p class=\"muted\">Source: <code>profile_*.txt</code>. A difference smaller than the 10% .. 90% spread is not a finding.</p>"
    write_page("profile.html", "Profile", body, run_name)
    return {"count": len(profiles)}


# Shapes of the serving client (runbook/env.sh, run_serving_client).
SHAPE_TEXT = {"long": "4,096 in / 1,024 out, random lengths", "short": "512 in / 512 out, random lengths",
              "decode": "128 in / 512 out, exact lengths", "decode2k": "128 in / 2,048 out, exact lengths",
              "aa1k": "1,000 in / 1,000 out, exact (Artificial Analysis's system load test shape)",
              "ix8k1k": "8,192 in / 1,024 out, exact (SemiAnalysis InferenceX's single-turn 8k/1k shape)",
              "ix1k8k": "1,024 in / 8,192 out, exact (SemiAnalysis InferenceX's 1k/8k shape, long outputs)"}
# Shapes run as concurrency sweeps: shown as throughput against speed per user, not as bars.
SWEEP_SHAPES = ("aa1k", "ix8k1k", "ix1k8k")


def pareto_chart(filename, title, subtitle, series, xlabel, ylabel):
    """series: [(name, color key, [(x, y, label)])]: one line per configuration, points labelled."""
    columns, rows = legend_layout([name for name, _, points in series if points], budget=125)
    legend_height = 0.28 * rows + 0.14
    height = 5.0 + legend_height
    fig, axis = plt.subplots(figsize=(9.2, height), dpi=130)
    for name, color, points in series:
        points = sorted(points)
        if not points:
            continue
        xs, ys, labels = zip(*points)
        axis.plot(xs, ys, color=COLOR[color], linewidth=2, marker="o", markersize=6, label=name)
        for x, y, label in points:
            axis.annotate(label, (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8, color=MUTED)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    fig.legend(*axis.get_legend_handles_labels(), frameon=False, loc="lower left", ncol=columns, fontsize=9,
               bbox_to_anchor=(0.01, 0.005))
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.02, 1 - 0.42 / height, subtitle, fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, legend_height / height, 1, 1 - 0.5 / height))
    fig.savefig(os.path.join(IMG, filename))
    plt.close(fig)
    return f"img/{filename}"
SERVING_NAMES = {"radix_fp32_window": ("fp32 + window, radix cache, 560 slots", "window"),
                 "radix_bf16_window": ("bf16 + window, radix cache", "bf16"),
                 "radix_leapquant": ("LeapQuant compressed, radix cache", "leapquant"),
                 "config0_sglang_defaults": ("SGLang defaults (server capped at 83 running requests)", "packed"),
                 "config1_fp32": ("fp32 state, no window", "packed"), "config2_fp32_window": ("fp32 + 16-token window", "window"),
                 "config3_bf16_window": ("bf16 + 16-token window", "bf16"), "config4_leapquant": ("LeapQuant", "leapquant")}


# ---- serving page: every metric of every arm -------------------------------------------------

# The numbers public serving benchmarks publish (SGLang's and vLLM's benchmark_serving, SemiAnalysis InferenceX,
# Artificial Analysis's system load test), in families. Every one is parsed from the run's logs; "-" means the log
# does not hold it.
SERVING_FAMILIES = [
    ("Throughput (one GPU: also the figure per GPU)", [
        "Output token throughput (tok/s)", "Total token throughput (tok/s)", "Input token throughput (tok/s)",
        "Request throughput (req/s)", "Peak output token throughput (tok/s)"]),
    ("Speed per user (interactivity)", [
        "Speed per user, median (tok/s)", "Speed per user, 90% of requests at least (tok/s)",
        "Speed per user, 99% of requests at least (tok/s)"]),
    ("Time to first token", ["Mean TTFT (ms)", "Median TTFT (ms)", "P90 TTFT (ms)", "P95 TTFT (ms)", "P99 TTFT (ms)"]),
    ("Time per output token (after the first)", [
        "Mean TPOT (ms)", "Median TPOT (ms)", "P90 TPOT (ms)", "P95 TPOT (ms)", "P99 TPOT (ms)"]),
    ("Time between tokens", [
        "Mean ITL (ms)", "Median ITL (ms)", "P90 ITL (ms)", "P95 ITL (ms)", "P99 ITL (ms)", "Max ITL (ms)"]),
    ("End-to-end latency", [
        "Mean E2E Latency (ms)", "Median E2E Latency (ms)", "P90 E2E Latency (ms)", "P95 E2E Latency (ms)",
        "P99 E2E Latency (ms)"]),
    ("Load, requests, tokens, memory", [
        "Max request concurrency", "Concurrency", "Peak concurrent requests", "Successful requests",
        "Benchmark duration (s)", "Total input tokens", "Total generated tokens", "Peak GPU memory (MiB)"]),
]
ALL_SERVING_METRICS = [metric for _, metrics in SERVING_FAMILIES for metric in metrics]
NO_RATIO = {"Max request concurrency", "Concurrency", "Peak concurrent requests", "Successful requests",
            "Total input tokens", "Total generated tokens"}
# The minimum set public benchmarks show for every load point: the headline table and the charts.
HEADLINE = [("Output token throughput (tok/s)", "Output tokens/s", 1.0, "{:,.0f}"),
            ("Speed per user, median (tok/s)", "Speed per user, tokens/s", 1.0, "{:.1f}"),
            ("Median TTFT (ms)", "TTFT median, s", 1000.0, "{:.2f}"),
            ("P99 TTFT (ms)", "TTFT p99, s", 1000.0, "{:.2f}"),
            ("Median TPOT (ms)", "TPOT median, ms", 1.0, "{:.1f}"),
            ("P99 ITL (ms)", "ITL p99, ms", 1.0, "{:.0f}"),
            ("Median E2E Latency (ms)", "E2E median, s", 1000.0, "{:.1f}"),
            ("P99 E2E Latency (ms)", "E2E p99, s", 1000.0, "{:.1f}"),
            ("Peak GPU memory (MiB)", "Peak GPU memory, GiB", 1024.0, "{:.1f}")]
CHART_METRICS = [
    ("Output token throughput (tok/s)", "Output token throughput", "Higher is better", "tokens/s (one GPU)", 1.0, "{:,.0f}"),
    ("Speed per user, median (tok/s)", "Speed per user (interactivity), median", "Higher is better",
     "tokens/s per request", 1.0, "{:.1f}"),
    ("Request throughput (req/s)", "Request throughput", "Higher is better", "requests/s", 1.0, "{:.2f}"),
    ("Median TTFT (ms)", "Time to first token, median", "Lower is better", "seconds", 1000.0, "{:.2f}"),
    ("P99 TTFT (ms)", "Time to first token, p99", "Lower is better", "seconds", 1000.0, "{:.2f}"),
    ("Median ITL (ms)", "Time between tokens, median", "Lower is better", "milliseconds", 1.0, "{:.1f}"),
    ("P99 ITL (ms)", "Time between tokens, p99 (steps with a prefill or a fold)", "Lower is better", "milliseconds",
     1.0, "{:.0f}"),
    ("Median E2E Latency (ms)", "End-to-end latency, median", "Lower is better", "seconds", 1000.0, "{:.1f}"),
    ("P99 E2E Latency (ms)", "End-to-end latency, p99", "Lower is better", "seconds", 1000.0, "{:.1f}"),
    ("Peak GPU memory (MiB)", "Peak GPU memory during the point", "Lower is better", "GiB", 1024.0, "{:.1f}"),
]
SHAPE_SHORT = {"long": "4,096/1,024 random", "short": "512/512 random", "decode": "128/512", "decode2k": "128/2,048",
               "aa1k": "1,000/1,000", "ix8k1k": "8,192/1,024", "ix1k8k": "1,024/8,192"}
SHAPE_ORDER = {shape: index for index, shape in enumerate(("decode", "decode2k", "short", "long", "aa1k", "ix8k1k",
                                                           "ix1k8k"))}

# Arm labels of the runbooks (serving_<label>_<shape>_c<N>.log; from round 9: <on|off>_<arm>_r<repetition>) and how
# the pages name them. A label not listed is shown as it is.
ARM_NAMES = {
    "bf16_window": "bf16 state + 16-token window", "fp32_window": "fp32 state + 16-token window",
    "fp32_packed": "fp32 state (SGLang's default)", "leapquant_cuda": "LeapQuant (CUDA fold)", "leapquant": "LeapQuant",
    "lq_field": "LeapQuant, slot copy field by field", "lq_fused": "LeapQuant, fused slot copy",
    "lq_aligned": "LeapQuant, aligned folds", "lq_auto": "LeapQuant, folds 'auto'", "lq_stagger": "LeapQuant, staggered folds",
    "lq_policy_default": "LeapQuant, fold policy default", "lq_policy_auto": "LeapQuant, fold policy 'auto'",
    "lq_policy_stagger": "LeapQuant, fold policy 'stagger'", "lq_reuse": "LeapQuant, step inputs built once",
    "lq_noreuse": "LeapQuant, step inputs built per layer", "lq_reuse_cudadec": "LeapQuant, step inputs built once, CUDA decode",
    "best_leapquant_cuda": "LeapQuant (CUDA fold), best settings",
    "best_bf16_lazy_int8_packed": "SGLang's best settings (best_bf16_lazy_int8_packed)",
    "sglang_default": "SGLang with no options (its defaults)",
    "sglang_best": "SGLang's best state settings (bf16, lazy extra buffers, int8 checkpoints)",
}
ARM_NAMES.update({key: value[0] for key, value in SERVING_NAMES.items()})
BASELINES = ("bf16_window", "config3_bf16_window", "radix_bf16_window", "best_bf16_lazy_int8_packed")
FIXED_COLORS = {"bf16_window": "bf16", "config3_bf16_window": "bf16", "radix_bf16_window": "bf16",
                "best_bf16_lazy_int8_packed": "bf16", "fp32_packed": "packed", "config1_fp32": "packed",
                "fp32_window": "window", "config2_fp32_window": "window", "radix_fp32_window": "window",
                "sglang_default": "packed"}
ARM_LABEL = re.compile(r"^(?:(on|off)_)?(.+?)(?:_r(\d+))?$")


def split_label(label):
    """'on_lq_fused_r1' -> ('on', 'lq_fused', 1); 'config4_leapquant' -> ('', 'config4_leapquant', 1)."""
    match = ARM_LABEL.match(label)
    return match.group(1) or "", match.group(2), int(match.group(3) or 1)


def order_arms(arms):
    """The baseline (bf16 + window when the run has it) first, then the other reference arms, then the rest."""
    arms = sorted(arms)
    base = next((arm for arm in BASELINES if arm in arms), None) or next((a for a in arms if "bf16" in a), arms[0])
    references = [a for a in arms if a != base and a in FIXED_COLORS]
    return [base] + references + [a for a in arms if a != base and a not in references]


def arm_colors(arms):
    """Known arms keep their colour on every page; the others take blue, then magenta, green, violet, red, in order."""
    free = ["leapquant", "slot5", "slot6", "slot7", "slot8"]
    colors = {}
    for arm in arms:
        if arm in FIXED_COLORS and FIXED_COLORS[arm] not in colors.values():
            colors[arm] = FIXED_COLORS[arm]
    for arm in arms:
        if arm not in colors:
            colors[arm] = free.pop(0) if free else "slot8"
    return colors


def metric_text(metric, value):
    if value is None or value != value:
        return "-"
    if metric.startswith("Speed per user"):
        return f"{value:.1f}"
    if metric.endswith("(req/s)") or metric == "Concurrency":
        return f"{value:.2f}"
    if metric.endswith(("(tok/s)", "(MiB)")) or "TTFT" in metric or "E2E" in metric or metric == "Max ITL (ms)" \
            or metric in NO_RATIO:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def ratio_text(metric, value, base):
    if metric in NO_RATIO or value is None or base is None or base <= 0:
        return ""
    return f"{value / base:.3f}"


def server_setup(label_list):
    """What a server held: state type, slots, state and KV memory, limits; from its log (server_<label>.log)."""
    path = None
    for label in sorted(label_list):
        found = find(f"server_{label}.log")
        if found:
            path = found[0]
            break
    if not path:
        return None
    text = read(path)
    setup = {"log": os.path.basename(path)}
    args = re.search(r"server_args=\{(.*?)\}\s*$", text, flags=re.M)
    args = args.group(1) if args else ""

    def arg(name):
        found = re.search(r"'" + name + r"': ('?)([^,']*)\1", args)
        return found.group(2) if found else None
    dtype = arg("mamba_ssm_dtype") or "?"
    state = "LeapQuant (compressed)" if arg("enable_linear_leapquant") == "True" else f"{dtype}"
    if arg("enable_linear_replayssm") == "True":
        state += f" + {arg('linear_replayssm_cache_len') or '?'}-token window"
    setup["state"] = state
    setup["radix"] = "off" if arg("disable_radix_cache") == "True" else f"on ({arg('mamba_radix_cache_strategy') or '?'})"
    setup["mem_fraction"] = arg("mem_fraction_static")
    mamba = re.search(r"max_mamba_cache_size: (\d+), conv_state size: ([\d.]+) ?GB, ssm_state size: ([\d.]+) ?GB", text)
    leapquant = re.search(r"LeapQuant decode state allocated: .*? ([\d.]+) GB", text)
    if mamba:
        setup["slots"] = int(mamba.group(1))
        setup["state_gb"] = float(mamba.group(2)) + float(mamba.group(3)) + (float(leapquant.group(1)) if leapquant else 0.0)
    kv = re.search(r"KV Cache is allocated\. dtype: (\S+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", text)
    if kv:
        setup["kv_tokens"] = int(kv.group(2))
        setup["kv_gb"] = float(kv.group(3)) + float(kv.group(4))
    limits = re.search(r"max_total_num_tokens=(\d+).*?max_running_requests=(\d+)", text)
    if limits:
        setup["max_running"] = int(limits.group(2))
    return setup


def parse_bursts():
    """burst_<label>.txt (eval/prefill_burst.py): prefill-only bursts, per server."""
    rows = {}
    for path in find("burst_*.txt"):
        label = os.path.basename(path)[len("burst_"):-len(".txt")]
        text = read(path)
        wall = re.search(r"median wall ([\d.]+) s over (\d+) bursts", text)
        rates = sorted(float(v.replace(",", "")) for v in re.findall(r"([\d,]+) prompt tok/s", text))
        latency = sorted(float(v) for v in re.findall(r"latency median ([\d.]+) s", text))
        if wall:
            rows[label] = (float(wall.group(1)), int(wall.group(2)), rates[len(rates) // 2] if rates else None,
                           latency[len(latency) // 2] if latency else None)
    return rows


def parse_turns():
    """turns_<label>.txt and .status (integrations/sglang/radix_turns_client.py): second turns from the radix cache."""
    rows = {}
    for path in find("turns_*.txt"):
        label = os.path.basename(path)[len("turns_"):-len(".txt")]
        text = read(path)
        status_path = path[:-4] + ".status"
        status = read(status_path).strip() if os.path.exists(status_path) else ""
        cached = [int(v) for v in re.findall(r"(\d+) from the cache", text)]
        same = re.findall(r"turn 2 (\d+)/(\d+)", text)
        rows[label] = ("PASS" if status == "0" else ("FAIL" if status else "?"), len(cached),
                       f"{min(cached)}-{max(cached)}" if cached else "-",
                       f"{sum(int(a) for a, _ in same)}/{sum(int(b) for _, b in same)}" if same else "-")
    return rows


def manifest_value(key):
    for folder in SOURCES:
        path = os.path.join(folder, "manifest.txt")
        if os.path.exists(path):
            found = re.search(r"^" + re.escape(key) + r":\s*(.+)$", read(path), flags=re.M)
            if found:
                return found.group(1).strip()
    return None


def gpu_text():
    """'NVIDIA H100 PCIe, 81559 MiB, 580.196.07' (manifest.txt) -> 'NVIDIA H100 PCIe (81,559 MiB, driver 580.196.07)'."""
    value = manifest_value("gpu")
    if not value:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 3 and parts[1].endswith("MiB") and parts[1][:-3].strip().isdigit():
        return f"{parts[0]} ({int(parts[1][:-3]):,} MiB, driver {parts[2]})"
    return value


def page_serving(results, run_name):
    runs = parse_serving(results)
    notes = os.path.join(results, "NOTES.txt")
    note = ("<div class=\"note\">" + "<br>".join(html.escape(line) for line in read(notes).strip().splitlines())
            + "</div>") if os.path.exists(notes) else ""
    definitions = (
        "<h2>Metrics</h2><ul>"
        "<li><b>Output token throughput</b>, tokens/s: generated tokens per second, all requests together. One GPU, so "
        "this is also the figure per GPU that public benchmarks plot.</li>"
        "<li><b>Total token throughput</b>: input plus generated tokens per second. <b>Request throughput</b>: completed "
        "requests per second. <b>Peak output token throughput</b>: the most tokens generated in any one-second window "
        "(the client's count).</li>"
        "<li><b>Speed per user</b> (interactivity), tokens/s: 1000 ÷ a request's time per output token. The median, "
        "and the speeds that 90% and 99% of requests reach or beat (from the p90 and p99 TPOT).</li>"
        "<li><b>TTFT</b>, time to first token: from sending the request to its first token, queueing and prefill "
        "included. <b>TPOT</b>, time per output token: a request's generation time after the first token divided by "
        "its other tokens. <b>ITL</b>, time between tokens: every gap between two consecutive tokens of a request (its "
        "tail shows the steps that carry a prefill or a fold). <b>E2E latency</b>: the whole request. Mean, median, "
        "p90, p95, p99 over the requests of the point.</li>"
        "<li><b>Concurrency</b>: the average number of requests in flight; <b>Max request concurrency</b>: the client's "
        "limit (the users of the point). <b>Peak concurrent requests</b>: the most requests active at some moment of "
        "any one-second window, as the client counts it; a request that ends and one that starts within the same "
        "second both count, so it can exceed the limit and is not the number running at once. <b>Peak GPU memory</b>: nvidia-smi's peak for the GPU while the point ran "
        "(sampled every 0.5 s), the server's preallocated pools included.</li></ul>")
    method = (
        "<h2>Method</h2><p>Closed loop, as in SemiAnalysis InferenceX and SGLang's own benchmarks: each point keeps a "
        "fixed number of requests in flight (the users) with no arrival-rate limit, random prompts "
        "(<code>sglang.benchmark.serving --dataset-name random</code>), exact input and output lengths unless the shape "
        "says random, three warm-up requests and a cache flush before the point; output lengths are forced (no early "
        "stop), so every server generates the same tokens. Open-loop load at fixed arrival rates, where it was run, is "
        "in its own section. Every number on this page is parsed from a log named in the tables; nothing is typed in."
        "</p><h3>What it does not show</h3><p>Answer quality (see the task accuracy page), other GPUs, and more than "
        "one GPU.</p>")
    if not runs:
        openloop = openloop_section()
        body = (openloop + note + definitions + method) if openloop else (
            "<h1>Serving</h1>" + note + pending("No serving logs (serving_<label>_<shape>_c<N>.log) in this run.")
            + definitions)
        write_page("serving.html", "Serving", body, run_name)
        return {}

    points, labels_of = {}, {}
    for (label, shape, conc), metrics in runs.items():
        mode, arm, rep = split_label(label)
        points.setdefault((mode, shape, conc), {}).setdefault(arm, []).append((rep, label, metrics))
        labels_of.setdefault(arm, set()).add(label)
    arms = order_arms(labels_of)
    colors = arm_colors(arms)
    base = arms[0]
    name = lambda arm: ARM_NAMES.get(arm, arm)
    modes = sorted({mode for mode, _, _ in points})
    keys = sorted(points, key=lambda k: (k[0], SHAPE_ORDER.get(k[1], 99), k[1], k[2]))

    def point_name(key, short=False):
        mode, shape, conc = key
        prefix = f"radix {mode} · " if len(modes) > 1 and mode else ""
        shape_text = SHAPE_SHORT.get(shape, shape) if short else SHAPE_TEXT.get(shape, shape)
        return f"{prefix}{shape_text} · {conc} users"

    def mean(key, arm, metric):
        values = [m[metric] for _, _, m in points[key].get(arm, []) if metric in m]
        return sum(values) / len(values) if values else None

    gpu = gpu_text() or "one GPU"
    sglang = manifest_value("sglang_commit")
    body = ("<h1>Serving: throughput, latency, speed per user</h1>"
            f"<p class=\"lede\">{html.escape(manifest_value('model') or 'Qwen3.5-9B')} on {html.escape(gpu)}, SGLang"
            f"{(' ' + html.escape(sglang)) if sglang else ''}. {len(arms)} servers, {len(keys)} load points. Baseline "
            f"for the ratios: <b>{html.escape(name(base))}</b> (ratio = server ÷ baseline: above 1 is better for "
            f"throughput and speed, below 1 for times and memory).</p>" + note)

    # 1. The headline table: the minimum set, every point, every server.
    rows = []
    for key in keys:
        for arm in arms:
            if arm not in points[key]:
                continue
            row = [point_name(key, short=True), name(arm)]
            for metric, _, scale, fmt in HEADLINE:
                value = mean(key, arm, metric)
                row.append(fmt.format(value / scale) if value is not None else "-")
            output, base_output = mean(key, arm, HEADLINE[0][0]), mean(key, base, HEADLINE[0][0])
            row.insert(3, f"{output / base_output:.3f}" if arm != base and output and base_output else "")
            rows.append(row)
    headers = ["Load point", "Server", HEADLINE[0][1], "÷ baseline"] + [label for _, label, _, _ in HEADLINE[1:]]
    body += ("<h2>Every load point at a glance</h2><p class=\"muted\">The numbers public benchmarks lead with. Means "
             "over the repetitions where a point was run more than once (each repetition is in the full table at the "
             "end).</p><div class=\"scroll\">" + table(headers, rows, numeric_from=2) + "</div>")

    # 2. Charts of the minimum set, one per metric, with the table of the chart.
    groups = [point_name(key, short=True).replace(" · ", "\n", 1) if len(modes) < 2 else point_name(key, short=True)
              for key in keys]
    body += "<h2>Charts</h2>"
    for metric, title, direction, unit, scale, fmt in CHART_METRICS:
        series = []
        for arm in arms:
            values = [mean(key, arm, metric) for key in keys]
            if any(v is not None for v in values):
                series.append((name(arm), colors[arm], [(v or 0.0) / scale for v in values]))
        if not series or not any(any(values) for _, _, values in series):
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")
        image = grouped_bars(f"serving_{slug}.png", title, f"{direction} · load point: input/output tokens and users",
                             groups, series, unit, value_format=fmt)
        chart_rows = [[point_name(key, short=True)] + [fmt.format(v / scale) if (v := mean(key, arm, metric)) is not None
                                                        else "-" for arm in arms] for key in keys]
        body += (f"<h3>{html.escape(title)}</h3>" + figure_block(image, title) + "<div class=\"scroll\">"
                 + table(["Load point"] + [name(arm) for arm in arms], chart_rows, numeric_from=1) + "</div>")

    # 3. Throughput against speed per user, for shapes run at several concurrencies (the public-benchmark plot).
    for mode in modes:
        for shape in sorted({k[1] for k in keys if k[0] == mode}, key=lambda s: SHAPE_ORDER.get(s, 99)):
            series = []
            for arm in arms:
                pts = []
                for key in keys:
                    if key[0] != mode or key[1] != shape:
                        continue
                    speed, output = mean(key, arm, "Speed per user, median (tok/s)"), \
                        mean(key, arm, "Output token throughput (tok/s)")
                    if speed and output:
                        pts.append((speed, output, str(key[2])))
                if len(pts) >= 2:
                    series.append((name(arm), colors[arm], pts))
            if series:
                image = pareto_chart(f"sweep_{mode or 'all'}_{shape}.png", "System throughput against speed per user",
                                     f"{SHAPE_TEXT.get(shape, shape)} · one point per number of users (labelled) · up "
                                     f"and to the right is better", series,
                                     "speed per user: output tokens/s per request (1000 / median TPOT)",
                                     "output throughput, tokens/s (one GPU)")
                body += (f"<h3>Concurrency sweep: {html.escape(point_name((mode, shape, 0), short=False).rsplit(' · ', 1)[0])}"
                         f"</h3>" + figure_block(image, f"Sweep {shape}"))

    # 4. Every metric, per load point.
    body += ("<h2>Every metric, per load point</h2><p class=\"muted\">All the numbers the benchmark client reports, "
             "grouped as public benchmark reports group them, with the ratio to the baseline.</p>")
    for key in keys:
        present = [arm for arm in arms if arm in points[key]]
        columns = []
        for arm in present:
            reps = points[key][arm]
            merged = {metric: mean(key, arm, metric) for metric in ALL_SERVING_METRICS if mean(key, arm, metric) is not None}
            merged["_client"] = reps[0][2].get("_client")
            columns.append((name(arm), merged))
        extra = [["Repetitions"] + [str(len(points[key][arm])) for arm in present]]
        body += (f"<details><summary><b>{html.escape(point_name(key))}</b></summary>"
                 + every_metric_table(columns, 0 if base in points[key] else None, extra) + "</details>")

    # 5. The servers: what each held.
    rows = []
    for arm in arms:
        setup = server_setup(labels_of[arm])
        if not setup:
            rows.append([name(arm)] + ["-"] * 7)
            continue
        rows.append([name(arm), setup.get("state", "-"), setup.get("radix", "-"),
                     f"{setup['slots']:,}" if "slots" in setup else "-",
                     f"{setup['state_gb']:.1f}" if "state_gb" in setup else "-",
                     f"{setup['kv_tokens']:,} ({setup['kv_gb']:.1f} GB)" if "kv_tokens" in setup else "-",
                     f"{setup['max_running']:,}" if "max_running" in setup else "-", setup["log"]])
    body += ("<h2>The servers</h2><p class=\"muted\">From each server's log: the state it kept, its pools and its "
             "limit on running requests.</p><div class=\"scroll\">"
             + table(["Server", "State", "Radix cache", "State slots", "State memory, GB", "KV cache, tokens",
                      "Max running requests", "Log"], rows, numeric_from=3)
             + "</div>")

    # 6. Checks run on the same servers.
    bursts = parse_bursts()
    if bursts:
        rows = [[name(split_label(label)[1]) + (f" (radix {split_label(label)[0]})" if split_label(label)[0] else ""),
                 f"{wall:.3f}", str(count), f"{rate:,.0f}" if rate else "-", f"{latency:.3f}" if latency else "-"]
                for label, (wall, count, rate, latency) in sorted(bursts.items())]
        body += ("<h2>Prefill-only bursts</h2><p class=\"muted\">Concurrent prompts with one output token each "
                 "(<code>eval/prefill_burst.py</code>, cache flushed before each burst): the prefill path alone."
                 "</p><div class=\"scroll\">" + table(["Server", "Median wall, s", "Bursts", "Prompt tokens/s (median)",
                                                       "Request latency (median), s"], rows, numeric_from=1) + "</div>")
    turns = parse_turns()
    if turns:
        rows = [[name(split_label(label)[1]), verdict, str(count), cached, same]
                for label, (verdict, count, cached, same) in sorted(turns.items())]
        body += ("<h2>Two-turn radix check</h2><p class=\"muted\">Second turns that start from a state taken from the "
                 "radix cache (<code>integrations/sglang/radix_turns_client.py</code>); same words: second-turn words "
                 "equal to the reference server's.</p><div class=\"scroll\">"
                 + table(["Server", "Verdict", "Conversations", "Turn-2 tokens from the cache", "Same words (turn 2)"],
                         rows, numeric_from=2) + "</div>")

    body += openloop_section()

    # 7. Everything, one row per log.
    raw_rows = []
    for (label, shape, conc), metrics in sorted(runs.items(), key=lambda item: (item[0][2], item[0][1], item[0][0])):
        raw_rows.append([label, shape, conc] + [metric_text(metric, metrics.get(metric)) for metric in ALL_SERVING_METRICS])
    body += ("<h2>Full measurements</h2><details><summary>One row per log "
             "(<code>serving_&lt;label&gt;_&lt;shape&gt;_c&lt;users&gt;.log</code>), every metric</summary>"
             "<div class=\"scroll\">" + table(["Label", "Shape", "Users"] + ALL_SERVING_METRICS, raw_rows, numeric_from=2)
             + "</div></details>")
    shapes = sorted({shape for _, shape, _ in keys}, key=lambda s: SHAPE_ORDER.get(s, 99))
    body += (definitions + method + "<h3>Shapes</h3><p>" + "; ".join(
        f"<b>{shape}</b>: {html.escape(SHAPE_TEXT.get(shape, shape))}" for shape in shapes) + ".</p>")
    write_page("serving.html", "Serving", body, run_name)
    return {"configs": arms}


def page_index(results, run_name, parts):
    gpu = "unknown GPU"
    gpu_file = os.path.join(results, "gpu.csv")
    if os.path.exists(gpu_file):
        lines = read(gpu_file).strip().splitlines()
        if len(lines) > 1:
            gpu = lines[1]
    elif gpu_text():
        gpu = gpu_text()
    steps = []
    for path in sorted(glob.glob(os.path.join(results, "steps", "*.passed")) + glob.glob(os.path.join(results, "steps", "*.failed"))):
        name, state = os.path.basename(path).rsplit(".", 1)
        steps.append((name, state))
    tiles = ""
    kernel = parts.get("kernel", {})
    if kernel.get("triton"):
        k = kernel["triton"]
        tiles += f"<div class=\"tile\"><b>{k['packed'] / k['ours']:.2f}×</b><span>LeapQuant (fast kernels) against SGLang's packed fp32 kernel, per token, batch {k['batch']} (below 1× is slower)</span></div>"
        if k.get("bf16"):
            tiles += f"<div class=\"tile\"><b>{k['bf16'] / k['ours']:.2f}×</b><span>LeapQuant (fast kernels) against SGLang's bf16 window kernel, per token, batch {k['batch']}</span></div>"
    elif kernel:
        k = kernel.get("fp32") or next(iter(kernel.values()))
        tiles += f"<div class=\"tile\"><b>{k['packed'] / k['ours']:.2f}×</b><span>LeapQuant against SGLang's packed kernel, per decode step, batch {k['batch']} (below 1× is slower)</span></div>"
        tiles += f"<div class=\"tile\"><b>{k['packed'] / k['append']:.2f}×</b><span>the same for a step that only appends (15 of 16 steps)</span></div>"
    accuracy = parts.get("accuracy", {})
    if accuracy:
        tiles += f"<div class=\"tile\"><b>{accuracy['wins']} of {accuracy['traces']}</b><span>real traces where LeapQuant is at least as accurate as bf16 with the window</span></div>"
    verification = parts.get("verification", {})
    if verification:
        tiles += f"<div class=\"tile\"><b>{verification['passed']} of {verification['total']}</b><span>real traces on which the GPU kernel passes verification</span></div>"
    body = ("<h1>LeapQuant on the GPU: what has been measured</h1>"
            f"<p class=\"lede\">Run <code>{html.escape(run_name)}</code> on {html.escape(gpu)}.</p>"
            f"<div class=\"tiles\">{tiles}</div>"
            "<h2>Reports</h2>" + table(["Report", "What it answers"], [
                [Html(f"<a href=\"{name}\">{label}</a>"), question] for name, label, question in [
                    ("kernel.html", "Kernel speed", "Is the LeapQuant kernel faster than the kernels SGLang ships?"),
                    ("accuracy.html", "Accuracy on real states", "Gate 2: is int8 with compensators as accurate as bf16 with the window?"),
                    ("tasks.html", "Task accuracy", "Does the model answer as well with LeapQuant as with the fp32 state?"),
                    ("verification.html", "Verification", "Does the GPU kernel compute what the reference core defines?"),
                    ("tuning.html", "Tuning", "Which build options are fastest and still correct?"),
                    ("profile.html", "Profile", "Which phase of the kernel costs the time?"),
                    ("serving.html", "Serving", "Gate 1 and the end-to-end result: throughput, time to first token, speed per user.")]
                if name in WRITTEN and has_data(WRITTEN[name][1])], numeric_from=9) +
            "<p class=\"muted\">Only the reports with measurements are listed. Measured in: "
            + ", ".join(f"<code>{html.escape(os.path.relpath(s, os.path.dirname(REPORTS)))}</code>" for s in SOURCES) + ".</p>"
            + ("<h2>Steps of the run</h2>" + table(["Step", "State"], steps, numeric_from=9) if steps else "")
            + status_table(results))
    write_page("index.html", "LeapQuant reports", body, run_name)


def status_table(results):
    """STATUS.txt of a detached run (runbook/*.sh): one line per step, as a table."""
    path = os.path.join(results, "STATUS.txt")
    if not os.path.exists(path):
        return ""
    rows = []
    for line in read(path).splitlines():
        match = re.match(r"(\S+) (PASS|FAIL) (\d+) (min|s)\s+(.*)", line)
        if match:
            rows.append([match.group(1), match.group(5), match.group(2), f"{match.group(3)} {match.group(4)}"])
    if not rows:
        return ""
    return "<h2>Steps of the run</h2>" + table(["Step", "What", "Verdict", "Time"], rows, numeric_from=9)


def write_list_of_reports():
    """reports/all_reports.html: every report folder, newest first. This one file is rewritten each time."""
    folders = sorted((name for name in os.listdir(REPORTS)
                      if os.path.isfile(os.path.join(REPORTS, name, "index.html"))), reverse=True)
    rows = "".join(f"<tr><td><a href=\"{name}/index.html\">{html.escape(name)}</a></td></tr>" for name in folders)
    older = ("<p class=\"muted\">The first report, made before reports had their own folders, is "
             "<a href=\"index.html\">here</a>.</p>") if os.path.isfile(os.path.join(REPORTS, "index.html")) else ""
    page = (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
            f"content=\"width=device-width, initial-scale=1\"><title>LeapQuant reports</title><style>{STYLE}</style></head>"
            f"<body><main><h1>LeapQuant reports</h1><p class=\"lede\">One folder per run and time, newest first.</p>"
            f"<table><thead><tr><th>Report (run, date and time)</th></tr></thead><tbody>{rows}</tbody></table>{older}"
            f"</main></body></html>")
    with open(os.path.join(REPORTS, "all_reports.html"), "w", encoding="utf-8") as handle:
        handle.write(page)


def main():
    global HERE, IMG
    import argparse
    parser = argparse.ArgumentParser(description="HTML reports from the results of a run.")
    parser.add_argument("results", help="results folder of the run")
    parser.add_argument("output", nargs="?", help="output folder (default: a new reports/<run>_<date>-<time>/)")
    parser.add_argument("--with", dest="extra", nargs="+", default=[], metavar="FOLDER",
                        help="more results folders to draw on (kernel checks, accuracy runs, replays); the run's own "
                             "folder comes first")
    args = parser.parse_args()
    results = os.path.abspath(args.results)
    SOURCES[:] = [results] + [os.path.abspath(folder) for folder in args.extra if os.path.isdir(folder)]
    run_name = os.path.basename(results.rstrip("/"))
    HERE = os.path.abspath(args.output) if args.output else \
        os.path.join(REPORTS, f"{run_name}_{datetime.now().strftime('%Y%m%d-%H%M')}")
    if os.path.exists(os.path.join(HERE, "index.html")):
        sys.exit(f"error: {HERE} already holds a report; it will not be overwritten")
    IMG = os.path.join(HERE, "img")
    os.makedirs(IMG, exist_ok=True)
    parts = {
        "kernel": page_kernel(results, run_name),
        "accuracy": page_accuracy(results, run_name),
        "tasks": page_tasks(results, run_name),
        "verification": page_verification(results, run_name),
        "tuning": page_tuning(results, run_name),
        "profile": page_profile(results, run_name),
        "serving": page_serving(results, run_name),
    }
    page_index(results, run_name, parts)
    write_pages()
    write_list_of_reports()
    written = [name for name in os.listdir(HERE) if name.endswith(".html")]
    print(f"wrote {len(written)} pages ({', '.join(sorted(written))}) and "
          f"{len(glob.glob(os.path.join(IMG, '*.png')))} charts to {HERE}")


if __name__ == "__main__":
    main()
