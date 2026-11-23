"""The LeapQuant status page as a local file (reports are written as local files).

    .venv/bin/python reports/build_status_page.py

Reads the synced results/ folders (copy them from the GPU machine first; results/jobs/LOG.txt is a copy of the host's
~/lqexp/jobs/LOG.txt for the schedule table), writes reports/status_<UTC date>-<time>/index.html, a new folder every
build (an earlier page is never overwritten), and refreshes the list reports/all_reports.html."""
import glob
import html
import json
import importlib.util
import math
import os
import re
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, "reports", "status_page_template.html")
BUILT = time.gmtime()
OUT = os.path.join(REPO, "reports", f"status_{time.strftime('%Y%m%d-%H%M', BUILT)}", "index.html")
if os.path.exists(OUT):
    sys.exit(f"error: {OUT} already exists; a status page is never overwritten (wait a minute and build again)")
R = lambda *p: os.path.join(REPO, "results", *p)

spec = importlib.util.spec_from_file_location("gr", os.path.join(REPO, "reports", "generate_reports.py"))
gr = importlib.util.module_from_spec(spec)
sys.argv = ["x"]
spec.loader.exec_module(gr)


def read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def table(headers, rows, numeric_from=1):
    head = "".join(f"<th class=\"{'num' if i >= numeric_from else ''}\">{html.escape(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f"<td class=\"{'num' if i >= numeric_from else ''}\">{c}</td>" for i, c in enumerate(row)) + "</tr>"
                   for row in rows)
    return f"<div class=\"scroll\"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


# ---- kernel numbers (batch 512) ----------------------------------------------------------------------------------
def batch512(text):
    part = text.split("batch 512")[-1] if "batch 512" in text else ""
    out = {}
    m = re.search(r"fast decode step ([\d.]+) ms.*?fast fold ([\d.]+) ms\s+->\s+([\d.]+) ms per token", part)
    if m:
        out["decode"], out["fold"], out["token"] = map(float, m.groups())
    for key, pattern in (("fp32_packed", r"sglang packed kernel, fp32 state: ([\d.]+)"),
                         ("fp32_window", r"sglang window kernel, fp32 state: ([\d.]+)"),
                         ("bf16_window", r"sglang window kernel, bf16 state: ([\d.]+)"),
                         ("bf16_packed", r"sglang packed kernel, bf16 state: ([\d.]+)")):
        m = re.search(pattern, part)
        if m:
            out[key] = float(m.group(1))
    m = re.search(r"fast store \(adopt\) of every row ([\d.]+) ms", part)
    if m:
        out["store"] = float(m.group(1))
    return out


def mean_of(paths, key):
    values = [batch512(read(p)).get(key) for p in paths]
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


base = batch512(read(R("fast_checks_final", "F04.log")))
triton_runs = glob.glob(R("fast_fold_v1", "F3_triton_*.log"))
cuda_runs = glob.glob(R("fast_fold_v1", "F3_cuda_*.log"))
rows_k = [("SGLang packed, fp32 state (default)", base.get("fp32_packed"), "base"),
          ("SGLang window, fp32 state", base.get("fp32_window"), "base"),
          ("SGLang packed, bf16 state", base.get("bf16_packed"), "base"),
          ("SGLang window, bf16 state (the bar)", base.get("bf16_window"), "base"),
          ("LeapQuant, Triton fold", mean_of(triton_runs, "token") or base.get("token"), "lq2"),
          ("LeapQuant, CUDA fold", mean_of(cuda_runs, "token"), "lq")]
extra = []
for label, folder, pattern in (("LeapQuant, CUDA fold v3 (prefetch)", "fast_fold_v3", "bench_v3_x0_*.txt"),
                               ("LeapQuant, CUDA fold + store (v6)", "fast_fold_v6", "bench_cuda_*.txt"),
                               ("LeapQuant, CUDA decode v5 (not adopted)", "fast_decode_v5", "bench_decode_cuda_*.txt"),
                               ("LeapQuant, CUDA decode v7 (not adopted)", "fast_decode_v7", "bench_cuda_s2_*.txt"),
                               ("LeapQuant, Triton decode, batched-GS fold", "fast_decode_v8", "bench_triton_*.txt"),
                               ("LeapQuant, CUDA decode v8, batched-GS fold", "fast_decode_v8", "bench_v8_*.txt"),
                               ("LeapQuant, CUDA decode v9, batched-GS fold", "g3", "v9_*.txt")):
    value = mean_of(glob.glob(R(folder, pattern)), "token")
    if value:
        extra.append((label, value, "lq"))
rows_k += extra
rows_k = [r for r in rows_k if r[1]]
bf16 = base.get("bf16_window")
fp32 = base.get("fp32_packed")
best_lq = min((v for label, v, s in rows_k if s == "lq"), default=None)

# SVG bar chart, to scale
W, row_h, left, right = 760, 30, 300, 70
scale_max = max(v for _, v, _ in rows_k) * 1.08
plot_w = W - left - right
H = row_h * len(rows_k) + 40
bars = []
for i, (label, value, style) in enumerate(rows_k):
    y = 10 + i * row_h
    w = plot_w * value / scale_max
    color = {"base": "var(--bar-base)", "lq": "var(--bar-lq)", "lq2": "var(--bar-lq2)"}[style]
    bars.append(f"<text x=\"{left - 10}\" y=\"{y + 17}\" text-anchor=\"end\" font-size=\"12.5\">{html.escape(label)}</text>"
                f"<rect x=\"{left}\" y=\"{y + 4}\" width=\"{w:.1f}\" height=\"{row_h - 10}\" rx=\"3\" fill=\"{color}\"></rect>"
                f"<text x=\"{left + w + 6:.1f}\" y=\"{y + 17}\" font-size=\"12\" class=\"num\">{value:.3f}</text>")
ticks = []
for t in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
    if t > scale_max:
        break
    x = left + plot_w * t / scale_max
    ticks.append(f"<line x1=\"{x:.1f}\" x2=\"{x:.1f}\" y1=\"6\" y2=\"{H - 26}\" stroke=\"var(--line)\" stroke-width=\"1\"></line>"
                 f"<text x=\"{x:.1f}\" y=\"{H - 10}\" text-anchor=\"middle\" class=\"tick\">{t:g}</text>")
chart = (f"<figure><div class=\"scroll\" style=\"border:0;background:transparent\"><svg viewBox=\"0 0 {W} {H}\" width=\"{W}\" "
         f"style=\"max-width:100%;height:auto\" role=\"img\" aria-label=\"Milliseconds per token at batch 512\">"
         + "".join(ticks) + "".join(bars) + "</svg></div><figcaption>Milliseconds per token, one layer, batch 512; "
         "shorter is faster. Grey: SGLang's kernels. Teal: LeapQuant.</figcaption></figure>")
krows = [[html.escape(label), f"{v:.3f}", f"{fp32 / v:.2f}×" if fp32 else "-", f"{bf16 / v:.2f}×" if bf16 else "-"]
         for label, v, _ in rows_k]
kernel_table = table(["Kernel", "ms per token", "against fp32 packed", "against bf16 window"], krows)

# ---- complete kernel cycle (test harness, job 110: results/harness/kernel-cycle) --------------------------------------
cycle = {}
try:
    with open(R("harness", "kernel-cycle", "kernel_cycle.json"), encoding="utf-8") as handle:
        cycle = json.load(handle)
except (OSError, ValueError):
    pass
cycle_html, cycle_512, cycle_512_fp32 = "", None, None
if cycle.get("batches"):
    crows = []
    for batch, b in sorted(cycle["batches"].items(), key=lambda kv: int(kv[0])):
        vb, vf = b.get("speed_vs_bf16_window", {}), b.get("speed_vs_fp32_window", {})
        crows.append([batch, f"{statistics.median(b['lq_ms_per_token']):.4f}",
                      f"{statistics.median(b['bf16_window_ms_per_token']):.4f}",
                      f"{statistics.median(b['fp32_window_ms_per_token']):.4f}",
                      f"{vb['value']:.3f}× ({vb['low']:.3f} .. {vb['high']:.3f})",
                      f"{vf['value']:.3f}× ({vf['low']:.3f} .. {vf['high']:.3f})",
                      "yes" if b.get("state_advanced_every_rep") else "<b>no</b>"])
        if str(batch) == "512":
            cycle_512, cycle_512_fp32 = vb, vf
    reps = max(int(b.get("repetitions", 0)) for b in cycle["batches"].values())
    cycle_html = (f"<h3>Complete cycle (test harness, {cycle.get('computed_utc', '')[:16].replace('T', ' ')} UTC)</h3>"
                  "<p>One CUDA graph per window of 16 tokens: 16 state-advancing decode steps and the fold, the start "
                  "state restored outside the timing; SGLang's window kernels timed the same way (16 steps, the last "
                  f"writes the state). {reps} repetitions, alternately LeapQuant first and SGLang first; the ratio is the "
                  "geometric mean of the per-repetition ratios with a 95% t interval on their logarithms. Milliseconds per "
                  "token = per step of the whole batch, one layer.</p>"
                  + table(["Batch", "LeapQuant ms", "bf16 window ms", "fp32 window ms", "against bf16 window (95%)",
                           "against fp32 window (95%)", "State advanced"], crows)
                  + "<p>LeapQuant is behind SGLang's bf16 window kernel at batches 1 and 8 and ahead from 32 up. Gate G3 "
                    "(1.25× at batch 512) is met by the complete cycle, which replaces the component sum below as its "
                    "evidence (results/harness/kernel-cycle).</p>")
# Correctness of the final kernels (test harness: results/harness/lifecycle and results/harness/sanitizers).
checks_html, parts = "", []
try:
    with open(R("harness", "lifecycle", "lifecycle_results.json"), encoding="utf-8") as handle:
        life = json.load(handle).get("tests") or []
except (OSError, ValueError, AttributeError):
    life = []
try:
    with open(R("harness", "sanitizers", "sanitizer_results.json"), encoding="utf-8") as handle:
        san = json.load(handle).get("records") or []
except (OSError, ValueError, AttributeError):
    san = []
if life:
    names = ", ".join(t["test"].replace("_test", "").replace("_", " ") for t in life)
    parts.append(f"lifecycle: {sum(t.get('status') == 'PASS' for t in life)} of {len(life)} tests pass ({names})")
if san:
    others = [f"{os.path.basename(r['log']).rsplit('.', 1)[0].replace('_', ' ')} {r.get('classification')}"
              for r in san if r.get("classification") != "clean"]
    parts.append(f"compute-sanitizer: {sum(r.get('classification') == 'clean' for r in san)} of {len(san)} runs clean"
                 + (" (" + "; ".join(others) + ")" if others else ""))
if parts:
    checks_html = ("<p><b>Correctness of the final kernels</b> (test harness, 15 Nov): " + "; ".join(parts) + "."
                   + (" Interrupted means stopped by the 10-minute limit, neither a pass nor a fail; the decode's race "
                      "check is clean at batch 1 (today) and batch 4 (14 Nov)." if any(r.get("classification") == "interrupted"
                                                                                       for r in san) else "")
                   + "</p>")
chart = cycle_html + checks_html + "<h3>Component timings, batch 512 (how the final configuration was chosen)</h3>" + chart
v6_runs = glob.glob(R("fast_fold_v6", "bench_cuda_*.txt")) or cuda_runs
pieces = {"decode": mean_of(v6_runs, "decode"), "fold": mean_of(v6_runs, "fold"), "store": mean_of(v6_runs, "store")}
kernel_table += (f"<p>LeapQuant's pieces at batch 512 (Triton decode, CUDA fold and store, v6): decode "
                 f"{pieces['decode']:.3f} ms per step, fold {pieces['fold']:.2f} ms every 16 tokens, store "
                 f"{pieces['store']:.2f} ms per prompt end. Gate G3 asks at most 0.306 ms per token (1.25× bf16 window): "
                 f"the fold at about 0.85 ms or the decode at about 0.239 ms. The paper reports about 1.47× bf16 window "
                 f"on B200.</p>") if pieces["decode"] else ""

# ---- serving (round 9, paired) -------------------------------------------------------------------------------------
gr.SOURCES[:] = [R("round9")]
runs = gr.parse_serving(R("round9"))
serving = ""
for mode, title in (("off", "Radix cache off: both 560 state slots, up to 512 running"),
                    ("on", "Radix cache on: as many slots as each fits (bf16 1,100: 366 running; LeapQuant 1,560: 512)")):
    keys = sorted({(s, c) for (cfg, s, c) in runs if cfg.startswith(mode + "_")}, key=lambda k: (k[1], k[0]))
    rows = []
    for shape, conc in keys:
        def avg(arm, metric):
            vals = [m.get(metric) for (cfg, s, c), m in runs.items()
                    if s == shape and c == conc and cfg.startswith(f"{mode}_{arm}_r") and m.get(metric)]
            return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)
        b, nb = avg("bf16_window", "Output token throughput (tok/s)")
        l, nl = avg("leapquant_cuda", "Output token throughput (tok/s)")
        if not b or not l:
            continue
        bi, _ = avg("bf16_window", "Median ITL (ms)")
        li, _ = avg("leapquant_cuda", "Median ITL (ms)")
        bp, _ = avg("bf16_window", "P99 ITL (ms)")
        lp, _ = avg("leapquant_cuda", "P99 ITL (ms)")
        bt, _ = avg("bf16_window", "Median TTFT (ms)")
        lt, _ = avg("leapquant_cuda", "Median TTFT (ms)")
        ratio = l / b
        cls = "pass" if ratio >= 1.02 else ("fail" if ratio < 0.98 else "open")
        rows.append([html.escape(gr.SHAPE_TEXT.get(shape, shape)), conc, f"{b:,.0f}", f"{l:,.0f}",
                     f"<span class=\"{cls}\">{ratio:.3f}</span>", f"{bi:.1f} / {li:.1f}", f"{bp:.0f} / {lp:.0f}",
                     f"{bt:,.0f} / {lt:,.0f}", f"{min(nb, nl)}"])
    if rows:
        serving += f"<h3>{html.escape(title)}</h3>" + table(
            ["Shape", "Users", "bf16 window tok/s", "LeapQuant tok/s", "Ratio", "Median ITL, ms", "p99 ITL, ms",
             "Median TTFT, ms", "Reps"], rows, numeric_from=1)
        if mode == "on":
            serving += ("<p>With the radix cache on, the slot counts were not set to equal memory: LeapQuant's 1,560 "
                        "slots take about 33 GB of state and leave it less KV cache than bf16 window, which costs it most "
                        "on the long outputs (128 in / 2,048 out), even though its decode step there is faster. A run with "
                        "slot counts set to equal bytes is queued.</p>")
serving += "<p class=\"src\">results/round9 (runbook/serving_round9.sh)</p>"

# ---- round 10: public-benchmark sweeps (bf16 window, LeapQuant, fp32 default) -------------------------------------
if os.path.isdir(R("round10")):
    gr.SOURCES[:] = [R("round10")]
    r10 = gr.parse_serving(R("round10"))
    rows = []
    for shape in ("aa1k", "ix8k1k"):
        for conc in sorted({c for (cfg, s, c) in r10 if s == shape}):
            get = lambda arm: r10.get((f"off_{arm}_r1", shape, conc), {})
            b, l, f = get("bf16_window"), get("leapquant_cuda"), get("fp32_packed")
            tb, tl, tf = (m.get("Output token throughput (tok/s)") for m in (b, l, f))
            if not tl:
                continue
            speed = lambda m: 1000.0 / m["Median TPOT (ms)"] if m.get("Median TPOT (ms)") else None
            rows.append([html.escape(gr.SHAPE_TEXT.get(shape, shape)), conc,
                         f"{tf:,.0f}" if tf else "-", f"{tb:,.0f}" if tb else "-", f"{tl:,.0f}",
                         f"{tl / tb:.3f}" if tb else "-", f"{tl / tf:.3f}" if tf else "-",
                         f"{speed(l):.1f}" if speed(l) else "-"])
    if rows:
        serving += ("<h3>Public-benchmark shapes (round 10): Artificial Analysis 1k/1k and InferenceX 8k/1k</h3>"
                    "<p>Concurrency sweeps, radix cache off, one run per point. fp32 packed is SGLang's default state and "
                    "the baseline the paper compares against.</p>"
                    + table(["Shape", "Users", "fp32 tok/s", "bf16 window tok/s", "LeapQuant tok/s", "vs bf16 window",
                             "vs fp32", "LeapQuant tok/s per user"], rows, numeric_from=1)
                    + "<p class=\"src\">results/round10 (runbook/serving_round10.sh)</p>")

# ---- round 12: the decode-heavy primary cell, final configuration, against bf16 window and fp32 ---------------------
r12 = {}
if os.path.isdir(R("round12")):
    gr.SOURCES[:] = [R("round12")]
    r12 = gr.parse_serving(R("round12"))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in r12}, key=lambda k: (k[0], k[1])):
        g12 = lambda arm: r12.get((f"off_{arm}_r1", s, c), {})
        b, l, f = g12("bf16_window"), g12("leapquant_cuda"), g12("fp32_packed")
        tb, tl, tf = (m.get("Output token throughput (tok/s)") for m in (b, l, f))
        if not (tb and tl and tf):
            continue
        second = {}
        if os.path.isdir(R("round12b")):
            gr.SOURCES[:] = [R("round12b")]
            r12b = gr.parse_serving(R("round12b"))
            second = {arm: r12b.get((f"off_{arm}_r1", s, c), {}).get("Output token throughput (tok/s)")
                      for arm in ("bf16_window", "leapquant_cuda", "fp32_packed")}
        both_f = [tl / tf] + ([second["leapquant_cuda"] / second["fp32_packed"]] if second.get("fp32_packed") and second.get("leapquant_cuda") else [])
        both_b = [tl / tb] + ([second["leapquant_cuda"] / second["bf16_window"]] if second.get("bf16_window") and second.get("leapquant_cuda") else [])
        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tf:,.0f}", f"{tb:,.0f}", f"{tl:,.0f}",
                     " / ".join(f"{x:.3f}" for x in both_b), "<b>" + " / ".join(f"{x:.3f}" for x in both_f) + "</b>",
                     f"{f.get('Median TPOT (ms)', 0):.1f} / {b.get('Median TPOT (ms)', 0):.1f} / {l.get('Median TPOT (ms)', 0):.1f}"])
    if rows:
        serving += ("<h3>Round 12: the decode-heavy primary cell (128 in / 512 out), final configuration</h3>"
                    "<p>Against fp32 packed (SGLang's default state, the paper's baseline) and bf16 window in one session; "
                    "radix cache off, 560 slots each; ratios of two independent sessions (round12, round12b), throughput of the first. The paper reports 1.47-1.57× end to end against "
                    "fp32 (B200).</p>"
                    + table(["Shape", "Users", "fp32 tok/s", "bf16 window tok/s", "LeapQuant tok/s", "vs bf16 window",
                             "vs fp32", "Median TPOT, ms (fp32 / bf16 / LQ)"], rows, numeric_from=1)
                    + "<p class=\"src\">results/round12 (runbook/serving_round10.sh, decode points)</p>")

# ---- round 13: the capacity regime (each configuration with the slots it fits) --------------------------------------
r13 = {}
if os.path.isdir(R("round13")):
    gr.SOURCES[:] = [R("round13")]
    r13 = gr.parse_serving(R("round13"))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in r13}, key=lambda k: (k[0], k[1])):
        g13 = lambda arm: r13.get((f"off_{arm}_r1", s, c), {})
        b, l, f = g13("bf16_window"), g13("leapquant_cuda"), g13("fp32_packed")
        tb, tl, tf = (m.get("Output token throughput (tok/s)") for m in (b, l, f))
        if not (tb and tl and tf):
            continue
        z = {}
        if os.path.isdir(R("round13-stagger")):
            gr.SOURCES[:] = [R("round13-stagger")]
            z = gr.parse_serving(R("round13-stagger")).get(("off_leapquant_cuda_r1", s, c), {})
        tz = z.get("Output token throughput (tok/s)")
        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tf:,.0f}", f"{tb:,.0f}",
                     f"{tl:,.0f} ({l.get('P99 ITL (ms)', 0):.0f} ms)",
                     f"{tz:,.0f} ({z.get('P99 ITL (ms)', 0):.0f} ms)" if tz else "-",
                     f"{tl / tb:.3f}" + (f" / {tz / tb:.3f}" if tz else ""),
                     f"<b>{tl / tf:.3f}</b>" + (f" / <b>{tz / tf:.3f}</b>" if tz else ""),
                     f"{f.get('Median TTFT (ms)', 0) / 1000:,.1f} / {b.get('Median TTFT (ms)', 0) / 1000:,.1f} / {l.get('Median TTFT (ms)', 0) / 1000:,.1f}"])
    if rows:
        serving += ("<h3>Round 13: the capacity regime</h3><p>The decode-heavy cell at 512 and 1,024 users, each configuration "
                    "with the state slots it fits on the 80 GB card (docs/DECISIONS.md): fp32 560 slots (50 MB each), bf16 "
                    "window and LeapQuant 1,100 (25 / 21 MB); up to 1,024 running, CUDA graphs to 1,024, radix cache off, "
                    "one run per point.</p>"
                    + table(["Shape", "Users", "fp32 tok/s", "bf16 window tok/s", "LeapQuant aligned (p99 ITL)",
                             "LeapQuant staggered (p99 ITL)", "vs bf16 (aligned / staggered)", "vs fp32 (aligned / staggered)",
                             "Median TTFT, s (fp32 / bf16 / LQ)"], rows, numeric_from=1)
                    + "<p>With staggered folds LeapQuant stays within the 100 ms p99 SLO at 1,024 users and serves 1.52× fp32. "
                       "A second session (round13b) gave 1.533× fp32 and 1.056× bf16 window at 1,024 users (first: 1.534×, "
                       "1.063×); its 512-user point was slow for LeapQuant (0.983× bf16 window, steps 8% slower in that one run; "
                       "six other sessions of this cell at 512 users gave 1.039-1.049×).</p>"
                    + "<p class=\"src\">results/round13, round13-stagger (runbook/serving_round13.sh)</p>")

# ---- fold schedule: aligned against staggered (the tail between tokens) ---------------------------------------------
if os.path.isdir(R("ab-stagger")):
    gr.SOURCES[:] = [R("ab-stagger")]
    st = gr.parse_serving(R("ab-stagger"))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in st}):
        g = lambda arm: st.get((f"off_{arm}_r1", s, c), {})
        b, d, z = g("bf16_window"), g("lq_policy_default"), g("lq_policy_stagger")
        tb, td, tz = (m.get("Output token throughput (tok/s)") for m in (b, d, z))
        if not (tb and td and tz):
            continue
        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tb:,.0f}", f"{td:,.0f}", f"{tz:,.0f}",
                     f"{td / tb:.3f} / {tz / tb:.3f}",
                     f"{b.get('P99 ITL (ms)', 0):.0f} / {d.get('P99 ITL (ms)', 0):.0f} / <b>{z.get('P99 ITL (ms)', 0):.0f}</b>"])
    if rows:
        serving += ("<h3>Fold schedule: aligned or staggered (the tail between tokens)</h3><p>Aligned folds (the default): "
                    "every request folds on the same step, every 16 steps: that step is the p99 between tokens. Staggered "
                    "(LEAPQUANT_FOLD_POLICY=stagger): each request folds on its own phase, about 1/16 of them per step. "
                    "Radix cache off, 512 users, one session.</p>"
                    + table(["Shape", "Users", "bf16 window tok/s", "LQ aligned tok/s", "LQ staggered tok/s",
                             "vs bf16 (aligned / staggered)", "p99 ITL, ms (bf16 / aligned / staggered)"], rows, numeric_from=1)
                    + "<p>Staggered folds bring LeapQuant's tail to bf16 window's level for 1-2% of throughput; it stays "
                      "3-4% ahead of bf16 window. Aligned stays the default (a numerical-policy item); staggered is the "
                      "choice for latency-bound serving.</p><p class=\"src\">results/ab-stagger (runbook/serving_ab_policy.sh)</p>")

# ---- radix cache on at equal state memory --------------------------------------------------------------------------
_eq_dir = "radix-equalmem-final" if os.path.isdir(R("radix-equalmem-final")) else "round9-equalmem"
if os.path.isdir(R(_eq_dir)):
    gr.SOURCES[:] = [R(_eq_dir)]
    eq = gr.parse_serving(R(_eq_dir))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in eq}, key=lambda k: (k[1], k[0])):
        b, l = eq.get(("on_bf16_window_r1", s, c), {}), eq.get(("on_leapquant_cuda_r1", s, c), {})
        tb, tl = b.get("Output token throughput (tok/s)"), l.get("Output token throughput (tok/s)")
        if not (tb and tl):
            continue
        ratio = tl / tb
        cls = "pass" if ratio >= 1.02 else ("fail" if ratio < 0.98 else "open")
        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tb:,.0f}", f"{tl:,.0f}",
                     f"<span class=\"{cls}\">{ratio:.3f}</span>",
                     f"{b.get('Median TPOT (ms)', 0):.1f} / {l.get('Median TPOT (ms)', 0):.1f}"])
    if rows:
        serving += ("<h3>Radix cache on, equal state memory</h3><p>bf16 window 1,100 slots (25.8 GB, at most 366 running) "
                    "against LeapQuant 1,310 slots (26.0 GB, at most 436 running). LeapQuant runs more requests, but at "
                    "these batch sizes a decode step is bound by the model's matrix products, so the extra requests add "
                    "little; and its prefill was about 20% slower than bf16 window's with the radix cache on, fixed since (B56, below). "
                    "Final configuration with folds on each request's own step, one session (results/radix-equalmem-final; "
                    "results/ab-trackflush repeats it within 0.004 per cell); this morning's main configuration was 1-5 "
                    "points lower.</p>"
                    + table(["Shape", "Users", "bf16 window tok/s", "LeapQuant tok/s", "Ratio",
                             "Median TPOT, ms (bf16 / LQ)"], rows, numeric_from=1)
                    + f"<p class=\"src\">results/{_eq_dir} (runbook/serving_round9.sh, LQ_SLOTS=1310)</p>")

# ---- radix cache on without the track-boundary folds (backlog B55) -------------------------------------------------
if os.path.isdir(R("ab-trackflush")):
    gr.SOURCES[:] = [R("ab-trackflush")]
    tf = gr.parse_serving(R("ab-trackflush"))
    tf_status = read(R("ab-trackflush", "STATUS.txt"))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in tf}, key=lambda k: (k[1], k[0])):
        g = lambda arm: tf.get((f"on_{arm}_r1", s, c), {})
        b, a, n, z = g("bf16_window"), g("lq_auto"), g("lq_aligned"), g("lq_stagger")
        tb = b.get("Output token throughput (tok/s)")
        if not tb:
            continue

        def cell(m):
            t = m.get("Output token throughput (tok/s)")
            if not t:
                return "-"
            ratio = t / tb
            cls = "pass" if ratio >= 1.02 else ("fail" if ratio < 0.98 else "open")
            return f"{t:,.0f} (<span class=\"{cls}\">{ratio:.3f}</span>)"

        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tb:,.0f}", cell(a), cell(n), cell(z),
                     " / ".join(f"{m.get('P99 ITL (ms)', 0):.0f}" if m else "-" for m in (b, a, n, z))])
    checks = []
    for arm, text in (("auto", "auto"), ("aligned", "aligned, no track folds"), ("stagger", "staggered, no track folds")):
        m = re.search(rf"^K_{arm} (PASS|FAIL)", tf_status, re.M)
        if m:
            checks.append(f"{text}: <span class=\"{'pass' if m.group(1) == 'PASS' else 'fail'}\">{m.group(1)}</span>")
    if rows:
        serving += ("<h3>Radix cache on without the track-boundary folds (B55)</h3><p>SGLang folds a request's window "
                    "whenever it reaches a radix track boundary (every 256 tokens), so that a snapshot there is current. "
                    "ReplaySSM, and so LeapQuant, runs only with the no_buffer radix strategy, and there nothing reads such a "
                    "snapshot: a finished request is cached up to its last fold. LEAPQUANT_TRACK_FLUSH=0 drops those folds. "
                    "Equal state memory (bf16 window 1,100 slots, LeapQuant 1,310), one session; LeapQuant tok/s with the "
                    "ratio to bf16 window.</p>"
                    + table(["Shape", "Users", "bf16 window tok/s", "LQ auto (track folds kept)",
                             "LQ aligned, no track folds", "LQ staggered, no track folds",
                             "p99 ITL, ms (bf16 / auto / aligned / staggered)"], rows, numeric_from=1)
                    + "<p><b>No effect</b>: without the track folds LeapQuant's ratios move by at most 0.004. The track "
                      "folds are not the radix-on gap. What is: LeapQuant's prefill. Per 8,192-token prefill batch it "
                      "makes 17.3k prompt tokens/s against bf16 window's 21.0k with the radix cache on (level with it off: "
                      "23-25k both), and its time to first token at 128 users is 724 against 563 ms (off: 533 against 532). "
                      "Decode steps at equal batch size are level or faster. Found and fixed: B56, below. Staggered folds bring "
                      "the decode shapes' p99 between tokens to bf16 window's level or below for 1-2% throughput.</p>"
                    + (f"<p>Two-turn radix check (second turn from a radix-cached decode state, against bf16 window's "
                       f"answers): {'; '.join(checks)}.</p>" if checks else "")
                    + "<p class=\"src\">results/ab-trackflush (runbook/serving_ab_trackflush.sh)</p>")

# ---- radix cache on: the slot copy at the end of each prefill (backlog B56) -----------------------------------------
if os.path.isdir(R("ab-slotcopy")):
    gr.SOURCES[:] = [R("ab-slotcopy")]
    sc = gr.parse_serving(R("ab-slotcopy"))
    sc_status = read(R("ab-slotcopy", "STATUS.txt"))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in sc}, key=lambda k: (k[1], k[0])):
        g = lambda arm: sc.get((f"on_{arm}_r1", s, c), {})
        b, f, u = g("bf16_window"), g("lq_field"), g("lq_fused")
        tb = b.get("Output token throughput (tok/s)")
        if not tb:
            continue

        def cell(m):
            t = m.get("Output token throughput (tok/s)")
            if not t:
                return "-"
            ratio = t / tb
            cls = "pass" if ratio >= 1.02 else ("fail" if ratio < 0.98 else "open")
            return f"{t:,.0f} (<span class=\"{cls}\">{ratio:.3f}</span>)"

        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tb:,.0f}", cell(f), cell(u),
                     " / ".join(f"{m.get('Median TTFT (ms)', 0) / 1000:.2f}" if m else "-" for m in (b, f, u))])
    bursts = []
    for arm, text in (("bf16_window", "bf16 window"), ("lq_field", "LeapQuant, copy field by field"),
                      ("lq_fused", "LeapQuant, fused copy")):
        m = re.search(r"median wall ([\d.]+) s", read(R("ab-slotcopy", f"burst_on_{arm}_r1.txt")))
        if m:
            bursts.append(f"{text} {float(m.group(1)):.2f} s")
    checks = []
    for arm, text in (("field", "copy field by field"), ("fused", "fused copy")):
        m = re.search(rf"^K_{arm} (PASS|FAIL)", sc_status, re.M)
        if m:
            checks.append(f"{text}: <span class=\"{'pass' if m.group(1) == 'PASS' else 'fail'}\">{m.group(1)}</span>")
    if rows:
        serving += ("<h3>Radix cache on: the state copy at the end of each prefill (B56)</h3><p>With the radix cache on, "
                    "SGLang copies every request's state slot into the prefix tree when its prefill ends. LeapQuant copied "
                    "it field by field: about 245 small kernels per request, 273 ms of GPU time in a burst of 256 prompts "
                    "(results/prefill-radix). Now one launch for all layers (LEAPQUANT_FUSED_SLOT_COPY=1, bit-identical). "
                    "Equal state memory (bf16 window 1,100 slots, LeapQuant 1,310), one session; LeapQuant tok/s with the "
                    "ratio to bf16 window.</p>"
                    + (f"<p>Prefill-only bursts (256 prompts × 128 tokens), median wall time: {'; '.join(bursts)}.</p>"
                       if bursts else "")
                    + table(["Shape", "Users", "bf16 window tok/s", "LQ, copy field by field", "LQ, fused copy",
                             "Median TTFT, s (bf16 / field / fused)"], rows, numeric_from=1)
                    + (f"<p>Two-turn radix check: {'; '.join(checks)}.</p>" if checks else "")
                    + "<p class=\"src\">results/ab-slotcopy (runbook/serving_ab_slotcopy.sh)</p>")

# ---- SGLang's own best state settings against LeapQuant (radix cache on) -----------------------------------------
if os.path.isdir(R("sglang-best-3")):
    gr.SOURCES[:] = [R("sglang-best-3")]
    sb = gr.parse_serving(R("sglang-best-3"))
    rows = []
    for (s, c) in sorted({(s, c) for (cfg, s, c) in sb}, key=lambda k: (k[1], k[0])):
        g, l = sb.get(("best_bf16_lazy_int8_packed", s, c), {}), sb.get(("best_leapquant_cuda", s, c), {})
        tg, tl = g.get("Output token throughput (tok/s)"), l.get("Output token throughput (tok/s)")
        if not (tg and tl):
            continue
        ratio = tl / tg
        cls = "pass" if ratio >= 1.02 else ("fail" if ratio < 0.98 else "open")
        rows.append([html.escape(gr.SHAPE_TEXT.get(s, s)), c, f"{tg:,.0f}", f"{tl:,.0f}",
                     f"<span class=\"{cls}\">{ratio:.3f}</span>",
                     f"{g.get('Median TTFT (ms)', 0) / 1000:,.1f} / {l.get('Median TTFT (ms)', 0) / 1000:,.1f}"])
    if rows:
        serving += ("<h3>SGLang's own best state settings, radix cache on</h3><p>SGLang: bf16 state, lazy extra buffers, int8 "
                    "checkpoints, its packed kernel (its window kernel is refused with lazy buffers); 600 active slots plus "
                    "1,200 int8 checkpoint slots, 29.7 GB, and SGLang then runs at most 150 requests. LeapQuant (final "
                    "configuration): 1,560 slots, 30.9 GB, up to 512 running.</p>"
                    + table(["Shape", "Users", "SGLang best tok/s", "LeapQuant tok/s", "Ratio",
                             "Median TTFT, s (SGLang / LQ)"], rows, numeric_from=1)
                    + "<p class=\"src\">results/sglang-best-3 (runbook/serving_sglang_best.sh)</p>")

# ---- round 11: the final configuration against bf16 window (fp32 from round 10) -----------------------------------
r11 = {}
if os.path.isdir(R("round11")):
    gr.SOURCES[:] = [R("round11")]
    r11 = gr.parse_serving(R("round11"))
    rows = []
    for shape in ("aa1k", "ix8k1k"):
        for conc in sorted({c for (cfg, s, c) in r11 if s == shape}):
            b = r11.get(("off_bf16_window_r1", shape, conc), {})
            l = r11.get(("off_leapquant_cuda_r1", shape, conc), {})
            tb, tl = b.get("Output token throughput (tok/s)"), l.get("Output token throughput (tok/s)")
            if not (tb and tl):
                continue
            b10 = r10.get(("off_bf16_window_r1", shape, conc), {}) if os.path.isdir(R("round10")) else {}
            l10 = r10.get(("off_leapquant_cuda_r1", shape, conc), {}) if os.path.isdir(R("round10")) else {}
            f10 = r10.get(("off_fp32_packed_r1", shape, conc), {}) if os.path.isdir(R("round10")) else {}
            t10 = (l10.get("Output token throughput (tok/s)") or 0) / (b10.get("Output token throughput (tok/s)") or 1)
            tf = f10.get("Output token throughput (tok/s)")
            ratio = tl / tb
            cls = "pass" if ratio >= 1.02 else ("fail" if ratio < 0.98 else "open")
            rows.append([html.escape(gr.SHAPE_TEXT.get(shape, shape)), conc, f"{tb:,.0f}", f"{tl:,.0f}",
                         f"<span class=\"{cls}\">{ratio:.3f}</span>", f"{t10:.3f}" if t10 else "-",
                         f"{tl / tf:.3f}" if tf else "-",
                         f"{b.get('Median TPOT (ms)', 0):.1f} / {l.get('Median TPOT (ms)', 0):.1f}",
                         f"{b.get('P99 ITL (ms)', 0):.0f} / {l.get('P99 ITL (ms)', 0):.0f}"])
    if rows:
        status11 = read(R("fast_decode_v9", "STATUS.txt"))
        m = re.search(r"round 11 decode: variant (\d+)", status11)
        serving += ("<h3>Round 11: the final configuration</h3><p>Batched-Gram-Schmidt CUDA fold, CUDA store, decode inputs "
                    f"built once per step (B51), CUDA decode v{m.group(1) if m else '8/9'}; against bf16 window in the same "
                    "session, radix cache off, 560 slots, one run per point. fp32 throughput is round 10's (same shapes and "
                    "slots).</p>"
                    + table(["Shape", "Users", "bf16 window tok/s", "LeapQuant tok/s", "vs bf16 window", "round 10: vs bf16",
                             "vs fp32 (round 10)", "Median TPOT, ms (bf16 / LQ)", "p99 ITL, ms (bf16 / LQ)"], rows,
                            numeric_from=1)
                    + "<p class=\"src\">results/round11 (runbook/serving_round10.sh from the experiment tree)</p>")

gate = None
gate_reps = {}
_ol_folders = [f for f in ("openloop-final", "openloop-final-2", "openloop-final-3") if os.path.isdir(R(f))]
if len(_ol_folders) >= 2:
    import statistics as _st
    for _rate in (9, 10):
        _gains = []
        for _f in _ol_folders:
            _b, _l = (read(R(_f, f"openloop_{_arm}_rate{_rate}.log")) for _arm in ("bf16_window", "leapquant_aligned"))
            _gb = re.search(r"Request goodput \(req/s\):\s+([\d.]+)", _b); _gl = re.search(r"Request goodput \(req/s\):\s+([\d.]+)", _l)
            if _gb and _gl and float(_gb.group(1)) > 0:
                _gains.append(float(_gl.group(1)) / float(_gb.group(1)) - 1)
        if len(_gains) >= 2:
            _m, _sd = _st.mean(_gains), _st.stdev(_gains)
            _half = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(_gains), 2.0) * _sd / len(_gains) ** 0.5
            gate_reps[_rate] = (_m, _m - _half, _m + _half)
# ---- open-loop goodput, final configuration -----------------------------------------------------------------------
if os.path.isdir(R("openloop-final")):
    gr.SOURCES[:] = [R("openloop-final")]
    olf = gr.parse_openloop()
    rows = []
    for rate in sorted({r for _, r in olf}):
        b, l = olf.get(("bf16_window", rate), {}), olf.get(("leapquant_aligned", rate), {})
        gb, gl = b.get("Request goodput (req/s)"), l.get("Request goodput (req/s)")
        if gb is None or gl is None:
            continue
        ratio = gl / gb if gb else float("nan")
        cls = "pass" if ratio >= 1.15 else ("open" if ratio >= 0.98 else "fail")
        rows.append([f"{rate:g}", f"{gb:.2f}", f"{gl:.2f}", f"<span class=\"{cls}\">{ratio:.2f}</span>",
                     f"{b.get('Median TPOT (ms)', 0):.1f} / {l.get('Median TPOT (ms)', 0):.1f}"])
        if gate is None and b.get("Median TPOT (ms)", 0) > 45:      # the first rate where bf16 window's goodput falls
            gate = (rate, ratio)
    if rows:
        serving += ("<h3>Open-loop goodput, final configuration</h3><p>Poisson arrivals, 1,000 in / 1,000 out, 600 requests "
                    "per rate; goodput counts requests within the SLO (first token within 2 s, at most 50 ms per token). bf16 "
                    "window in the same session.</p>"
                    + table(["Offered, req/s", "bf16 window goodput", "LeapQuant goodput", "Ratio",
                             "Median TPOT, ms (bf16 / LQ)"], rows, numeric_from=0)
                    + (f"<p>Three sessions at 8-10 req/s: goodput gain {100 * gate_reps[9][0]:+.0f}% at 9 req/s (95% "
                       f"{100 * gate_reps[9][1]:+.0f}..{100 * gate_reps[9][2]:+.0f}%) and {100 * gate_reps[10][0]:+.0f}% at 10 req/s (95% "
                       f"{100 * gate_reps[10][1]:+.0f}..{100 * gate_reps[10][2]:+.0f}%). The service gate (+15%, lower bound above +5%) is "
                       f"met at 10 req/s and not established at 9, where bf16 window's goodput starts to fall: the sheet's "
                       f"'capacity boundary' needs a precise rate. LeapQuant is faster per token at every rate.</p>" if gate_reps.get(9) and gate_reps.get(10) else "")
                    + "<p class=\"src\">results/openloop-final (runbook/serving_openloop.sh)</p>")

# ---- open-loop goodput (morning) -----------------------------------------------------------------------------------
if os.path.isdir(R("openloop")) and not os.path.isdir(R("openloop-final")):
    gr.SOURCES[:] = [R("openloop")]
    ol = gr.parse_openloop()
    rates = sorted({r for _, r in ol})
    rows = []
    for rate in rates:
        cells = [f"{rate:g}"]
        for arm in ("bf16_window", "leapquant_aligned", "leapquant_own_step"):
            m = ol.get((arm, rate), {})
            cells.append(f"{m['Request goodput (req/s)']:.2f}" if "Request goodput (req/s)" in m else "-")
        rows.append(cells)
    if rows:
        ol_data = dict(ol)                     # (arm, rate) -> the client's report

        def olv(arm, rate, key):
            return ol_data.get((arm, rate), {}).get(key)
        notes = []
        tp = [(r, olv("bf16_window", r, "Median TPOT (ms)"), olv("leapquant_aligned", r, "Median TPOT (ms)")) for r in rates]
        tp = [(r, b, l) for r, b, l in tp if b and l]
        if tp:
            below = [l / b - 1 for r, b, l in tp if b <= 50]
            if below:
                notes.append(f"Below saturation both meet the SLO on every request; LeapQuant's median time per token is "
                             f"{100 * min(below):+.0f}% to {100 * max(below):+.0f}% against bf16 window's (a fixed cost "
                             f"per decode step, backlog B51: a fix is being measured).")
        boundary = [r for r in rates if (olv("bf16_window", r, "Median TPOT (ms)") or 0) > 50]
        if boundary:
            r = boundary[0]
            gb, gl = olv("bf16_window", r, "Request goodput (req/s)"), olv("leapquant_aligned", r, "Request goodput (req/s)")
            if gb and gl:
                notes.append(f"At {r:g} req/s both cross the 50 ms limit: goodput {gl:.2f} (LeapQuant) against {gb:.2f} "
                             f"({100 * (gl / gb - 1):+.0f}%, one run). Service gate (+15% at the capacity boundary): "
                             f"{'met' if gl / gb >= 1.15 else 'not met'}. Folds on each request's own step: no gain.")
        serving += ("<h3>Open-loop goodput</h3><p>Poisson arrivals at a fixed rate, 1,000 in / 1,000 out; requests per "
                    "second that met the SLO (first token within 2 s, at most 50 ms per token).</p>"
                    + table(["Offered, req/s", "bf16 window", "LeapQuant, folds aligned", "LeapQuant, own-step folds"],
                            rows, numeric_from=1) + "".join(f"<p>{html.escape(n)}</p>" for n in notes)
                    + "<p class=\"src\">results/openloop (runbook/serving_openloop.sh)</p>")

# ---- paper-lane accuracy -----------------------------------------------------------------------------------------
paper_lane = ""
reference_rows = []
for line in read(R("accuracy-paper", "paper_fp32_window.txt")).splitlines():
    m = re.match(r"\s+(aime\w+|beyond_aime)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", line)
    if m:
        reference_rows.append([m.group(1), m.group(2), m.group(3), f"{m.group(4)}% ± {m.group(5)}", m.group(6),
                               m.group(8), f"{int(m.group(9)):,}"])
if reference_rows:
    paper_lane += ("<h3>Reference: fp32 window (done, 193 min)</h3>"
                   + table(["Task", "Answers", "Correct", "Accuracy", "No answer", "Cut at 81,920 tokens",
                            "Tokens per answer"], reference_rows, numeric_from=1))
# The strict lane re-scored under the fixed-budget rule (results/accuracy-strict/fixed_budget_20261115, 15 Nov 06:20 UTC):
# the bench's own tables there counted one capped LeapQuant answer as correct; these are the rule's numbers, computed by
# the bench's own compare() on the saved answers.
FIXED = R("accuracy-strict", "fixed_budget_20261115")
rescored, answer_tokens = {}, {}
for part in read(os.path.join(FIXED, "strict_lane_capped_is_wrong.txt")).split("\n=== ")[1:]:
    header, _, body = part.partition("\n")
    m = re.match(r"(\w+) against (\w+), capped = wrong", header)
    if m:
        rescored[(m.group(1), m.group(2))] = body
        continue
    m = re.match(r"(\w+) \(", header)
    if m:
        for line in body.splitlines():
            row = re.match(r"\s+(aime\w+)\s+\d+\s+\d+\s+[\d.]+%\s+[\d.]+%\s+\d+\s+\d+\s+\d+\s+(\d+)", line)
            if row:
                answer_tokens[(m.group(1), row.group(1))] = int(row.group(2))
strict_pairs = {"paper_strict_leapquant_vs_fp32.txt": ("leapquant", "fp32_window"),
                "paper_strict_bf16_window_vs_fp32.txt": ("bf16_window", "fp32_window"),
                "paper_strict_leapquant_vs_bf16_window.txt": ("leapquant", "bf16_window")}
losses = []
for line in read(os.path.join(FIXED, "bf16_window_losses_by_cap.txt")).splitlines():
    m = re.match(r"(\w+)\s+(\w+): lost\s+(\d+) \(arm capped in (\d+)\), won\s+(\d+) \(fp32 capped in (\d+)\); "
                 r"capped arm\s+(\d+), fp32\s+(\d+), both (\d+)", line)
    if m:
        losses.append(m.groups())
for folder, name, label in (("accuracy-strict", "paper_strict_leapquant_vs_fp32.txt", "LeapQuant against fp32 (retraction-free)"),
                            ("accuracy-strict", "paper_strict_bf16_window_vs_fp32.txt", "bf16 window against fp32 (retraction-free)"),
                            ("accuracy-strict", "paper_strict_leapquant_vs_bf16_window.txt", "LeapQuant against bf16 window (retraction-free)"),
                            ("accuracy-paper", "paper_bf16_window_vs_fp32.txt", "bf16 window against fp32"),
                            ("accuracy-paper", "paper_leapquant_vs_fp32.txt", "LeapQuant against fp32"),
                            ("accuracy-paper", "paper_leapquant_vs_bf16_window.txt", "LeapQuant against bf16 window")):
    text = read(R(folder, name))
    is_rescored = folder == "accuracy-strict" and strict_pairs.get(name) in rescored
    if is_rescored:
        text, label = rescored[strict_pairs[name]], label + ", capped answers counted wrong"
    if losses and name == "paper_bf16_window_vs_fp32.txt":
        # after the strict lane's tables: where the disagreements with fp32 come from
        lrows = [[arm.replace("_", " "), task, f"{lost} ({lost_capped} capped)", f"{won} ({won_capped} capped in fp32)",
                  f"{cap_arm} / {cap_fp32} / {both}",
                  f"{answer_tokens.get((arm, task), 0):,} / {answer_tokens.get(('fp32_window', task), 0):,}"]
                 for arm, task, lost, lost_capped, won, won_capped, cap_arm, cap_fp32, both in losses]
        bf = {task: (int(lost), int(lost_capped), int(cap_arm), int(cap_fp32)) for arm, task, lost, lost_capped, _, _, cap_arm, cap_fp32, _ in losses if arm == "bf16_window"}
        paper_lane += ("<h3>Where the strict lane's disagreements with fp32 come from</h3>"
                       + table(["Arm", "Task", "Lost against fp32", "Won against fp32", "Capped: arm / fp32 / both",
                                "Tokens per answer: arm / fp32"], lrows, numeric_from=2)
                       + (f"<p>Every AIME 2025 sample the bf16 window lost against fp32 hit the 81,920-token cap "
                          f"({bf['aime2025'][1]} of {bf['aime2025'][0]}; AIME 2024 {bf['aime2024'][1]} of {bf['aime2024'][0]}), "
                          f"and so did every sample LeapQuant lost. The bf16 window capped {bf['aime2025'][2]} of 120 AIME 2025 "
                          f"answers (fp32 {bf['aime2025'][3]}) and wrote longer answers: its deficit there consists entirely of "
                          "answers that ran out of room, none of them a finished wrong answer. Whether those answers finish, "
                          "and finish correctly, with more room is what the quality job's 131,072-token budget tests "
                          "(results/accuracy-strict/fixed_budget_20261115).</p>"
                          if "aime2025" in bf and "aime2024" in bf else ""))
        losses = []
    rows = []
    for line in text.splitlines():
        m = re.match(r"\s+(aime\w+|beyond_aime)\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+(\d+)\s+(\d+)\s+([+-][\d.]+)%\s+([+-][\d.]+ \.\. [+-][\d.]+)\s+(\w+)", line)
        if m:
            rows.append([m.group(1), m.group(2), m.group(3) + "%", m.group(4) + "%", m.group(7), m.group(8), m.group(9)])
    if rows:
        paper_lane += f"<h3>{label}</h3>" + table(["Task", "Samples", "Candidate", "Baseline", "Difference",
                                                  "95% interval (question-clustered)", "Verdict"], rows, numeric_from=1)
        # The fixed-budget rule (15 Nov 05:00 UTC): a capped answer is wrong, whatever it contains. The tables above are the
        # bench's own output; this note recomputes the candidate's accuracy from the raw answers under the rule.
        raw = R(folder, name.replace("_vs_fp32.txt", ".json").replace("_vs_bf16_window.txt", ".json"))
        if os.path.exists(raw) and name.endswith("_vs_fp32.txt") and not is_rescored:
            run = json.load(open(raw))
            fixes = []
            for task, items in run["tasks"].items():
                wrong_caps = sum(1 for i in items if i.get("truncated") and i["correct"])
                if wrong_caps:
                    n, c = len(items), sum(i["correct"] for i in items)
                    fixes.append(f"{task}: {c - wrong_caps}/{n} = {100 * (c - wrong_caps) / n:.2f}% (the table counts "
                                 f"{wrong_caps} capped answer{'s' if wrong_caps > 1 else ''} as correct: {100 * c / n:.2f}%)")
            if fixes:
                paper_lane += ("<p><b>Fixed-budget rule</b> (capped answers count as wrong, enforced from 15 Nov 05:00 UTC): "
                               + "; ".join(fixes) + ". The verdicts do not change (all INCONCLUSIVE).</p>")

# ---- accuracy (product lane) ---------------------------------------------------------------------------------------
accuracy = ("<p>Product lane (results/accuracy-v7): temperature 0, answers up to 8,192 tokens, 1,000 questions per task, "
            "paired against fp32, margin 1 point, Newcombe paired interval. The paper's own lane (thinking mode, temperature "
            "1.0, up to 81,920 tokens, 4 samples per question): see below.</p>"
            + table(["Task", "fp32", "LeapQuant", "Difference (95% interval)", "Verdict", "bf16 window"],
                    [["GSM8K", "94.00%", "93.50%", "-0.50 (-1.88 .. +0.86)", "<span class=\"open\">inconclusive</span>", "93.50%"],
                     ["MMLU-Pro", "65.60%", "67.00%", "+1.40 (-0.82 .. +3.62)", "<span class=\"pass\">noninferior</span>", "65.90%"],
                     ["AIME 2025 (8,192 tokens: 29 of 30 cut)", "3.33%", "6.67%", "+3.33 (-9.30 .. +17.35)",
                      "<span class=\"open\">inconclusive</span>", "6.67%"]], numeric_from=1)
            + "<p>fp32 against itself is inconclusive on GSM8K too: at 1,000 questions a 1-point margin cannot be decided "
              "there. On the state level, LeapQuant's error is far below bf16 window's on layers 0-13 and above it on "
              "layers 14 and 20-30 (results/layer_map).</p>")
if paper_lane:
    accuracy += ("<h3>The paper's lane (results/accuracy-paper, results/accuracy-strict)</h3><p>Thinking mode, "
                 "temperature 1.0, answers up to 81,920 tokens, 4 samples per question; the interval is clustered by "
                 "question. The bf16-window and LeapQuant arms run retraction-free (B58): in the reference arm the KV "
                 "cache ran full and SGLang retracted requests 516 times, and a retracted request's state is rebuilt "
                 "from scratch when it resumes, which would reset a compressed state's accumulated error. So those arms "
                 "hold a 24-slot state pool (1.39 million KV tokens instead of 0.50) and 16 requests at a time, which "
                 "fit at full length.</p>" + paper_lane)

decisions = ("<ul>"
             "<li><b>15 Nov, 06:20 UTC</b>: Asif accepted the <b>GPQA-Diamond terms</b>; the bench now loads all 198 "
             "questions on the host (the quality job's 30 have prompts of at most 911 tokens, inside its 1,536-token "
             "allowance), so job 118 runs all four of the paper's tasks. <b>Reports are local files</b> from now on: "
             "this page is reports/status_&lt;time&gt;/index.html, a new folder at every build.</li>"
             "<li><b>15 Nov, 04:50 UTC, Asif's decisions</b>:<b>one fit sweep</b> is the numerical policy (B52; fold 20% "
             "faster, nine traces pass) and <b>staggered folds are the default</b> (B53; the p99 between tokens at bf16 "
             "window's level for 1-2% throughput); the product candidate is <code>lq_final</code>. Quality answers get "
             "<b>131,072 tokens</b> (the paper's 81,920 cut 11-12% of answers), still retraction-free (B58). "
             "<b>$2.00 per hour</b> for the host: cost per million tokens is now computed. A <b>docker-sandboxed "
             "LiveCodeBench scorer</b> is approved and built. <b>All tests within 12 hours</b>, the public matrix "
             "before the remaining quality runs: the plan was cut to fit (see Next).</li>"
             "<li><b>15 Nov, 05:00 UTC: fixed-budget rule enforced</b>: one capped answer of the strict lane's LeapQuant "
             "arm had been scored correct by the fallback parser; capped answers now count as wrong everywhere "
             "(bench, decision, checker). LeapQuant AIME 2025: 87.5%, not 88.3%.</li>"
             "<li><b>15 Nov: historical claims re-graded by the harness rules</b>: the 1.271× kernel result is a "
             "component sum (decode + fold/16), order-dependent (1.244 with the slower fold timing), recorded under "
             "plan-5 numbering; the complete-cycle timing decides it. Serving ratios against fp32 are single runs, partly "
             "across sessions, on H100/SGLang: not comparable with the paper's B200/vLLM averages. 'Equal memory' "
             "was not equal (peaks 71.3-71.9 vs 67.0-67.5 GiB). The open-loop gate never applied the p99 token-gap "
             "limit and reused one arrival seed.</li>"
             "<li><b>CUDA fold adopted</b> for the fast path: 1.85× the Triton fold, same fold loss, all checks pass.</li>"
             "<li><b>CUDA store adopted</b> (v6): 1.7 against 3.4 ms at 512 rows; a shared-memory race found by the race "
             "checker in v4 was fixed before any serving use (v6: 0 hazards).</li>"
             "<li><b>CUDA decode v5 not adopted</b>: correct (9 traces pass) but 0.371 against Triton's 0.253 ms per step; "
             "a pipelined rewrite (v7) is being measured.</li>"
             "<li><b>Paper-lane accuracy moved to the end of the queue</b>: it takes hours; the serving runs go first.</li>"
             "<li><b>Fold with batched Gram-Schmidt adopted</b> (6-9% faster, identical fold loss, 9 traces pass); the main "
             "runs use it from the open-loop run on.</li>"
             "<li><b>CUDA decode v7 not adopted</b> (correct, slower: 0.334 against 0.253 ms); its profile showed ~6,300 "
             "instructions per head and block-wide barriers.</li>"
             "<li><b>CUDA decode v9 adopted for the final configuration</b> (v8's design, one warp per head, the int8 levels "
             "loaded straight into tensor-core fragments, plus 4 adjacent value rows per lane): 0.241 against Triton's "
             "0.253 ms at 512 rows; sanitizers clean; gate G3 met with it (1.271×, paired).</li>"
             "<li><b>Decode inputs built once per step adopted</b> (B51): the low-load serving gap against bf16 window "
             "shrank from 4.5-6% to 1.4-2%; LeapQuant ahead from 64 users up.</li>"
             "<li><b>Folds on each request's own step: no gain</b> in the open-loop run (radix cache off); aligned folds "
             "stay there. <b>With the radix cache on, own-step folds are 2-3% faster</b> (results/radix-align0): track-"
             "boundary folds desynchronise rows and aligned folds then pull everyone into early folds.</li>"
             "<li><b>SGLang's own best state settings</b> (bf16, lazy extra buffers, int8 checkpoints) run 150 requests in "
             "about the state memory where LeapQuant runs 512: LeapQuant +23-25% at 512 users, 0.95 at 128.</li>"
             "<li><b>Staggered folds</b> (opt-in): the fold-step tail gone (p99 between tokens at bf16 window's level) "
             "for 1-2% throughput. <b>Decode v10</b> (earlier loads of the next head) 2% slower: not used. The fold policy "
             "that drops track flushes with the radix cache off: no effect.</li>"
             "<li><b>Retraction under KV pressure: pass</b> (answers agree with fp32 window). <b>Exact kernel requalified</b> "
             "after the low-bit change: every fold bit-identical on the 9 traces.</li>"
             "<li><b>Layer switch fixed</b> (LeapQuant on some layers only, B30 / TQ.5): the integration sized SGLang's "
             "replay ring for one slot, and the window kernel of the other layers wrote past it. With the full ring, "
             "LeapQuant on layer 30 only and on layers 0-13 only both serve and agree with fp32 window as the all-layer "
             "server does; a compressed subset is refused as intended (results/layer-switch-2).</li>"
             "<li><b>Radix track-boundary folds dropped: no effect</b> (B55): with SGLang's no_buffer radix strategy "
             "nothing reads those snapshots, so LEAPQUANT_TRACK_FLUSH=0 removes them safely (two-turn radix check passes), "
             "but throughput moves by at most 0.004. The radix-on gap was LeapQuant's prefill there.</li>"
             "<li><b>Radix-on prefill fixed</b> (B56): SGLang copies every request's state slot into the prefix tree "
             "when its prefill ends, and LeapQuant copied it field by field (about 245 small kernels per request). Now "
             "one launch, bit-identical: a 256-prompt prefill burst 2.02 -> 1.61 s (bf16 window 1.64 s), radix-on "
             "LeapQuant 0.97-1.03× bf16 window at equal state memory (was 0.93-1.02), time to first token at 512 users "
             "below bf16 window's (results/ab-slotcopy).</li>"
             "<li><b>One sweep instead of two</b> (fold 20% faster, fold loss +0.66%; the 9 recorded traces still pass, "
             "C 0.861 against 0.840 with two; results/fold_sweeps1): adopted by Asif on 15 Nov.</li>"
             "<li><b>Rotation rejected</b> (state-cache plan, gate GR): a Hadamard rotation of keys and state was worse than "
             "none on 8 of 9 real traces.</li>"
             "<li><b>8-bit, rank 4 kept</b>: 8 compensators on the upper layers gave no task gain; 6- and 4-bit with the "
             "paper's rank cost more bytes than 8-bit here (steps per group, one byte per level).</li>"
             "<li><b>Rejected after measuring</b>: persistent decode, the fold without its lo products, a Cholesky QR in the "
             "fold, staggered folds in a benchmark where all requests start together.</li>"
             "<li><b>Verdicts tightened</b>: the paired interval no longer collapses to 0..0 on small samples (a false AIME "
             "pass is now inconclusive).</li>"
             "<li><b>Paper-lane accuracy made retraction-free</b> (B58, Asif's go-ahead 22:30 UTC): the fp32 reference "
             "was kept; the bf16-window and LeapQuant arms replaced by arms in which no request can be retracted "
             "(0 so far).</li>"
             "<li><b>Reports show every serving metric</b>: the per-run reports (reports/, 15 regenerated) now give, for "
             "every load point and server, throughput (output, total, input, requests, peak), speed per user, TTFT, "
             "TPOT, time between tokens and end-to-end latency (mean, median, p90, p95, p99, max), peak GPU memory, "
             "the client's settings, each server's pools, with charts. Before, runs from round 9 on showed four "
             "metrics, and A/B runs with other arm names showed dashes.</li></ul>")

# ---- the 12-hour schedule, from a copy of the host's job log (results/jobs/LOG.txt) ------------------------------------
# Planned minutes per job are the estimates behind the schedule of 15 Nov 04:47 UTC (640 minutes from 05:45).
PLAN = [("110", 25, "Complete state-advancing kernel cycle (16 steps + fold, one CUDA graph), one sweep, against SGLang's window kernels; 10 interleaved repetitions"),
        ("111", 20, "Lifecycle checks (restore on outputs, slot copy, fold/store, engine)"),
        ("112", 30, "compute-sanitizer at batches 1 and 512"),
        ("113", 15, "One real baseline/candidate pair on all three lanes"),
        ("114", 20, "GPU bytes, slots, running and waiting for 8 arms"),
        ("115", 30, "5-block pilot at 128/512, 512 users"),
        ("116", 105, "InferenceX fixed sequence (pinned client): 1k/1k at 1-512 users, 8k/1k at 4 and 64; lq_final, bf16 window, fp32"),
        ("117", 110, "AA-SLT method (1000/1000, 180 s steady): 1-1,024 users; same three arms"),
        ("118", 230, "Quality: fp32, bf16 window, LeapQuant on AIME 2026 (30), BeyondAIME (15), GPQA-Diamond (30), LiveCodeBench v6 (12); paper sampling, 131,072 tokens, one seed, 12 at a time, retraction-free"),
        ("119", 25, "Long-context NLL through the server (1k-32k)"),
        ("120", 30, "Compounding: 8 single layers, those 8 together, all 24"),
        ("121", 230, "Quality re-run of 118 (FlashInfer sampling kernel fixed; arms with failed requests now fail)"),
        ("122", 300, "Speed repeats: AA-SLT at 1, 64, 256, 512 users, three arms, 5 paired blocks"),
        ("123", 60, "Qwen3.8-27B first contact (B75): kernel agreement and memcheck at its head layout, one server per arm, balanced slot counts"),
        ("124", 180, "9B slot-tuned control (B77): each arm with slots balanced for 1k/1k (LeapQuant 1,008, bf16 window 927, fp32 609); AA-SLT 256/512, 5 blocks")]
# A job's exit code alone can mislead: these are the outcomes found by reading the results.
OUTCOME = {"118": "<b>failed</b>: every request failed (the server died compiling FlashInfer's sampling kernel); exit 0 hid it; re-run as 121",
           "120": "ran, but <b>measured nothing</b>: every delta exactly 0 (the per-layer mirror state is never read back); needs a redesign"}
LIMIT = 16 * 60 + 30                                  # Asif: all tests within 12 hours, by about 16:30 UTC
starts, ends = {}, {}
for line in read(R("jobs", "LOG.txt")).splitlines():
    m = re.match(r"(\d{3})[a-z]?_\S+ (start|end) (\d\d):(\d\d)(?: exit (\d+))?", line)
    if m:
        minute = int(m.group(3)) * 60 + int(m.group(4))
        if m.group(2) == "start":
            starts[m.group(1)] = minute
            ends.pop(m.group(1), None)                # a re-run replaces the earlier attempt
        else:
            ends[m.group(1)] = (minute, int(m.group(5) or 0))
clock = lambda minute: f"{(minute // 60) % 24:02d}:{minute % 60:02d}"
# The job log has times without dates: a job that starts after midnight UTC gets 24 h added, so it sorts after the
# evening's jobs (the schedule began at 05:45, so a start before 05:00 belongs to the next day).
for table_ in (starts, ends):
    for job in list(table_):
        value = table_[job]
        minute = value[0] if isinstance(value, tuple) else value
        if int(job) > 120 and minute < 5 * 60:
            table_[job] = (minute + 1440, value[1]) if isinstance(value, tuple) else minute + 1440
now = BUILT.tm_hour * 60 + BUILT.tm_min
if now < 5 * 60:
    now += 1440
cursor, srows = None, []
for job, minutes, what in PLAN:
    if job in ends:
        end, code = ends[job]
        status = OUTCOME.get(job) or ("done" if code == 0 else f"<b>failed (exit {code})</b>")
        when, cursor = f"{clock(starts.get(job, end))}-{clock(end)}", end
    elif job in starts:
        cursor = max(starts[job] + minutes, now)
        status, when = "running", f"{clock(starts[job])}, ends about {clock(cursor)}"
    else:
        begin = cursor if cursor is not None else now
        cursor = begin + minutes
        status, when = "queued", f"about {clock(begin)}"
    srows.append([job, status, when, what])
nxt = ("<p><b>Job queue</b>: the 12-hour schedule "
       "ran jobs 110-120 from 05:45 to 13:54 UTC; 121-124 were queued after it. Times are UTC, actual for jobs that "
       "ran, estimated for the rest (planned minutes per job), from the job log at "
       f"{clock(now)} UTC.</p>"
       + table(["Job", "Status", "UTC", "What"], srows, numeric_from=99)
       + (f"<p>Projected end of the queue: about <b>{clock(cursor)} UTC</b>"
          + (" on 16 Nov" if cursor >= 1440 else "") + ". The GPU sat idle from 13:54 to 18:31 (job 118's failure "
          "was not caught until Asif asked for status).</p>" if cursor is not None else "")
       + "<ul><li>The cuts: three arms in the public matrix instead of six (the others get the memory snapshot only); "
       "fewer concurrency points; one seed and subsets in the quality runs. At that size the quality runs can catch a "
       "large loss (the paper's BF16 lost 16 points on AIME 2026) but cannot show 'within 1 point': every task is "
       "expected INCONCLUSIVE under the frozen rule. The full qualification (3 seeds, all questions, 4 arms) needs "
       "about 60 GPU-hours, and even that would leave intervals of roughly 4-9 points per task: at temperature 1.0 a "
       "1-point verdict per task needs about 6,000-12,000 paired answers, some 200 GPU-hours per task (corrected 15 Nov "
       "19:05 UTC).</li>"
       "<li>Strict paper lane (B58) done at 06:10 UTC, 0 retractions in either arm (see Accuracy): LeapQuant 87.5% / "
       "85.8% on AIME 2025 / 2024, bf16 window 75.8% / 84.2%, fp32 window 85.8% / 88.3%, capped answers counted wrong. "
       "Every LeapQuant-fp32 comparison is INCONCLUSIVE at the 1-point margin; LeapQuant is noninferior to the bf16 "
       "window on AIME 2025 (+11.7 points, 95% +0.8 .. +22.6).</li>"
       "<li>Every harness job ends with a checked campaign report in a new folder (~/lqexp/reports/harness-campaign_*); "
       "the first holds the carried history (397 client logs from 33 experiments; 14,551 values re-derived).</li>"
       "<li>Open for Asif: X2, a local git tag of the code behind the published numbers (recommended after these "
       "runs); X1, the authors' vLLM plugin (not recommended now). For the full quality qualification later: one "
       "GPQA-Diamond question has a 2,793-token prompt, above the 1,536 the retraction-free concurrency assumes "
       "(not in this job's 30).</li></ul>")

# ---- test harness (15 Nov): the final configuration's runs and the checked campaign reports ---------------------------
def metric(values, name, field="value"):
    entry = values.get(name)
    return entry.get(field) if isinstance(entry, dict) else None


def run_tps(values):
    steady = metric(values, "successful_steady_output_tps")
    return steady if steady is not None else metric(values, "finite_cohort_output_tps")


harness_html = ""
pair_runs = {}
for path in glob.glob(R("harness", "validation_pair", "b*", "*", "*", "metrics.json")):
    arm, cell = path.split(os.sep)[-3:-1]
    try:
        with open(path, encoding="utf-8") as handle:
            pair_runs[(cell, arm)] = json.load(handle)["metrics"]
    except (OSError, ValueError, KeyError):
        pass
hrows = []
for cell in sorted({c for c, _ in pair_runs}):
    b, l = pair_runs.get((cell, "bf16_window")), pair_runs.get((cell, "lq_final"))
    if not b or not l or not run_tps(b) or not run_tps(l):
        continue

    def both(name, field="value", scale=1.0, digits=1):
        x, y = metric(b, name, field), metric(l, name, field)
        return f"{x * scale:,.{digits}f} / {y * scale:,.{digits}f}" if x is not None and y is not None else "-"
    hrows.append([html.escape(cell), f"{run_tps(b):,.0f}", f"{run_tps(l):,.0f}", f"{run_tps(l) / run_tps(b):.3f}",
                  both("tpot_native_ms", "p99"), both("ttft_ms", "p99", digits=0),
                  both("gpu_memory_used_mib", "max", 1 / 1024), both("gpu_energy_j_per_output_token", digits=3),
                  both("usd_per_million_output_tokens", digits=3)])
if hrows:
    harness_html += ("<h3>Final configuration through the test harness: validation pair (15 Nov, job 113)</h3>"
                     "<p>lq_final (one fit sweep, staggered folds, fused slot copy, CUDA fold, CUDA decode v9) against "
                     "bf16 window on the three lanes, one block per cell, so no interval yet: the 5-block pilot and the "
                     "public matrix follow. Paired columns are bf16 window / LeapQuant; cost at $2.00 per hour.</p>"
                     + table(["Cell", "bf16 window tok/s", "LeapQuant tok/s", "Ratio", "TPOT p99, ms", "TTFT p99, ms",
                              "Peak GPU memory, GiB", "GPU J per output token", "USD per million output tokens"], hrows))
# Paired comparisons of lq_final from the newest checked campaign report copied into reports/ (the pilot's 5 blocks,
# the matrix's single blocks). Values are the report's own (the checker reproduced them).
summaries = sorted(glob.glob(os.path.join(REPO, "reports", "harness-campaign_*", "summary.json")))
if summaries:
    try:
        with open(summaries[-1], encoding="utf-8") as handle:
            comps = json.load(handle).get("paired_comparisons") or []
    except (OSError, ValueError):
        comps = []
    crows, wave_note = [], False
    ours = ("lq_final", "lq_final_eqmem")
    for c in comps:
        if not isinstance(c.get("geometric_mean"), (int, float)) or c.get("plan_id") == "matrix_infx":
            continue                                   # the matrix has its own table below
        ratios, low, high = c.get("ratios") or [], c.get("low"), c.get("high")
        lead, other, mean = c.get("candidate"), c.get("baseline"), c["geometric_mean"]
        if lead not in ours and other in ours:         # report oriented the other way: show LeapQuant over the other
            lead, other, mean = other, lead, 1 / mean
            ratios = [1 / r for r in ratios]
            low, high = (1 / high, 1 / low) if isinstance(low, (int, float)) and isinstance(high, (int, float)) else (None, None)
        elif lead not in ours:
            continue
        interval = (f"{low:.3f} .. {high:.3f}" if isinstance(low, (int, float)) else "none (one block: screening)")
        crows.append([html.escape(c["plan_id"]), html.escape(c["cell_id"].split(":")[-1]),
                      html.escape(f"{lead} / {other}"), str(c.get("n_blocks")),
                      f"{mean:.3f}", interval, ", ".join(f"{r:.3f}" for r in ratios)])
        wave_note = wave_note or (c["plan_id"].startswith("pilot") and any(r < 0.9 for r in ratios))
    if crows:
        harness_html += (f"<h3>Paired comparisons (checked report {html.escape(os.path.basename(os.path.dirname(summaries[-1])))})"
                         "</h3><p>Successful output tokens per second, candidate over baseline in the same block; geometric "
                         "mean of the per-block ratios with a t interval on their logarithms.</p>"
                         + table(["Plan", "Cell", "Candidate / baseline", "Blocks", "Ratio", "95% interval",
                                  "Per block"], crows, numeric_from=3)
                         + ("<p><b>The pilot's block 4</b> (0.821): that LeapQuant run locked into synchronized waves. "
                            "About 362 of 364 running requests, admitted together with exactly 512 output tokens each, "
                            "finished in the same instant; the server fell to 2 running and refilled in a trickle of small "
                            "prefills, every wave (94 prefill batches against 53, median time per token 38.1 against "
                            "30.5 ms). The client and the token stream were normal and nothing was retracted. It is a "
                            "property of a closed-loop benchmark in which every request has the same length, it can hit "
                            "either arm, and it is kept in the result, which is therefore inconclusive (the other four "
                            "blocks give 1.02-1.06). Backlog B73: how often each arm locks, and a confirmation design that "
                            "cannot lock.</p>" if wave_note else ""))
# InferenceX matrix (job 116): the three arms side by side at every point, from the copied metrics.json files.
infx = {}
for path in glob.glob(R("harness", "matrix_infx", "b*", "*", "*", "metrics.json")):
    arm, cell = path.split(os.sep)[-3:-1]
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc.get("status") == "measured":
            infx[(cell, arm)] = doc["metrics"]
    except (OSError, ValueError, KeyError):
        pass
ARMS = [("lq_final", "LeapQuant"), ("bf16_window", "bf16 window"), ("fp32_packed", "fp32 packed")]


def infx_order(cell):
    m = re.match(r"infx_(\d+)k1k_c(\d+)$", cell)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 0)


trows, lrows = [], []
for cell in sorted({c for c, _ in infx}, key=infx_order):
    got = {arm: infx.get((cell, arm)) for arm, _ in ARMS}
    if not got["lq_final"]:
        continue
    isl, users = infx_order(cell)
    label = f"{isl}k / 1k, {users}"
    tps = {arm: metric(m, "output_tps_per_gpu") if m else None for arm, m in got.items()}

    def ratio(other):
        return f"{tps['lq_final'] / tps[other]:.3f}" if tps.get(other) and tps["lq_final"] else "-"

    def each(name, field="value", fmt="{:,.0f}"):
        return " / ".join(fmt.format(metric(got[a], name, field)) if got[a] and metric(got[a], name, field) is not None
                          else "-" for a, _ in ARMS)
    trows.append([label] + [f"{tps[a]:,.1f}" if tps[a] else "-" for a, _ in ARMS]
                 + [ratio("bf16_window"), ratio("fp32_packed"), each("server_running_requests", "p50"),
                    each("gpu_energy_j_per_output_token", fmt="{:.3f}")])
    lrows.append([label, each("user_speed_post_first_token_tps", "p50", "{:.1f}"), each("tpot_native_ms", "p50", "{:.1f}"),
                  each("tpot_native_ms", "p99", "{:.1f}"), each("ttft_ms", "p50", "{:,.0f}"), each("ttft_ms", "p99", "{:,.0f}")])
if trows:
    harness_html += ("<h3>Public matrix: InferenceX fixed sequence (job 116, pinned client)</h3>"
                     "<p>One run per point (a screening block, so no interval: differences of 1-2% are within run-to-run "
                     "noise). Input/output 1,024/1,024 tokens at 1-512 concurrent users and 8,192/1,024 at 4 and 64; "
                     "prompts 10 &times; users, warmups 2 &times; users (verified in the pinned client's log; its export "
                     "carries only the measured cohort, so the exported warmup count reads 0), fixed lengths. "
                     "Throughput is output tokens per "
                     "second on the one GPU. Every run measured, no failed requests, no retractions. Triples are "
                     "LeapQuant / bf16 window / fp32 packed.</p>"
                     + table(["Input / output, users", "LeapQuant tok/s", "bf16 window tok/s", "fp32 packed tok/s",
                              "LeapQuant / bf16", "LeapQuant / fp32", "Running at once (median)",
                              "GPU J per output token"], trows)
                     + "<p>Latency, the same runs (medians and p99 over requests):</p>"
                     + table(["Input / output, users", "Per-user speed, tok/s (median)", "TPOT median, ms",
                              "TPOT p99, ms", "TTFT median, ms", "TTFT p99, ms"], lrows)
                     + "<p class=\"warn\"><b>Caveat found at 23:00 UTC (backlog B77): slot configuration.</b> These runs "
                       "gave LeapQuant and bf16 window 1,100 state slots each and fp32 560. At 1k/1k bf16 window cannot "
                       "use its slots (its KV pool binds at 240 requests while the slots hold 25.8 GiB), and fp32 is "
                       "short of its best. With each arm's slots balanced for 1k/1k (from these runs' own server logs) "
                       "the estimate is bf16 window ~308 requests, LeapQuant ~335, fp32 ~203: LeapQuant's capacity edge "
                       "over bf16 window would be about +9%, not +27%, and the high-load throughput gains below may "
                       "shrink. Job 124 measures it (AA-SLT, 256/512 users, 5 blocks).</p>"
                     + "<ul><li><b>Where LeapQuant gains</b>: memory. LeapQuant and bf16 window have state room for 366 "
                       "requests, fp32 packed for 186. For the first two the KV-cache pool binds first at these lengths, "
                       "and the compressed state leaves LeapQuant the larger pool: 623,450 tokens against 492,156 (fp32 "
                       "packed has 495,468, but its 186-request state room binds before it). At 1k/1k LeapQuant runs up to 305 "
                       "requests at once against 240 and 186; at 8k/1k with 64 users it holds all 64 against 53 for "
                       "both others.</li>"
                       "<li><b>Where it loses</b>: at 1 and 4 users fp32 packed is the fastest (94.9 against 89.6 tok/s "
                       "at one user): LeapQuant's time per token is 0.5-0.6 ms longer (11.1 against 10.5 ms at one "
                       "user). bf16 window, which uses the same 16-token replay (window) path, pays nearly the same, "
                       "which points to that path rather than the compression. Backlog B74.</li>"
                       "<li>At 512 users LeapQuant trades per-user speed for total throughput (it runs a bigger batch). "
                       "The pilot's wave lock does not appear: running counts sit at each arm's cap through every run "
                       "(the low points are the end-of-run drain).</li></ul>")
# AA-SLT matrix (job 117): the same three arms, AA's published method run locally.
aa, aa_status = {}, {}
for path in glob.glob(R("harness", "matrix_aa_slt", "b*", "*", "*", "metrics.json")):
    arm, cell = path.split(os.sep)[-3:-1]
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        aa_status[(cell, arm)] = doc.get("status")
        if doc.get("status") == "measured":
            aa[(cell, arm)] = doc["metrics"]
    except (OSError, ValueError, KeyError):
        pass


def aa_users(cell):
    m = re.match(r"aa_slt_1k1k_c(\d+)$", cell)
    return int(m.group(1)) if m else 0


sys.path.insert(0, os.path.join(REPO, "harness"))
from lqh import eligibility as elig_mod                 # noqa: E402  (derived service eligibility, not yet checked)


def service(cell_dir_glob):
    marks = []
    for arm, _ in ARMS:
        found = glob.glob(cell_dir_glob.replace("ARM", arm))
        if not found:
            marks.append("-")
            continue
        row = elig_mod.classify(__import__("pathlib").Path(found[0]))
        marks.append("yes" if row["service_eligible"] else ("invalid" if row["execution_status"] != "measured" else "no"))
    return " / ".join(marks)


arows = []
for cell in sorted({c for c, _ in aa_status}, key=aa_users):
    got = {arm: aa.get((cell, arm)) for arm, _ in ARMS}
    eligible_marks = service(R("harness", "matrix_aa_slt", "b1", "ARM", cell))
    if not got["lq_final"]:
        arows.append([str(aa_users(cell)), "-", "-", "-", "-", "-", "-", "-", "-",
                      "invalid: client failure storm (V053/V054)" if aa_users(cell) >= 1024 else eligible_marks])
        continue
    tps = {arm: run_tps(m) if m else None for arm, m in got.items()}

    def aratio(other):
        return f"{tps['lq_final'] / tps[other]:.3f}" if tps.get(other) and tps["lq_final"] else "-"

    def aeach(name, field="value", fmt="{:,.0f}"):
        return " / ".join(fmt.format(metric(got[a], name, field)) if got[a] and metric(got[a], name, field) is not None
                          else "-" for a, _ in ARMS)
    arows.append([str(aa_users(cell))] + [f"{tps[a]:,.1f}" if tps[a] else "-" for a, _ in ARMS]
                 + [aratio("bf16_window"), aratio("fp32_packed"), aeach("user_speed_post_first_token_tps", "p50", "{:.1f}"),
                    aeach("ttft_ms", "p99", "{:,.0f}"), aeach("server_running_requests", "p50"), eligible_marks])
if arows:
    harness_html += ("<h3>Public matrix: Artificial Analysis method (job 117, run locally)</h3>"
                     "<p>AA's published system load test: 1,000-token prompts and answers, each finished request "
                     "replaced at once, a ramp of at least 30 s, then a 180 s steady phase; throughput is the successful "
                     "output tokens per second in the steady phase. One run per point; an unofficial local run of the "
                     "published method, not an AA score. Triples are LeapQuant / bf16 window / fp32 packed. The slot caveat "
                     "above applies from 256 users. <b>Corrections (15 Nov 23:00 UTC)</b>: the 1,024-user runs were a "
                     "client failure storm (99.86-99.90% of 2.3-2.9 million attempts failed with 'Too many open files'; "
                     "the client's open-file limit was 1,024) and are invalid; 512 users is an overload point (no arm "
                     "has latency-qualified goodput there).</p>"
                     + table(["Users", "LeapQuant tok/s", "bf16 window tok/s", "fp32 packed tok/s", "LeapQuant / bf16",
                              "LeapQuant / fp32", "Per-user speed, tok/s (median)", "TTFT p99, ms",
                              "Running at once (median)", "Service-eligible"], arows)
                     + "<p><b>Service-eligible</b> (multi-GPU benchmark plan's suggested internal gates, derived by "
                       "python -m lqh.eligibility, not yet reproduced by the independent checker): the SLO's request "
                       "predicate (TTFT &le; 2 s, TPOT &le; 50 ms) met by &ge; 95% of attempted requests, &ge; 99% "
                       "completed, error rate &le; 1%, p99 token gap &le; 100 ms. Quality is a separate gate (pending), "
                       "so no point is publication-eligible yet.</p>")
# ---- Qwen3.8-27B (backlog B75): first contact (job 123) and screening (job 125) --------------------------------------
q38_plan = R("q38", "probe", "plan.json")
if os.path.isfile(q38_plan):
    with open(q38_plan, encoding="utf-8") as handle:
        q38 = json.load(handle)
    k1 = read(R("q38", "probe", "K1.log"))
    m = re.search(r"HV=48 H=16.*?batch 128\s+decode step, fast vs exact\s+output difference: median (\S+)\s+p99 (\S+)\s+worst (\S+);",
                  k1, re.S)
    agree = q38.get("agreement", {})
    prow = []
    for arm, label in (("leapquant", "LeapQuant"), ("bf16_window", "bf16 window"), ("fp32", "fp32 packed")):
        a = q38["arms"].get(arm) or {}
        if "balanced_requests" not in a:
            continue
        diffs = agree.get(arm)
        same = f"{sum(1 for d in diffs if d is None)} of {len(diffs)}" if diffs else "reference"
        prow.append([label, f"{a['budget_gib']:.1f}", f"{a['state_bytes_per_slot'] / 2**20:.1f}",
                     f"{a['balanced_requests']} ({a['balanced_slots']} slots)", same])
    harness_html += ("<h3>Qwen3.8-27B (FP8): first contact (job 123, 15 Nov 23:36-23:50 UTC)</h3>"
                     "<p>The newest released model with Gated DeltaNet layers that fits one H100 (5 Aug 2026): 48 "
                     "linear-attention layers of 48 value heads over 16 key heads (3:1; the 9B has 2:1), 16 attention "
                     "layers. Every arm served greedy and sampled answers without an error; LeapQuant's log shows all "
                     "48 layers on the compressed path (CUDA decode v9). Kernel agreement at this head layout, fast "
                     "against exact, batch 128: "
                     + (f"median {m.group(1)}, p99 {m.group(2)}, worst {m.group(3)}" if m else "see K1.log")
                     + " (the same as the 9B layout in the same process). Memcheck: no memory-access error; two CUDA "
                       "API errors from Triton's kernel loader, re-run with job 112's convention in job 125.</p>"
                     + table(["Arm", "State + KV budget, GiB", "Per state slot, MiB", "Requests at once, 1k/1k (balanced)",
                              "Greedy completions identical to fp32"], prow)
                     + "<p>LeapQuant fits 11% more 1k/1k requests than bf16 window and 82% more than fp32 in the same "
                       "memory. These arms (300, 270 and 165 slots) run the screening sweep in job 125.</p>")
q38_runs = {}
for path in glob.glob(R("q38", "harness", "screen_aa_q38", "b*", "*", "*", "metrics.json")):
    arm, cell = path.split(os.sep)[-3:-1]
    try:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        q38_runs[(cell, arm)] = doc
    except (OSError, ValueError):
        pass
if q38_runs:
    QARMS = [("lq_final_q38", "LeapQuant"), ("bf16_window_q38", "bf16 window"), ("fp32_packed_q38", "fp32 packed")]
    qrows = []
    for cell in sorted({c for c, _ in q38_runs}, key=aa_users):
        docs = {a: q38_runs.get((cell, a)) for a, _ in QARMS}
        tps = {a: (run_tps(d["metrics"]) if d and d.get("status") == "measured" else None) for a, d in docs.items()}
        marks = []
        for a, _ in QARMS:
            found = glob.glob(R("q38", "harness", "screen_aa_q38", "b1", a, cell))
            marks.append(("yes" if elig_mod.classify(__import__("pathlib").Path(found[0]))["service_eligible"] else "no")
                         if found else "-")

        def qr(other):
            return f"{tps['lq_final_q38'] / tps[other]:.3f}" if tps.get(other) and tps.get("lq_final_q38") else "-"
        qrows.append([str(aa_users(cell))] + [f"{tps[a]:,.1f}" if tps[a] else "-" for a, _ in QARMS]
                     + [qr("bf16_window_q38"), qr("fp32_packed_q38"), " / ".join(marks)])
    harness_html += ("<h3>Qwen3.8-27B: screening sweep (job 125, AA-SLT method, one block)</h3>"
                     "<p>Each arm with the state slots balanced for 1k/1k from the probe. One run per point: screening, "
                     "no interval.</p>"
                     + table(["Users", "LeapQuant tok/s", "bf16 window tok/s", "fp32 packed tok/s", "LeapQuant / bf16",
                              "LeapQuant / fp32", "Service-eligible (L / B / F)"], qrows))
report_rows = []
for line in read(R("harness", "REPORTS.txt")).splitlines():
    m = re.match(r"(\d{8}-\d{6}) (.+?): (\S+) check exit (\d+)", line)
    if m:
        name = os.path.basename(m.group(3))
        link = (f"<a href=\"../{name}/index.html\">{html.escape(name)}</a>"
                if os.path.isfile(os.path.join(REPO, "reports", name, "index.html")) else html.escape(name))
        report_rows.append([f"{m.group(1)[9:11]}:{m.group(1)[11:13]}", html.escape(m.group(2)), link,
                            "<span class=\"pass\">all values reproduced</span>" if m.group(4) == "0"
                            else "<span class=\"fail\">check failed</span>"])
        continue
    m = re.match(r"(\d{8}-\d{6}) recheck of (\S+) .*\"ok\": (true|false)", line)
    if m:
        for row in report_rows:
            if m.group(2) in row[2]:
                row[3] += (f"; re-checked {m.group(1)[9:11]}:{m.group(1)[11:13]}: "
                           + ("<span class=\"pass\">all values reproduced</span>" if m.group(3) == "true"
                              else "<span class=\"fail\">still failing</span>"))
        continue
    m = re.match(r"(\d{8}-\d{6}) note on (\S+) ", line)
    if m:
        for row in report_rows:
            if m.group(2) in row[2]:
                row[3] += f"; cause fixed at {m.group(1)[9:11]}:{m.group(1)[11:13]} (summary.txt)"
if report_rows:
    harness_html += ("<h3>Checked campaign reports</h3><p>Every harness job ends with a report in a new folder showing "
                     "every metric of every run; the independent checker recomputes each shown value from the raw "
                     "records (results/harness/REPORTS.txt). The report after job 113 failed only because GPU energy, "
                     "shown for the first time, had no second implementation in the checker; added at 06:48 UTC, and "
                     "the re-check reproduces all 12,595 values.</p>"
                     + table(["UTC", "After", "Report", "Checker"], report_rows[::-1], numeric_from=99))

# ---- verdict strip --------------------------------------------------------------------------------------------------
def card(label, value, note, cls, word):
    return (f"<div class=\"verdict\"><div class=\"label\">{html.escape(label)}</div><div class=\"value\">{value}</div>"
            f"<div class=\"note\"><span class=\"pill {cls}\">{word}</span> {note}</div></div>")


def paired_ratios(found):
    out = []
    for (cfg, s, c), m in found.items():
        if cfg.startswith("off_leapquant_cuda_r"):
            twin = found.get(("off_bf16_window_" + cfg.split("_")[-1], s, c))
            if twin and m.get("Output token throughput (tok/s)") and twin.get("Output token throughput (tok/s)"):
                out.append(m["Output token throughput (tok/s)"] / twin["Output token throughput (tok/s)"])
    return out


off_ratios = paired_ratios(runs)
serving_source = "round 9, radix off, CUDA fold"
if os.path.isdir(R("round10")):
    r10_ratios = paired_ratios(r10)
    if len(r10_ratios) >= 6:
        off_ratios, serving_source = r10_ratios, f"round 10, {len(r10_ratios)} load points, CUDA fold and store"
if r11:
    r11_ratios = paired_ratios(r11)
    if len(r11_ratios) >= 5:
        off_ratios, serving_source = r11_ratios, f"round 11, {len(r11_ratios)} load points, final configuration"
fp32_ratios = []
fp32_source = "round 10"
if os.path.isdir(R("round10")):
    lq_runs = r11 if r11 else r10
    fp32_source = "round 11 against round 10's fp32" if r11 else "round 10"
    for (cfg, s, c), m in lq_runs.items():
        if cfg == "off_leapquant_cuda_r1":
            f = r10.get(("off_fp32_packed_r1", s, c), {})
            if m.get("Output token throughput (tok/s)") and f.get("Output token throughput (tok/s)"):
                fp32_ratios.append(m["Output token throughput (tok/s)"] / f["Output token throughput (tok/s)"])
    for (cfg, s, c), m in r12.items():             # round 12: fp32 in the same session
        if cfg == "off_leapquant_cuda_r1":
            f = r12.get(("off_fp32_packed_r1", s, c), {})
            if m.get("Output token throughput (tok/s)") and f.get("Output token throughput (tok/s)"):
                fp32_ratios.append(m["Output token throughput (tok/s)"] / f["Output token throughput (tok/s)"])
    for (cfg, s, c), m in r13.items():             # round 13: fp32 in the same session, each with the slots it fits
        if cfg == "off_leapquant_cuda_r1":
            f = r13.get(("off_fp32_packed_r1", s, c), {})
            if m.get("Output token throughput (tok/s)") and f.get("Output token throughput (tok/s)"):
                fp32_ratios.append(m["Output token throughput (tok/s)"] / f["Output token throughput (tok/s)"])
    if r12:
        fp32_source += " and round 12 (decode cell, same session)"
    if r13:
        fp32_source += " and round 13 (capacity regime)"
# Gate G3 from the paired confirmation (runbook/g3_confirm.sh), if there is one: the arm with the highest ratio.
g3_best = None
for line in (read(R("g3-final", "G3.txt")) or read(R("g3", "G3.txt"))).splitlines():
    m = re.match(r"(\w+)\s+n=(\d+)\s+LeapQuant ([\d.]+) ms\s+bf16 window ([\d.]+) ms\s+ratio ([\d.]+) \(95% ([\d.]+) \.\. ([\d.]+)\)\s+(.+)$", line)
    if m and (g3_best is None or float(m.group(5)) > g3_best[4]):
        g3_best = (m.group(1), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)),
                   float(m.group(6)), float(m.group(7)), m.group(8).strip())
serving_mean = sum(off_ratios) / len(off_ratios) if off_ratios else None
serving_cls, serving_word = (("open", "open") if serving_mean is None else ("pass", "ahead") if serving_mean >= 1.02
                             else ("fail", "behind") if serving_mean < 0.98 else ("open", "level"))
verdicts = "".join([
    card("Kernel against fp32", f"{fp32 / best_lq:.2f}×" if fp32 and best_lq else "-",
         "against SGLang's default fp32 kernel (packed), component timing, batch 512; paper: 2.52× on B200"
         + (f". Complete cycle against the faster fp32 window kernel: {cycle_512_fp32['value']:.2f}×"
            if cycle_512_fp32 else ""), "pass", "exceeds"),
    (card("Kernel against bf16 window (complete cycle)", f"{cycle_512['value']:.3f}×",
          f"batch 512, {cycle['batches']['512'].get('repetitions', '?')} repetitions, 95% {cycle_512['low']:.3f} .. "
          f"{cycle_512['high']:.3f}; gate p5.G3 1.25×, paper ~1.47× on B200; behind at batches 1 and 8",
          "pass" if cycle_512["low"] >= 1.25 else "open", "gate met" if cycle_512["low"] >= 1.25 else "below the gate")
     if cycle_512 else
     card("Kernel against bf16 window (component sum)", f"{g3_best[4]:.3f}×",
          f"{g3_best[0]}, {g3_best[1]} runs, decode + fold/16 summed, order-dependent; the complete-cycle timing "
          f"(queued) decides gate v6 G2",
          "open", "to be re-timed")
     if g3_best else
     card("Kernel against bf16 window", f"{bf16 / best_lq:.3f}×" if bf16 and best_lq else "-", "gate G3: 1.25×; paper ~1.47×",
          "pass" if bf16 and best_lq and bf16 / best_lq >= 1.25 else "open", "met" if bf16 and best_lq and bf16 / best_lq >= 1.25 else "open")),
    card("Serving against bf16 window", f"{min(off_ratios):.2f}-{max(off_ratios):.2f}×" if off_ratios else "-",
         f"mean {sum(off_ratios) / len(off_ratios):.2f} ({serving_source})" if off_ratios else "",
         serving_cls, serving_word),
    card("Serving against fp32 (the paper's baseline)",
         f"up to {max(fp32_ratios):.2f}×" if fp32_ratios else "-",
         (f"mean {sum(fp32_ratios) / len(fp32_ratios):.2f} over {len(fp32_ratios)} points ({fp32_source}); paper 1.47-1.57× "
          f"end to end") if fp32_ratios else "round 10 pending",
         "open", "not comparable with the paper (H100/SGLang, single runs)" if fp32_ratios else "open"),
    (card("Service gate: open-loop goodput", f"{100 * gate_reps[10][0]:+.0f}% at 10 req/s",
          f"95% {100 * gate_reps[10][1]:+.0f}..{100 * gate_reps[10][2]:+.0f}%, 3 sessions; at 9 req/s {100 * gate_reps[9][0]:+.0f}% "
          f"({100 * gate_reps[9][1]:+.0f}..{100 * gate_reps[9][2]:+.0f}%); gate +15%",
          "open", "not established: p99 token-gap limit not applied, one arrival seed")
     if gate_reps.get(9) and gate_reps.get(10) else ""),
    card("State memory per slot", "21.3 MB", "fp32 50.3 MB, bf16 25.2 MB", "pass", "2.4× less than fp32"),
    card("Accuracy (strict paper lane)", "inconclusive",
         "LeapQuant vs fp32 window: AIME 2025 +1.7 points (-7.4..+10.8), AIME 2024 -2.5 (-11.0..+6.0). bf16 window vs "
         "fp32: AIME 2025 -10.0 (-19.4..-0.6), every lost sample capped. Capped = wrong; 0 retractions",
         "open", "cannot exclude a 1-point loss"),
])

# ---- 15 Nov evening: quality on the paper's four tasks (job 121), long context (119), compounding (120) -------------
newer_accuracy = ""
try:
    with open(R("harness", "quality-12h-r2", "quality_decision.json"), encoding="utf-8") as handle:
        qd = json.load(handle)
except (OSError, ValueError):
    qd = None
if qd:
    TASKS = [("aime2026", "AIME 2026"), ("beyond_aime", "BeyondAIME"), ("gpqa_diamond", "GPQA-Diamond"),
             ("livecodebench_v6", "LiveCodeBench v6")]
    qrows = []
    for key, label in TASKS:
        lq, bf = qd["arms"].get("leapquant", {}).get(key), qd["arms"].get("bf16_window", {}).get(key)
        if not lq:
            continue
        n = lq["n_reference"]

        def pts(x):
            return f"{100 * x:+.1f}"
        qrows.append([label, str(n), f"{round(lq['accuracy_reference'] * n)} ({100 * lq['accuracy_reference']:.1f}%)",
                      f"{round(lq['accuracy_arm'] * n)} ({100 * lq['accuracy_arm']:.1f}%)",
                      f"{pts(lq['difference'])} ({pts(lq['low'])} .. {pts(lq['high'])})", lq["verdict"],
                      f"{round(bf['accuracy_arm'] * n)} ({100 * bf['accuracy_arm']:.1f}%)" if bf else "-",
                      f"{100 * lq['cap_rate_arm']:.0f}% / {100 * bf['cap_rate_arm']:.0f}%" if bf else "-"])
    newer_accuracy += ("<h3>The paper's four tasks (job 121, 15 Nov 18:32-22:11 UTC)</h3>"
                       "<p>Paper sampling (thinking on, temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5), "
                       "answers up to 131,072 tokens, one sample per question, subsets (AIME 2026 all 30, BeyondAIME 15, "
                       "GPQA-Diamond 30, LiveCodeBench v6 12), retraction-free (0 retractions in every arm), no failed "
                       "request. Capped answers count as wrong. Difference: LeapQuant minus fp32 in points, Newcombe "
                       "paired 95% interval; the frozen rule needs the whole interval above -1 point.</p>"
                       + table(["Task", "Questions", "fp32 correct", "LeapQuant correct", "LeapQuant - fp32, points (95%)",
                                "Verdict (1-point margin)", "bf16 window correct", "Answers capped, LeapQuant / bf16"],
                               qrows)
                       + "<ul><li><b>No large loss</b>: LeapQuant matched or beat fp32 on every task (73 against 69 of "
                         "87 correct); our fp32 AIME 2026 score (86.7%) is close to the paper's FP32 (87.9). At one "
                         "sample per question the intervals are 12-19 points wide on each side, so every task is "
                         "INCONCLUSIVE at the 1-point margin, as expected for this size.</li>"
                         "<li><b>bf16 window</b> also scored 73 of 87, but 3 of its 15 BeyondAIME answers ran out of "
                         "the 131,072-token budget (20%, above the 5% target).</li>"
                         "<li>The first attempt (job 118, 12:33) lost every request: the server died compiling "
                         "FlashInfer's sampling kernel (no curand.h in the host's CUDA 13.0 toolkit). Fixed by an "
                         "include folder with only the curand headers; arms with failed requests now fail.</li></ul>")
try:
    with open(R("harness", "longcontext", "nll_compare_c16.json"), encoding="utf-8") as handle:
        nll16 = json.load(handle)
except (OSError, ValueError):
    nll16 = None
if nll16:
    BUCKETS = [("L32768:1-1024", "1-1k"), ("L32768:1024-2048", "1k-2k"), ("L32768:2048-4096", "2k-4k"),
               ("L32768:4096-8192", "4k-8k"), ("L32768:8192-16384", "8k-16k"), ("L32768:16384-32768", "16k-32k")]
    cand = {os.path.basename(k).split("_c")[0].replace("nll_", ""): v for k, v in nll16["candidates"].items()}
    nrows = []
    for key, label in BUCKETS:
        row = [label]
        for arm in ("leapquant", "bf16_window"):
            b = (cand.get(arm) or {}).get(key)
            row.append(f"{100 * b['relative_perplexity_change']:+.2f}% "
                       f"({100 * (math.exp(b['low']) - 1):+.2f} .. {100 * (math.exp(b['high']) - 1):+.2f})" if b else "-")
        nrows.append(row)
    newer_accuracy += ("<h3>Long context (job 119): perplexity against fp32 by position</h3>"
                       "<p>Eight PG-19 books of 33,000 tokens, teacher-forced through the server with 16-token prefill "
                       "chunks (the state is compressed every window, the closest prefill analogue of generation). "
                       "Change in perplexity against the fp32 state, 95% interval over books. With the default 8,192-token "
                       "chunks LeapQuant matches fp32 to within 0.01%.</p>"
                       + table(["Positions", "LeapQuant", "bf16 window"], nrows)
                       + "<p>The difference grows with position: +0.07% at 8k-16k and +0.29% at 16k-32k for LeapQuant, "
                         "about twice bf16 window's drift. Small, but measurable.</p>"
                         "<p><b>Compounding (job 120)</b> measured nothing: every layer-by-layer difference was exactly 0, "
                         "all 24 layers included, because the per-layer test needs the mirror state, which is never read "
                         "back between prefill chunks. It needs a redesign before a re-run.</p>")
accuracy = newer_accuracy + accuracy

page = read(TEMPLATE)
for key, value in (("__UPDATED__", time.strftime("%d %b %Y, %H:%M UTC", BUILT).upper()), ("__VERDICTS__", verdicts),
                   ("__KERNEL_CHART__", chart), ("__KERNEL_TABLE__", kernel_table), ("__SERVING__", harness_html + serving),
                   ("__ACCURACY__", accuracy), ("__DECISIONS__", decisions), ("__NEXT__", nxt),
                   ("__SELF__", os.path.relpath(OUT, REPO))):
    page = page.replace(key, value)
leftover = sorted(set(re.findall(r"__[A-Z][A-Z_]+__", page)))
if not page.strip() or leftover:
    sys.exit(f"error: the template is missing or placeholders were left unfilled: {leftover}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "x", encoding="utf-8") as handle:       # "x": fails rather than overwrite an earlier page
    handle.write(page)
gr.write_list_of_reports()                             # reports/all_reports.html, the list of every report folder
print("wrote", OUT, len(page), "chars; kernels:", [(l, round(v, 3)) for l, v, _ in rows_k], "; serving ratios:",
      [round(r, 3) for r in off_ratios])
