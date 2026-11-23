"""Evidence records that bind claims to artifacts (prompt rev. 3, sections 6, 14 and 16).

  python -m lqh.evidence --out harness/evidence

Writes
  paper_crosswalk.json       one row per claimed paper comparison: paper version, page/figure/cell, metric
                             boundary, reported value, local value, matched fields, missing fields, status
                             (matched / partial / not_comparable / pending)
  claims_registry.json       every engineering claim with evidence_kind (measured_raw, reported_aggregate,
                             derived_from_reported, unverified_assertion, planned), source artifacts, candidate or
                             build, metric boundary and derivation; unknowns stay null with a reason
  memory_reconciliation.json what the historical runs allocated, what "equal memory" did and did not hold, and the
                             measurement that replaces it (plan memory_per_arm)
Local values are recomputed from raw files where they exist (results/g3-final per-run files).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .arith import bytes_and_capacity, conditional_cycle, time_and_speed
from .util import harness_root, read_json, repo_root, sha256_file, utc_now, write_json

PAPER = {"version": "arXiv 2609.38166v2 (10 Nov 2026)", "pdf_sha256": "044e839c2ea0b1a6cb4893fd2dc2b373c0d5d74c55fac25863777d5df2acc32f"}


def g3_final_means(root: Path) -> dict:
    """Mean per-token times at batch 512 from results/g3-final/<arm>_<n>.txt (component sums)."""
    pattern = {"lq": re.compile(r"->\s+([0-9.]+) ms per token with a fold every"),
               "bf16_window": re.compile(r"sglang window kernel, bf16 state: ([0-9.]+) ms per token"),
               "fp32_window": re.compile(r"sglang window kernel, fp32 state: ([0-9.]+) ms per token"),
               "fp32_packed": re.compile(r"sglang packed kernel, fp32 state: ([0-9.]+) ms per token")}
    out = {}
    for arm in ("adopted", "v9"):
        values: dict[str, list[float]] = {k: [] for k in pattern}
        files = []
        for n in range(1, 11):
            file = root / "results" / "g3-final" / f"{arm}_{n}.txt"
            if not file.exists():
                continue
            section = file.read_text(encoding="utf-8", errors="replace").split("batch 512", 1)[-1]
            files.append(str(file.relative_to(root)))
            for key, regex in pattern.items():
                found = regex.search(section)
                if found:
                    values[key].append(float(found.group(1)))
        out[arm] = {k: (sum(v) / len(v) if v else None) for k, v in values.items()}
        out[arm]["runs"] = len(files)
        out[arm]["files_sha256"] = {f: sha256_file(root / f) for f in files}
    return out


def crosswalk(cells: dict, g3: dict) -> list[dict]:
    def c(cid):
        return cells[cid]
    v9 = g3.get("v9", {})
    rows = [
        {"paper_cell": "headline.e2e_avg_speedup.v2_pdf_html", "reported": c("headline.e2e_avg_speedup.v2_pdf_html")["reported_value"],
         "metric_boundary": "end-to-end offline output throughput vs FP32-state vLLM, mean of the 24 Figure 9 cells (B200 15, RTX PRO 6000 9)",
         "local_value": None, "local_null_reason": "no offline-serving lane on the paper's datasets has been run (paper_reproduction lane planned)",
         "matches": {"hardware": False, "model": "partial (Qwen3.5-9B is one of the averaged models)", "engine": False,
                     "precision": True, "workload": False, "cache": "unknown", "graph": "unknown"},
         "missing_fields": ["vLLM/FLA/TileLang versions", "request counts", "prefix caching setting for Figure 8/9"],
         "status": "not_comparable", "note": "the arXiv abstract page still says 1.47x (v1's 18-cell mean); the full v2 PDF says 1.57x; neither is a universal target"},
        {"paper_cell": "headline.e2e_avg_speedup.abs_page_metadata", "reported": 1.47, "metric_boundary": "v1 abstract leftover",
         "local_value": None, "local_null_reason": "superseded by the v2 PDF (discrepancy retained)", "matches": {}, "missing_fields": [],
         "status": "not_comparable"},
        {"paper_cell": "fig7.gdn.b200.bs512", "reported": c("fig7.gdn.b200.bs512")["reported_value"],
         "metric_boundary": "single GDN layer decode kernel, 32 heads, d_k = d_v = 128, batch 512, LeapQuant TileLang vs FLA FP32 kernel, B200",
         "local_value": (v9.get("fp32_window") / v9["lq"]) if v9.get("fp32_window") and v9.get("lq") else None,
         "local_boundary": "H100 PCIe, one layer, component sum (decode + fold/16), LeapQuant CUDA v9 vs SGLang's FP32 WINDOW kernel (not FLA's packed kernel), 10 runs (results/g3-final)",
         "local_value_against_sglang_fp32_packed": (v9.get("fp32_packed") / v9["lq"]) if v9.get("fp32_packed") and v9.get("lq") else None,
         "matches": {"hardware": False, "model_layer_shape": True, "batch": True, "dtype": True, "full_cycle": False,
                     "baseline_kernel": False},
         "missing_fields": ["full state-advancing cycle timing (queued: harness kernel-cycle job)", "FLA FP32 kernel on H100"],
         "status": "partial"},
        {"paper_cell": "fig8.qwen3.5-9b.b200.bs512", "reported": c("fig8.qwen3.5-9b.b200.bs512")["reported_value"],
         "metric_boundary": "full decode step in vLLM, 4K context, batch 512, CUDA graphs, vs FP32-state vLLM, B200",
         "local_value": None, "local_null_reason": "pure-decode lane not run (planned); client concurrency in serving runs is not the decode batch",
         "matches": {"hardware": False, "engine": False}, "missing_fields": ["pure-decode measurement on H100/SGLang"], "status": "pending"},
        {"paper_cell": "fig9.qwen3.5-9b.b200.gpqa", "reported": c("fig9.qwen3.5-9b.b200.gpqa")["reported_value"],
         "metric_boundary": "offline output throughput, GPQA prompts, vLLM, vs FP32-state vLLM, B200",
         "local_value": None, "local_null_reason": "offline lane not run; GPQA gated for this account until terms are accepted",
         "matches": {"hardware": False, "engine": False}, "missing_fields": ["dataset access", "offline harness lane"], "status": "pending"},
    ]
    for task, local in (("aime2026", "results/harness/quality-aime2026 (job 102)"), ("beyondaime", "results/harness/quality-beyond-aime (job 103)"),
                        ("gpqa_diamond", "BLOCKED: gated dataset (job 103b checks access)"), ("lcb_v6", "BLOCKED: no sandboxed scorer")):
        for arm in ("fp32", "bf16", "8bit.ours"):
            cid = f"tab1.qwen3.5-9b.{task}.{arm}"
            if cid not in cells:
                continue
            rows.append({"paper_cell": cid, "reported": cells[cid]["reported_value"], "unit": "%",
                         "metric_boundary": "task accuracy, Qwen3.5-9B, 3 seeds, Table 5 sampling (T 1.0, top-p 0.95, top-k 20, presence 1.5, 81,920 tokens, early stop on repetition)",
                         "local_value": None, "local_null_reason": f"pending: {local}",
                         "matches": {"model": True, "sampling": "partial (early repetition stop not reproduced)", "seeds": True,
                                     "engine": False, "hardware": False,
                                     "bf16_variant": "paper BF16 = rounded every step; local arm bf16_rounded matches it"},
                         "missing_fields": ["paper's early-stop rule", "dataset versions (local: MathArena/aime_2026, ByteDance-Seed/BeyondAIME)"],
                         "status": "pending"})
    rows.append({"paper_cell": "headline.state_memory_traffic_reduction", "reported": 3.4, "unit": "x vs FP32 (4 bytes/element)",
                 "metric_boundary": "stored window-boundary state per element: 1.19 bytes vs 4",
                 "local_value": bytes_and_capacity(4.0, 1.19)["ideal_capacity_x"],
                 "local_boundary": "arithmetic on the reported 1.19 bytes (derived_from_reported); measured server-log allocation: LeapQuant 1.69 vs bf16 window 2.38 bytes per entry (historical inventory)",
                 "matches": {"definition": True}, "missing_fields": ["measured equal-total-memory serving capacity (plan memory_per_arm)"],
                 "status": "partial"})
    for row in rows:
        row["paper"] = PAPER
    return rows


def claims(g3: dict, sanitizer_path: Path) -> list[dict]:
    sanitizer = read_json(sanitizer_path) if sanitizer_path.exists() else {}
    resolved = next((r for r in sanitizer.get("claims", []) if "exit" in r.get("claim", "")), {})
    estimate = conditional_cycle(0.2524, 1.27)
    return [
        {"claim": "CUDA fold 1.27 ms at batch 512 (fold v1)", "value": 1.27, "unit": "ms per fold",
         "evidence_kind": "measured_raw", "artifacts": ["results/fast_fold_v1/F3_cuda_*.log line 31"],
         "build": "fast_fold_v1 (superseded by v3/v4/v6 and batched Gram-Schmidt)", "boundary": "the fold alone, one call per graph, boundary state restored before each replay; NOT a complete state-advancing window",
         "status": "historical measurement of a superseded kernel; not a full-cycle result"},
        {"claim": "0.332 ms per layer-token", "value": estimate, "unit": "ms/token", "evidence_kind": "derived_from_reported",
         "derivation": "0.2524 ms decode + 1.27 ms fold / 16 = 0.331775 ms (conditional model)",
         "comparison": time_and_speed(0.3821, estimate), "artifacts": ["summary.txt lines 509-510"],
         "status": "an estimate, not a timed result; the full-cycle timing is order-of-work step 1 (job 100)"},
        {"claim": "G3 kernel gate MET: 1.271x bf16 window per token at batch 512", "value": (g3.get("v9", {}).get("bf16_window") or 0) /
         (g3.get("v9", {}).get("lq") or 1) if g3.get("v9", {}).get("lq") else None, "unit": "x",
         "evidence_kind": "measured_raw", "artifacts": list((g3.get("v9") or {}).get("files_sha256", {}).keys()),
         "boundary": "component sum (decode + fold/16), 10 paired runs, plan-5 gate numbering",
         "caveats": ["run order changes the fold time (1.04-1.07 ms after the Triton decode, 0.95 ms after v9); with the slower fold v9's ratio is 1.244 < 1.25",
                     "plan-5 G3 is not v6 G3 (State); its PASS is not imported into the v6 registry"],
         "status": "recorded as history; v6.G2 is decided by the full-cycle timing (job 100)"},
        {"claim": "memory-check run exited with code 1", "evidence_kind": "measured_raw",
         "artifacts": ["results/fast_fold_v1/F1a_rerun.log", "harness/evidence/sanitizer_results_historical.json"],
         "resolution": resolved.get("resolution"), "status": "resolved: API errors only (Triton launcher), no memory-access report; current checks queued (job 100b)"},
        {"claim": "LeapQuant stores 1.19 bytes per boundary-state entry vs 2 (bf16)", "evidence_kind": "derived_from_reported",
         "derivation": bytes_and_capacity(2.0, 1.19), "measured_alternative": "server logs: 1.69 (LeapQuant) vs 2.38 (bf16 window incl. ring) bytes per entry",
         "status": "storage accounting only; not equal total GPU memory or SLO-qualified capacity"},
        {"claim": "1,310 LeapQuant slots = equal memory with 1,100 bf16-window slots", "evidence_kind": "unverified_assertion",
         "facts": {"slot_increase": 1310 / 1100 - 1, "measured_peaks_GiB": {"bf16_window": "71.3-71.9", "leapquant": "67.0-67.5"},
                   "not_counted": "bf16 window replay ring buffers 4.89 GiB"},
         "status": "configured slots, not equal measured bytes; replaced by plan memory_per_arm (job 104)"},
        {"claim": "round 13: all 1,024 requests running", "evidence_kind": "unverified_assertion",
         "facts": {"server_max_running_seen": 929}, "status": "contradicted by the server log (929)"},
    ]


def memory(inventory_path: Path) -> dict:
    inventory = read_json(inventory_path)
    topic = next(t for t in inventory["topics"] if t["id"] == 4)
    return {"schema": "lqh.memory_reconciliation.v1", "computed_utc": utc_now(),
            "historical_findings": topic.get("problems", []),
            "categories_required": ["weights", "ordinary attention KV", "active recurrent state", "window records",
                                    "compensators", "scales", "checkpoint copies", "convolution state",
                                    "temporary dequantisation", "scratch", "graph pools", "fallback buffers"],
            "categories_logged_by_sglang": ["weights (Load weight end mem usage)", "KV K/V sizes", "recurrent ssm/conv state",
                                            "LeapQuant decode state (window, compensators, scales, levels together)",
                                            "CUDA graph mem usage", "ReplaySSM ring (window kernel records)"],
            "not_separable_from_logs": ["LeapQuant compensators vs scales vs window records (one allocation)",
                                        "temporary dequantisation and scratch (transient, inside the allocator)"],
            "snapshots": "every harness run records nvidia-smi samples (whole device) and snapshots before model load, "
                         "after pools/graphs, at steady load, at the peak and after drain; allocation vs reservation is "
                         "not separable from nvidia-smi (torch allocator statistics are not exported by SGLang)",
            "measurement_plan": "plan memory_per_arm (job 104): all 13 candidates, decode 128/512 at 512 users, allocations, "
                                "peaks, configured slots, admitted running and waiting requests per arm",
            "inventory_sha256": sha256_file(inventory_path)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.evidence")
    parser.add_argument("--out", default=str(harness_root() / "evidence"))
    args = parser.parse_args(argv)
    out = Path(args.out)
    root = repo_root()
    cells = {c["cell_id"]: c for c in read_json(harness_root() / "config" / "sources" / "paper_cells.json")["cells"]}
    g3 = g3_final_means(root)
    write_json(out / "paper_crosswalk.json", {"schema": "lqh.paper_crosswalk.v1", "computed_utc": utc_now(),
                                              "label": "H100 port evaluated with paper-inspired methodology",
                                              "rows": crosswalk(cells, g3)})
    write_json(out / "claims_registry.json", {"schema": "lqh.claims_registry.v1", "computed_utc": utc_now(),
                                              "g3_final_recomputed": g3,
                                              "claims": claims(g3, out / "sanitizer_results_historical.json")})
    write_json(out / "memory_reconciliation.json", memory(out / "historical_inventory.json"))
    print({"g3_final_v9": {k: v for k, v in g3.get("v9", {}).items() if k != "files_sha256"}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
