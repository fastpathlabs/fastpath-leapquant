"""Summary of runbook/harness_kernel_cycle.sh: the complete, state-advancing layer cycle per token, LeapQuant against
SGLang's bf16 and fp32 window kernels, paired within each repetition (same process, alternating order).

  python -m lqh.kernel_cycle RESULTS_DIR      -> kernel_cycle.json (evidence_kind measured_raw)

Gate p5.G3 (plan 5, kept in history; v6 G2 Kernel uses it): LeapQuant per token at least 20% below the bf16 window
per token at batch 512, i.e. ratio bf16 / LeapQuant >= 1.25, judged on the lower bound of the paired 95% interval.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from .stats import paired_ratio
from .util import sha256_file, utc_now, write_json

BATCH = re.compile(r"^\s*batch (\d+)\s*$")
LQ = re.compile(r"full cycle\s+LeapQuant: ([0-9.]+) ms per window of (\d+) tokens = ([0-9.]+) ms per token.*"
                r"boundary state advanced by the cycle: (True|False)")
SG = re.compile(r"full cycle\s+sglang window kernel, (fp32|bf16) state: ([0-9.]+) ms per token")


def parse(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    batch = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = BATCH.match(line)
        if m:
            batch = int(m.group(1))
            out.setdefault(batch, {})
            continue
        if batch is None:
            continue
        m = LQ.search(line)
        if m:
            out[batch].update(lq_ms_per_token=float(m.group(3)), window=int(m.group(2)),
                              state_advanced=m.group(4) == "True")
        m = SG.search(line)
        if m:
            out[batch][f"sglang_{m.group(1)}_window_ms_per_token"] = float(m.group(2))
    return out


def summarise(results: Path) -> dict:
    reps = sorted(results.glob("rep_*.txt"), key=lambda p: int(p.stem.split("_")[1]))
    per_rep = {int(p.stem.split("_")[1]): parse(p) for p in reps}
    batches = sorted({b for r in per_rep.values() for b in r})
    doc = {"schema": "lqh.kernel_cycle.v1", "computed_utc": utc_now(), "evidence_kind": "measured_raw",
           "method": "one CUDA graph per window: 16 state-advancing decode steps + fold, start state restored outside "
                     "the timing; SGLang window kernels the same way, same process; order alternated by repetition",
           "artifacts": {p.name: sha256_file(p) for p in reps + [results / "manifest.txt"] if p.exists()},
           "batches": {}}
    for batch in batches:
        rows = [(n, r.get(batch, {})) for n, r in sorted(per_rep.items())]
        complete = [(n, r) for n, r in rows if "lq_ms_per_token" in r and "sglang_bf16_window_ms_per_token" in r]
        entry = {"repetitions": len(complete), "state_advanced_every_rep": all(r.get("state_advanced") for _, r in complete),
                 "lq_ms_per_token": [r["lq_ms_per_token"] for _, r in complete],
                 "bf16_window_ms_per_token": [r["sglang_bf16_window_ms_per_token"] for _, r in complete],
                 "fp32_window_ms_per_token": [r.get("sglang_fp32_window_ms_per_token") for _, r in complete]}
        # Speed ratio bf16 / LeapQuant (> 1: LeapQuant faster), paired per repetition.
        entry["speed_vs_bf16_window"] = paired_ratio([r["lq_ms_per_token"] for _, r in complete],
                                                     [r["sglang_bf16_window_ms_per_token"] for _, r in complete])
        fp32 = [(r["lq_ms_per_token"], r.get("sglang_fp32_window_ms_per_token")) for _, r in complete
                if r.get("sglang_fp32_window_ms_per_token")]
        if fp32:
            entry["speed_vs_fp32_window"] = paired_ratio([a for a, _ in fp32], [b for _, b in fp32])
        if batch == 512:
            low = entry["speed_vs_bf16_window"].get("low")
            entry["gate_p5_G3"] = {"threshold": 1.25, "comparator": "lower 95% bound of bf16/LeapQuant >= 1.25",
                                   "observed_low": low,
                                   "status": "MET" if low is not None and low >= 1.25 else
                                   ("NOT_MET" if entry["speed_vs_bf16_window"].get("high") is not None
                                    and entry["speed_vs_bf16_window"]["high"] < 1.25 else "INCONCLUSIVE")}
        doc["batches"][str(batch)] = entry
    return doc


def main(argv=None) -> int:
    results = Path((argv or sys.argv[1:])[0])
    doc = summarise(results)
    write_json(results / "kernel_cycle.json", doc)
    for batch, entry in doc["batches"].items():
        ratio = entry["speed_vs_bf16_window"]
        lq = entry["lq_ms_per_token"]
        print(f"batch {batch:>4}: {entry['repetitions']} reps, LeapQuant median "
              f"{sorted(lq)[len(lq) // 2] if lq else float('nan'):.4f} ms/token, bf16 window / LeapQuant "
              f"{ratio.get('geometric_mean') or float('nan'):.4f} (95% {ratio.get('low') or float('nan'):.4f} .. "
              f"{ratio.get('high') or float('nan'):.4f}), state advanced: {entry['state_advanced_every_rep']}"
              + (f", gate p5.G3: {entry['gate_p5_G3']['status']}" if "gate_p5_G3" in entry else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
