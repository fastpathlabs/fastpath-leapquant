#!/usr/bin/env python3
"""
ablation_table.py

Tabulates the replays of plan 5, step S20 (precision ablations): one leapquant_replay output per
(setting, trace), named <setting>_<prompt>_trace_layer_<NN>.txt, as runbook / s20 writes them.

    python3 kernel/ablation_table.py results/s20_ablations

For every trace: the output error of LeapQuant under each setting, next to bf16 rounded every
window (SGLang's bf16 state with the window) from the same replay, and the bytes per entry.
"""

import glob
import os
import re
import sys

SETTINGS = [("base", "window 16, 4 compensators, 8 bits"), ("w8", "window 8"),
            ("b6", "6 bits, 4 compensators, one step per value column"),
            ("b4", "4 bits, 4 compensators, one step per value column"),
            ("r2", "2 compensators"), ("r8", "8 compensators"),
            ("b6p", "6 bits: 6 compensators (paper), one step per 8 key rows (ours)"),
            ("b4p", "4 bits: 6 compensators (paper), one step per 8 key rows (ours)"),
            ("r0", "0 compensators"), ("r0rot", "0 compensators, Hadamard-rotated keys and state"),
            ("r2rot", "2 compensators, rotated"), ("r4rot", "4 compensators, rotated")]


def parse(path):
    text = open(path, encoding="utf-8").read()
    lq = re.search(r"\(= LeapQuant\)\s+([\d.]+) \|.*\|\s*([\d.]+)%\s*$", text, flags=re.M)
    bf = re.search(r"bf16, rounded every window\s+([\d.]+) \|.*\|\s*([\d.]+)%\s*$", text, flags=re.M)
    ok = "(ok)" in text
    if not lq or not bf:
        return None
    return {"bytes": float(lq.group(1)), "error": float(lq.group(2)), "bf16": float(bf.group(2)), "ok": ok}


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "results/s20_ablations"
    results = {}
    for path in glob.glob(os.path.join(folder, "*.txt")):
        name = os.path.basename(path)[:-4]
        setting, _, trace = name.partition("_")
        parsed = parse(path)
        if parsed:
            results.setdefault(trace, {})[setting] = parsed
    if not results:
        sys.exit(f"no replay outputs in {folder}")
    names = [key for key, _ in SETTINGS]
    print("Output error against fp32, % (all heads and tokens). bf16 = bf16 state rounded every window.")
    print(f"{'trace':<28}" + "".join(f"{key:>9}" for key in names) + f"{'bf16':>9}")
    wins = {key: 0 for key in names}
    for trace in sorted(results):
        row = results[trace]
        bf16 = next(iter(row.values()))["bf16"]
        cells = []
        for key in names:
            if key in row:
                cells.append(f"{row[key]['error']:>9.3f}")
                wins[key] += row[key]["error"] <= bf16
            else:
                cells.append(f"{'-':>9}")
        print(f"{trace:<28}" + "".join(cells) + f"{bf16:>9.3f}")
    print(f"{'bytes per entry':<28}" + "".join(
        f"{next((r[key]['bytes'] for r in results.values() if key in r), float('nan')):>9.2f}" for key in names) + f"{2.0:>9.2f}")
    print(f"{'traces at least as good as bf16':<28}" + "".join(f"{wins[key]:>9}" for key in names))
    print("\nSettings: " + "; ".join(f"{key} = {text}" for key, text in SETTINGS))
    bad = [trace for trace, row in results.items() if not all(value["ok"] for value in row.values())]
    if bad:
        print(f"replay self-check NOT ok on: {', '.join(bad)}")


if __name__ == "__main__":
    main()
