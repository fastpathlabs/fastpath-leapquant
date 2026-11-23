#!/usr/bin/env python3
"""
profile_summary.py

Summarises torch-profiler traces (Chrome trace JSON, optionally .gz) written by SGLang's /start_profile: the profiled
span, the GPU's busy time and idle time within it, and the kernels and CPU operations that take the most time.
With two traces it prints them side by side (the second against the first).

    python3 eval/profile_summary.py TRACE [TRACE2] [--top 25]
"""

import argparse
import collections
import gzip
import json
import re


def load(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["traceEvents"] if isinstance(data, dict) else data


def short(name, width=70):
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"<.*>", "<...>", name) if len(name) > width else name
    return name[:width]


def summarise(path, top):
    events = [e for e in load(path) if e.get("ph") == "X" and "dur" in e]
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu_ops = [e for e in events if e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    if not kernels:
        return {"path": path, "error": "no GPU events"}
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    # GPU busy time: union of kernel intervals (all streams).
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    busy, current_start, current_end = 0.0, None, None
    for a, b in intervals:
        if current_end is None or a > current_end:
            if current_end is not None:
                busy += current_end - current_start
            current_start, current_end = a, b
        else:
            current_end = max(current_end, b)
    busy += current_end - current_start
    by_kernel = collections.Counter()
    count = collections.Counter()
    for e in kernels:
        by_kernel[short(e["name"])] += e["dur"]
        count[short(e["name"])] += 1
    by_cpu = collections.Counter()
    for e in cpu_ops:
        by_cpu[short(e["name"])] += e["dur"]
    return {"path": path, "span_ms": (end - start) / 1000, "busy_ms": busy / 1000, "kernels": len(kernels),
            "by_kernel": by_kernel, "count": count, "by_cpu": by_cpu}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    results = [summarise(path, args.top) for path in args.traces]
    for r in results:
        if "error" in r:
            print(r["path"], r["error"])
            continue
        print(f"{r['path']}\n  span {r['span_ms']:.1f} ms, GPU busy {r['busy_ms']:.1f} ms "
              f"({100 * r['busy_ms'] / r['span_ms']:.0f}%), idle {r['span_ms'] - r['busy_ms']:.1f} ms, "
              f"{r['kernels']} GPU events")
    good = [r for r in results if "error" not in r]
    if not good:
        return
    names = collections.Counter()
    for r in good:
        names.update(r["by_kernel"])
    print("\nGPU time by kernel, ms (count):")
    for name, _ in names.most_common(args.top):
        cells = "  ".join(f"{r['by_kernel'].get(name, 0) / 1000:9.2f} ({r['count'].get(name, 0):5d})" for r in good)
        print(f"  {cells}  {name}")
    cpu = collections.Counter()
    for r in good:
        cpu.update(r["by_cpu"])
    print("\nCPU time by operation, ms:")
    for name, _ in cpu.most_common(args.top):
        cells = "  ".join(f"{r['by_cpu'].get(name, 0) / 1000:9.2f}" for r in good)
        print(f"  {cells}  {name}")


if __name__ == "__main__":
    main()
