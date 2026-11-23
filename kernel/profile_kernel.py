#!/usr/bin/env python3
"""
profile_kernel.py

Says where the time of the LeapQuant kernel goes, phase by phase.

    python3 kernel/profile_kernel.py --device cuda --batch 128
    python3 kernel/profile_kernel.py --device cuda --batch 512 --sweeps 2 --label "2 sweeps"

How it measures
    The kernel can be told to stop at a named point of its work
    (LqPool.profile_stop_phase). The script times the kernel stopped at each
    point in turn. The time of a phase is the difference between two
    neighbouring stops.

    Rules that make the numbers trustworthy, each learned from a measurement
    that went wrong without it:
      - The pool is put back to the same realistic state before EVERY timed
        call (a fold changes the state; a second fold of a folded state does
        different work).
      - Only the kernel is timed, with GPU-side timers (CUDA events) on a GPU.
        Restoring the state is outside the timed span.
      - The stops are measured round-robin, not one after the other, so that a
        slow minute on the machine hits all of them alike.
      - Every number is a median over `--repeats` calls, printed with the
        spread between the 10% and the 90% quantile. A difference smaller
        than the spread is not a finding.
      - The script prints what else is running on the GPU. Do not trust a
        profile taken while another job uses it.

Writes one JSON line per run to --log (default build/profile_log.jsonl), so
that successive versions of the kernel can be compared.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_kernel import Kernel, StatePool  # noqa: E402

DECODE_STOPS = [(11, "decays of the window records"), (12, "fade the two input vectors"),
                (13, "key matches (records, compensators)"), (0, "int8 matrix, correction, output")]
FOLD_STOPS = [(1, "rebuild the full-precision matrix"), (2, "compensator 1: starting guess"),
              (21, "  sweep 1: key_side = M * value_side"), (22, "  sweep 1: length of key_side (1 thread)"),
              (23, "  sweep 1: normalise key_side"), (24, "  sweep 1: value_side = M^T * key_side"),
              (25, "  sweep 1: length of value_side (1 thread)"),
              (3, "compensator 1: the other sweeps"), (4, "compensator 1: store and subtract"),
              (5, "compensators 2 .. R"), (6, "row scales (smoothing)"), (0, "column steps and rounding")]


class Timer:
    """Times one call: CUDA events on a GPU, the wall clock on a CPU."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.device = device

    def milliseconds(self, function):
        if self.cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            function()
            end.record()
            torch.cuda.synchronize(self.device)
            return start.elapsed_time(end)
        start = time.perf_counter()
        function()
        return 1000.0 * (time.perf_counter() - start)


def summarise(samples):
    ordered = sorted(samples)
    pick = lambda share: ordered[min(len(ordered) - 1, int(share * len(ordered)))]
    return pick(0.5), pick(0.1), pick(0.9)


def other_gpu_users():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
        return [line for line in out.splitlines() if line.strip() and str(os.getpid()) not in line.split(",")[0]]
    except Exception:
        return []


def main():
    parser = argparse.ArgumentParser(description="Phase-by-phase timing of the LeapQuant kernel.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--compensators", type=int, default=4)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--fit", choices=["together", "sequential"], default="together",
                        help="how the compensators are fitted (Settings::fit_compensators_together)")
    parser.add_argument("--label", default="")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()
    device = torch.device(args.device)
    B, HV, H, K, V, L = args.batch, args.hv, args.h, args.k, args.v, args.window
    generator = torch.Generator().manual_seed(0)
    timer = Timer(device)
    kernel = Kernel(device)

    def random(*shape):
        return torch.randn(shape, generator=generator).to(device)

    # ---- a realistic pool: adopted from structured states, then a full window of real records ----
    pool = StatePool(B + 1, HV, K, V, window_length=L, compensator_count=args.compensators,
                     power_iteration_sweeps=args.sweeps, device=device,
                     fit_together=args.fit == "together")
    slot = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    state = 0.05 * random(B, HV, V, K)
    for pattern in range(3):
        state += random(B, HV, V, 1) * random(B, HV, 1, K) / (pattern + 1)
    kernel.store(pool, slot, state)
    decay = (1.0 - 10.0 ** (-3.0 + 1.7 * torch.rand((B, HV), generator=generator))).to(device)
    beta = (0.2 + 0.8 * torch.rand((B, HV), generator=generator)).to(device)
    key = torch.nn.functional.normalize(random(B, H, K), dim=-1).contiguous()
    query = (torch.nn.functional.normalize(random(B, H, K), dim=-1) * K ** -0.5).contiguous()
    value = random(B, HV, V)
    out = torch.empty((B, HV, V), dtype=torch.float32, device=device)
    scratch = torch.empty((B, HV, V, K), dtype=torch.float32, device=device)
    cursor = torch.zeros(B, dtype=torch.int32, device=device)
    for position in range(L - 1):
        cursor.fill_(position)
        kernel.decode_step(pool, slot, cursor, decay, beta, key, query, value, out)
    cursor.fill_(L - 1)       # the window holds L-1 records: the next token fills it, the fold then folds L records

    fields = StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS
    before_decode = {name: getattr(pool, name).clone() for name in fields}
    kernel.decode_step(pool, slot, cursor, decay, beta, key, query, value, out)
    before_fold = {name: getattr(pool, name).clone() for name in fields}

    def restore(snapshot):
        for name in fields:
            getattr(pool, name).copy_(snapshot[name])

    def decode_call():
        kernel.decode_step(pool, slot, cursor, decay, beta, key, query, value, out)

    def fold_call():
        kernel.fold(pool, slot, cursor, tokens_added=1, scratch=scratch)

    def measure(stops, snapshot, call):
        samples = {stop: [] for stop, _ in stops}
        for repeat in range(args.repeats + 3):
            for stop, _ in stops:
                restore(snapshot)
                pool.profile_stop_phase = stop
                elapsed = timer.milliseconds(call)
                if repeat >= 3:                      # the first rounds warm up
                    samples[stop].append(elapsed)
        pool.profile_stop_phase = 0
        return samples

    decode_samples = measure(DECODE_STOPS, before_decode, decode_call)
    fold_samples = measure(FOLD_STOPS, before_fold, fold_call)
    # In a server the requests do not fold on the same step unless the engine aligns them:
    # each step, about 1 row in L folds. How long does such a fold launch take?
    some = torch.zeros(B, dtype=torch.int32, device=device)
    some[: max(1, B // L)] = L - 1
    few = []
    for _ in range(args.repeats):
        restore(before_fold)
        few.append(timer.milliseconds(lambda: kernel.fold(pool, slot, some, tokens_added=1, scratch=scratch)))
    restore(before_fold)
    empty = [timer.milliseconds(lambda: kernel.fold(pool, slot, torch.zeros_like(cursor), tokens_added=1, scratch=scratch))
             for _ in range(args.repeats)]

    # ---- report -------------------------------------------------------------------------------
    print(f"LeapQuant kernel profile on {device}: batch {B}, {HV} heads, state {K} x {V}, window {L}, "
          f"{args.compensators} compensators, {args.sweeps} sweeps, fit {args.fit}   {args.label}")
    print(f"  kernel library: {kernel.library_file}")
    others = other_gpu_users() if device.type == "cuda" else []
    if others:
        print(f"  WARNING: other processes are using this GPU ({'; '.join(others)}). These numbers are not clean.")
    record = {"label": args.label, "batch": B, "sweeps": args.sweeps, "compensators": args.compensators,
              "library": kernel.library_file, "other_gpu_users": len(others), "decode": {}, "fold": {}}

    def table(title, stops, samples, key_name):
        print(f"\n  {title}")
        print(f"    {'phase':<40} {'ms':>8} {'share':>6}   {'cumulative ms (10% .. 90%)':>28}")
        total = summarise(samples[0])[0]
        previous = 0.0
        for stop, name in stops:
            median, low, high = summarise(samples[stop])
            phase = median - previous
            print(f"    {name:<40} {phase:8.3f} {100 * phase / total:5.1f}%   {median:10.3f} ({low:.3f} .. {high:.3f})")
            record[key_name][name] = round(phase, 4)
            previous = median
        record[key_name]["total"] = round(total, 4)
        return total

    decode_total = table("decode step (append only)", DECODE_STOPS, decode_samples, "decode")
    fold_total = table(f"fold of a full window (all {B} rows fold)", FOLD_STOPS, fold_samples, "fold")
    empty_median = summarise(empty)[0]
    per_step = decode_total + (fold_total + (L - 1) * empty_median) / L
    record.update({"fold_launch_no_rows": round(empty_median, 4), "per_step": round(per_step, 4)})
    few_median = summarise(few)[0]
    unaligned = decode_total + few_median
    record.update({"fold_one_in_L_rows": round(few_median, 4), "per_step_unaligned": round(unaligned, 4)})
    print(f"\n  fold launch in which {max(1, B // L)} of {B} rows fold: {few_median:.3f} ms   "
          f"(a full fold divided by {L} would be {fold_total / L:.3f} ms)")
    print(f"  -> per decode step if requests fold on different steps: {decode_total:.3f} + {few_median:.3f} = {unaligned:.3f} ms")
    print(f"  fold launch in which no row folds: {empty_median:.3f} ms")
    print(f"  per decode step if all requests fold on the same step: {decode_total:.3f} + {fold_total:.3f} / {L} "
          f"+ launches = {per_step:.3f} ms")

    log = args.log or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "profile_log.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
