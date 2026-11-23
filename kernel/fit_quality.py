#!/usr/bin/env python3
"""
fit_quality.py

Step S19 of plan 5, the "best-possible check": on real states from a trace, how much
of what four compensators COULD capture does our fit capture?

    python3 kernel/fit_quality.py traces/reasoning/trace_layer_30.bin
    python3 kernel/fit_quality.py traces/*/trace_layer_*.bin --every 512 --sweeps 2

It answers the question plan 5 asks about layer 30. LeapQuant loses there to a
bf16 state. Either
  (a) the state has no few dominant patterns, so no four compensators can help much, or
  (b) there are dominant patterns and our fit misses them.
The fix differs: more compensators or another format for (a), a better fit for (b).

For sampled fp32 states (the exact state of the model, replayed from the trace):
    top-4 share   energy in the best four patterns / energy of the state, from an
                  exact singular value decomposition. Low = case (a).
    captured      energy our four compensators remove / energy the best four would
                  remove. Low = case (b). 100% = the fit is as good as it can be.
    top-8 share   the same for eight patterns: what rank 8 could buy.

Runs on a CPU. Needs numpy and build/libleapquant_reference (make kernel).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_kernel import ReferenceLayer, Trace, load_reference_library  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Captured energy of the compensator fit against the exact optimum.")
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--every", type=int, default=512, help="sample the state every N tokens")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=8, help="power-iteration sweeps of the fit under test")
    parser.add_argument("--compensators", type=int, default=4)
    args = parser.parse_args()
    library = load_reference_library()
    rank = args.compensators

    print(f"fit under test: {rank} compensators, one at a time, {args.sweeps} sweeps, start = longest row")
    print(f"{'trace':<34} {'samples':>7} | {'top-' + str(rank) + ' share':>11} {'top-' + str(2 * rank) + ' share':>11} | "
          f"{'captured: median':>16} {'worst head':>11} {'heads < 90%':>12}")
    for path in args.traces:
        trace = Trace(path, args.max_tokens)
        # One layer of fp32 truth heads replays the trace; a second, never stepped, is only used to fit.
        truth = ReferenceLayer(library, trace, 16, rank, 8, args.sweeps, True)
        fitter = ReferenceLayer(library, trace, 16, rank, 8, args.sweeps, True)
        if trace.initial_state is not None:
            truth.load_state(trace.initial_state, which=2)
        share_low, share_high, captured = [], [], []
        for token in range(trace.tokens):
            truth.step(*trace.token(token))
            if (token + 1) % args.every != 0:
                continue
            states = truth.state(which=2).astype(np.float64)                      # [H, K, V]
            singular = np.linalg.svd(states, compute_uv=False)                     # [H, min(K, V)]
            energy = (singular ** 2).sum(axis=1)
            best = (singular[:, :rank] ** 2).sum(axis=1)
            share_low.append(best / energy)
            share_high.append((singular[:, :2 * rank] ** 2).sum(axis=1) / energy)
            # Our fit on the same states: the compensators as stored (fp16).
            fitter.load_state(states.astype(np.float32), which=1)
            boundary = fitter.boundary_state()
            keys = boundary["compensator_keys"].float().numpy().astype(np.float64)          # [H, R, K]
            corrections = boundary["compensator_corrections"].float().numpy().astype(np.float64)   # [H, R, V]
            removed = np.einsum("hrk,hrv->hkv", keys, corrections)
            left = ((states - removed) ** 2).sum(axis=(1, 2))
            captured.append((energy - left) / np.maximum(best, 1e-300))
        if not captured:
            print(f"{os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path):<34} too short for --every {args.every}")
            continue
        share_low, share_high, captured = np.concatenate(share_low), np.concatenate(share_high), np.concatenate(captured)
        name = os.path.basename(os.path.dirname(os.path.abspath(path))) + "/" + os.path.basename(path)
        print(f"{name:<34} {len(captured):>7} | {100 * np.median(share_low):>10.1f}% {100 * np.median(share_high):>10.1f}% | "
              f"{100 * np.median(captured):>15.1f}% {100 * captured.min():>10.1f}% {100 * (captured < 0.9).mean():>11.1f}%")
    print("\ntop-N share low (say under 50%): little for compensators to capture, whatever the fit.")
    print("captured well under 100%: the fit misses patterns that are there.")


if __name__ == "__main__":
    main()
