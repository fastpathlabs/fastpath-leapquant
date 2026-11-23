#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
profile_fast_fold.py

One fold of a full window at a given batch (default 512 rows x 32 heads), for Nsight Compute:

    sudo -E ncu --kernel-name regex:fast_fold_kernel --launch-count 1 --set full \
        python3 kernel/profile_fast_fold.py --backend cuda

The pool is filled the way a server fills it: a stored state, then 16 decode steps (records). Warm-up
folds run first on a copy, so the profiled launch is not the first one (no JIT or library loading in it).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leapquant_triton as fast  # noqa: E402
from leapquant_kernel import StatePool  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--backend", choices=["triton", "cuda"], default="cuda")
    parser.add_argument("--compensators", type=int, default=4)
    args = parser.parse_args()
    fast.FOLD_BACKEND = args.backend
    device = torch.device("cuda")
    B, HV, H, K, V, L = args.batch, 32, 16, 128, 128, 16
    g = torch.Generator().manual_seed(0)
    pool = StatePool(B + 1, HV, K, V, window_length=L, compensator_count=args.compensators, bits_per_entry=8,
                     power_iteration_sweeps=2, device=device, scale_group=0)
    slot = torch.arange(1, B + 1, dtype=torch.int32, device=device)
    state = torch.zeros(B + 1, HV, V, K, device=device)
    state[1:] = (0.05 * torch.randn(B, HV, V, K, generator=g)).to(device)
    for p in range(3):
        state[1:] += (torch.randn(B, HV, V, 1, generator=g) * torch.randn(B, HV, 1, K, generator=g)).to(device) / (p + 1)
    fast.store(pool, slot, state)
    del state
    mixed = torch.randn(B, 2 * H * K + HV * V, generator=g).to(device, torch.bfloat16)
    a = (0.5 * torch.randn(B, HV, generator=g)).to(device, torch.bfloat16)
    b = torch.randn(B, HV, generator=g).to(device, torch.bfloat16)
    A_log = (0.3 * torch.randn(HV, generator=g) - 4).to(device)
    dt_bias = (0.1 * torch.randn(HV, generator=g)).to(device)
    out = torch.empty(B, HV, V, device=device)
    write_pos = torch.zeros(B, dtype=torch.int32, device=device)
    for position in range(L):
        write_pos.fill_(position)
        fast.decode_step_raw(pool, slot, write_pos, mixed, a, b, A_log, dt_bias, K ** -0.5, H, out)
    full = torch.full((B,), L, dtype=torch.int32, device=device)
    saved = {name: getattr(pool, name).clone() for name in StatePool.BOUNDARY_FIELDS}
    for _ in range(3):                          # warm-up
        fast.fold(pool, slot, full)
        for name, tensor in saved.items():
            getattr(pool, name).copy_(tensor)
    torch.cuda.synchronize()
    fast.fold(pool, slot, full)                 # the profiled fold
    torch.cuda.synchronize()
    print("done")


if __name__ == "__main__":
    main()
