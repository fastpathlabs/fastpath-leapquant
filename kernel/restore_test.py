#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
restore_test.py

State-cache plan v2, task TC.3 (backlog B32): a restore must give the same OUTPUTS, not only the same stored
bytes, in fresh buffers and in buffers that lie elsewhere (the paper behind that plan found logits that changed
after a restore because another buffer layout chose other kernels).

A request is prefilled and decoded for two windows; at the window boundary its LeapQuant state is saved to the
host (EngineState.get_cpu_slots). The uninterrupted run continues for 64 tokens. Then the saved state is restored
    A. into a fresh engine, another slot, the request alone in its batch;
    B. into an engine created after other allocations (its tensors lie elsewhere), another slot, at another
       position in a batch with two other requests and a padded row;
and the same 64 tokens are decoded. Every output of every layer must equal the uninterrupted run's bit for bit.
Heads are 128 x 128, so with LEAPQUANT_FOLD_BACKEND=cuda the CUDA fold is exercised (two folds in the 64 tokens).

    python3 kernel/restore_test.py        (GPU; LEAPQUANT_FOLD_BACKEND=cuda to cover the CUDA fold)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_engine import EngineState, fast_kernels_available  # noqa: E402

LAYERS, SLOTS, HV, H, K, V, L = 2, 9, 4, 2, 128, 128, 16
WARMUP_STEPS, STEPS = 32, 64
failures = []


def check(condition, text):
    print(f"  {'pass' if condition else 'FAIL'}  {text}")
    if not condition:
        failures.append(text)


def main():
    device = torch.device("cuda")
    if not fast_kernels_available(device):
        sys.exit("restore_test.py needs a CUDA device and triton")
    generator = torch.Generator().manual_seed(11)
    A_log = [(0.3 * torch.randn(HV, generator=generator) - 4.0).to(device) for _ in range(LAYERS)]
    dt_bias = [(0.1 * torch.randn(HV, generator=generator)).to(device) for _ in range(LAYERS)]
    scale = K ** -0.5

    def stream(steps, seed):
        g = torch.Generator().manual_seed(seed)
        return [[(torch.randn((1, 2 * H * K + HV * V), generator=g).to(device=device, dtype=torch.bfloat16),
                  (0.5 * torch.randn((1, HV), generator=g)).to(device=device, dtype=torch.bfloat16),
                  torch.randn((1, HV), generator=g).to(device=device, dtype=torch.bfloat16))
                 for _ in range(LAYERS)] for _ in range(steps)]

    def new_engine():
        cursor = torch.zeros(SLOTS, dtype=torch.int32, device=device)
        engine = EngineState(LAYERS, SLOTS, HV, H, K, V, max_rows=8, window_length=L, device=device, cursor=cursor,
                             padding_slot=0, fast=True, compressed_only=True)
        return engine, cursor

    def prefill(engine, slot, seed):
        """A random full-precision state stands in for the prefill kernel's result."""
        g = torch.Generator().manual_seed(seed)
        ci = torch.tensor([slot], dtype=torch.int64, device=device)
        engine.reset_slots(ci)
        for layer in range(LAYERS):
            staging, rows = engine.extend_begin(layer, ci)
            state = 0.05 * torch.randn((HV, V, K), generator=g)
            for p in range(3):
                state += torch.randn((HV, V, 1), generator=g) * torch.randn((HV, 1, K), generator=g) / (p + 1)
            staging[int(rows[0])] = state.to(device)
            engine.extend_end(layer, ci, staging, rows)

    def decode(engine, cursor, batch_slots, inputs_of, steps):
        """Aligned decode as the SGLang patch does it. inputs_of[slot][step][layer] = (mixed_qkv, a, b) for one row.
        Returns {slot: [step][layer] output} for the non-padded rows."""
        slots = torch.tensor(batch_slots, dtype=torch.int32, device=device)
        live = [b for b, s in enumerate(batch_slots) if s != 0]
        outputs = {batch_slots[b]: [] for b in live}
        for step in range(steps):
            write_pos = cursor[slots.long()].clone()
            any_full = bool((write_pos[torch.tensor(live, device=device)] == L - 1).any())
            force = torch.full((len(batch_slots),), 1 if any_full else 0, dtype=torch.int32, device=device)
            per_layer = []
            for layer in range(LAYERS):
                rows = [inputs_of[s][step][layer] if s != 0 else inputs_of["pad"][step][layer] for s in batch_slots]
                mixed_qkv = torch.cat([r[0] for r in rows])
                a = torch.cat([r[1] for r in rows])
                b = torch.cat([r[2] for r in rows])
                per_layer.append(engine.decode(layer, mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale, None, slots,
                                               write_pos, force).clone())
            for b in live:
                outputs[batch_slots[b]].append([per_layer[layer][b] for layer in range(LAYERS)])
            flushed = (write_pos + 1 >= L) | (force != 0)
            next_pos = torch.where(flushed, torch.zeros_like(write_pos), write_pos + 1)
            for b in live:
                cursor[batch_slots[b]] = next_pos[b]
        return outputs

    warmup = stream(WARMUP_STEPS, 101)
    tail = stream(STEPS, 102)
    pad = stream(WARMUP_STEPS + STEPS, 199)

    # ---- the uninterrupted request (slot 3) ------------------------------------------------------------
    engine, cursor = new_engine()
    prefill(engine, 3, 7)
    decode(engine, cursor, [3], {3: warmup, "pad": pad}, WARMUP_STEPS)
    check(int(cursor[3]) == 0, f"after {WARMUP_STEPS} tokens the request sits at a window boundary")
    saved = engine.get_cpu_slots(torch.tensor([3], device=device))
    reference = decode(engine, cursor, [3], {3: tail, "pad": pad}, STEPS)[3]

    def same(outputs, label):
        worst = 0.0
        equal = True
        for step in range(STEPS):
            for layer in range(LAYERS):
                mine, theirs = outputs[step][layer], reference[step][layer]
                equal = equal and torch.equal(mine, theirs)
                worst = max(worst, float((mine - theirs).abs().max()))
        check(equal, f"{label}: the next {STEPS} tokens' outputs are bit-identical (largest difference {worst:.1e})")

    # ---- A. fresh engine, another slot, alone -----------------------------------------------------------
    fresh, fresh_cursor = new_engine()
    fresh.load_cpu_slots(saved, torch.tensor([5], device=device))
    fresh_cursor[5] = 0
    same(decode(fresh, fresh_cursor, [5], {5: tail, "pad": pad}, STEPS)[5], "A. restored into a fresh engine, slot 5")

    # ---- B. moved buffers, another slot and batch position, other requests beside it ---------------------
    ballast = [torch.empty(int(size), dtype=torch.uint8, device=device) for size in (3 << 20, 7 << 20, 1 << 20)]
    moved, moved_cursor = new_engine()
    check(moved.pools[0].rounded_residual.data_ptr() != fresh.pools[0].rounded_residual.data_ptr()
          and moved.pools[0].rounded_residual.data_ptr() != engine.pools[0].rounded_residual.data_ptr(),
          "B's pool lies at another address")
    prefill(moved, 1, 21)
    prefill(moved, 2, 22)
    moved.load_cpu_slots(saved, torch.tensor([7], device=device))
    moved_cursor[1] = moved_cursor[2] = moved_cursor[7] = 0
    others = {1: stream(STEPS, 301), 2: stream(STEPS, 302), 7: tail, "pad": pad}
    same(decode(moved, moved_cursor, [1, 7, 0, 2], others, STEPS)[7],
         "B. restored into moved buffers, slot 7, batch row 2 of 4 with two other requests and a padded row")
    del ballast
    print(f"\n{'PASS' if not failures else f'FAIL ({len(failures)} checks failed)'}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
