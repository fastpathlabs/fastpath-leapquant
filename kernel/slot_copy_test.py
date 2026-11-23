#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
slot_copy_test.py   (needs a CUDA GPU and triton)

EngineState.copy_slots with the fused slot copy (leapquant_triton.SlotRows, LEAPQUANT_FUSED_SLOT_COPY=1, the default)
against the same copies done field by field on clones: every boundary tensor of every layer, the slot flags and the
cursor must come out bit-identical, and no other slot may change. One pair (the fused launch), several pairs (the
gather-then-scatter path), a slot onto itself, CPU and GPU indices in int32 and int64, layers with different
compensator counts, and the switch off.

    python3 kernel/slot_copy_test.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_engine import EngineState            # noqa: E402
from leapquant_kernel import StatePool              # noqa: E402

SLOTS, LAYERS, HEADS = 9, 3, 4


def build(fused):
    os.environ["LEAPQUANT_FUSED_SLOT_COPY"] = "1" if fused else "0"
    cursor = torch.zeros(SLOTS, dtype=torch.int32, device="cuda")
    state = EngineState(layers=LAYERS, slots=SLOTS, value_heads=HEADS, key_heads=2, key_channels=128,
                        value_channels=128, max_rows=SLOTS, compensator_count=[4, 8, 4], device="cuda",
                        cursor=cursor, padding_slot=0, fast=True, compressed_only=True)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    for pool in state.pools:
        for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS:
            tensor = getattr(pool, name)
            if tensor.dtype == torch.int8:
                tensor.copy_(torch.randint(-127, 128, tensor.shape, generator=generator, device="cuda",
                                           dtype=torch.int32).to(torch.int8))
            else:
                tensor.copy_(torch.randn(tensor.shape, generator=generator, device="cuda").to(tensor.dtype))
    state.valid.copy_(torch.randint(0, 2, state.valid.shape, generator=generator, device="cuda", dtype=torch.int32))
    cursor.copy_(torch.randint(0, 16, cursor.shape, generator=generator, device="cuda", dtype=torch.int32))
    return state


def snapshot(state):
    saved = {"valid": state.valid.clone(), "cursor": state.cursor.clone()}
    for index, pool in enumerate(state.pools):
        for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS:
            saved[(index, name)] = getattr(pool, name).clone()
    return saved


def expected_after(saved, src, dst):
    out = {key: value.clone() for key, value in saved.items()}
    src_t, dst_t = torch.tensor(src, device="cuda"), torch.tensor(dst, device="cuda")
    for key in out:
        if isinstance(key, tuple) and key[1] in StatePool.BOUNDARY_FIELDS:
            out[key][dst_t] = saved[key][src_t]
    out["valid"][:, dst_t] = saved["valid"][:, src_t]
    out["cursor"][dst_t] = 0
    return out


def compare(state, expected, what):
    now = snapshot(state)
    bad = [str(key) for key in expected if not torch.equal(now[key], expected[key])]
    print(f"{'PASS' if not bad else 'FAIL'}  {what}" + (f"  differs: {bad}" if bad else ""))
    return not bad


def main():
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    ok = True
    for fused in (True, False):
        state = build(fused)
        label = "fused" if fused else "field by field"
        if fused and state._slot_rows is None:
            print("FAIL  the fused copy is not in use")
            ok = False
        cases = [
            ([3], [5], "one pair, CPU int64", lambda v: torch.tensor(v)),
            ([7], [2], "one pair, GPU int32", lambda v: torch.tensor(v, dtype=torch.int32, device="cuda")),
            ([4], [4], "a slot onto itself, GPU int64", lambda v: torch.tensor(v, device="cuda")),
            ([1, 2], [6, 8], "two pairs, GPU int64", lambda v: torch.tensor(v, device="cuda")),
            ([6, 8, 1], [3, 1, 5], "three pairs, overlapping", lambda v: torch.tensor(v, device="cuda")),
        ]
        for src, dst, what, make in cases:
            expected = expected_after(snapshot(state), src, dst)
            state.copy_slots(make(src), make(dst))
            torch.cuda.synchronize()
            ok &= compare(state, expected, f"{label}: {what}")
    print("SLOT COPY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
