#!/usr/bin/env python3
"""
compressed_test.py

Tests the compressed-only mode of leapquant_engine.EngineState (integration phase 2):
no fp32 state per slot; prefill runs on a staging tensor and is compressed at its end;
decode reads and writes only the LeapQuant pool. Needs a GPU (the fast kernels).

The test engine does what SGLang does with --disable-radix-cache:
    - a new request's slot is reset (clear_slots -> reset_slots), then prefilled
    - a long prompt is prefilled in two chunks; the second chunk continues from the first
    - decode steps advance one cursor per slot; when any window is full, all fold (as the
      SGLang patch aligns them)
    - batches carry padded rows (slot 0)

Checked
    1. A new request's staging row is zero; a padded row maps to staging row 0.
    2. After a prefill chunk the stored state is the prefilled fp32 state, up to the
       rounding of one store; the next chunk starts from that state.
    3. Decode outputs stay close to an fp32 engine that ran the same tokens (limit 2%).
    4. Padded rows give zeros; slots are marked as stored.
    5. A prefix-cache hit (copy_slots, as SGLang's MambaPool.copy_from calls it) copies the
       boundary state: the copy's prefill starts from the source's state, and a new request
       in a cleared slot still starts from zero.

Run:   python3 kernel/compressed_test.py          (after `make kernel`, on a GPU machine)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_engine import EngineState, fast_kernels_available  # noqa: E402
from leapquant_kernel import StatePool  # noqa: E402

LAYERS, SLOTS, HV, H, K, V, L = 2, 9, 4, 2, 32, 48, 16
DECODE_STEPS = 60
failures = []


def check(condition, text):
    if not condition:
        failures.append(text)
        print(f"    FAILED: {text}")


def fp32_step(state, decay, beta, key, query, value):
    """One Gated DeltaNet token in fp32, state [HV, V, K]; key, query [H, K]; decay, beta [HV]."""
    key_per_head = key.repeat_interleave(HV // H, dim=0)
    query_per_head = query.repeat_interleave(HV // H, dim=0)
    alpha = decay[:, None]
    held = torch.einsum("hvk,hk->hv", state, key_per_head) * alpha
    correction = beta[:, None] * (value - held)
    output = torch.einsum("hvk,hk->hv", state, query_per_head) * alpha + \
        correction * (key_per_head * query_per_head).sum(dim=-1, keepdim=True)
    return output, state * alpha[:, :, None] + correction[:, :, None] * key_per_head[:, None, :]


def prefill_tokens(state, tokens, generator, device):
    """Runs `tokens` random tokens through the fp32 recurrence (stands in for SGLang's prefill kernel)."""
    for _ in range(tokens):
        key = torch.nn.functional.normalize(torch.randn((H, K), generator=generator), dim=-1).to(device)
        query = torch.nn.functional.normalize(torch.randn((H, K), generator=generator), dim=-1).to(device) * K ** -0.5
        value = torch.randn((HV, V), generator=generator).to(device)
        decay = (0.9 + 0.1 * torch.rand(HV, generator=generator)).to(device)
        beta = torch.rand(HV, generator=generator).to(device)
        _, state = fp32_step(state, decay, beta, key, query, value)
    return state


def relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def main():
    device = torch.device("cuda")
    if not fast_kernels_available(device):
        sys.exit("compressed_test.py needs a CUDA device and triton")
    generator = torch.Generator().manual_seed(7)
    cursor = torch.zeros(SLOTS, dtype=torch.int32, device=device)
    engine = EngineState(LAYERS, SLOTS, HV, H, K, V, max_rows=8, window_length=L, device=device, cursor=cursor,
                         padding_slot=0, fast=True, compressed_only=True)
    check(engine.scratch is None, "no fp32 working memory is allocated")
    A_log = [(0.3 * torch.randn(HV, generator=generator) - 4.0).to(device) for _ in range(LAYERS)]
    dt_bias = [(0.1 * torch.randn(HV, generator=generator)).to(device) for _ in range(LAYERS)]
    scale = K ** -0.5
    truth = {}                                  # slot -> [fp32 state per layer]

    def prefill(cache_indices, tokens_of_row, generator):
        """One prefill batch of every layer. tokens_of_row[b]: tokens of row b (0 for a padded row)."""
        ci = torch.tensor(cache_indices, dtype=torch.int64, device=device)
        new_slots = {slot for slot in cache_indices if slot != 0 and slot not in truth}
        worst_start, worst_store = 0.0, 0.0
        for layer in range(LAYERS):
            staging, rows = engine.extend_begin(layer, ci)
            check(staging.shape[0] == len(cache_indices) + 1, "one staging row per batch row, plus the padding row")
            for b, slot in enumerate(cache_indices):
                row = int(rows[b])
                if slot == 0:
                    check(row == 0, "a padded row uses staging row 0")
                    continue
                if slot in new_slots:
                    check(not staging[row].any(), f"slot {slot}: a new request starts from zero")
                else:
                    worst_start = max(worst_start, relative(staging[row], truth[slot][layer]))
                    staging[row] = truth[slot][layer]       # the prefill kernel continues from its input state;
                    # here the fp32 truth continues and the staged (decompressed) start is checked above.
                staging[row] = prefill_tokens(staging[row].clone(), tokens_of_row[b], generator, device)
            engine.extend_end(layer, ci, staging, rows)
            for b, slot in enumerate(cache_indices):
                if slot == 0:
                    continue
                truth.setdefault(slot, [None] * LAYERS)[layer] = staging[int(rows[b])].clone()
                stored = torch.empty((1, HV, V, K), device=device)
                engine.kernel.export(engine.pools[layer], torch.tensor([slot], dtype=torch.int32, device=device), None, stored)
                worst_store = max(worst_store, relative(stored[0], truth[slot][layer]))
        return worst_start, worst_store

    # ---- prefill: three new requests, one padded row; then a second chunk ---------------------
    engine.reset_slots(torch.tensor([3, 1, 5], device=device))
    _, store_error = prefill([3, 0, 1], [40, 0, 25], generator)
    print(f"  1. prefill of two new requests (and a padded row): stored state off by at most {100 * store_error:.3f}%")
    check(store_error < 0.01, "the stored state is the prefilled state up to the rounding of one store (limit 1%)")
    start_error, store_error = prefill([3, 5], [30, 50], generator)
    print(f"  2. second chunk of slot 3 starts {100 * start_error:.3f}% from where its first chunk ended; "
          f"stored states off by at most {100 * store_error:.3f}%")
    check(0.0 < start_error < 0.01, "a later chunk continues from the stored state (limit 1%)")
    check(store_error < 0.01, "the second chunk is stored (limit 1%)")
    valid = engine.valid.cpu()
    check(bool((valid[:, [1, 3, 5]] == 1).all()) and int(valid[:, 0].sum()) == 0, "stored slots are marked, the padding slot is not")

    # ---- decode ------------------------------------------------------------------------------------
    rows = [3, 1, 0, 5]
    slots = torch.tensor(rows, dtype=torch.int32, device=device)
    fp32 = {slot: [state.clone() for state in truth[slot]] for slot in (1, 3, 5)}
    squared_error = squared_truth = 0.0
    padded_zero = True
    folds = 0
    for step in range(DECODE_STEPS):
        write_pos = cursor[slots.long()].clone()
        any_full = bool((write_pos[torch.tensor(rows, device=device) != 0] == L - 1).any())
        force = torch.full((len(rows),), 1 if any_full else 0, dtype=torch.int32, device=device)
        folds += int(any_full)
        for layer in range(LAYERS):
            mixed_qkv = torch.randn((len(rows), 2 * H * K + HV * V), generator=generator).to(device=device, dtype=torch.bfloat16)
            a = (0.5 * torch.randn((len(rows), HV), generator=generator)).to(device=device, dtype=torch.bfloat16)
            b = torch.randn((len(rows), HV), generator=generator).to(device=device, dtype=torch.bfloat16)
            output = engine.decode(layer, mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale, None, slots, write_pos, force)
            decay, beta, key, query, value = engine.gates(mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale)
            for row, slot in enumerate(rows):
                if slot == 0:
                    padded_zero = padded_zero and not output[row].any()
                    continue
                expected, fp32[slot][layer] = fp32_step(fp32[slot][layer], decay[row], beta[row], key[row], query[row], value[row])
                squared_error += float(((output[row] - expected) ** 2).sum())
                squared_truth += float((expected ** 2).sum())
        flushed = (write_pos + 1 >= L) | (force != 0)
        next_pos = torch.where(flushed, torch.zeros_like(write_pos), write_pos + 1)
        for row, slot in enumerate(rows):
            if slot != 0:
                cursor[slot] = next_pos[row]
    error = (squared_error / squared_truth) ** 0.5
    print(f"  3. {DECODE_STEPS} decode steps, {folds} aligned folds: output error against fp32 {100 * error:.3f}%   (limit 2%)")
    check(folds >= 3, "the decode steps folded")
    check(error < 0.02, "decode outputs stay close to fp32")
    print(f"  4. padded row gives zeros: {'yes' if padded_zero else 'NO'}")
    check(padded_zero, "padded rows give zeros")
    # ---- 5. a prefix-cache hit: slot 1's state (folded just now) is copied into slot 7 -------------
    cursor[1] = 0
    engine.copy_slots(torch.tensor([1], device=device), torch.tensor([7], device=device))
    worst_copy = 0.0
    copied_exactly = True
    for layer in range(LAYERS):
        pool = engine.pools[layer]
        for name in StatePool.BOUNDARY_FIELDS:
            copied_exactly = copied_exactly and torch.equal(getattr(pool, name)[7], getattr(pool, name)[1])
        staging, rows = engine.extend_begin(layer, torch.tensor([7], dtype=torch.int64, device=device))
        source = engine.extend_begin(layer, torch.tensor([1], dtype=torch.int64, device=device))[0][1]
        worst_copy = max(worst_copy, float((staging[int(rows[0])] - source).abs().max()))
    print(f"  5. prefix-cache hit: boundary state copied bit for bit: {'yes' if copied_exactly else 'NO'}; "
          f"the copy's prefill starts from the source's state (largest difference {worst_copy:.1e})")
    check(copied_exactly, "a prefix-cache hit copies the boundary state bit for bit")
    check(worst_copy == 0.0, "a copied slot starts its prefill from the source's boundary state")
    check(int(cursor[7]) == 0, "the copy's window starts empty")
    engine.reset_slots(torch.tensor([7], device=device))
    staging, rows = engine.extend_begin(0, torch.tensor([7], dtype=torch.int64, device=device))
    check(not staging[int(rows[0])].any(), "a cleared slot starts from zero again")

    per_slot = sum(engine.bytes_allocated() for _ in [0]) / SLOTS
    print(f"     LeapQuant state per slot (all layers): {per_slot / 1024:.1f} KiB; fp32 would be "
          f"{LAYERS * HV * V * K * 4 / 1024:.1f} KiB")
    print(f"\n{'PASS' if not failures else f'FAIL ({len(failures)} checks failed)'}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
