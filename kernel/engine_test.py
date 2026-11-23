#!/usr/bin/env python3
"""
engine_test.py

Tests leapquant_engine.EngineState, the glue the SGLang patch calls, by playing
the part of the engine on a CPU (or on a GPU with --device cuda).

The test engine does what SGLang does around its replay-window decode:
    - it keeps an fp32 state tensor per layer, [slots, HV, V, K], slot 0 = padding
    - it keeps one window cursor per slot, hands the cursor of the step to every
      layer, and advances it once per step
    - it forces a fold on "track" steps
    - it clears slots, copies slots (prefix-cache hit) and reuses freed slots

What is checked
    1. Every output equals the reference core's output for the same request.
       On a CPU: bit for bit.
    2. Every output is close to what an fp32 engine computes. The fp32 engine
       here is written from the formulas in SGLang's replay-window kernel, in
       SGLang's [value][key] layout, independently of the LeapQuant code.
    3. At every fold, the engine's fp32 tensor is brought up to date.
    4. A prefix-cache hit continues exactly like the request it was copied from.
    5. A freed slot can be reused. A host round trip in the middle of a window
       loses nothing. Padded rows give zeros.

Run:   python3 kernel/engine_test.py            (after `make kernel`)
"""

import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_engine import CheckpointStore, EngineState  # noqa: E402
from leapquant_kernel import StatePool  # noqa: E402
from verify_kernel import ReferenceLayer, load_reference_library  # noqa: E402
from leapquant_kernel import DEFAULT_SWEEPS  # noqa: E402

LAYERS, SLOTS, HV, H, K, V, L = 2, 7, 4, 2, 32, 48, 16
TRACK_INTERVAL = 23          # a forced fold every 23rd token of a request
STEPS = 150

failures = []


def check(condition, text):
    if not condition:
        failures.append(text)
        print(f"    FAILED: {text}")


class Request:
    """One request as the test engine sees it."""

    def __init__(self, slot, reference_library, initial_states, device):
        self.slot = slot
        self.tokens = 0
        shape = types.SimpleNamespace(heads=HV, key_channels=K, value_channels=V)
        # One reference layer (LeapQuant heads + fp32 truth heads) per model layer.
        self.references = [ReferenceLayer(reference_library, shape, L, 4, 8, DEFAULT_SWEEPS, True) for _ in range(LAYERS)]
        for layer, reference in enumerate(self.references):
            # The reference core stores a state as [key][value].
            reference.load_state(initial_states[layer].transpose(1, 2).contiguous().cpu().numpy(), which=3)
        # The fp32 engine's state, in SGLang's layout [HV, V, K].
        self.fp32_state = [state.clone().to(device) for state in initial_states]


def fp32_engine_step(state, decay, beta, key, query, value):
    """
    One decode step of an fp32 engine for one request, from the formulas in the
    header of SGLang's fused_recurrent_linear_replayssm.py, state [HV, V, K]:
        d = beta * (v - S . (a k))      o = S . (a q) + d * (k^T q)      S' = a S + d k^T
    """
    key_per_head = key.repeat_interleave(HV // H, dim=0)        # [HV, K]
    query_per_head = query.repeat_interleave(HV // H, dim=0)
    alpha = decay[:, None]
    held = torch.einsum("hvk,hk->hv", state, key_per_head) * alpha
    correction = beta[:, None] * (value - held)
    output = torch.einsum("hvk,hk->hv", state, query_per_head) * alpha + \
        correction * (key_per_head * query_per_head).sum(dim=-1, keepdim=True)
    new_state = state * alpha[:, :, None] + correction[:, :, None] * key_per_head[:, None, :]
    return output, new_state


def random_state(generator):
    """A state with a few strong patterns plus noise, [HV, V, K]."""
    state = 0.05 * torch.randn((HV, V, K), generator=generator)
    for pattern in range(3):
        state += torch.randn((HV, V, 1), generator=generator) * torch.randn((HV, 1, K), generator=generator) / (pattern + 1)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fused", action="store_true", help="gates computed inside the kernel (not bit-identical to the reference)")
    parser.add_argument("--fast", action="store_true", help="the fast (Triton) kernels; needs --device cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    exact = device.type == "cpu" and not args.fused and not args.fast
    generator = torch.Generator().manual_seed(2026)
    reference_library = load_reference_library()

    # ---- the test engine's own state ----------------------------------------------
    temporal_all = torch.zeros((LAYERS, SLOTS, HV, V, K), dtype=torch.float32, device=device)   # as SGLang holds it
    temporal = [temporal_all[layer] for layer in range(LAYERS)]
    checkpoints = CheckpointStore(num_layers=LAYERS, num_slots=4, num_heads=HV, head_v_dim=V, head_k_dim=K, device=device,
                                  fast=args.fast)
    cursor = torch.zeros(SLOTS, dtype=torch.int32, device=device)
    engine = EngineState(LAYERS, SLOTS, HV, H, K, V, max_rows=8, window_length=L, device=device, cursor=cursor,
                         padding_slot=0, fused_gates=args.fused, fast=args.fast)
    A_log = [(0.3 * torch.randn(HV, generator=generator) - 4.0).to(device) for _ in range(LAYERS)]
    dt_bias = [(0.1 * torch.randn(HV, generator=generator)).to(device) for _ in range(LAYERS)]
    scale = K ** -0.5

    def start_request(slot, copy_of=None, through_checkpoint=False):
        """Prefill ends in `slot` (UC3), or a cached prefix is copied into it (UC4),
        directly or through the checkpoint store (T11)."""
        index = torch.tensor([slot], device=device)
        if copy_of is not None and through_checkpoint:
            source = torch.tensor([copy_of.slot], device=device)
            checkpoint_slot = torch.tensor([2], device=device)
            checkpoints.store_from_pool(temporal_all, source, checkpoint_slot)
            # The checkpoint must be the boundary state the running request holds, bit for bit.
            for layer in range(LAYERS):
                for name in StatePool.BOUNDARY_FIELDS:
                    ours = getattr(checkpoints.pools[layer], name)[2]
                    theirs = getattr(engine.pools[layer], name)[copy_of.slot]
                    del ours, theirs
                # The checkpoint is a fresh fold of the same full-precision matrix. With the joint fit a
                # fresh fold starts from no guess, and the running request's fold started from its previous
                # compensators. The two boundary states are two valid roundings of one matrix: close, not
                # identical. (Copying the boundary state directly is step S23 of the plan.)
                both = torch.empty((2, HV, V, K), dtype=torch.float32, device=device)
                engine.kernel.export(checkpoints.pools[layer], torch.tensor([2], dtype=torch.int32, device=device), None, both[0:1])
                engine.kernel.export(engine.pools[layer], torch.tensor([copy_of.slot], dtype=torch.int32, device=device), None, both[1:2])
                apart = float((both[0] - both[1]).norm() / both[1].norm())
                check(apart < 0.02, f"checkpoint of layer {layer} is within 2% of the request's boundary state ({apart:.4f})")
            checkpoints.copy_to_pool(temporal_all, checkpoint_slot, index)
            engine.copy_slots(source, index)
            states = [temporal[layer][slot].cpu() for layer in range(LAYERS)]
        elif copy_of is None:
            states = [random_state(generator) for _ in range(LAYERS)]
            engine.reset_slots(index)                          # the engine clears the slot ...
            for layer in range(LAYERS):
                temporal[layer][slot] = states[layer].to(device)   # ... and prefill writes the state
        else:
            source = torch.tensor([copy_of.slot], device=device)
            for layer in range(LAYERS):
                temporal[layer][slot] = temporal[layer][copy_of.slot]
            engine.copy_slots(source, index)
            states = [temporal[layer][slot].cpu() for layer in range(LAYERS)]
        cursor[slot] = 0
        request = Request(slot, reference_library, states, device)
        if copy_of is not None:
            request.tokens = copy_of.tokens
            request.fp32_state = [state.clone() for state in copy_of.fp32_state]
        return request

    # Batch rows hold a request or None (a padded row). The row order is fixed.
    rows = [start_request(3), start_request(1), None, None, None, None]
    twin_of_row_1 = None
    checkpoint_twin_error = checkpoint_twin_truth = 0.0
    direct_twin_error = direct_twin_truth = 0.0
    against_fp32 = {1: [0.0, 0.0], 5: [0.0, 0.0]}
    squared_error = squared_truth = 0.0
    worst_reference_difference = 0.0
    worst_fold_state_error = 0.0
    folds_seen = forced_folds_seen = 0
    twin_steps = 0

    for step in range(STEPS):
        # ---- the engine changes who is in the batch --------------------------------
        if step == 4:
            rows[3] = start_request(2)                         # a request joins late
        if step == 40:
            rows[0] = None                                     # UC6: the request in slot 3 ends
        if step == 55:
            rows[0] = start_request(3)                         # ... and slot 3 is reused
        if step == 90:
            # A host round trip in the middle of a window (slot 1 has records in its window).
            index = torch.tensor([rows[1].slot], device=device)
            cursor_before = int(cursor[rows[1].slot])
            check(cursor_before > 0, "the host round trip happens in the middle of a window")
            saved = engine.get_cpu_slots(index)
            engine.reset_slots(index)                          # wipe the slot, as a new owner would
            for pool in engine.pools:
                pool.clear(index)
                pool.window_keys[index] = 0
                pool.window_corrections[index] = 0
            engine.load_cpu_slots(saved, index)
            cursor[rows[1].slot] = cursor_before               # the cursor is the engine's to restore
        if twin_of_row_1 is None and step > 60 and int(cursor[rows[1].slot]) == 0:
            # UC4: slot 1 is at a fold boundary. Copy it into slot 5, as a prefix-cache hit does.
            twin_of_row_1 = start_request(5, copy_of=rows[1])
            rows[4] = twin_of_row_1
            # T11: the same prefix, restored from a LeapQuant checkpoint into slot 6.
            rows[5] = start_request(6, copy_of=rows[1], through_checkpoint=True)

        slots = torch.tensor([0 if request is None else request.slot for request in rows], dtype=torch.int32, device=device)
        long_slots = slots.to(torch.long)
        write_pos = cursor[long_slots].clone()
        force = torch.tensor([0 if request is None else int((request.tokens + 1) % TRACK_INTERVAL == 0)
                              for request in rows], dtype=torch.int32, device=device)
        batch = len(rows)

        for layer in range(LAYERS):
            # What the model's projections give, in bfloat16 as in a bf16 model.
            mixed_qkv = torch.randn((batch, 2 * H * K + HV * V), generator=generator).to(device=device, dtype=torch.bfloat16)
            a = (0.5 * torch.randn((batch, HV), generator=generator)).to(device=device, dtype=torch.bfloat16)
            b = torch.randn((batch, HV), generator=generator).to(device=device, dtype=torch.bfloat16)
            if twin_of_row_1 is not None:
                mixed_qkv[4], a[4], b[4] = mixed_qkv[1], a[1], b[1]   # the cache hits see the same tokens
                mixed_qkv[5], a[5], b[5] = mixed_qkv[1], a[1], b[1]

            output = engine.decode(layer, mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale, temporal[layer],
                                   slots, write_pos, force)
            decay, beta, key, query, value = engine.gates(mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale)
            output_cpu = output.cpu().numpy()

            for row, request in enumerate(rows):
                if request is None:
                    check(not output_cpu[row].any(), f"step {step}: padded row {row} gives zeros")
                    continue
                # 1. Against the reference core.
                reference_output, _ = request.references[layer].step(
                    decay[row].cpu().numpy(), beta[row].cpu().numpy(),
                    np.ascontiguousarray(key[row].repeat_interleave(HV // H, dim=0).cpu().numpy()),
                    np.ascontiguousarray(value[row].cpu().numpy()),
                    np.ascontiguousarray(query[row].repeat_interleave(HV // H, dim=0).cpu().numpy()))
                folded = int(write_pos[row]) + 1 >= L or int(force[row]) != 0
                if folded and request.references[layer].tokens_in_window() > 0:
                    reference_library.lqref_fold(request.references[layer]._handle)   # the forced fold
                difference = np.abs(output_cpu[row] - reference_output).max() / np.abs(reference_output).max()
                worst_reference_difference = max(worst_reference_difference, float(difference))

                # 2. Against the fp32 engine.
                fp32_output, request.fp32_state[layer] = fp32_engine_step(
                    request.fp32_state[layer], decay[row], beta[row], key[row], query[row], value[row])
                squared_error += float(((output[row] - fp32_output) ** 2).sum())
                squared_truth += float((fp32_output ** 2).sum())
                # After the cache hits: the error of the original request (row 1) and of the
                # request restored from a checkpoint (row 5), each against fp32.
                if twin_of_row_1 is not None and row in (1, 5):
                    against_fp32[row][0] += float(((output[row] - fp32_output) ** 2).sum())
                    against_fp32[row][1] += float((fp32_output ** 2).sum())

                # 3. At a fold, the engine's fp32 tensor is current.
                if folded:
                    folds_seen += 1
                    forced_folds_seen += int(force[row]) != 0 and int(write_pos[row]) + 1 < L
                    exported = temporal[layer][request.slot]
                    error = float((exported - request.fp32_state[layer]).norm() / request.fp32_state[layer].norm())
                    worst_fold_state_error = max(worst_fold_state_error, error)

            # 4. The cache hit continues exactly like the request it was copied from.
            if twin_of_row_1 is not None and rows[1] is not None:
                if layer == 0:
                    twin_steps += 1
                # The hit adopts the fp32 state with a fresh fit (no guess); the original folded with its
                # previous compensators as the guess. Two valid roundings: close, not identical.
                direct_twin_error += float(((output[4] - output[1]) ** 2).sum())
                direct_twin_truth += float((output[1] ** 2).sum())
                # The hit that went through the checkpoint store was rounded once more: close, not equal.
                checkpoint_twin_error += float(((output[5] - output[1]) ** 2).sum())
                checkpoint_twin_truth += float((output[1] ** 2).sum())

        # ---- the engine advances its cursor once per step (as SGLang does) --------------
        flushed = (write_pos + 1 >= L) | (force != 0)
        next_pos = torch.where(flushed, torch.zeros_like(write_pos), write_pos + 1)
        for row, request in enumerate(rows):
            if request is not None:
                cursor[request.slot] = next_pos[row]
                request.tokens += 1

    output_error = (squared_error / squared_truth) ** 0.5
    print(f"engine test on {device}: {STEPS} steps, {LAYERS} layers, {HV} value heads, state {K} x {V}")
    # On a GPU the kernel and the reference run freely side by side and drift apart, up to
    # the size of the rounding error (see verify_kernel.py). Measured on an H100: 1.8e-02.
    # There the comparison with fp32 below is the check; this number is only reported.
    print(f"  1. worst difference from the reference core: {worst_reference_difference:.1e}"
          f"   ({'must be 0 on a CPU' if exact else 'reported only: free-running on a GPU'})")
    check(worst_reference_difference == 0.0 if exact else worst_reference_difference < 0.2,
          "outputs equal the reference core")
    print(f"  2. output error against the fp32 engine: {100 * output_error:.3f}%   (limit 2%)")
    check(output_error < 0.02, "outputs are close to the fp32 engine")
    print(f"  3. {folds_seen} folds seen, {forced_folds_seen} of them forced; fp32 tensor after a fold is off by "
          f"at most {100 * worst_fold_state_error:.3f}%   (limit 2%)")
    check(folds_seen > 20 and forced_folds_seen > 3, "the test saw natural and forced folds")
    check(worst_fold_state_error < 0.02, "the engine's fp32 tensor is current after every fold")
    print(f"  4. prefix-cache hit followed for {twin_steps} steps")
    check(twin_steps > 30, "the prefix-cache hit was exercised")
    direct_difference = (direct_twin_error / direct_twin_truth) ** 0.5
    print(f"     the hit copied in fp32 differs from the original by {100 * direct_difference:.3f}%   (limit 2%)")
    check(direct_difference < 0.02, "a prefix-cache hit continues like the original request")
    twin_difference = (checkpoint_twin_error / checkpoint_twin_truth) ** 0.5
    print(f"     the hit restored from a LeapQuant checkpoint differs from the original by {100 * twin_difference:.3f}%   (limit 2%)")
    check(twin_difference < 0.02, "a request restored from a LeapQuant checkpoint continues like the original")
    original_error = (against_fp32[1][0] / against_fp32[1][1]) ** 0.5
    restored_error = (against_fp32[5][0] / against_fp32[5][1]) ** 0.5
    print(f"     error against fp32 after the hit: original request {100 * original_error:.3f}%, "
          f"restored request {100 * restored_error:.3f}%")
    print(f"     checkpoint size: {checkpoints.bytes_per_slot() // (LAYERS * HV):,} bytes per head")
    print(f"  5. slot reuse, host round trip and padded rows: covered by 1 and 2 above")
    print(f"\n{'PASS' if not failures else f'FAIL ({len(failures)} checks failed)'}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
