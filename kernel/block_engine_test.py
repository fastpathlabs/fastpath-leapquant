#!/usr/bin/env python3
"""
block_engine_test.py

Tests leapquant_engine.BlockEngineState, the glue of the vLLM patch, by playing
the part of vLLM on a CPU (or on a GPU with --device cuda).

The test engine does what vLLM does with prefix caching off:
    - the fp32 state of a request lives in one block of a large block pool,
      with padding between blocks; block 0 is the "no block" id of padded rows
    - nothing is called when a request ends; its block is simply handed to the
      next request, or left alone
    - a step is decode-only, or mixed: the decode rows first, then the prefill
      of new requests, which writes their blocks

What is checked
    1. Every output equals the reference core's output for the same request
       (bit for bit on a CPU), over many requests that come and go, with far
       more blocks used over time than the LeapQuant pool has slots.
    2. Every output is close to what an fp32 engine computes.
    3. No decode row is ever left without a slot.
    4. A pool that is too small is reported by assign(), not silently wrong.

Run:   python3 kernel/block_engine_test.py            (after `make kernel`)
"""

import argparse
import os
import random
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine_test import fp32_engine_step, random_state, HV, H, K, V, L, LAYERS  # noqa: E402
from leapquant_engine import BlockEngineState  # noqa: E402
from verify_kernel import ReferenceLayer, load_reference_library  # noqa: E402
from leapquant_kernel import DEFAULT_SWEEPS  # noqa: E402

BLOCKS = 300
PADDING_FLOATS = 37           # unused floats between two blocks, as in a padded KV page
LQ_SLOTS = 1 + 7              # slot 0 = none
MAX_ACTIVE = 4
STEPS = 260

failures = []


def check(condition, text):
    if not condition:
        failures.append(text)
        print(f"    FAILED: {text}")


class Request:
    def __init__(self, block, reference_library, states, tokens_to_generate):
        self.block = block
        self.remaining = tokens_to_generate
        shape = types.SimpleNamespace(heads=HV, key_channels=K, value_channels=V)
        self.references = [ReferenceLayer(reference_library, shape, L, 4, 8, DEFAULT_SWEEPS, True) for _ in range(LAYERS)]
        for layer, reference in enumerate(self.references):
            reference.load_state(states[layer].transpose(1, 2).contiguous().cpu().numpy(), which=3)
        self.fp32_state = [state.clone() for state in states]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fused", action="store_true", help="gates computed inside the kernel (not bit-identical to the reference)")
    parser.add_argument("--fast", action="store_true", help="the fast (Triton) kernels; needs --device cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    exact = device.type == "cpu" and not args.fused and not args.fast
    generator = torch.Generator().manual_seed(7)
    choose = random.Random(7)
    reference_library = load_reference_library()

    # The block pool of each layer: [blocks, HV, V, K] with padding between blocks.
    entry = HV * V * K
    buffers = [torch.zeros((BLOCKS, entry + PADDING_FLOATS), dtype=torch.float32, device=device) for _ in range(LAYERS)]
    block_state = [buffer[:, :entry].view(BLOCKS, HV, V, K) for buffer in buffers]
    check(not block_state[0].is_contiguous(), "the test's block pool is padded, like vLLM's")

    state = BlockEngineState(LAYERS, BLOCKS, HV, K, V, lq_slots=LQ_SLOTS, max_rows=MAX_ACTIVE + 2, window_length=L,
                             device=device, idle_steps=1, fused_gates=args.fused, fast=args.fast)
    A_log = [(0.3 * torch.randn(HV, generator=generator) - 4.0).to(device) for _ in range(LAYERS)]
    dt_bias = [(0.1 * torch.randn(HV, generator=generator)).to(device) for _ in range(LAYERS)]
    scale = K ** -0.5

    active = []
    blocks_ever_used = set()
    last_freed_block = None
    squared_error = squared_truth = 0.0
    worst_reference_difference = 0.0
    requests_started = decode_rows = 0

    def prefill_new_requests(count):
        """Prefill of `count` new requests ends: their blocks are written, then assign() is called."""
        nonlocal requests_started, last_freed_block
        new_requests = []
        for _ in range(count):
            in_use = {request.block for request in active + new_requests}
            if last_freed_block is not None and last_freed_block not in in_use and choose.random() < 0.4:
                block = last_freed_block                       # a freed block is handed out again at once
            else:
                block = choose.choice([b for b in range(1, BLOCKS) if b not in in_use])
            states = [random_state(generator).to(device) for _ in range(LAYERS)]
            for layer in range(LAYERS):
                block_state[layer][block] = states[layer]
            new_requests.append(Request(block, reference_library, states, choose.randint(5, 70)))
            blocks_ever_used.add(block)
            requests_started += 1
        indices = torch.tensor([request.block for request in new_requests], dtype=torch.int32, device=device)
        for layer in range(LAYERS):
            state.assign(layer, indices)
        active.extend(new_requests)

    prefill_new_requests(2)
    for step in range(STEPS):
        # ---- decode rows of this step: the active requests in shuffled order, plus padded rows ----
        decoding = list(active)
        choose.shuffle(decoding)
        rows = decoding + [None] * choose.randint(0, 2)
        if rows:
            indices = torch.tensor([0 if request is None else request.block for request in rows], dtype=torch.int32, device=device)
            for layer in range(LAYERS):
                batch = len(rows)
                mixed_qkv = torch.randn((batch, 2 * H * K + HV * V), generator=generator).to(device=device, dtype=torch.bfloat16)
                a = (0.5 * torch.randn((batch, HV), generator=generator)).to(device=device, dtype=torch.bfloat16)
                b = torch.randn((batch, HV), generator=generator).to(device=device, dtype=torch.bfloat16)
                output = state.decode(layer, mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale, block_state[layer], indices)
                decay, beta, key, query, value = state.engine.gates(mixed_qkv, a, b, A_log[layer], dt_bias[layer], scale)
                output_cpu = output.cpu().numpy()
                for row, request in enumerate(rows):
                    if request is None:
                        check(not output_cpu[row].any(), f"step {step}: padded row gives zeros")
                        continue
                    reference_output, _ = request.references[layer].step(
                        decay[row].cpu().numpy(), beta[row].cpu().numpy(),
                        np.ascontiguousarray(key[row].repeat_interleave(HV // H, dim=0).cpu().numpy()),
                        np.ascontiguousarray(value[row].cpu().numpy()),
                        np.ascontiguousarray(query[row].repeat_interleave(HV // H, dim=0).cpu().numpy()))
                    difference = np.abs(output_cpu[row] - reference_output).max() / np.abs(reference_output).max()
                    worst_reference_difference = max(worst_reference_difference, float(difference))
                    fp32_output, request.fp32_state[layer] = fp32_engine_step(
                        request.fp32_state[layer], decay[row], beta[row], key[row], query[row], value[row])
                    squared_error += float(((output[row] - fp32_output) ** 2).sum())
                    squared_truth += float((fp32_output ** 2).sum())
                    decode_rows += layer == 0

        # ---- requests end without any notice to the LeapQuant state ----
        for request in list(active):
            request.remaining -= 1
            if request.remaining <= 0:
                active.remove(request)
                last_freed_block = request.block

        # ---- the same step can be mixed: prefill of new requests after the decode rows ----
        room = MAX_ACTIVE - len(active)
        if room > 0 and choose.random() < 0.35:
            prefill_new_requests(choose.randint(1, min(2, room)))

    unmapped = int(state.unmapped_rows.sum())
    output_error = (squared_error / squared_truth) ** 0.5
    print(f"block engine test on {device}: {STEPS} steps, {requests_started} requests in {len(blocks_ever_used)} "
          f"different blocks, {decode_rows} decode rows, {LQ_SLOTS - 1} LeapQuant slots")
    print(f"  1. worst difference from the reference core: {worst_reference_difference:.1e}"
          f"   ({'must be 0 on a CPU' if exact else 'reported only: free-running on a GPU'})")
    check(worst_reference_difference == 0.0 if exact else worst_reference_difference < 0.2, "outputs equal the reference core")
    check(len(blocks_ever_used) > 3 * (LQ_SLOTS - 1), "the test used many more blocks than the pool has slots")
    print(f"  2. output error against the fp32 engine: {100 * output_error:.3f}%   (limit 2%)")
    check(output_error < 0.02, "outputs are close to the fp32 engine")
    print(f"  3. decode rows without a slot: {unmapped}")
    check(unmapped == 0, "no decode row was left without a slot")

    # 4. A pool that is too small must be reported.
    small = BlockEngineState(1, BLOCKS, HV, K, V, lq_slots=1 + 2, max_rows=4, window_length=L, device=device, idle_steps=1, fused_gates=args.fused, fast=args.fast)
    small.assign(0, torch.tensor([10, 11], dtype=torch.int32, device=device))
    indices = torch.tensor([10, 11], dtype=torch.int32, device=device)
    mixed_qkv = torch.randn((2, 2 * H * K + HV * V), generator=generator).to(device=device, dtype=torch.bfloat16)
    a = torch.zeros((2, HV), dtype=torch.bfloat16, device=device)
    small.decode(0, mixed_qkv, a, a, A_log[0], dt_bias[0], scale, block_state[0], indices)
    try:
        small.assign(0, torch.tensor([12], dtype=torch.int32, device=device))   # both slots decoded in the last step
        reported = False
    except RuntimeError:
        reported = True
    print(f"  4. a pool that is too small is reported: {'yes' if reported else 'NO'}")
    check(reported, "assign() raises when it would take the slot of a request that is still decoding")

    print(f"\n{'PASS' if not failures else f'FAIL ({len(failures)} checks failed)'}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
