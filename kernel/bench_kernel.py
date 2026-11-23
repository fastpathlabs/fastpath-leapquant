#!/usr/bin/env python3
"""
bench_kernel.py

Stage E of the implementation plan: per-step latency of the LeapQuant kernel at
several batch sizes, and, with --with-sglang, of SGLang's two decode kernels in
the same harness.

    python3 kernel/bench_kernel.py --device cuda --batch-sizes 32 128 512
    python3 kernel/bench_kernel.py --device cuda --batch-sizes 32 128 512 --with-sglang   (in the SGLang environment)
    python3 kernel/bench_kernel.py --device cpu --batch-sizes 2 --repeats 3                (smoke test)

What one "step" is
    One decode token for every request of the batch, for ONE layer.

    A window of L tokens is L-1 steps that only append a record, and one step
    that also folds. The two kinds are timed separately and combined:
        per step = ((L - 1) * append step + fold step) / L
    This is the same accounting SGLang's bench_gdn_replayssm_decode.py uses.

Rows printed
    leapquant, kernel only    lq_decode_step + lq_fold, inputs already gated
    leapquant, with gates     EngineState.decode(): the same plus the gate and
                              normalisation arithmetic (a few small tensor ops)
                              and the write-back of the fp32 state at each fold.
                              This is what the SGLang patch runs, and the row to
                              compare with SGLang's kernels, which include the
                              gates.
    sglang packed             fused_recurrent_gated_delta_rule_packed_decode:
                              reads and writes the fp32 state every step
    sglang window L=16        fused_recurrent_linear_replayssm_decode

Timing is wall-clock around `repeats` calls with a device synchronise at the
end, after `warmup` calls. It includes the Python launch cost of every call.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_engine import EngineState  # noqa: E402
from leapquant_kernel import Kernel, StatePool  # noqa: E402


def time_ms(function, device, warmup, repeats):
    for _ in range(warmup):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return 1000.0 * (time.perf_counter() - start) / repeats


def amortised(append_ms, fold_ms, window_length):
    return ((window_length - 1) * append_ms + fold_ms) / window_length


def bench_leapquant(batch, args, device):
    HV, H, K, V, L = args.hv, args.h, args.k, args.v, args.window
    slots = batch + 1                                   # slot 0 is the padding slot, as in SGLang
    generator = torch.Generator().manual_seed(0)

    def random(*shape, dtype=torch.float32):
        return torch.randn(shape, generator=generator).to(device=device, dtype=dtype)

    # ---- kernel only ---------------------------------------------------------------
    pool = StatePool(slots, HV, K, V, window_length=L, device=device)
    kernel = Kernel(device)
    slot = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    kernel.store(pool, slot, random(batch, HV, V, K))
    decay = (1.0 - 10.0 ** (-3.0 + 1.7 * torch.rand((batch, HV), generator=generator))).to(device)
    beta = (0.2 + 0.8 * torch.rand((batch, HV), generator=generator)).to(device)
    key = torch.nn.functional.normalize(random(batch, H, K), dim=-1).contiguous()
    query = (torch.nn.functional.normalize(random(batch, H, K), dim=-1) * K ** -0.5).contiguous()
    value = random(batch, HV, V)
    out = torch.empty((batch, HV, V), dtype=torch.float32, device=device)
    scratch = torch.empty((batch, HV, V, K), dtype=torch.float32, device=device)
    cursor = torch.zeros(batch, dtype=torch.int32, device=device)

    # Fill the window with real records, so the timed steps read a realistic window.
    for position in range(L - 1):
        cursor.fill_(position)
        kernel.decode_step(pool, slot, cursor, decay, beta, key, query, value, out)

    def kernel_step():
        kernel.decode_step(pool, slot, cursor, decay, beta, key, query, value, out)
        kernel.fold(pool, slot, cursor, tokens_added=1, scratch=scratch)

    cursor.fill_((L - 1) // 2)
    kernel_append = time_ms(kernel_step, device, args.warmup, args.repeats)
    cursor.fill_(L - 1)
    kernel_fold = time_ms(kernel_step, device, args.warmup, args.repeats)

    # ---- with gates: what the SGLang patch runs --------------------------------------
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    engine = EngineState(1, slots, HV, H, K, V, max_rows=batch, window_length=L, device=device, padding_slot=0)
    temporal = random(slots, HV, V, K)
    mixed_qkv = random(batch, 2 * H * K + HV * V, dtype=dtype)
    a = (0.5 * random(batch, HV)).to(dtype)
    b = random(batch, HV, dtype=dtype)
    A_log = 0.3 * random(HV)
    dt_bias = 0.1 * random(HV)
    for position in range(L - 1):
        cursor.fill_(position)
        engine.decode(0, mixed_qkv, a, b, A_log, dt_bias, K ** -0.5, temporal, slot, cursor)

    def engine_step():
        engine.decode(0, mixed_qkv, a, b, A_log, dt_bias, K ** -0.5, temporal, slot, cursor)

    cursor.fill_((L - 1) // 2)
    engine_append = time_ms(engine_step, device, args.warmup, args.repeats)
    cursor.fill_(L - 1)
    engine_fold = time_ms(engine_step, device, args.warmup, args.repeats)

    return [
        ("leapquant, kernel only", amortised(kernel_append, kernel_fold, L), kernel_append, kernel_fold),
        ("leapquant, with gates", amortised(engine_append, engine_fold, L), engine_append, engine_fold),
    ]
    # "with gates" is EngineState.decode() as the SGLang patch calls it: gates inside the kernel
    # (fused_gates=True, the default), adoption check, fold and write-back of the fp32 state.


def bench_sglang(batch, args, device):
    """SGLang's packed decode and window decode, set up as in its own
    benchmark/kernels/attention/bench_gdn_replayssm_decode.py."""
    from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
    from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )

    HV, H, K, V, L = args.hv, args.h, args.k, args.v, args.window
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    mixed_qkv = torch.randn(batch, 2 * H * K + HV * V, device=device, dtype=dtype)
    a = torch.randn(batch, HV, device=device, dtype=dtype) * 0.5
    b = torch.randn(batch, HV, device=device, dtype=dtype)
    A_log = (torch.randn(HV, device=device, dtype=torch.float32) * 0.3).contiguous()
    dt_bias = (torch.randn(HV, device=device, dtype=torch.float32) * 0.1).contiguous()
    cache_indices = torch.arange(batch, device=device, dtype=torch.int32)
    scale = K ** -0.5

    state = torch.randn(batch, HV, V, K, device=device, dtype=torch.float32)
    out = mixed_qkv.new_empty(batch, 1, HV, V)

    def packed_step():
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=state,
            out=out, ssm_state_indices=cache_indices, use_qk_l2norm_in_kernel=True)

    packed = time_ms(packed_step, device, args.warmup, args.repeats)

    window_state = torch.randn(batch, HV, V, K, device=device, dtype=torch.float32)
    d_cache = torch.zeros(batch, HV, L, V, device=device, dtype=dtype)
    k_cache = torch.zeros(batch, H, L, K, device=device, dtype=dtype)
    g_cache = torch.zeros(batch, HV, L, device=device, dtype=torch.float32)
    write_pos = torch.empty(batch, device=device, dtype=torch.int32)
    window_out = mixed_qkv.new_empty(batch, 1, HV, V)

    def window_step():
        fused_recurrent_linear_replayssm_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=window_state,
            d_cache=d_cache, k_cache=k_cache, g_cache=g_cache, out=window_out, ssm_state_indices=cache_indices,
            write_pos=write_pos, use_qk_l2norm_in_kernel=True)

    write_pos.fill_((L - 1) // 2)
    append = time_ms(window_step, device, args.warmup, args.repeats)
    write_pos.fill_(L - 1)
    flush = time_ms(window_step, device, args.warmup, args.repeats)
    return [
        ("sglang packed", packed, packed, packed),
        (f"sglang window L={L}", amortised(append, flush, L), append, flush),
    ]


# ---- complete windows (plan 5, step S07) ---------------------------------------------------

def cycle_ms(step_at, reset, window_length, device, cycles):
    """
    Times complete windows: positions 0 .. L-1 in order, with the cursor advancing as in a
    server, starting every window from the same state (`reset` runs outside the timed span).
    Returns the median time of one step, averaged over a window.
    """
    cuda = device.type == "cuda"
    totals = []
    for cycle in range(cycles + 1):
        reset()
        if cuda:
            torch.cuda.synchronize(device)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            begin = time.perf_counter()
        for position in range(window_length):
            step_at(position)
        if cuda:
            end.record()
            torch.cuda.synchronize(device)
            elapsed = start.elapsed_time(end)
        else:
            elapsed = 1000.0 * (time.perf_counter() - begin)
        if cycle > 0:                               # the first window warms up
            totals.append(elapsed / window_length)
    totals.sort()
    return totals[len(totals) // 2]


def bench_full_cycles(batch, args, device):
    """
    Every kernel runs whole windows of L steps on the SAME projections, from an equal
    starting point: an fp32 state for SGLang's kernels, the boundary state made from that
    same fp32 state for LeapQuant, and EMPTY record buffers for both.
    Printed with each row: the format of the projections, the state, the records.
    """
    HV, H, K, V, L = args.hv, args.h, args.k, args.v, args.window
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    generator = torch.Generator().manual_seed(0)

    def random(*shape, kind=torch.float32):
        return torch.randn(shape, generator=generator).to(device=device, dtype=kind)

    slots = batch + 1
    state0 = 0.05 * random(slots, HV, V, K)
    for pattern in range(3):
        state0 += random(slots, HV, V, 1) * random(slots, HV, 1, K) / (pattern + 1)
    mixed_qkv = random(batch, 2 * H * K + HV * V, kind=dtype)
    a = (0.5 * random(batch, HV)).to(dtype)
    b = random(batch, HV, kind=dtype)
    A_log = (0.3 * random(HV) - 4.0).contiguous()
    dt_bias = (0.1 * random(HV)).contiguous()
    scale = K ** -0.5
    rows = []

    # LeapQuant, as the engine runs it (gates in the kernel, adoption check, fold, fp32 write-back).
    engine = EngineState(1, slots, HV, H, K, V, max_rows=batch, window_length=L, device=device, padding_slot=0)
    slot = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    temporal = state0.clone()
    cursor = torch.zeros(batch, dtype=torch.int32, device=device)
    engine.decode(0, mixed_qkv, a, b, A_log, dt_bias, scale, temporal, slot, cursor)        # adopts every slot
    pool = engine.pools[0]
    fields = StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS
    engine.kernel.store(pool, slot, state0[1:].clone())                                    # boundary of state0
    snapshot = {name: getattr(pool, name).clone() for name in fields}

    def reset_leapquant():
        for name in fields:
            getattr(pool, name).copy_(snapshot[name])

    def leapquant_step(position):
        cursor.fill_(position)
        engine.decode(0, mixed_qkv, a, b, A_log, dt_bias, scale, temporal, slot, cursor)

    rows.append(("leapquant, with gates", f"{args.dtype} / int8+fp16 / fp16",
                 cycle_ms(leapquant_step, reset_leapquant, L, device, args.cycles)))

    if args.with_sglang:
        from sglang.kernels.ops.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode
        from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
            fused_recurrent_linear_replayssm_decode,
        )
        indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
        for state_name, state_dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
            state = state0.to(state_dtype)
            start_state = state.clone()
            out = mixed_qkv.new_empty(batch, 1, HV, V)

            def reset_packed():
                state.copy_(start_state)

            def packed_step(_position):
                fused_recurrent_gated_delta_rule_packed_decode(
                    mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=state,
                    out=out, ssm_state_indices=indices, use_qk_l2norm_in_kernel=True)

            d_cache = torch.zeros(slots, HV, L, V, device=device, dtype=dtype)
            k_cache = torch.zeros(slots, H, L, K, device=device, dtype=dtype)
            g_cache = torch.zeros(slots, HV, L, device=device, dtype=torch.float32)
            write_pos = torch.zeros(batch, device=device, dtype=torch.int32)

            def reset_window():
                state.copy_(start_state)
                d_cache.zero_()
                k_cache.zero_()
                g_cache.zero_()

            def window_step(position):
                write_pos.fill_(position)
                fused_recurrent_linear_replayssm_decode(
                    mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=state,
                    d_cache=d_cache, k_cache=k_cache, g_cache=g_cache, out=out, ssm_state_indices=indices,
                    write_pos=write_pos, use_qk_l2norm_in_kernel=True)

            try:
                rows.append(("sglang packed", f"{args.dtype} / {state_name} / -",
                             cycle_ms(packed_step, reset_packed, L, device, args.cycles)))
                rows.append((f"sglang window L={L}", f"{args.dtype} / {state_name} / {args.dtype}",
                             cycle_ms(window_step, reset_window, L, device, args.cycles)))
            except Exception as problem:      # a state format a kernel does not accept is reported, not hidden
                rows.append((f"sglang, {state_name} state", "not supported by the kernel", float("nan")))
                print(f"    ({state_name} state: {type(problem).__name__}: {str(problem)[:120]})")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Per-step latency of the LeapQuant kernel.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--hv", type=int, default=32, help="value heads (Qwen3.5-9B: 32)")
    parser.add_argument("--h", type=int, default=16, help="key heads (Qwen3.5-9B: 16)")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="dtype of the layer's projections")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--with-sglang", action="store_true", help="also time SGLang's packed and window kernels")
    parser.add_argument("--full-cycles", action="store_true",
                        help="time complete windows (positions 0..L-1, advancing cursor, equal starting state, "
                             "empty record buffers), with SGLang's kernels on fp32, bf16 and fp16 states")
    parser.add_argument("--cycles", type=int, default=20, help="windows timed per kernel with --full-cycles")
    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"LeapQuant kernel benchmark on {device}: HV={args.hv} H={args.h} K={args.k} V={args.v} "
          f"window={args.window} projections={args.dtype}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print("  milliseconds per decode step of one layer, whole batch\n")
    if args.full_cycles:
        print(f"  complete windows of {args.window} steps; median of {args.cycles} windows; all requests fold on the same step")
        print(f"  {'batch':>5}  {'kernel':<24} {'projections / state / records':<32} {'ms per step':>12}")
        for batch in args.batch_sizes:
            for name, formats, value in bench_full_cycles(batch, args, device):
                print(f"  {batch:>5}  {name:<24} {formats:<32} {value:12.4f}")
            print()
        return
    print(f"  {'batch':>5}  {'kernel':<24} {'per step':>9} {'append step':>12} {'fold step':>10} {'vs packed':>10}")
    for batch in args.batch_sizes:
        rows = bench_leapquant(batch, args, device)
        if args.with_sglang:
            rows = bench_sglang(batch, args, device) + rows
        packed = rows[0][1] if args.with_sglang else None
        for name, per_step, append, fold in rows:
            speedup = f"{packed / per_step:9.2f}x" if packed else f"{'':>10}"
            print(f"  {batch:>5}  {name:<24} {per_step:9.4f} {append:12.4f} {fold:10.4f} {speedup}")
        print()


if __name__ == "__main__":
    main()
