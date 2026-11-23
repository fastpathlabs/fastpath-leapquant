#!/usr/bin/env python3
"""
bench_triton.py

Checks and times the fast (Triton, tensor-core) kernels of kernel/leapquant_triton.py
against the exact kernel, on the same state pool format.

    python3 kernel/bench_triton.py --batch-sizes 32 128 512            (needs a GPU and triton)
    python3 kernel/bench_triton.py --batch-sizes 512 --with-sglang

For every batch size:
1. Two pools get the same boundary state (made by the exact store kernel from a random state).
2. One window is decoded token by token, by the exact kernel on one pool and by the fast
   kernel on the other. Compared per token: the outputs and the records both wrote. After
   each token the fast pool takes the records of the exact one, so every difference printed
   is the difference of ONE step.
3. Both pools fold. Compared: how far each new boundary state is from the full-precision
   state it was made from (the fold loss), and how much of the best possible rank-R
   approximation the compensators capture.
4. Times, with GPU-side timers and without the host's launch overhead (see gpu_ms): the
   fast decode step at every window position, the fast fold, the exact decode step and the
   exact fold, and with --with-sglang SGLang's kernels the same way.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leapquant_triton as fast  # noqa: E402
from leapquant_kernel import Kernel, StatePool  # noqa: E402


def gpu_ms(function, device, repeats, before=None):
    """
    GPU time of one call of function(), in ms, without the host's launch overhead (as in a
    CUDA graph, which is how a server runs the step):
      - before is None: the call can be repeated as it is; ten calls are captured in one CUDA
        graph; median over `repeats` replays, divided by ten.
      - otherwise: the call is captured in a CUDA graph once; before() runs before every replay,
        outside the timed part; median of `repeats` replays.
    """
    function()
    torch.cuda.synchronize(device)
    calls = 10 if before is None else 1          # calls per graph: repeatable calls are captured ten times
    if before is None:
        before = lambda: None                    # noqa: E731
    before()
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        function()                       # warm-up on the side stream, as graph capture requires
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    before()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            function()
    times = []
    for _ in range(repeats):
        before()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize(device)
        times.append(start.elapsed_time(end) / calls)
    times.sort()
    return times[len(times) // 2]


def copy_fields(source, destination, names):
    for name in names:
        getattr(destination, name).copy_(getattr(source, name))


def per_head_difference(candidate, reference):
    """Relative difference per head, over the last dimension(s) of [B, HV, ...] tensors."""
    difference = (candidate.float() - reference.float()).flatten(2).norm(dim=-1)
    return difference / reference.float().flatten(2).norm(dim=-1).clamp_min(1e-30)


def captured_share(pool, slot, true_state, heads):
    """Energy the compensators capture, as a share of the best rank-R approximation, for the first `heads` heads."""
    index = slot.long()
    rank = pool.compensator_count
    matrix = true_state.flatten(0, 1)[:heads].double()                                   # [n, V, K]
    compensators = torch.einsum("brv,brk->bvk", pool.compensator_corrections[index].flatten(0, 1)[:heads].double(),
                                pool.compensator_keys[index].flatten(0, 1)[:heads].double())
    total = (matrix ** 2).sum(dim=(1, 2))
    captured = total - ((matrix - compensators) ** 2).sum(dim=(1, 2))
    best = (torch.linalg.svdvals(matrix)[:, :rank] ** 2).sum(dim=1)
    return float((captured / best.clamp_min(1e-300)).mean()), float((captured / best.clamp_min(1e-300)).min())


def main():
    parser = argparse.ArgumentParser(description="Check and time the Triton decode step and fold.")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--hv", type=int, default=32)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--v", type=int, default=128)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="dtype of the projections")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--with-sglang", action="store_true")
    parser.add_argument("--nk", type=int, default=None, help="K tiles of the Triton decode step")
    parser.add_argument("--block-v", type=int, default=None, help="value channels per program of the decode step")
    parser.add_argument("--warps", type=int, default=1, help="warps per program of the decode step")
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--sweeps", type=int, default=None, help="sweeps of the fast fold's fit (default: the pool's)")
    parser.add_argument("--fold-warps", type=int, default=None)
    parser.add_argument("--fold-programs", type=int, default=None, help="programs per multiprocessor of the fold and store")
    parser.add_argument("--fit-bf16", action="store_true", help="the fast fold's fit in bf16 without rescaling")
    parser.add_argument("--fold-one-pass", action="store_true",
                        help="the fold's records and compensators as one fp16 product each (no lo part)")
    parser.add_argument("--decode-persistent", type=int, default=0, metavar="PROGRAMS_PER_SM",
                        help="decode with a fixed number of programs per multiprocessor walking the heads (0: one per head)")
    parser.add_argument("--fold-backend", choices=["triton", "cuda"], default=None,
                        help="the fast fold's backend (default: LEAPQUANT_FOLD_BACKEND, else triton)")
    parser.add_argument("--decode-backend", choices=["triton", "cuda"], default=None,
                        help="the fast decode step's backend (default: LEAPQUANT_DECODE_BACKEND, else triton)")
    parser.add_argument("--windows", type=int, default=3, help="windows decoded and folded before the comparison of the fold")
    parser.add_argument("--full-cycle", action="store_true",
                        help="also time one complete, state-advancing window as ONE CUDA graph: the window's decode "
                             "steps (cursor 0..L-1, each with its own inputs, appending its record) and the fold, from "
                             "a valid start state restored outside the timed part (and SGLang's window kernels the "
                             "same way with --with-sglang)")
    parser.add_argument("--cycle-order", choices=["lq-first", "sglang-first"], default="lq-first",
                        help="which full cycle is timed first (alternate it between repetitions: the time of a "
                             "kernel can depend on what ran before it)")
    parser.add_argument("--no-exact-times", action="store_true")
    parser.add_argument("--decode-variant", choices=["dot", "fma"], default=None,
                        help="tensor-core walk over the int8 matrix (dot) or fp32 multiply-and-sum (fma); default: the module's")
    args = parser.parse_args()
    fast.FOLD_FIT_BF16 = args.fit_bf16
    fast.FOLD_EXACT_M = not args.fold_one_pass
    if args.fold_backend:
        fast.FOLD_BACKEND = args.fold_backend
    if args.decode_backend:
        fast.DECODE_BACKEND = args.decode_backend
    if args.decode_persistent:
        fast.DECODE_PERSISTENT = True
        fast.DECODE_PROGRAMS_PER_SM = args.decode_persistent
    device = torch.device("cuda")
    HV, H, K, V, L = args.hv, args.h, args.k, args.v, args.window
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    generator = torch.Generator().manual_seed(0)
    kernel = Kernel(device)
    print(f"Triton kernels on {torch.cuda.get_device_name(device)}: HV={HV} H={H} K={K} V={V} window={L} projections={args.dtype}")
    print(f"dtypes: projections {args.dtype}; state int8 levels, fp32 steps and scales, fp16 compensators and records, "
          f"fp16 log decays; fold backend {fast.FOLD_BACKEND}; decode backend {fast.DECODE_BACKEND}")

    def random(*shape, kind=torch.float32):
        return torch.randn(shape, generator=generator).to(device=device, dtype=kind)

    for batch in args.batch_sizes:
        slots = batch + 1
        exact_pool = StatePool(slots, HV, K, V, window_length=L, device=device)
        fast_pool = StatePool(slots, HV, K, V, window_length=L, device=device)
        slot = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
        state = 0.05 * random(batch, HV, V, K)
        for pattern in range(3):
            state += random(batch, HV, V, 1) * random(batch, HV, 1, K) / (pattern + 1)
        start_state = state.clone()
        kernel.store(exact_pool, slot, state)                  # destroys `state`
        # The fast store, from the same state, into the fast pool: compared, then replaced by the exact one.
        source = torch.zeros(slots, HV, V, K, device=device)
        source[1:] = start_state
        fast.store(fast_pool, slot, source)
        stored = torch.empty(batch, HV, V, K, device=device)
        store_loss = {}
        for name, pool in (("exact", exact_pool), ("fast", fast_pool)):
            kernel.export(pool, slot, None, stored)
            store_loss[name] = per_head_difference(stored, start_state)
        store_ms = gpu_ms(lambda: fast.store(fast_pool, slot, source, programs_per_sm=args.fold_programs), device,
                          max(3, args.repeats // 4))
        skip_all = torch.ones(slots, dtype=torch.int32, device=device)
        store_skip_ms = gpu_ms(lambda: fast.store(fast_pool, slot, source, skip_slot=skip_all), device, args.repeats)
        del source
        copy_fields(exact_pool, fast_pool, StatePool.BOUNDARY_FIELDS)
        A_log = (0.3 * random(HV) - 4.0).contiguous()
        dt_bias = (0.1 * random(HV)).contiguous()
        scale = K ** -0.5
        exact_out = torch.empty(batch, HV, V, device=device, dtype=torch.float32)
        fast_out = torch.empty(batch, HV, V, device=device, dtype=torch.float32)
        write_pos = torch.zeros(batch, dtype=torch.int32, device=device)
        true_state = torch.empty(batch, HV, V, K, device=device, dtype=torch.float32)
        after = torch.empty_like(true_state)
        full = torch.full((batch,), L, dtype=torch.int32, device=device)

        worst_output, worst_record, per_position = 0.0, 0.0, []
        output_differences = []
        exact_step_ms = []
        for window in range(args.windows):
            last = window == args.windows - 1
            for position in range(L):
                mixed_qkv = random(batch, 2 * H * K + HV * V, kind=dtype)
                a = (0.5 * random(batch, HV)).to(dtype)
                b = random(batch, HV, kind=dtype)
                write_pos.fill_(position)

                def exact_step():
                    kernel.decode_step_raw(exact_pool, slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale, H, exact_out)

                def fast_step():
                    fast.decode_step_raw(fast_pool, slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale, H, fast_out,
                                         nk=args.nk, block_v=args.block_v, num_warps=args.warps, num_stages=args.stages,
                                         use_dot=None if args.decode_variant is None else args.decode_variant == "dot")

                if last:                    # every call writes the same record again, so it can be repeated
                    per_position.append(gpu_ms(fast_step, device, args.repeats))
                    if not args.no_exact_times:
                        exact_step_ms.append(gpu_ms(exact_step, device, 3))
                exact_step()
                fast_step()
                difference = per_head_difference(fast_out, exact_out)
                output_differences.append(difference.flatten())
                worst_output = max(worst_output, float(difference.max()))
                for name in StatePool.WINDOW_FIELDS:
                    mine = getattr(fast_pool, name)[1:, :, position].float()
                    theirs = getattr(exact_pool, name)[1:, :, position].float()
                    relative = (mine - theirs).abs().max() / theirs.abs().max().clamp_min(1e-30)
                    worst_record = max(worst_record, float(relative))
                copy_fields(exact_pool, fast_pool, StatePool.WINDOW_FIELDS)     # one-step differences only

            # The fold. Both pools hold the same boundary state and the same records here.
            kernel.export(exact_pool, slot, full, true_state)
            saved = {name: getattr(fast_pool, name).clone() for name in StatePool.BOUNDARY_FIELDS}

            def restore_fast():
                for name, tensor in saved.items():
                    getattr(fast_pool, name).copy_(tensor)

            def restore_exact():
                for name, tensor in saved.items():
                    getattr(exact_pool, name).copy_(tensor)

            write_pos.fill_(L - 1)
            if last:
                fast_fold_ms = gpu_ms(lambda: fast.fold(fast_pool, slot, full, sweeps=args.sweeps, num_warps=args.fold_warps, programs_per_sm=args.fold_programs),
                                      device, max(3, args.repeats // 4), before=restore_fast)
                exact_fold_ms = None
                if not args.no_exact_times:
                    exact_fold_ms = gpu_ms(lambda: kernel.fold(exact_pool, slot, write_pos, tokens_added=1), device, 3,
                                           before=restore_exact)
            restore_fast()
            restore_exact()
            exported = torch.zeros(slots, HV, V, K, device=device)
            fast.fold(fast_pool, slot, full, sweeps=args.sweeps, num_warps=args.fold_warps, export_by_slot=exported,
                      programs_per_sm=args.fold_programs)
            export_difference = per_head_difference(exported[1:], true_state)
            del exported
            if fast.FOLD_BACKEND != "triton":
                # A fold that exports runs in Triton whatever the backend: fold again, without export, with the
                # backend under test, so that the losses below are its own.
                restore_fast()
                fast.fold(fast_pool, slot, full, sweeps=args.sweeps, num_warps=args.fold_warps,
                          programs_per_sm=args.fold_programs)
            kernel.fold(exact_pool, slot, write_pos, tokens_added=1)
            losses = {}
            shares = {}
            for name, pool in (("exact", exact_pool), ("fast", fast_pool)):
                kernel.export(pool, slot, None, after)
                losses[name] = per_head_difference(after, true_state)
                shares[name] = captured_share(pool, slot, true_state, min(batch * HV, 256))
            if last:
                break
            # Next window: both continue from the exact pool's boundary state.
            copy_fields(exact_pool, fast_pool, StatePool.BOUNDARY_FIELDS)

        # The fast export (decompression) against the exact one: the boundary state alone, and with
        # the 16 records still in the window buffer (an arithmetic check: both read the same numbers).
        exact_state = torch.empty(batch, HV, V, K, device=device)
        fast_state = torch.empty(batch, HV, V, K, device=device)
        kernel.export(fast_pool, slot, None, exact_state)
        fast.export(fast_pool, slot, fast_state)
        export_plain = float(per_head_difference(fast_state, exact_state).max())
        kernel.export(fast_pool, slot, full, exact_state)
        fast.export(fast_pool, slot, fast_state, tokens_in_window=full)
        export_records = float(per_head_difference(fast_state, exact_state).max())
        del exact_state, fast_state

        differences = torch.cat(output_differences)
        decode_ms = sum(per_position) / L
        print(f"\n  batch {batch}")
        print(f"    decode step, fast vs exact   output difference: median {float(differences.median()):.1e}  "
              f"p99 {float(differences.quantile(0.99)) if differences.numel() < 1e7 else float(differences[:10_000_000].quantile(0.99)):.1e}  "
              f"worst {worst_output:.1e};   records written: worst relative difference {worst_record:.1e}")
        print(f"    fold loss (new boundary state vs full precision)   exact: mean {float(losses['exact'].mean()):.5f} worst {float(losses['exact'].max()):.5f}"
              f"   fast: mean {float(losses['fast'].mean()):.5f} worst {float(losses['fast'].max()):.5f}"
              f"   fast / exact (mean): {float(losses['fast'].mean() / losses['exact'].mean()):.4f}")
        print(f"    store loss (boundary state vs the state stored)   exact: mean {float(store_loss['exact'].mean()):.5f} "
              f"worst {float(store_loss['exact'].max()):.5f}   fast: mean {float(store_loss['fast'].mean()):.5f} "
              f"worst {float(store_loss['fast'].max()):.5f}   fast / exact (mean): "
              f"{float(store_loss['fast'].mean() / store_loss['exact'].mean()):.4f}")
        print(f"    full-precision state exported by the fast fold vs exact export: worst relative difference "
              f"{float(export_difference.max()):.1e}")
        print(f"    fast export vs exact export: worst relative difference {export_plain:.1e} (boundary state), "
              f"{export_records:.1e} (with 16 records)")
        print(f"    share of the best rank-{exact_pool.compensator_count} energy captured (mean, lowest)   "
              f"exact: {shares['exact'][0]:.5f}, {shares['exact'][1]:.5f}   fast: {shares['fast'][0]:.5f}, {shares['fast'][1]:.5f}")
        print(f"    time   fast decode step {decode_ms:.4f} ms (position 0: {per_position[0]:.4f}, last: {per_position[-1]:.4f})"
              f"   fast fold {fast_fold_ms:.3f} ms   ->  {decode_ms + fast_fold_ms / L:.4f} ms per token with a fold every {L} tokens")
        print(f"    time   fast store (adopt) of every row {store_ms:.3f} ms; with every row skipped {store_skip_ms:.4f} ms")
        if exact_step_ms:
            exact_ms = sum(exact_step_ms) / L
            print(f"    time   exact decode step {exact_ms:.4f} ms   exact fold {exact_fold_ms:.3f} ms   ->  "
                  f"{exact_ms + exact_fold_ms / L:.4f} ms per token")

        if args.with_sglang:
            from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
                fused_recurrent_linear_replayssm_decode,
            )
            from sglang.kernels.ops.attention.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule_packed_decode,
            )
            inputs = [(random(batch, 2 * H * K + HV * V, kind=dtype), (0.5 * random(batch, HV)).to(dtype), random(batch, HV, kind=dtype))
                      for _ in range(L)]
            for name, state_dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
                sstate = torch.zeros(slots, HV, V, K, device=device, dtype=state_dtype)
                sstate[1:] = start_state.to(state_dtype)
                sd = torch.zeros(slots, HV, L, V, device=device, dtype=dtype)
                sk = torch.zeros(slots, H, L, K, device=device, dtype=dtype)
                sg = torch.zeros(slots, HV, L, device=device, dtype=torch.float32)
                sout = torch.empty(batch, 1, HV, V, device=device, dtype=dtype)
                times = []
                for position in range(L):
                    mixed_qkv, a, b = inputs[position]
                    write_pos.fill_(position)
                    saved_state = sstate.clone() if position == L - 1 else None

                    def restore_state():
                        if saved_state is not None:
                            sstate.copy_(saved_state)       # its last position rewrites the state

                    def sglang_step():
                        fused_recurrent_linear_replayssm_decode(
                            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=sstate,
                            d_cache=sd, k_cache=sk, g_cache=sg, out=sout, ssm_state_indices=slot, write_pos=write_pos,
                            use_qk_l2norm_in_kernel=True)

                    times.append(gpu_ms(sglang_step, device, args.repeats, before=restore_state))
                print(f"    time   sglang window kernel, {name} state: {sum(times) / L:.4f} ms per token "
                      f"(append steps {sum(times[:-1]) / (L - 1):.4f}, the step that writes the state {times[-1]:.4f})")
                try:
                    pout = torch.empty(batch, 1, HV, V, device=device, dtype=dtype)
                    mixed_qkv, a, b = inputs[0]
                    packed_ms = gpu_ms(lambda: fused_recurrent_gated_delta_rule_packed_decode(
                        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=sstate,
                        out=pout, ssm_state_indices=slot, use_qk_l2norm_in_kernel=True), device, args.repeats)
                    print(f"    time   sglang packed kernel, {name} state: {packed_ms:.4f} ms per token")
                except Exception as error:      # the packed kernel's signature differs between SGLang versions
                    print(f"    (sglang packed kernel not timed: {type(error).__name__}: {error})")
        if args.full_cycle:
            # A complete, state-advancing window (test harness prompt rev. 3, sections 0, 6 and 14): one CUDA graph
            # holds the L decode steps, cursor 0..L-1, each with its own inputs and appending its own record, and the
            # fold (gate preparation, append, fold, fit, scales, quantisation). Before every replay the pool returns
            # to the same valid start state, outside the timed part, so every replay times the same real work rather
            # than one buffer rewritten in place. Per token = graph time / L.
            cycle_inputs = [(random(batch, 2 * H * K + HV * V, kind=dtype), (0.5 * random(batch, HV)).to(dtype),
                             random(batch, HV, kind=dtype)) for _ in range(L)]
            start_fields = {name: getattr(fast_pool, name).clone()
                            for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS}

            def restore_cycle():
                for name, tensor in start_fields.items():
                    getattr(fast_pool, name).copy_(tensor)

            def lq_cycle():
                for position in range(L):
                    mixed_qkv, a, b = cycle_inputs[position]
                    write_pos.fill_(position)
                    fast.decode_step_raw(fast_pool, slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale, H, fast_out,
                                         nk=args.nk, block_v=args.block_v, num_warps=args.warps, num_stages=args.stages,
                                         use_dot=None if args.decode_variant is None else args.decode_variant == "dot")
                fast.fold(fast_pool, slot, full, sweeps=args.sweeps, num_warps=args.fold_warps,
                          programs_per_sm=args.fold_programs)

            cycle_repeats = max(5, args.repeats // 2)
            cycle_result = {}

            def time_leapquant():
                cycle_ms = gpu_ms(lq_cycle, device, cycle_repeats, before=restore_cycle)
                restore_cycle()
                lq_cycle()
                torch.cuda.synchronize(device)
                advanced = any(not torch.equal(getattr(fast_pool, name), start_fields[name])
                               for name in StatePool.BOUNDARY_FIELDS)
                cycle_result["lq"] = cycle_ms
                print(f"    full cycle   LeapQuant: {cycle_ms:.4f} ms per window of {L} tokens = {cycle_ms / L:.4f} ms "
                      f"per token (one CUDA graph: {L} state-advancing decode steps + fold; start state restored "
                      f"outside the timing; median of {cycle_repeats} replays; boundary state advanced by the cycle: "
                      f"{advanced}; fold backend {fast.FOLD_BACKEND}, decode backend {fast.DECODE_BACKEND}; "
                      f"order {args.cycle_order})")

            def time_sglang():
                if not args.with_sglang:
                    return
                from sglang.kernels.ops.attention.linear.fused_recurrent_linear_replayssm import (
                    fused_recurrent_linear_replayssm_decode,
                )
                for name, state_dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
                    sstate = torch.zeros(slots, HV, V, K, device=device, dtype=state_dtype)
                    sstate[1:] = start_state.to(state_dtype)
                    start_sstate = sstate.clone()
                    sd = torch.zeros(slots, HV, L, V, device=device, dtype=dtype)
                    sk = torch.zeros(slots, H, L, K, device=device, dtype=dtype)
                    sg = torch.zeros(slots, HV, L, device=device, dtype=torch.float32)
                    sout = torch.empty(batch, 1, HV, V, device=device, dtype=dtype)

                    def restore_sglang():
                        sstate.copy_(start_sstate)
                        sd.zero_()
                        sk.zero_()
                        sg.zero_()

                    def sglang_cycle():
                        for position in range(L):
                            mixed_qkv, a, b = cycle_inputs[position]
                            write_pos.fill_(position)
                            fused_recurrent_linear_replayssm_decode(
                                mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
                                initial_state=sstate, d_cache=sd, k_cache=sk, g_cache=sg, out=sout,
                                ssm_state_indices=slot, write_pos=write_pos, use_qk_l2norm_in_kernel=True)

                    sglang_ms = gpu_ms(sglang_cycle, device, cycle_repeats, before=restore_sglang)
                    cycle_result[name] = sglang_ms
                    print(f"    full cycle   sglang window kernel, {name} state: {sglang_ms / L:.4f} ms per token "
                          f"(same method: one graph of {L} steps, the last writes the state)")

            if args.cycle_order == "lq-first":
                time_leapquant()
                time_sglang()
            else:
                time_sglang()
                time_leapquant()
            if "bf16" in cycle_result:
                print(f"    full cycle   ratio sglang bf16 window / LeapQuant per token: "
                      f"{cycle_result['bf16'] / cycle_result['lq']:.4f}  (time reduction "
                      f"{1 - cycle_result['lq'] / cycle_result['bf16']:.2%})")
        del exact_pool, fast_pool
        torch.cuda.empty_cache()
    print()


if __name__ == "__main__":
    main()
