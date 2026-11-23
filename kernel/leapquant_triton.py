# SPDX-License-Identifier: Apache-2.0
"""
leapquant_triton.py

The FAST kernels of LeapQuant, in Triton, on the GPU's tensor cores: the decode step, the
fold and the store (adoption of a full-precision state). They read and write the same state
pool (leapquant_kernel.StatePool) as the exact CUDA kernels, so the two sets can be mixed and
each can check the other. The exact kernels are bit-identical to the reference core; these
are the ones a server runs.

    decode_step_raw / decode_step    one token for every row: read the int8 state and the window,
                                     append one record, write the output
    fold                             close the windows that are due: boundary state + records ->
                                     new boundary state
    store                            full-precision states -> boundary states
    records_to_fold                  which rows fold after this token (tensor operations only)

Where the decode step comes from
    Its structure (one program per head, the K-tiled walk, the gate arithmetic, the record ring
    and its cursor) is SGLang's buffered decode kernel,
    `fused_recurrent_linear_replayssm_decode_kernel` in
    sglang/kernels/ops/attention/linear/fused_recurrent_linear_replayssm.py
    (Apache-2.0; itself ported from vLLM's ReplaySSM kernel). Three things differ:

    1. The state is not loaded as fp32 or bf16. It is read from the LeapQuant boundary
       state: int8 levels, two scale vectors, compensators. The int8 load is a quarter
       of the bytes of an fp32 load.
    2. The kernel never writes the state. Every step appends one record, also the last
       step of a window. The fold then turns boundary state plus records into a new
       boundary state.
    3. The state tile is never formed. The int8 matrix is multiplied directly with the two
       read vectors on the tensor cores, and a rank-1 term  c k^T  is read as  c (k . q).

How exact they are
    Not bit-identical to the reference core: the sums are added in another order. But the
    decode step and the full-precision state of the fold round nothing that the reference
    does not round: the int8 levels are exact in fp16, every weighted factor of a tensor-core
    product is split in two fp16 parts (hi + lo), the accumulation is fp32, and the new record
    enters the output as it is stored, as in the reference. The fit of the compensators uses
    fp16 operands (precision is not needed there: what it misses stays in the remainder, which
    is formed from the compensators as stored). kernel/verify_kernel.py --fast checks the
    kernels against the reference core on recorded traces; kernel/bench_triton.py against the
    exact kernels and SGLang's.

The record of one token (the same as the exact kernel writes, see ../leapquant.h):
    window_keys         [slots, HV, L, K]  fp16   beta * key
    window_corrections  [slots, HV, L, V]  fp16   value - held
    window_log_decays   [slots, HV, L]     fp16   log(decay)
"""

import os
import sys
import warnings

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_head(
    i_v,
    i_n,
    i_hv,
    mixed_qkv,  # [B, 2*H*K + HV*V] packed (q | k | v)
    a,  # [B, HV]  raw gate input, or the decay itself if PRE_GATED
    b,  # [B, HV]  raw beta input, or beta itself if PRE_GATED
    A_log,  # [HV]   (not read if PRE_GATED)
    dt_bias,  # [HV]   (not read if PRE_GATED)
    o,  # [B, HV, V] output
    residual,  # int8 [slots, HV, V, K]
    channel_step,  # fp32 [slots, HV, V]
    row_scale,  # fp32 [slots, HV, K]
    comp_c,  # fp16 [slots, HV, R, V]   compensator corrections (value sides)
    comp_k,  # fp16 [slots, HV, R, K]   compensator keys
    win_c,  # fp16 [slots, HV, L, V]   window corrections
    win_k,  # fp16 [slots, HV, L, K]   window keys (beta * key)
    win_g,  # fp16 [slots, HV, L]      window log decays
    slot,  # [B] int32; negative = padded row
    write_pos,  # [B] int32: records already in the window, 0 .. L-1
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    PRE_GATED: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    """The decode step of one head of one row (see leapquant_decode_kernel)."""
    i_h = i_hv // (HV // H)

    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, BC)
    o_r = tl.arange(0, BR)
    o_n = tl.arange(0, 16)
    mask_v = o_v < V

    state_idx = tl.load(slot + i_n).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    b_write_pos = tl.load(write_pos + i_n).to(tl.int64)
    # A padded row, or a cursor out of range: the output is zero and nothing is appended.
    valid = (state_idx >= 0) & (b_write_pos >= 0) & (b_write_pos < L)
    if valid:
        head = state_idx * HV + i_hv
        cache_valid = o_c < b_write_pos

        # ---- gates of the current token ---------------------------------------------------
        if PRE_GATED:
            alpha_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
            beta_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
            g_val = tl.log(alpha_val)
        else:
            # g = -exp(A_log) * softplus(a + dt_bias); decay = exp(g); beta = sigmoid(b), rounded to b's dtype
            A_log_val = tl.load(A_log + i_hv).to(tl.float32)
            b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
            beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)
            x = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32) + tl.load(dt_bias + i_hv).to(tl.float32)
            softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
            g_val = -tl.exp(A_log_val) * softplus_x
            alpha_val = tl.exp(g_val)
        # The decay is stored as its logarithm in fp16. The new state is read with the decay AS STORED.
        g_stored = g_val.to(tl.float16)
        alpha_stored = tl.exp(g_stored.to(tl.float32))

        # ---- the window: decay each record has received from the records after it ---------
        b_g_all = tl.load(win_g + head * L + o_c, mask=cache_valid, other=0.0).to(tl.float32)
        b_g_total = tl.sum(b_g_all, axis=0)
        b_fade = tl.where(cache_valid, tl.exp(b_g_total - tl.cumsum(b_g_all, axis=0)), 0.0)  # [BC]
        b_window_fade = tl.exp(b_g_total)

        p_mix = mixed_qkv + i_n * stride_mixed_qkv_tok
        if USE_QK_L2NORM:
            o_kf = tl.arange(0, BK)
            mask_kf = o_kf < K
            qf = tl.load(p_mix + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
            kf = tl.load(p_mix + H * K + i_h * K + o_kf, mask=mask_kf, other=0).to(tl.float32)
            q_rnorm = 1.0 / tl.sqrt(tl.sum(qf * qf) + 1e-6)
            k_rnorm = 1.0 / tl.sqrt(tl.sum(kf * kf) + 1e-6)
        else:
            q_rnorm = 1.0
            k_rnorm = 1.0

        # ---- one walk over the K tiles ------------------------------------------------------
        #   b_int8      [BV, 16]  the int8 matrix times four weight columns, on tensor cores
        #   b_rec_q/k   [BC]      (key of record j) . q   and   (key of record j) . k
        #   b_comp_q/k  [BR]      the same for the compensators
        b_int8 = tl.zeros([BV, 16], dtype=tl.float32)
        b_acc_q = tl.zeros([BV], dtype=tl.float32)  # the same, CUDA-core variant (USE_DOT=False)
        b_acc_k = tl.zeros([BV], dtype=tl.float32)
        b_rec_q = tl.zeros([BC], dtype=tl.float32)
        b_rec_k = tl.zeros([BC], dtype=tl.float32)
        b_comp_q = tl.zeros([BR], dtype=tl.float32)
        b_comp_k = tl.zeros([BR], dtype=tl.float32)
        new_kq = tl.zeros([1], dtype=tl.float32)  # (new record's key, as stored) . q
        for kk in range(NK):
            o_kt = kk * BKT + tl.arange(0, BKT)
            mask_kt = o_kt < K
            q_c = tl.load(p_mix + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * q_rnorm * scale
            k_c = tl.load(p_mix + H * K + i_h * K + o_kt, mask=mask_kt, other=0).to(tl.float32) * k_rnorm

            # The record of this token, key side: beta * key, rounded to fp16.
            b_new_key = (beta_val * k_c).to(tl.float16)
            tl.store(win_k + (head * L + b_write_pos) * K + o_kt, b_new_key, mask=mask_kt & (i_v == 0))
            new_kq += tl.sum(b_new_key.to(tl.float32) * q_c)

            # The int8 matrix. Its levels are whole numbers up to 127: exact in fp16. Each weight
            # (key_row_scale * q, key_row_scale * k) is split into an fp16 value and an fp16
            # remainder, w = hi + lo, so the tensor-core product loses nothing against fp32.
            p_res = residual + head * V * K + o_v[:, None] * K + o_kt[None, :]
            b_scale = tl.load(row_scale + head * K + o_kt, mask=mask_kt, other=0.0).to(tl.float32)
            w_q = b_scale * q_c
            w_k = b_scale * k_c
            if USE_DOT:
                b_levels = tl.load(p_res, mask=mask_v[:, None] & mask_kt[None, :], other=0).to(tl.float16)  # [BV, BKT]
                # Each weight vector is first divided by its largest entry, so that the split keeps
                # 22 bits also where the weights are small (fp16 has no precision left below 6e-5).
                s_q = tl.max(tl.abs(w_q), axis=0) + 1e-30
                s_k = tl.max(tl.abs(w_k), axis=0) + 1e-30
                w_q = w_q / s_q
                w_k = w_k / s_k
                w_q_hi = w_q.to(tl.float16)
                w_k_hi = w_k.to(tl.float16)
                w_q_lo = (w_q - w_q_hi.to(tl.float32)).to(tl.float16)
                w_k_lo = (w_k - w_k_hi.to(tl.float32)).to(tl.float16)
                b_weights = tl.where(
                    o_n[None, :] == 0, w_q_hi[:, None],
                    tl.where(o_n[None, :] == 1, w_q_lo[:, None],
                             tl.where(o_n[None, :] == 2, w_k_hi[:, None],
                                      tl.where(o_n[None, :] == 3, w_k_lo[:, None], tl.zeros([BKT, 16], dtype=tl.float16)))))
                b_tile_scale = tl.where(o_n < 2, s_q, tl.where(o_n < 4, s_k, 0.0))  # [16]
                b_int8 += tl.dot(b_levels, b_weights, out_dtype=tl.float32) * b_tile_scale[None, :]  # [BV, 16], fp32 accumulation
            else:
                # CUDA cores: the levels as fp32, two multiply-and-sum passes. Plain fp32 arithmetic.
                b_lv = tl.load(p_res, mask=mask_v[:, None] & mask_kt[None, :], other=0).to(tl.float32)  # [BV, BKT]
                b_acc_q += tl.sum(b_lv * w_q[None, :], axis=1)
                b_acc_k += tl.sum(b_lv * w_k[None, :], axis=1)

            # Keys of the window records and of the compensators against q and k.
            p_k = win_k + (head * L + o_c[:, None]) * K + o_kt[None, :]
            b_k_all = tl.load(p_k, mask=cache_valid[:, None] & mask_kt[None, :], other=0).to(tl.float32)  # [BC, BKT]
            b_rec_q += tl.sum(b_k_all * q_c[None, :], axis=1)
            b_rec_k += tl.sum(b_k_all * k_c[None, :], axis=1)
            p_ck = comp_k + (head * R + o_r[:, None]) * K + o_kt[None, :]
            b_ck = tl.load(p_ck, mask=(o_r < R)[:, None] & mask_kt[None, :], other=0).to(tl.float32)  # [BR, BKT]
            b_comp_q += tl.sum(b_ck * q_c[None, :], axis=1)
            b_comp_k += tl.sum(b_ck * k_c[None, :], axis=1)

        # ---- state^T q and state^T k for the state BEFORE this token ------------------------
        #   ((int8 part) * step + compensators) * decay of the whole window + window records
        # The value-side tiles are loaded only now, so that they are not held through the K loop.
        p_c = win_c + (head * L + o_c[None, :]) * V + o_v[:, None]
        b_c_faded = tl.load(p_c, mask=mask_v[:, None] & cache_valid[None, :], other=0).to(tl.float32) * b_fade[None, :]  # [BV, BC]
        p_cc = comp_c + (head * R + o_r[None, :]) * V + o_v[:, None]
        b_cc = tl.load(p_cc, mask=mask_v[:, None] & (o_r < R)[None, :], other=0).to(tl.float32)  # [BV, BR]
        b_step = tl.load(channel_step + head * V + o_v, mask=mask_v, other=0.0).to(tl.float32)  # [BV]
        b_value = tl.load(mixed_qkv + i_n * stride_mixed_qkv_tok + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        if USE_DOT:
            b_int8_q = tl.sum(tl.where(o_n[None, :] < 2, b_int8, 0.0), axis=1) * b_step
            b_int8_k = tl.sum(tl.where((o_n[None, :] >= 2) & (o_n[None, :] < 4), b_int8, 0.0), axis=1) * b_step
        else:
            b_int8_q = b_acc_q * b_step
            b_int8_k = b_acc_k * b_step
        b_state_q = (b_int8_q + tl.sum(b_cc * b_comp_q[None, :], axis=1)) * b_window_fade + tl.sum(b_c_faded * b_rec_q[None, :], axis=1)
        b_state_k = (b_int8_k + tl.sum(b_cc * b_comp_k[None, :], axis=1)) * b_window_fade + tl.sum(b_c_faded * b_rec_k[None, :], axis=1)

        # ---- the token (equations 1 and 3) ----------------------------------------------------
        #   held = state^T (decay * key);  correction = value - held, stored in fp16
        #   output = state_after^T q = correction_as_stored * (stored key . q) + decay_as_stored * state^T q
        b_correction = (b_value - alpha_val * b_state_k).to(tl.float16)
        tl.store(win_c + (head * L + b_write_pos) * V + o_v, b_correction, mask=mask_v)
        if i_v == 0:
            tl.store(win_g + head * L + b_write_pos, g_stored)
        b_o = b_correction.to(tl.float32) * tl.sum(new_kq) + alpha_stored * b_state_q
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    else:
        tl.store(p_o, tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty), mask=mask_v)


@triton.jit
def leapquant_decode_kernel(

    mixed_qkv,  # [B, 2*H*K + HV*V] packed (q | k | v)
    a,  # [B, HV]  raw gate input, or the decay itself if PRE_GATED
    b,  # [B, HV]  raw beta input, or beta itself if PRE_GATED
    A_log,  # [HV]   (not read if PRE_GATED)
    dt_bias,  # [HV]   (not read if PRE_GATED)
    o,  # [B, HV, V] output
    residual,  # int8 [slots, HV, V, K]
    channel_step,  # fp32 [slots, HV, V]
    row_scale,  # fp32 [slots, HV, K]
    comp_c,  # fp16 [slots, HV, R, V]   compensator corrections (value sides)
    comp_k,  # fp16 [slots, HV, R, K]   compensator keys
    win_c,  # fp16 [slots, HV, L, V]   window corrections
    win_k,  # fp16 [slots, HV, L, K]   window keys (beta * key)
    win_g,  # fp16 [slots, HV, L]      window log decays
    slot,  # [B] int32; negative = padded row
    write_pos,  # [B] int32: records already in the window, 0 .. L-1
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    BR: tl.constexpr,
    NK: tl.constexpr,
    BKT: tl.constexpr,
    PRE_GATED: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    USE_DOT: tl.constexpr,
    B,
    PERSISTENT: tl.constexpr,
):
    """
    One decode token for every row: one program per (value block, row, head), or with PERSISTENT
    a fixed number of programs that walk the (row, head) pairs (one value block per head).
    """
    if PERSISTENT:
        for item in range(tl.program_id(0), B * HV, tl.num_programs(0)):
            _decode_head(0, item // HV, item % HV, mixed_qkv, a, b, A_log, dt_bias, o, residual, channel_step, row_scale, comp_c, comp_k, win_c, win_k, win_g, slot, write_pos, scale, stride_mixed_qkv_tok, stride_a_tok, stride_b_tok, H, HV, K, V, R, L, BK, BV, BC, BR, NK, BKT, PRE_GATED, USE_QK_L2NORM, USE_DOT)
    else:
        _decode_head(tl.program_id(0), tl.program_id(1), tl.program_id(2), mixed_qkv, a, b, A_log, dt_bias, o, residual, channel_step, row_scale, comp_c, comp_k, win_c, win_k, win_g, slot, write_pos, scale, stride_mixed_qkv_tok, stride_a_tok, stride_b_tok, H, HV, K, V, R, L, BK, BV, BC, BR, NK, BKT, PRE_GATED, USE_QK_L2NORM, USE_DOT)


@triton.jit
def _work_list_kernel(slot, records, skip_slot, order, count, B, BLOCK: tl.constexpr, FOR_STORE: tl.constexpr,
                      HAS_SKIP: tl.constexpr):
    """One program: the rows that need work, in batch order, packed to the front of `order`;
    their number in count[0]. Fold: rows with records > 0. Store: rows whose slot is not skipped."""
    o = tl.arange(0, BLOCK)
    m = o < B
    s = tl.load(slot + o, mask=m, other=-1)
    if FOR_STORE:
        if HAS_SKIP:
            skipped = tl.load(skip_slot + tl.maximum(s, 0), mask=m & (s >= 0), other=1)
            due = m & (s >= 0) & (skipped == 0)
        else:
            due = m & (s >= 0)
    else:
        r = tl.load(records + o, mask=m, other=0)
        due = m & (s >= 0) & (r > 0)
    due_int = due.to(tl.int32)
    position = tl.cumsum(due_int, axis=0) - 1
    tl.store(order + position, o, mask=due)
    tl.store(count, tl.sum(due_int, axis=0))


@triton.jit
def _orthonormal_columns(U, o_c, R: tl.constexpr):
    """Makes the first R columns of U [BK, BC] orthonormal, left to right (Gram-Schmidt).
    A column that is (nearly) a combination of the ones before it is set to zero."""
    u_start = tl.sum(U * U, axis=0)  # [BC]
    for p in range(R):
        is_p = o_c == p
        u = tl.sum(tl.where(is_p[None, :], U, 0.0), axis=1)  # [BK]
        u_squared = tl.sum(u * u, axis=0)
        keep = u_squared > 1e-6 * tl.sum(tl.where(is_p, u_start, 0.0), axis=0)
        u = tl.where(keep, u * tl.rsqrt(tl.where(keep, u_squared, 1.0)), 0.0)
        along = tl.sum(U * u[:, None], axis=0)  # [BC]: u . every column
        U = tl.where(is_p[None, :], u[:, None], U - u[:, None] * tl.where(o_c > p, along, 0.0)[None, :])
    return U


@triton.jit
def _write_boundary_state(M, X, U, head, residual, channel_step, row_scale, comp_c, comp_k,
                          o_v, o_k, o_c, mask_v, mask_k,
                          K: tl.constexpr, V: tl.constexpr, R: tl.constexpr, BK: tl.constexpr,
                          LEVELS: tl.constexpr, USE_SMOOTHING: tl.constexpr):
    """
    The end of a fold or a store: from the full-precision state M [BV, BK] (pool layout) and
    a fit  M ~ X U^T  (X [BV, BC], U [BK, BC] with orthonormal columns), the new boundary state.
    As the reference core: compensators in fp16, subtracted as stored; smoothing; one step per
    value channel; rounding to the nearest level (ties to even).
    """
    comp_valid = o_c < R
    # Each pair gets sqrt(strength) on both sides (as the reference core does), then fp16.
    x_squared = tl.sum(X * X, axis=0)  # [BC]
    present = (x_squared > 0.0) & comp_valid
    balance = tl.sqrt(tl.sqrt(tl.where(present, x_squared, 1.0)))
    new_k = tl.where(present[None, :], U * balance[None, :], 0.0).to(tl.float16)  # [BK, BC]
    new_c = tl.where(present[None, :], X / balance[None, :], 0.0).to(tl.float16)  # [BV, BC]
    # The products of two fp16 numbers are exact in fp32; the sum over BC is in fp32. Negating an
    # fp16 value is exact, so M - new_c new_k^T is accumulated in place.
    remainder = tl.dot(-new_c, tl.trans(new_k), acc=M, out_dtype=tl.float32)

    if USE_SMOOTHING:
        new_scale = tl.maximum(tl.sqrt(tl.sum(tl.abs(remainder), axis=0) / V), 1e-6)  # [BK]
    else:
        new_scale = tl.full([BK], 1.0, dtype=tl.float32)
    remainder = remainder / new_scale[None, :]
    new_step = tl.max(tl.abs(remainder), axis=1) / LEVELS  # [BV]
    has_step = new_step > 0.0
    ratio = remainder / tl.where(has_step, new_step, 1.0)[:, None]
    # Round to nearest, ties to even: adding and subtracting 1.5 * 2^23 does that in fp32.
    rounded = (ratio + 12582912.0) - 12582912.0
    rounded = tl.minimum(tl.maximum(rounded, -LEVELS), LEVELS)
    rounded = tl.where(has_step[:, None], rounded, 0.0)

    p_res = residual + head * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_res, rounded.to(tl.int8), mask=mask_v[:, None] & mask_k[None, :])
    tl.store(channel_step + head * V + o_v, new_step, mask=mask_v)
    tl.store(row_scale + head * K + o_k, new_scale, mask=mask_k)
    tl.store(comp_k + (head * R + o_c[:, None]) * K + o_k[None, :], tl.trans(new_k), mask=comp_valid[:, None] & mask_k[None, :])
    tl.store(comp_c + (head * R + o_c[:, None]) * V + o_v[None, :], tl.trans(new_c), mask=comp_valid[:, None] & mask_v[None, :])


@triton.jit
def _levels(p_res, o_v, o_k, mask_vk, K: tl.constexpr):
    """The int8 levels of one head as fp16 (exact). Read again at every use (from L2): holding
    the tile in registers in both operand layouts cost about 128 registers per thread."""
    return tl.load(p_res + o_v[:, None] * K + o_k[None, :], mask=mask_vk, other=0, volatile=True).to(tl.float16)


@triton.jit
def _state_times(p_res, o_v, o_k, mask_vk, U, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c, K: tl.constexpr):
    """X = M U for the end-of-window state M of the fold, which is never built (see the fold).
    fp16 operands: precision is not needed for the fit."""
    w = U * scale[:, None]
    w_max = tl.max(tl.abs(w), axis=0) + 1e-30
    X = tl.dot(_levels(p_res, o_v, o_k, mask_vk, K), (w / w_max[None, :]).to(tl.float16),
               out_dtype=tl.float32) * (step[:, None] * (w_max * fade_all)[None, :])
    u_h = U.to(tl.float16)
    s_rec = tl.dot(rec_k, u_h, out_dtype=tl.float32) * fade[:, None]
    s_old = tl.dot(old_k, u_h, out_dtype=tl.float32) * fade_all
    s_rec_max = tl.max(tl.abs(s_rec), axis=0) + 1e-30
    s_old_max = tl.max(tl.abs(s_old), axis=0) + 1e-30
    X += tl.dot(tl.trans(rec_c), (s_rec / s_rec_max[None, :]).to(tl.float16), out_dtype=tl.float32) * s_rec_max[None, :]
    X += tl.dot(tl.trans(old_c), (s_old / s_old_max[None, :]).to(tl.float16), out_dtype=tl.float32) * s_old_max[None, :]
    return X


@triton.jit
def _fp16_sweep(p_res, o_v, o_k, mask_vk, X, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c, o_c,
                R: tl.constexpr, K: tl.constexpr):
    """One sweep of the fit (U = M^T X, orthonormal U, X = M U) with fp16 operands, each
    rescaled by its largest entry first."""
    # U = M^T X
    w = X * step[:, None]
    w_max = tl.max(tl.abs(w), axis=0) + 1e-30
    U = tl.dot(tl.trans(_levels(p_res, o_v, o_k, mask_vk, K)), (w / w_max[None, :]).to(tl.float16),
               out_dtype=tl.float32) * (scale[:, None] * (w_max * fade_all)[None, :])
    x_max = tl.max(tl.abs(X), axis=0) + 1e-30
    x_h = (X / x_max[None, :]).to(tl.float16)
    t_rec = tl.dot(rec_c, x_h, out_dtype=tl.float32) * (fade[:, None] * x_max[None, :])  # [BC records, BC vectors]
    t_old = tl.dot(old_c, x_h, out_dtype=tl.float32) * (fade_all * x_max)[None, :]
    t_rec_max = tl.max(tl.abs(t_rec), axis=0) + 1e-30
    t_old_max = tl.max(tl.abs(t_old), axis=0) + 1e-30
    U += tl.dot(tl.trans(rec_k), (t_rec / t_rec_max[None, :]).to(tl.float16), out_dtype=tl.float32) * t_rec_max[None, :]
    U += tl.dot(tl.trans(old_k), (t_old / t_old_max[None, :]).to(tl.float16), out_dtype=tl.float32) * t_old_max[None, :]

    U = _orthonormal_columns(U, o_c, R)
    X = _state_times(p_res, o_v, o_k, mask_vk, U, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c, K)
    return X, U


@triton.jit
def _bf16_sweep(p_res, o_v, o_k, mask_vk, X, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c, o_c,
                R: tl.constexpr, K: tl.constexpr):
    """The same sweep with bf16 operands: bf16 has fp32's range, so nothing is rescaled (every
    rescaling is a reduction across the warps). The levels are exact in bf16 too; the fit needs
    no precision (what it misses stays in the remainder)."""
    levels = tl.load(p_res + o_v[:, None] * K + o_k[None, :], mask=mask_vk, other=0, volatile=True).to(tl.bfloat16)
    U = tl.dot(tl.trans(levels), (X * step[:, None]).to(tl.bfloat16), out_dtype=tl.float32) * (scale * fade_all)[:, None]
    x_b = X.to(tl.bfloat16)
    t_rec = tl.dot(rec_c.to(tl.bfloat16), x_b, out_dtype=tl.float32) * fade[:, None]
    t_old = tl.dot(old_c.to(tl.bfloat16), x_b, out_dtype=tl.float32) * fade_all
    U = tl.dot(tl.trans(rec_k).to(tl.bfloat16), t_rec.to(tl.bfloat16), acc=U, out_dtype=tl.float32)
    U = tl.dot(tl.trans(old_k).to(tl.bfloat16), t_old.to(tl.bfloat16), acc=U, out_dtype=tl.float32)
    U = _orthonormal_columns(U, o_c, R)
    levels = tl.load(p_res + o_v[:, None] * K + o_k[None, :], mask=mask_vk, other=0, volatile=True).to(tl.bfloat16)
    X = tl.dot(levels, (U * scale[:, None]).to(tl.bfloat16), out_dtype=tl.float32) * (step * fade_all)[:, None]
    u_b = U.to(tl.bfloat16)
    s_rec = tl.dot(rec_k.to(tl.bfloat16), u_b, out_dtype=tl.float32) * fade[:, None]
    s_old = tl.dot(old_k.to(tl.bfloat16), u_b, out_dtype=tl.float32) * fade_all
    X = tl.dot(tl.trans(rec_c).to(tl.bfloat16), s_rec.to(tl.bfloat16), acc=X, out_dtype=tl.float32)
    X = tl.dot(tl.trans(old_c).to(tl.bfloat16), s_old.to(tl.bfloat16), acc=X, out_dtype=tl.float32)
    return X, U


@triton.jit
def leapquant_fold_kernel(
    residual,  # int8 [slots, HV, V, K]    read, then rewritten
    channel_step,  # fp32 [slots, HV, V]   read, then rewritten
    row_scale,  # fp32 [slots, HV, K]      read, then rewritten
    comp_c,  # fp16 [slots, HV, R, V]      read, then rewritten
    comp_k,  # fp16 [slots, HV, R, K]      read, then rewritten
    win_c,  # fp16 [slots, HV, L, V]       read
    win_k,  # fp16 [slots, HV, L, K]       read
    win_g,  # fp16 [slots, HV, L]          read
    slot,  # [B] int32; negative = padded row
    records,  # [B] int32: records in the window to fold; 0 = this row does not fold
    export,  # fp32 [slots, HV, V, K] or a dummy: the full-precision state of every head that folds
    order,  # [B] int32: the rows that fold first (see _work_list_kernel)
    count,  # [1] int32: how many rows fold
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    SWEEPS: tl.constexpr,
    LEVELS: tl.constexpr,
    USE_SMOOTHING: tl.constexpr,
    HAS_EXPORT: tl.constexpr,
    FIT_BF16: tl.constexpr,
    EXACT_M: tl.constexpr,
):
    """
    The fold of one head (one program per head; the whole matrix is one tile).

    The state at the end of the window is never built for the fit. It is
        M = F * (step * levels * row_scale  +  sum_r cc_r ck_r^T)  +  sum_j f_j c_j k_j^T
    with f_j the decay record j received from the records after it and F the decay of the
    whole window. A product M x or M^T u is one tensor-core pass over the int8 levels plus
    two small products with the 16 records and the compensators.

    1. Fit. Block iteration on R vectors, started from the value sides of the compensators
       of the last fold:  U = M^T X, make U orthonormal, X = M U.  Then  M ~ X U^T.
       Precision is not needed here (fp16 weights): whatever the fit misses stays in the
       remainder, which is formed from the compensators AS STORED.
    2. Remainder = M - (new compensators as stored), on tensor cores. Every weighted factor
       is split in two fp16 parts, so this loses nothing against fp32.
    3. Smoothing, one step per value channel, rounding: as in the reference core.
    """
    # A fixed number of programs (so the launch can be captured in a CUDA graph) walk the
    # (row, head) pairs of the rows that fold; with no row due, every program ends at once.
    total = tl.load(count) * HV
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        i_n = tl.load(order + item // HV)
        i_hv = item % HV
        state_idx = tl.load(slot + i_n).to(tl.int64)
        n = tl.load(records + i_n).to(tl.int64)
        head = state_idx * HV + i_hv

        o_v = tl.arange(0, BV)
        o_k = tl.arange(0, BK)
        o_c = tl.arange(0, BC)
        mask_v = o_v < V
        mask_k = o_k < K
        rec_valid = (o_c < n) & (o_c < L)
        comp_valid = o_c < R

        # ---- decay of the window -------------------------------------------------------------
        b_g = tl.load(win_g + head * L + o_c, mask=rec_valid, other=0.0).to(tl.float32)
        g_total = tl.sum(b_g, axis=0)
        fade = tl.where(rec_valid, tl.exp(g_total - tl.cumsum(b_g, axis=0)), 0.0)  # [BC]
        fade_all = tl.exp(g_total)

        # ---- the stored state ----------------------------------------------------------------
        p_res = residual + head * V * K
        mask_vk = mask_v[:, None] & mask_k[None, :]
        step = tl.load(channel_step + head * V + o_v, mask=mask_v, other=0.0).to(tl.float32)  # [BV]
        scale = tl.load(row_scale + head * K + o_k, mask=mask_k, other=0.0).to(tl.float32)  # [BK]
        # Rows of these four tiles are records (or compensators); fp16 as stored.
        rec_c = tl.load(win_c + (head * L + o_c[:, None]) * V + o_v[None, :], mask=rec_valid[:, None] & mask_v[None, :], other=0.0)  # [BC, BV]
        rec_k = tl.load(win_k + (head * L + o_c[:, None]) * K + o_k[None, :], mask=rec_valid[:, None] & mask_k[None, :], other=0.0)  # [BC, BK]
        old_c = tl.load(comp_c + (head * R + o_c[:, None]) * V + o_v[None, :], mask=comp_valid[:, None] & mask_v[None, :], other=0.0)  # [BC, BV]
        old_k = tl.load(comp_k + (head * R + o_c[:, None]) * K + o_k[None, :], mask=comp_valid[:, None] & mask_k[None, :], other=0.0)  # [BC, BK]

        # ---- 1. the fit ----------------------------------------------------------------------
        if SWEEPS == 0:
            # Half sweep: the key sides of the stored compensators are kept (made orthonormal) and
            # only the value sides are fitted again, X = M U: the best value sides for those keys.
            U = _orthonormal_columns(tl.trans(old_k).to(tl.float32), o_c, R)
            X = _state_times(p_res, o_v, o_k, mask_vk, U, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c, K)
        # Start vectors: value sides of the old compensators. A missing one (all zero) is replaced
        # by the correction of one of the newest records.
        else:
            X = tl.trans(old_c).to(tl.float32)  # [BV, BC]; column p = vector p; columns >= R stay zero
            missing = (tl.sum(X * X, axis=0) == 0.0) & comp_valid & (n - 1 - o_c >= 0)
            p_alt = win_c + (head * L + (n - 1 - o_c)[None, :]) * V + o_v[:, None]
            X += tl.load(p_alt, mask=mask_v[:, None] & missing[None, :], other=0.0).to(tl.float32)
            U = tl.zeros([BK, BC], dtype=tl.float32)
        for _ in range(SWEEPS):
            if FIT_BF16:
                X, U = _bf16_sweep(p_res, o_v, o_k, mask_vk, X, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c,
                                   o_c, R, K)
            else:
                X, U = _fp16_sweep(p_res, o_v, o_k, mask_vk, X, scale, step, fade, fade_all, rec_k, old_k, rec_c, old_c,
                                   o_c, R, K)

        # ---- 2. the full-precision state M, on tensor cores ----------------------------------------
        # Every weighted factor is split in two fp16 parts (hi + lo), so this loses nothing against fp32.
        a = tl.trans(rec_c).to(tl.float32) * fade[None, :]  # [BV, BC]
        a_hi = a.to(tl.float16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
        c = tl.trans(old_c).to(tl.float32) * fade_all
        c_hi = c.to(tl.float16)
        c_lo = (c - c_hi.to(tl.float32)).to(tl.float16)
        # (Accumulated in place: a separate [BV, BK] result per product would double the registers.)
        M = _levels(p_res, o_v, o_k, mask_vk, K).to(tl.float32) * (step * fade_all)[:, None] * scale[None, :]
        M = tl.dot(a_hi, rec_k, acc=M, out_dtype=tl.float32)
        if EXACT_M:
            M = tl.dot(a_lo, rec_k, acc=M, out_dtype=tl.float32)
        M = tl.dot(c_hi, old_k, acc=M, out_dtype=tl.float32)
        if EXACT_M:
            M = tl.dot(c_lo, old_k, acc=M, out_dtype=tl.float32)
        if HAS_EXPORT:
            tl.store(export + (state_idx * HV + i_hv) * V * K + o_v[:, None] * K + o_k[None, :], M, mask=mask_vk)

        # ---- 3. new compensators, smoothing, steps, rounding ----------------------------------------
        _write_boundary_state(M, X, U, head, residual, channel_step, row_scale, comp_c, comp_k,
                              o_v, o_k, o_c, mask_v, mask_k, K, V, R, BK, LEVELS, USE_SMOOTHING)


@triton.jit
def leapquant_store_kernel(
    source,  # fp32 [entries, HV, V, K]; entries `source_stride` elements apart
    residual,  # int8 [slots, HV, V, K]    written
    channel_step,  # fp32 [slots, HV, V]   written
    row_scale,  # fp32 [slots, HV, K]      written
    comp_c,  # fp16 [slots, HV, R, V]      written
    comp_k,  # fp16 [slots, HV, R, K]      written
    slot,  # [B] int32; negative = padded row
    source_index,  # [B] int32: entry of `source` per row (or a dummy: the entry of the row's slot)
    order,  # [B] int32: the rows to store first (see _work_list_kernel)
    count,  # [1] int32: how many rows to store
    source_stride,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    SWEEPS: tl.constexpr,
    LEVELS: tl.constexpr,
    USE_SMOOTHING: tl.constexpr,
    HAS_INDEX: tl.constexpr,
):
    """
    A full-precision state becomes a boundary state (adoption of a request; a checkpoint).
    One program per head. As the reference core's store without previous compensators: the
    fit starts from the longest key rows of the matrix and runs two more sweeps.
    """
    total = tl.load(count) * HV
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        i_n = tl.load(order + item // HV)
        i_hv = item % HV
        state_idx = tl.load(slot + i_n).to(tl.int64)
        if HAS_INDEX:
            entry = tl.load(source_index + i_n).to(tl.int64)
        else:
            entry = state_idx
        head = state_idx * HV + i_hv

        o_v = tl.arange(0, BV)
        o_k = tl.arange(0, BK)
        o_c = tl.arange(0, BC)
        mask_v = o_v < V
        mask_k = o_k < K
        M = tl.load(source + entry * source_stride + i_hv * V * K + o_v[:, None] * K + o_k[None, :],
                    mask=mask_v[:, None] & mask_k[None, :], other=0.0).to(tl.float32)  # [BV, BK]

        # The fit, in fp16 on tensor cores (precision is not needed here: see the fold).
        m_max = tl.max(tl.max(tl.abs(M), axis=1), axis=0) + 1e-30
        Mh = (M / m_max).to(tl.float16)
        # Start vectors: key rows (columns of this [V, K] tile), chosen greedily: the longest, then
        # the one with the most length left outside the directions already chosen (a pivoted QR).
        # Plain "the R longest" can pick R near-copies of the strongest pattern.
        length = tl.sum(M * M, axis=0)  # [BK]: squared length left, per key row
        X = tl.zeros([BV, BC], dtype=tl.float32)  # the chosen directions, orthonormal
        for p in range(R):
            longest = tl.max(length, axis=0)
            index = tl.argmax(length, axis=0)
            column = tl.sum(tl.where(o_k[None, :] == index, M, 0.0), axis=1)  # [BV]
            column = column - tl.sum(X * tl.sum(X * column[:, None], axis=0)[None, :], axis=1)
            column_squared = tl.sum(column * column, axis=0)
            use = (longest > 0.0) & (column_squared > 0.0)
            column = tl.where(use, column * tl.rsqrt(tl.where(use, column_squared, 1.0)), 0.0)
            X = tl.where(o_c[None, :] == p, column[:, None], X)
            along = tl.sum(column[:, None] * M, axis=0)  # [BK]: each key row along the new direction
            length = tl.where(o_k == index, -1.0, length - along * along)
        U = tl.zeros([BK, BC], dtype=tl.float32)
        for _ in range(SWEEPS + 2):
            x_max = tl.max(tl.abs(X), axis=0) + 1e-30
            U = tl.dot(tl.trans(Mh), (X / x_max[None, :]).to(tl.float16), out_dtype=tl.float32) * (m_max * x_max)[None, :]
            U = _orthonormal_columns(U, o_c, R)
            X = tl.dot(Mh, U.to(tl.float16), out_dtype=tl.float32) * m_max

        _write_boundary_state(M, X, U, head, residual, channel_step, row_scale, comp_c, comp_k,
                              o_v, o_k, o_c, mask_v, mask_k, K, V, R, BK, LEVELS, USE_SMOOTHING)


@triton.jit
def leapquant_export_kernel(
    residual, channel_step, row_scale, comp_c, comp_k, win_c, win_k, win_g,
    slot,  # [B] int32; negative = skipped row (its output is left as it is)
    tokens,  # [B] int32: records in the window to include (or a dummy: none)
    out,  # fp32 [B, HV, V, K]
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    L: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BC: tl.constexpr,
    HAS_TOKENS: tl.constexpr,
):
    """
    The state of one head in full precision (decompression): the boundary state, plus the
    first `tokens` records of the window if given. As the exact kernel's export, with the sums
    of the compensators and records on tensor cores (products of fp16 values are exact in fp32;
    the faded corrections are split in two fp16 parts).
    """
    i_n = tl.program_id(0)
    i_hv = tl.program_id(1)
    state_idx = tl.load(slot + i_n).to(tl.int64)
    if state_idx < 0:
        return
    head = state_idx * HV + i_hv
    o_v = tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    o_c = tl.arange(0, BC)
    mask_v = o_v < V
    mask_k = o_k < K
    mask_vk = mask_v[:, None] & mask_k[None, :]
    comp_valid = o_c < R
    levels = tl.load(residual + head * V * K + o_v[:, None] * K + o_k[None, :], mask=mask_vk, other=0).to(tl.float32)
    step = tl.load(channel_step + head * V + o_v, mask=mask_v, other=0.0)
    scale = tl.load(row_scale + head * K + o_k, mask=mask_k, other=0.0)
    M = levels * step[:, None] * scale[None, :]
    cc = tl.load(comp_c + (head * R + o_c[None, :]) * V + o_v[:, None], mask=mask_v[:, None] & comp_valid[None, :], other=0.0)  # [BV, BC]
    ck = tl.load(comp_k + (head * R + o_c[:, None]) * K + o_k[None, :], mask=comp_valid[:, None] & mask_k[None, :], other=0.0)  # [BC, BK]
    M = tl.dot(cc, ck, acc=M, out_dtype=tl.float32)
    if HAS_TOKENS:
        n = tl.minimum(tl.maximum(tl.load(tokens + i_n).to(tl.int64), 0), L)
        rec_valid = o_c < n
        b_g = tl.load(win_g + head * L + o_c, mask=rec_valid, other=0.0).to(tl.float32)
        g_total = tl.sum(b_g, axis=0)
        fade = tl.where(rec_valid, tl.exp(g_total - tl.cumsum(b_g, axis=0)), 0.0)
        a = tl.load(win_c + (head * L + o_c[None, :]) * V + o_v[:, None], mask=mask_v[:, None] & rec_valid[None, :],
                    other=0.0).to(tl.float32) * fade[None, :]  # [BV, BC]
        a_hi = a.to(tl.float16)
        a_lo = (a - a_hi.to(tl.float32)).to(tl.float16)
        rk = tl.load(win_k + (head * L + o_c[:, None]) * K + o_k[None, :], mask=rec_valid[:, None] & mask_k[None, :], other=0.0)
        M = M * tl.exp(g_total)
        M = tl.dot(a_hi, rk, acc=M, out_dtype=tl.float32)
        M = tl.dot(a_lo, rk, acc=M, out_dtype=tl.float32)
    tl.store(out + ((i_n * HV + i_hv) * V + o_v[:, None]) * K + o_k[None, :], M, mask=mask_vk)


# Tiling found fastest on an H100 (14 Nov 2026, batch 512): one program per head covering all
# value channels, K tiles of 32 (four for K = 128), one warp, three pipeline stages: 0.254 ms per step.
DEFAULT_K_TILE = 32
DEFAULT_BLOCK_V = 128


# The tensor-core walk over the int8 matrix (True) or plain fp32 multiply-and-sum (False).
DEFAULT_USE_DOT = True


# A fixed number of decode programs walking the heads (True) or one program per head (False);
# programs per streaming multiprocessor when True. Under evaluation, 14 Nov 2026.
DECODE_PERSISTENT = False
DECODE_PROGRAMS_PER_SM = 12

# A register cap for the decode (Triton's maxnreg), or None. One warp per head at 166 registers fits 12 warps
# per multiprocessor (Nsight: DRAM 66%); a cap trades spills for more warps. Under evaluation, 14 Nov 2026.
DECODE_MAXNREG = int(os.environ["LEAPQUANT_DECODE_MAXNREG"]) if os.environ.get("LEAPQUANT_DECODE_MAXNREG") else None


_DECODE_NOTED = set()


def _note_decode(what):
    """Says once per process which decode kernel runs (a log line, so that a run shows what it measured)."""
    if what not in _DECODE_NOTED:
        _DECODE_NOTED.add(what)
        print(f"LeapQuant decode: {what}", file=sys.stderr, flush=True)


def _require_one_step_per_column(pool):
    """The kernels below multiply the whole key sum by one step per value channel."""
    if getattr(pool, "scale_group_count", 1) > 1:
        raise ValueError("the fast kernels keep one step per value column; a scale group uses the exact kernel")


def _launch(pool, mixed_qkv, a, b, A_log, dt_bias, scale, slot, write_pos, out, key_heads, pre_gated, l2norm,
            nk, block_v, num_warps, num_stages, use_dot=None, persistent=None):
    _require_one_step_per_column(pool)
    HV, V, K, R, L = pool.value_heads, pool.value_channels, pool.key_channels, pool.compensator_count, pool.window_length
    H = key_heads
    B = mixed_qkv.shape[0]
    if mixed_qkv.dim() != 2 or mixed_qkv.shape[1] < 2 * H * K + HV * V or mixed_qkv.stride(1) != 1:
        raise ValueError(f"mixed_qkv must be [B, >= {2 * H * K + HV * V}] with unit stride in its last dimension")
    if HV % H != 0:
        raise ValueError("key_heads must divide value_heads")
    if R < 1:
        raise ValueError("the Triton decode step needs at least one compensator slot")
    for name, tensor, dtype in (("slot", slot, torch.int32), ("write_pos", write_pos, torch.int32)):
        if tensor.dtype != dtype or tensor.shape[0] != B or not tensor.is_contiguous():
            raise TypeError(f"{name} must be a contiguous {dtype} tensor of length {B}")
    if tuple(out.shape) != (B, HV, V) or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous [{B}, {HV}, {V}] tensor")
    if DECODE_BACKEND == "cuda" and out.dtype in (torch.float32, torch.float16, torch.bfloat16):
        module = _cuda_fold_module()
        if module is not None and hasattr(module, "decode") and module.decode_supports(pool, mixed_qkv, a, b, out):
            _note_decode(f"CUDA decode, variant {getattr(module, 'DECODE_VARIANT', '?')}")
            return module.decode(pool, mixed_qkv, a, b, A_log, dt_bias, scale, slot, write_pos, out, H, pre_gated, l2norm)
        _note_decode("Triton decode (LEAPQUANT_DECODE_BACKEND=cuda, but this call is outside the CUDA decode's limits)")
    BK = triton.next_power_of_2(K)
    nk = nk or max(1, BK // DEFAULT_K_TILE)
    BKT = BK // nk
    if BKT < 16:
        raise ValueError("the K tile must be at least 16 wide for the tensor-core product")
    BV = min(triton.next_power_of_2(V), block_v or DEFAULT_BLOCK_V)
    BC = max(16, triton.next_power_of_2(L))
    BR = max(2, triton.next_power_of_2(R))       # elementwise only: no tensor-core minimum
    persistent = DECODE_PERSISTENT if persistent is None else persistent
    if persistent:
        BV = triton.next_power_of_2(V)                     # one value block per head
        grid = (_programs(out.device, B * HV, DECODE_PROGRAMS_PER_SM),)
    else:
        grid = (triton.cdiv(V, BV), B, HV)
    extra = {"maxnreg": DECODE_MAXNREG} if DECODE_MAXNREG else {}
    leapquant_decode_kernel[grid](
        mixed_qkv, a, b, A_log, dt_bias, out,
        pool.rounded_residual, pool.value_channel_step, pool.key_row_scale,
        pool.compensator_corrections, pool.compensator_keys,
        pool.window_corrections, pool.window_keys, pool.window_log_decays,
        slot, write_pos, scale,
        stride_mixed_qkv_tok=mixed_qkv.stride(0), stride_a_tok=a.stride(0), stride_b_tok=b.stride(0),
        H=H, HV=HV, K=K, V=V, R=R, L=L, BK=BK, BV=BV, BC=BC, BR=BR, NK=nk, BKT=BKT,
        PRE_GATED=pre_gated, USE_QK_L2NORM=l2norm, USE_DOT=DEFAULT_USE_DOT if use_dot is None else use_dot,
        B=B, PERSISTENT=persistent, num_warps=num_warps, num_stages=num_stages, **extra,
    )
    return out


def decode_step_raw(pool, slot, tokens_in_window, mixed_qkv, a, b, A_log, dt_bias, scale, key_heads, out,
                    nk=None, block_v=None, num_warps=1, num_stages=3, use_dot=None):  # defaults: see DEFAULT_K_TILE
    """
    One decode token for every row, from what the model's projections give (the fast kernel).
    Same arguments and same effect on the pool as leapquant_kernel.Kernel.decode_step_raw.
    `out` may be float32 or the model's dtype.
    """
    return _launch(pool, mixed_qkv, a, b, A_log, dt_bias, scale, slot, tokens_in_window, out, key_heads,
                   pre_gated=False, l2norm=True, nk=nk, block_v=block_v, num_warps=num_warps, num_stages=num_stages,
                   use_dot=use_dot)


def decode_step(pool, slot, tokens_in_window, decay, beta, key, query, value, out,
                nk=None, block_v=None, num_warps=1, num_stages=3, use_dot=None):
    """
    The same step from gated inputs, with the arguments of leapquant_kernel.Kernel.decode_step:
    decay, beta [B, HV]; key, query [B, H, K] as the recurrence uses them; value [B, HV, V].
    """
    rows = slot.shape[0]
    key_heads = key.shape[1]
    packed = torch.cat([query.reshape(rows, -1), key.reshape(rows, -1), value.reshape(rows, -1)], dim=1).contiguous()
    return _launch(pool, packed, decay, beta, decay, decay, 1.0, slot, tokens_in_window, out, key_heads,
                   pre_gated=True, l2norm=False, nk=nk, block_v=block_v, num_warps=num_warps, num_stages=num_stages,
                   use_dot=use_dot)


# Measured on an H100 (kernel/bench_triton.py --fold-warps): 4 warps per head is fastest.
DEFAULT_FOLD_WARPS = 4
# The fold's fit in bf16 without rescaling (True) or in fp16 with rescaling (False). Measured on an
# H100, 14 Nov 2026 (results/fit_bf16): no faster (2.39 ms at batch 512 either way), the same fold
# loss, all 9 prompt traces pass. Kept off: fp16 keeps 3 more bits in the fit.
FOLD_FIT_BF16 = False
# The state the fold rounds: the records' and compensators' terms each as two fp16 products (hi + lo:
# nothing lost against fp32; True) or as one (the faded values rounded to fp16 first, about 5e-4 of
# those terms; False). Under evaluation, 14 Nov 2026.
FOLD_EXACT_M = True
# Programs per streaming multiprocessor of the fold and store launches (they walk a work list).
DEFAULT_FOLD_PROGRAMS_PER_SM = 4
# The fold's backend: "triton" (leapquant_fold_kernel above) or "cuda" (kernel/leapquant_fast_fold.cu through
# kernel/leapquant_fast_cuda.py: the same contract, written by hand for sm_90). Pools outside the CUDA kernel's
# limits, and folds that export the full-precision state, take the Triton fold either way.
FOLD_BACKEND = os.environ.get("LEAPQUANT_FOLD_BACKEND", "triton")
# The decode step's backend: "triton" (leapquant_decode_kernel) or "cuda" (kernel/leapquant_fast_decode.cu, built
# into the same library as the CUDA fold). Rows outside the CUDA kernel's limits take the Triton step.
DECODE_BACKEND = os.environ.get("LEAPQUANT_DECODE_BACKEND", "triton")
_FAST_CUDA = None


def _cuda_fold_module():
    """kernel/leapquant_fast_cuda.py if its library loads, else None (warned once)."""
    global _FAST_CUDA
    if _FAST_CUDA is None:
        try:
            import leapquant_fast_cuda as module
        except ImportError:
            from . import leapquant_fast_cuda as module
        if module.available():
            _FAST_CUDA = module
        else:
            warnings.warn(f"LEAPQUANT_FOLD_BACKEND=cuda but {module.library_path()} does not load "
                          f"(make fast-fold); the Triton fold runs")
            _FAST_CUDA = False
    return _FAST_CUDA or None
_SM_COUNT = {}


def _programs(device, work_items, per_sm):
    device = torch.device(device)
    if device.type != "cuda":          # Triton's interpreter (TRITON_INTERPRET=1) runs on the CPU
        return max(1, min(work_items, 4 * (per_sm or DEFAULT_FOLD_PROGRAMS_PER_SM)))
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return max(1, min(work_items, _SM_COUNT[index] * (per_sm or DEFAULT_FOLD_PROGRAMS_PER_SM)))


def _work_list(slot, records=None, skip_slot=None, for_store=False):
    """The rows that need work, packed to the front, and their number (both on the device)."""
    B = slot.shape[0]
    order = torch.empty(max(B, 1), dtype=torch.int32, device=slot.device)
    count = torch.empty(1, dtype=torch.int32, device=slot.device)
    _work_list_kernel[(1,)](slot, records if records is not None else slot, skip_slot if skip_slot is not None else slot,
                            order, count, B, BLOCK=max(16, triton.next_power_of_2(B)), FOR_STORE=for_store,
                            HAS_SKIP=skip_slot is not None)
    return order, count


def fold_work_list(slot, records):
    """fold()'s work list (order, count) for these rows: the same for every layer of a decode step, so a caller
    may build it once per step and pass it to fold(work=...)."""
    return _work_list(slot, records=records)


def records_to_fold(tokens_in_window, window_length, force=None, tokens_added=1):
    """
    For fold(): the number of records each row folds now. A row folds when its window is
    full after this token, or when force != 0 and its window is not empty. Other rows get 0.
    Tensor operations only, so it can be captured in a CUDA graph.
    """
    records = tokens_in_window + tokens_added
    due = records >= window_length
    if force is not None:
        due = due | (force != 0)
    return torch.where(due, records, torch.zeros_like(records))


def fold(pool, slot, records, sweeps=None, num_warps=None, num_stages=1, export_by_slot=None, programs_per_sm=None,
         work=None):
    """
    Closes the window of every row with records[row] > 0 (the fast kernel): boundary state
    plus `records` window records become a new boundary state. Rows with 0 are not touched.
    The caller resets the cursor of the rows that folded (see records_to_fold).
    `work`: the work list from fold_work_list(slot, records), or None to build it here.

        slot            int32 [B]   pool slot per row; negative = padded row
        records         int32 [B]   see records_to_fold()
        sweeps          block-iteration sweeps of the compensator fit; default: the pool's setting.
                        0 = a half sweep: the compensators' key sides are kept and only their value
                        sides are fitted again (about 0.5 sweep of work)
        export_by_slot  float32 [slots, HV, V, K] or None. Every row that folds also writes its
                        full-precision state (before rounding) to the entry of its slot.
    """
    _require_one_step_per_column(pool)
    HV, V, K, R, L = pool.value_heads, pool.value_channels, pool.key_channels, pool.compensator_count, pool.window_length
    B = slot.shape[0]
    if getattr(pool, "scale_group_count", 1) > 1:
        raise ValueError("the fast kernels keep one step per value column; a scale group uses the exact kernel")
    for name, tensor in (("slot", slot), ("records", records)):
        if tensor.dtype != torch.int32 or tensor.shape[0] != B or not tensor.is_contiguous():
            raise TypeError(f"{name} must be a contiguous int32 tensor of length {B}")
    if R < 1:
        raise ValueError("the Triton fold needs at least one compensator slot")
    if export_by_slot is not None:
        if (export_by_slot.dtype != torch.float32 or tuple(export_by_slot.shape) != (pool.slots, HV, V, K)
                or not export_by_slot.is_contiguous()):
            raise ValueError(f"export_by_slot must be a contiguous float32 [{pool.slots}, {HV}, {V}, {K}] tensor")
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))
    BC = max(16, triton.next_power_of_2(max(L, R)))
    if B == 0:
        return
    order, count = work if work is not None else _work_list(slot, records=records)
    if FOLD_BACKEND == "cuda":
        module = _cuda_fold_module()
        if module is not None and module.supports(pool, export_by_slot):
            module.fold(pool, slot, records, order, count,
                        max(0, sweeps if sweeps is not None else pool.power_iteration_sweeps))
            return
    leapquant_fold_kernel[(_programs(slot.device, B * HV, programs_per_sm),)](
        pool.rounded_residual, pool.value_channel_step, pool.key_row_scale,
        pool.compensator_corrections, pool.compensator_keys,
        pool.window_corrections, pool.window_keys, pool.window_log_decays,
        slot, records, export_by_slot if export_by_slot is not None else records, order, count,
        HV=HV, K=K, V=V, R=R, L=L, BK=BK, BV=BV, BC=BC,
        SWEEPS=max(0, sweeps if sweeps is not None else pool.power_iteration_sweeps),
        LEVELS=float((1 << (pool.bits_per_entry - 1)) - 1), USE_SMOOTHING=pool.use_smoothing,
        HAS_EXPORT=export_by_slot is not None, FIT_BF16=FOLD_FIT_BF16, EXACT_M=FOLD_EXACT_M,
        num_warps=num_warps or DEFAULT_FOLD_WARPS, num_stages=num_stages,
    )


def store(pool, slot, source, skip_slot=None, source_index=None, sweeps=None, num_warps=None, programs_per_sm=None):
    """
    Turns full-precision states into boundary states (the fast kernel; use case UC3).
    Unlike leapquant_kernel.Kernel.store, `source` is not changed and no working memory is used.

        slot          int32 [B]   pool slot per row; negative = padded row
        source        float32 [entries, HV, V, K]; each entry contiguous, any stride between entries.
                      Row r reads entry slot[r], or source_index[r] if that is given.
        skip_slot     int32 [slots] or None: rows whose slot has a nonzero entry are skipped
        source_index  int32 [B] or None
    The caller sets the cursor of the stored slots to 0.
    """
    _require_one_step_per_column(pool)
    HV, V, K, R = pool.value_heads, pool.value_channels, pool.key_channels, pool.compensator_count
    B = slot.shape[0]
    if getattr(pool, "scale_group_count", 1) > 1:
        raise ValueError("the fast kernels keep one step per value column; a scale group uses the exact kernel")
    if slot.dtype != torch.int32 or not slot.is_contiguous():
        raise TypeError("slot must be a contiguous int32 tensor")
    if source.dtype != torch.float32 or tuple(source.shape[1:]) != (HV, V, K) or tuple(source.stride()[1:]) != (V * K, K, 1):
        raise ValueError(f"source must be float32 [entries, {HV}, {V}, {K}] with contiguous entries")
    if source_index is None and source.shape[0] < pool.slots:
        raise ValueError("source must have an entry for every slot when source_index is not given")
    for name, tensor, length in (("skip_slot", skip_slot, pool.slots), ("source_index", source_index, B)):
        if tensor is not None and (tensor.dtype != torch.int32 or tensor.shape[0] != length or not tensor.is_contiguous()):
            raise TypeError(f"{name} must be a contiguous int32 tensor of length {length}")
    if R < 1:
        raise ValueError("the Triton store needs at least one compensator slot")
    if B == 0:
        return
    BK = max(16, triton.next_power_of_2(K))
    BV = max(16, triton.next_power_of_2(V))
    BC = max(16, triton.next_power_of_2(R))
    order, count = _work_list(slot, skip_slot=skip_slot, for_store=True)
    if FOLD_BACKEND == "cuda":
        module = _cuda_fold_module()
        if module is not None and module.store_supports(pool):
            module.store(pool, slot, source, source_index, order, count,
                         max(1, sweeps if sweeps is not None else pool.power_iteration_sweeps) + 2)
            return
    leapquant_store_kernel[(_programs(slot.device, B * HV, programs_per_sm),)](
        source, pool.rounded_residual, pool.value_channel_step, pool.key_row_scale,
        pool.compensator_corrections, pool.compensator_keys,
        slot, source_index if source_index is not None else slot, order, count,
        source.stride(0),
        HV=HV, K=K, V=V, R=R, BK=BK, BV=BV, BC=BC,
        SWEEPS=max(1, sweeps if sweeps is not None else pool.power_iteration_sweeps),
        LEVELS=float((1 << (pool.bits_per_entry - 1)) - 1), USE_SMOOTHING=pool.use_smoothing,
        HAS_INDEX=source_index is not None,
        num_warps=num_warps or DEFAULT_FOLD_WARPS,
    )


def export(pool, slot, out, tokens_in_window=None, num_warps=8):
    """
    The state of every row in full precision (decompression; the fast kernel). Same arguments
    and result as leapquant_kernel.Kernel.export, for any number of compensators:

        slot              int32 [B]   pool slot per row; negative = skipped (its out entry is left alone)
        out               float32 [B, HV, V, K]   filled
        tokens_in_window  int32 [B] or None: records of the window to include (None: none)
    """
    _require_one_step_per_column(pool)
    HV, V, K, R, L = pool.value_heads, pool.value_channels, pool.key_channels, pool.compensator_count, pool.window_length
    B = slot.shape[0]
    if getattr(pool, "scale_group_count", 1) > 1:
        raise ValueError("the fast kernels keep one step per value column; a scale group uses the exact kernel")
    if slot.dtype != torch.int32 or not slot.is_contiguous():
        raise TypeError("slot must be a contiguous int32 tensor")
    if out.dtype != torch.float32 or tuple(out.shape) != (B, HV, V, K) or not out.is_contiguous():
        raise ValueError(f"out must be a contiguous float32 [{B}, {HV}, {V}, {K}] tensor")
    if tokens_in_window is not None:
        if not pool.with_window:
            raise ValueError("this pool keeps no window")
        if tokens_in_window.dtype != torch.int32 or tokens_in_window.shape[0] != B or not tokens_in_window.is_contiguous():
            raise TypeError(f"tokens_in_window must be a contiguous int32 tensor of length {B}")
    if B == 0:
        return out
    window = (pool.window_corrections, pool.window_keys, pool.window_log_decays) if pool.with_window else \
        (pool.compensator_corrections, pool.compensator_keys, pool.value_channel_step)   # dummies, never read
    leapquant_export_kernel[(B, HV)](
        pool.rounded_residual, pool.value_channel_step, pool.key_row_scale,
        pool.compensator_corrections, pool.compensator_keys, *window,
        slot, tokens_in_window if tokens_in_window is not None else slot, out,
        HV=HV, K=K, V=V, R=R, L=max(L, 1), BK=max(16, triton.next_power_of_2(K)), BV=max(16, triton.next_power_of_2(V)),
        BC=max(16, triton.next_power_of_2(max(L, R))), HAS_TOKENS=tokens_in_window is not None, num_warps=num_warps,
    )
    return out


# ---- slot copies: every boundary tensor of every layer in one launch ---------------------------------------------------
# SGLang's radix cache copies a request's state slot once per request with the cache on (MambaPool.copy_from ->
# EngineState.copy_slots). Copied field by field that is about 245 small kernels per request (24 layers x 5 fields x
# gather + scatter, plus flags): 62,643 index kernels and 273 ms of GPU time in a burst of 256 prompts
# (results/prefill-radix). One launch here, like SGLang's own fused copy of the conv state (mamba_slot_fused.py).
@triton.jit
def _slot_rows_copy_kernel(ptr_arr, words_arr, stride_arr, src_arr, dst_arr, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    tid = tl.program_id(0)                      # which tensor
    pid = tl.program_id(1)                      # which (source, destination) pair
    cid = tl.program_id(2)                      # which chunk of the row
    words = tl.load(words_arr + tid)            # 4-byte words in one slot's row of this tensor
    start = cid * CHUNK
    if start < words:
        base = tl.load(ptr_arr + tid).to(tl.pointer_type(tl.int32))
        stride = tl.load(stride_arr + tid)      # words from one slot to the next
        src_row = base + tl.load(src_arr + pid).to(tl.int64) * stride
        dst_row = base + tl.load(dst_arr + pid).to(tl.int64) * stride
        for offset in range(0, CHUNK, BLOCK):
            cols = start + offset + tl.arange(0, BLOCK)
            mask = cols < words
            tl.store(dst_row + cols, tl.load(src_row + cols, mask=mask), mask=mask)


class SlotRows:
    """
    Rows of several tensors indexed by slot (their first dimension), copied from slot to slot in one launch.
    Each slot's row must be contiguous and a whole number of 4-byte words; the tensors are kept alive here.

        rows = SlotRows([pool.rounded_residual, ...])
        rows.copy(src, dst)        # int32/int64 [n] on the tensors' device; n pairs, all copied in one launch

    Sources and destinations must not overlap when n > 1 (the pairs run in parallel); one pair may copy a slot onto
    itself. The caller checks that.
    """

    CHUNK = 4096        # words (16 KB) per program
    BLOCK = 1024

    def __init__(self, tensors):
        if not tensors:
            raise ValueError("no tensors")
        pointers, words, strides = [], [], []
        for tensor in tensors:
            row = tensor[0]
            row_bytes = row.numel() * tensor.element_size()
            stride_bytes = tensor.stride(0) * tensor.element_size()
            if not row.is_contiguous() or row_bytes % 4 or stride_bytes % 4 or tensor.data_ptr() % 4:
                raise ValueError("each slot's row must be contiguous and a whole number of 4-byte words")
            pointers.append(tensor.data_ptr())
            words.append(row_bytes // 4)
            strides.append(stride_bytes // 4)
        device = tensors[0].device
        self.tensors = list(tensors)
        self.pointers = torch.tensor(pointers, dtype=torch.int64, device=device)
        self.words = torch.tensor(words, dtype=torch.int64, device=device)
        self.strides = torch.tensor(strides, dtype=torch.int64, device=device)
        self.chunks = max(1, -(-max(words) // self.CHUNK))
        self.bytes_per_slot = 4 * sum(words)

    def copy(self, src, dst):
        pairs = src.shape[0]
        if pairs == 0:
            return
        if dst.shape[0] != pairs:
            raise ValueError("src and dst differ in length")
        _slot_rows_copy_kernel[(len(self.tensors), pairs, self.chunks)](
            self.pointers, self.words, self.strides, src, dst, CHUNK=self.CHUNK, BLOCK=self.BLOCK)
