// SPDX-License-Identifier: Apache-2.0
//
// leapquant_fast_decode.cu
//
// The decode step of the serving path, written by hand for sm_90: what leapquant_triton._decode_head does (same
// pool, same record format, same gate arithmetic), on CUDA cores, one warp per (row, head), with every load of the
// int8 matrix a coalesced 16-byte vector. The decode is a bandwidth problem (about 19.5 KB per head per token
// against a few thousand multiply-adds); this layout keeps 16 KB per warp in flight.
//
// Lane l = 8 * rg + c works on key channels 16c .. 16c+15 (its weights stay in registers) and on value rows
// v = 4i + rg, i = 0..31: 32 loads of 16 bytes per lane. Each lane keeps 64 partial sums (row i, query and key);
// a reduce-scatter over the 8 lanes of a row group leaves lane c the complete sums of rows i = 4c .. 4c+3.
// Records and compensators are dotted with the query and the key first, summed over the 8 lanes of a group,
// and kept in a few bytes of shared memory for the last part.
//
// Precision: the int8 levels times fp32 weights in fp32 (exact products of 8-bit integers and fp32 values, fp32
// sums); the record of the token is written exactly as the Triton step writes it (beta * key rounded to fp16, the
// correction rounded to fp16, log decay in fp16), and the output uses the correction and the decay as stored.
//
// Limits: key and value channels 128, window at most 16 records, compensators at most 8.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace lqfd {

constexpr int K = 128;
constexpr int V = 128;
constexpr int L = 16;
constexpr int RMAX = 8;
constexpr int WARPS = 4;                 // warps (work items) per block
constexpr unsigned FULL = 0xffffffffu;

enum DType : int32_t { F32 = 0, F16 = 1, BF16 = 2 };

struct Args {
    const void* mixed_qkv;      // [B, >= 2*H*K + HV*V]: q | k | v, row stride stride_qkv elements
    const void* a;              // [B, HV] raw gate input, or the decay (pre_gated)
    const void* b;              // [B, HV] raw beta input, or beta (pre_gated)
    const float* A_log;         // [HV] fp32
    const float* dt_bias;       // [HV] fp32
    void* out;                  // [B, HV, V]
    const int8_t* residual;     // [slots, HV, V, K]
    const float* channel_step;  // [slots, HV, V]
    const float* row_scale;     // [slots, HV, K]
    const __half* comp_c;       // [slots, HV, R, V]
    const __half* comp_k;       // [slots, HV, R, K]
    __half* win_c;              // [slots, HV, WL, V]   record written
    __half* win_k;              // [slots, HV, WL, K]   record written
    __half* win_g;              // [slots, HV, WL]      record written
    const int32_t* slot;        // [B]
    const int32_t* write_pos;   // [B] records already in the window
    long long stride_qkv;
    long long stride_a;
    long long stride_b;
    float scale;
    int32_t rows;               // B
    int32_t key_heads;          // H
    int32_t value_heads;        // HV
    int32_t key_channels;
    int32_t value_channels;
    int32_t window_length;      // WL
    int32_t compensators;       // R
    int32_t in_dtype;           // dtype of mixed_qkv, a, b
    int32_t out_dtype;
    int32_t pre_gated;
    int32_t l2norm;
    int32_t variant;            // 0: v5 (one warp per head); 2, 3, 4: v7 with that many pipeline stages
    int32_t reserved;
};

__device__ __forceinline__ float load_as_float(const void* base, long long index, int dtype) {
    if (dtype == BF16) return __bfloat162float(static_cast<const __nv_bfloat16*>(base)[index]);
    if (dtype == F16) return __half2float(static_cast<const __half*>(base)[index]);
    return static_cast<const float*>(base)[index];
}

// x rounded to `dtype` and back (the Triton step rounds sigmoid(b) to b's dtype).
__device__ __forceinline__ float round_to(float x, int dtype) {
    if (dtype == BF16) return __bfloat162float(__float2bfloat16_rn(x));
    if (dtype == F16) return __half2float(__float2half_rn(x));
    return x;
}

__device__ __forceinline__ float group_sum(float v) {   // over the 8 lanes of a row group (lane bits 0..2)
    v += __shfl_xor_sync(FULL, v, 1);
    v += __shfl_xor_sync(FULL, v, 2);
    v += __shfl_xor_sync(FULL, v, 4);
    return v;
}

// The two fp16 values of a 32-bit word, as floats.
__device__ __forceinline__ float half_lo(uint32_t word) { return __half2float(__ushort_as_half(static_cast<unsigned short>(word & 0xffffu))); }
__device__ __forceinline__ float half_hi(uint32_t word) { return __half2float(__ushort_as_half(static_cast<unsigned short>(word >> 16))); }

// Sum over this lane's 16 key channels of (fp16 vector row) * x, for the query and the key at once.
__device__ __forceinline__ void dot16(const __half* row, const float (&q)[16], const float (&k)[16], float& sq, float& sk) {
    const uint4 r0 = *reinterpret_cast<const uint4*>(row);
    const uint4 r1 = *reinterpret_cast<const uint4*>(row + 8);
    const uint32_t w[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float lo = half_lo(w[i]), hi = half_hi(w[i]);
        sq = fmaf(lo, q[2 * i], sq);
        sk = fmaf(lo, k[2 * i], sk);
        sq = fmaf(hi, q[2 * i + 1], sq);
        sk = fmaf(hi, k[2 * i + 1], sk);
    }
}

__global__ void __launch_bounds__(WARPS * 32) fast_decode_kernel(const Args a) {
    __shared__ float sums[WARPS][2 * (L + RMAX)];    // per warp: record dots (q, k), then compensator dots (q, k)
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int item = blockIdx.x * WARPS + warp;
    const int HV = a.value_heads;
    if (item >= a.rows * HV) return;
    const int n = item / HV;
    const int hv = item - n * HV;
    const int c = lane & 7;          // key chunk: 16c .. 16c+15
    const int rg = lane >> 3;        // row group: rows 4i + rg
    const int H = a.key_heads;
    const int h = hv / (HV / H);
    const int WL = a.window_length;
    const int R = a.compensators;
    const int state_slot = a.slot[n];
    const int wp = a.write_pos[n];
    const bool valid = state_slot >= 0 && wp >= 0 && wp < WL;
    const long long out_base = (static_cast<long long>(n) * HV + hv) * V;
    if (!valid) {                    // a padded row (or a cursor out of range): output zero, nothing appended
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int v = lane + 32 * j;
            if (a.out_dtype == BF16) static_cast<__nv_bfloat16*>(a.out)[out_base + v] = __float2bfloat16_rn(0.f);
            else if (a.out_dtype == F16) static_cast<__half*>(a.out)[out_base + v] = __float2half_rn(0.f);
            else static_cast<float*>(a.out)[out_base + v] = 0.f;
        }
        return;
    }
    const long long head = static_cast<long long>(state_slot) * HV + hv;
    float* const rec_s = sums[warp];                // [0, L): record . q   [L, 2L): record . k
    float* const comp_s = sums[warp] + 2 * L;       // [0, RMAX): compensator . q   [RMAX, 2 RMAX): . k

    // ---- gates of the current token (as the Triton step: beta rounded to b's dtype; decay stored as fp16 log) ----
    float alpha, beta, g;
    if (a.pre_gated) {
        alpha = load_as_float(a.a, n * a.stride_a + hv, a.in_dtype);
        beta = load_as_float(a.b, n * a.stride_b + hv, a.in_dtype);
        g = logf(alpha);
    } else {
        const float bv = load_as_float(a.b, n * a.stride_b + hv, a.in_dtype);
        beta = round_to(1.f / (1.f + expf(-bv)), a.in_dtype);
        const float x = load_as_float(a.a, n * a.stride_a + hv, a.in_dtype) + a.dt_bias[hv];
        const float softplus = x <= 20.f ? logf(1.f + expf(x)) : x;
        g = -expf(a.A_log[hv]) * softplus;
        alpha = expf(g);
    }
    const __half g_stored = __float2half_rn(g);
    const float alpha_stored = expf(__half2float(g_stored));

    // ---- decay each record received from the records after it (lane j holds fade_j) ---------------------------
    const float gj = (lane < wp) ? __half2float(a.win_g[head * WL + lane]) : 0.f;
    float cum = gj;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
        const float o = __shfl_up_sync(FULL, cum, off);
        if (lane >= off) cum += o;
    }
    const float g_total = __shfl_sync(FULL, cum, 15);
    const float fade = (lane < wp) ? expf(g_total - cum) : 0.f;
    const float window_fade = expf(g_total);

    // ---- query and key of this key head, this lane's 16 channels ----------------------------------------------
    const long long qkv = n * a.stride_qkv;
    float q[16], k[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q[i] = load_as_float(a.mixed_qkv, qkv + h * K + 16 * c + i, a.in_dtype);
        k[i] = load_as_float(a.mixed_qkv, qkv + H * K + h * K + 16 * c + i, a.in_dtype);
    }
    float q_rnorm = 1.f, k_rnorm = 1.f;
    if (a.l2norm) {
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sq += q[i] * q[i];
            sk += k[i] * k[i];
        }
        q_rnorm = 1.f / sqrtf(group_sum(sq) + 1e-6f);
        k_rnorm = 1.f / sqrtf(group_sum(sk) + 1e-6f);
    }
    // The record of this token, key side: beta * key rounded to fp16; its dot with q uses the stored value.
    float new_kq = 0.f;
    uint32_t key_words[8];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        q[i] = q[i] * q_rnorm * a.scale;
        k[i] = k[i] * k_rnorm;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __half lo = __float2half_rn(beta * k[2 * i]), hi = __float2half_rn(beta * k[2 * i + 1]);
        new_kq = fmaf(__half2float(lo), q[2 * i], new_kq);
        new_kq = fmaf(__half2float(hi), q[2 * i + 1], new_kq);
        key_words[i] = static_cast<uint32_t>(__half_as_ushort(lo)) | (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
    }
    new_kq = group_sum(new_kq);

    // ---- records and compensators: their keys against q and k, summed over the 8 lanes of a group -------------
#pragma unroll
    for (int j = 0; j < L; ++j) {
        float sq = 0.f, sk = 0.f;
        if (j < wp) dot16(a.win_k + (head * WL + j) * K + 16 * c, q, k, sq, sk);
        sq = group_sum(sq);
        sk = group_sum(sk);
        if (lane == 0) {
            rec_s[j] = sq;
            rec_s[L + j] = sk;
        }
    }
#pragma unroll
    for (int r = 0; r < RMAX; ++r) {
        float sq = 0.f, sk = 0.f;
        if (r < R) dot16(a.comp_k + (head * R + r) * K + 16 * c, q, k, sq, sk);
        sq = group_sum(sq);
        sk = group_sum(sk);
        if (lane == 0) {
            comp_s[r] = sq;
            comp_s[RMAX + r] = sk;
        }
    }
    if (rg == 0) {                   // the new record's key (after the records above were read)
        uint4* dst = reinterpret_cast<uint4*>(a.win_k + (head * WL + wp) * K + 16 * c);
        dst[0] = make_uint4(key_words[0], key_words[1], key_words[2], key_words[3]);
        dst[1] = make_uint4(key_words[4], key_words[5], key_words[6], key_words[7]);
    }

    // ---- the int8 matrix: weights key_row_scale * q and key_row_scale * k -------------------------------------
    {
        const float4* sc = reinterpret_cast<const float4*>(a.row_scale + head * K + 16 * c);
#pragma unroll
        for (int i4 = 0; i4 < 4; ++i4) {
            const float4 s = __ldg(sc + i4);
            q[4 * i4] *= s.x;
            q[4 * i4 + 1] *= s.y;
            q[4 * i4 + 2] *= s.z;
            q[4 * i4 + 3] *= s.w;
            k[4 * i4] *= s.x;
            k[4 * i4 + 1] *= s.y;
            k[4 * i4 + 2] *= s.z;
            k[4 * i4 + 3] *= s.w;
        }
    }
    float pq[32], pk[32];
    const int8_t* zrow = a.residual + head * (V * K) + rg * K + 16 * c;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int4 z = __ldg(reinterpret_cast<const int4*>(zrow + (4 * i) * K));
        const int words[4] = {z.x, z.y, z.z, z.w};
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float level = static_cast<float>(static_cast<int8_t>((words[w] >> (8 * e)) & 0xff));
                sq = fmaf(level, q[4 * w + e], sq);
                sk = fmaf(level, k[4 * w + e], sk);
            }
        }
        pq[i] = sq;
        pk[i] = sk;
    }
    // Reduce-scatter over the 8 lanes of the row group (lane bits 2, 1, 0 in turn): lane c ends with the complete
    // sums of rows i = 4c + s, s = 0..3.
    {
        const bool b2 = (c >> 2) & 1, b1 = (c >> 1) & 1, b0 = c & 1;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float keep_q = b2 ? pq[16 + i] : pq[i], give_q = b2 ? pq[i] : pq[16 + i];
            const float keep_k = b2 ? pk[16 + i] : pk[i], give_k = b2 ? pk[i] : pk[16 + i];
            pq[i] = keep_q + __shfl_xor_sync(FULL, give_q, 4);
            pk[i] = keep_k + __shfl_xor_sync(FULL, give_k, 4);
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float keep_q = b1 ? pq[8 + i] : pq[i], give_q = b1 ? pq[i] : pq[8 + i];
            const float keep_k = b1 ? pk[8 + i] : pk[i], give_k = b1 ? pk[i] : pk[8 + i];
            pq[i] = keep_q + __shfl_xor_sync(FULL, give_q, 2);
            pk[i] = keep_k + __shfl_xor_sync(FULL, give_k, 2);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float keep_q = b0 ? pq[4 + i] : pq[i], give_q = b0 ? pq[i] : pq[4 + i];
            const float keep_k = b0 ? pk[4 + i] : pk[i], give_k = b0 ? pk[i] : pk[4 + i];
            pq[i] = keep_q + __shfl_xor_sync(FULL, give_q, 1);
            pk[i] = keep_k + __shfl_xor_sync(FULL, give_k, 1);
        }
    }
    __syncwarp();                    // the record and compensator sums in shared memory are visible to the warp

    // ---- this lane's 4 value rows: state^T q and state^T k before the token, then the token -------------------
    const long long value_base = qkv + 2LL * H * K + static_cast<long long>(hv) * V;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const int v = 4 * (4 * c + s) + rg;
        const float step = a.channel_step[head * V + v];
        float sq = pq[s] * step, sk = pk[s] * step;
#pragma unroll
        for (int r = 0; r < RMAX; ++r) {
            if (r < R) {
                const float cc = __half2float(a.comp_c[(head * R + r) * V + v]);
                sq = fmaf(cc, comp_s[r], sq);
                sk = fmaf(cc, comp_s[RMAX + r], sk);
            }
        }
        sq *= window_fade;
        sk *= window_fade;
#pragma unroll
        for (int j = 0; j < L; ++j) {
            const float fj = __shfl_sync(FULL, fade, j);
            if (j < wp) {
                const float cf = __half2float(a.win_c[(head * WL + j) * V + v]) * fj;
                sq = fmaf(cf, rec_s[j], sq);
                sk = fmaf(cf, rec_s[L + j], sk);
            }
        }
        const float value = load_as_float(a.mixed_qkv, value_base + v, a.in_dtype);
        const __half correction = __float2half_rn(value - alpha * sk);
        a.win_c[(head * WL + wp) * V + v] = correction;
        const float o = __half2float(correction) * new_kq + alpha_stored * sq;
        if (a.out_dtype == BF16) static_cast<__nv_bfloat16*>(a.out)[out_base + v] = __float2bfloat16_rn(o);
        else if (a.out_dtype == F16) static_cast<__half*>(a.out)[out_base + v] = __float2half_rn(o);
        else static_cast<float*>(a.out)[out_base + v] = o;
    }
    if (lane == 0) a.win_g[head * WL + wp] = g_stored;
}

// =====================================================================================================================
// v7: the same step, pipelined. A block of 4 warps walks the (row, head) items (grid stride); the inputs of the next
// S-1 items are copied into shared memory (cp.async, 16 bytes per copy) while the current one is computed, so that a
// head's loads never wait on its own arithmetic (v5's weakness: one warp per head, every load in the open).
//   The int8 matrix on tensor cores: each weight vector w (key_row_scale * q, key_row_scale * k) is divided by its
//   unit s = max|w| / 127 and split into four int8 pieces, w / s = P0 + P1/128 + P2/128^2 + P3/128^3, which holds
//   the fp32 value exactly (7 + 3 * 7 bits >= 24); m16n8k32 int8 products with int32 sums are exact, and one MMA
//   column per piece covers q and k (4 + 4 = 8 columns). No conversion of the levels.
//   Record and compensator keys against q and k on fp16 tensor cores, q and k as fp16 hi + lo (as the fold does).
//   Each warp does the key-side work of the item itself (norms, pieces, record dots: a few hundred instructions),
//   so the only block-wide barriers are the two of the pipeline. Warp w owns value rows 32w .. 32w+31.
// Shared memory tiles read by ldmatrix (the levels, the record and compensator keys) are XOR-swizzled by 16-byte
// chunk: chunk c of row r is stored at chunk c ^ (r & 7), so the 8 rows of each 8x8 matrix hit 8 different banks.
// Requires: window length 16, mixed_qkv 16-byte aligned with a 16-byte multiple row stride (checked by the caller).
// =====================================================================================================================
namespace v7 {
constexpr int THREADS = 128;
constexpr int ZB = 0;                          // int8 [V][K] levels (swizzled, 8 chunks per row)
constexpr int RK = ZB + V * K;                 // fp16 [L][K] record keys (swizzled, 16 chunks per row)
constexpr int CK = RK + L * K * 2;             // fp16 [RMAX][K] compensator keys (swizzled)
constexpr int RC = CK + RMAX * K * 2;          // fp16 [L][V] record corrections
constexpr int CC = RC + L * V * 2;             // fp16 [RMAX][V] compensator corrections
constexpr int ST = CC + RMAX * V * 2;          // fp32 [V] value channel steps
constexpr int SC = ST + V * 4;                 // fp32 [K] key row scales
constexpr int QB = SC + K * 4;                 // q | k | v of the row, input dtype (up to 4 bytes): 512 bytes each
constexpr int KB = QB + K * 4;
constexpr int VB = KB + K * 4;
constexpr int GB = VB + V * 4;                 // fp16 [L] log decays
constexpr int AB = GB + L * 2;                 // the 16-byte chunk that holds a[n, hv]
constexpr int BB = AB + 16;                    // the 16-byte chunk that holds b[n, hv]
constexpr int IB = BB + 16;                    // int [2]: slot, records in the window
constexpr int STAGE = (IB + 16 + 127) / 128 * 128;
constexpr int W_ROW = 144;                     // bytes per piece row: 128 + 16, conflict-free B fragments
constexpr int XW = 0;                          // per warp: int8 [8][W_ROW] weight pieces (q: rows 0-3, k: rows 4-7)
constexpr int XZ = XW + 8 * W_ROW;             // fp32 [2][32]: the int8 part of this warp's rows (q, k)
constexpr int XR = XZ + 2 * 32 * 4;            // fp32 [2][L]: record keys . q and . k (q and k normalized)
constexpr int XC = XR + 2 * L * 4;             // fp32 [2][RMAX]: compensator keys . q and . k
constexpr int XF = XC + 2 * RMAX * 4;          // fp32 [L]: decay each record received from the records after it
constexpr int WSCR = (XF + L * 4 + 15) / 16 * 16;
template <int S>
constexpr int smem_bytes() { return S * STAGE + 4 * WSCR; }
static_assert(STAGE % 128 == 0 && RK % 128 == 0 && CK % 128 == 0 && RC % 16 == 0 && CC % 16 == 0 && ST % 16 == 0 &&
              SC % 16 == 0 && QB % 16 == 0 && GB % 16 == 0 && AB % 16 == 0 && XZ % 16 == 0, "aligned regions");

__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t address) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(address));
}
__device__ __forceinline__ void mma_s8(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                       uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float value_at(const unsigned char* base, int i, int dtype) {
    if (dtype == BF16) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[i]);
    if (dtype == F16) return __half2float(reinterpret_cast<const __half*>(base)[i]);
    return reinterpret_cast<const float*>(base)[i];
}
__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {     // lo: the lower-indexed element
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}
__device__ __forceinline__ float lo_part(float x) { return x - __half2float(__float2half_rn(x)); }
__device__ __forceinline__ void store_out(const Args& a, long long index, float o) {
    if (a.out_dtype == BF16) static_cast<__nv_bfloat16*>(a.out)[index] = __float2bfloat16_rn(o);
    else if (a.out_dtype == F16) static_cast<__half*>(a.out)[index] = __float2half_rn(o);
    else static_cast<float*>(a.out)[index] = o;
}
__device__ __forceinline__ const unsigned char* gate_address(const void* base, long long n, long long stride, int hv,
                                                             int esize) {
    return static_cast<const unsigned char*>(base) + (n * stride + hv) * esize;
}

// The copies of one item into a stage (every thread its share), then one commit (empty for no item / a padded row).
__device__ __forceinline__ void issue(const Args& a, unsigned char* stage_base, long long item, long long total, int sl,
                                      int wp, int esize) {
    const int tid = threadIdx.x;
    if (item < total) {
        const int HV = a.value_heads;
        const int n = static_cast<int>(item / HV);
        const int hv = static_cast<int>(item - static_cast<long long>(n) * HV);
        if (tid == 0) {
            reinterpret_cast<int*>(stage_base + IB)[0] = sl;
            reinterpret_cast<int*>(stage_base + IB)[1] = wp;
        }
        if (sl >= 0 && wp >= 0 && wp < L) {
            const int H = a.key_heads;
            const int h = hv / (HV / H);
            const int R = a.compensators;
            const long long head = static_cast<long long>(sl) * HV + hv;
            const unsigned char* z = reinterpret_cast<const unsigned char*>(a.residual + head * (V * K));
#pragma unroll
            for (int j = 0; j < (V * K / 16) / THREADS; ++j) {
                const int i = tid + THREADS * j;
                const int r = i >> 3, c = i & 7;
                cp16(stage_base + ZB + r * 128 + ((c ^ (r & 7)) << 4), z + 16 * i);
            }
            const unsigned char* rk = reinterpret_cast<const unsigned char*>(a.win_k + head * (L * K));
            const unsigned char* rc = reinterpret_cast<const unsigned char*>(a.win_c + head * (L * V));
            for (int i = tid; i < wp * 16; i += THREADS) {
                const int r = i >> 4, c = i & 15;
                cp16(stage_base + RK + r * 256 + ((c ^ (r & 7)) << 4), rk + 16 * i);
                cp16(stage_base + RC + 16 * i, rc + 16 * i);
            }
            const unsigned char* ck = reinterpret_cast<const unsigned char*>(a.comp_k + head * (R * K));
            const unsigned char* cc = reinterpret_cast<const unsigned char*>(a.comp_c + head * (R * V));
            for (int i = tid; i < R * 16; i += THREADS) {
                const int r = i >> 4, c = i & 15;
                cp16(stage_base + CK + r * 256 + ((c ^ (r & 7)) << 4), ck + 16 * i);
                cp16(stage_base + CC + 16 * i, cc + 16 * i);
            }
            const int vec = 8 * esize;                    // 16-byte chunks per q, k or v vector
            const unsigned char* row = static_cast<const unsigned char*>(a.mixed_qkv) + n * a.stride_qkv * esize;
            const int small = 64 + 3 * vec + 4;
            for (int i = tid; i < small; i += THREADS) {
                const unsigned char* src;
                unsigned char* dst;
                if (i < 32) {
                    src = reinterpret_cast<const unsigned char*>(a.channel_step + head * V) + 16 * i;
                    dst = stage_base + ST + 16 * i;
                } else if (i < 64) {
                    src = reinterpret_cast<const unsigned char*>(a.row_scale + head * K) + 16 * (i - 32);
                    dst = stage_base + SC + 16 * (i - 32);
                } else if (i < 64 + 3 * vec) {
                    const int j = i - 64, which = j / vec, c = j - which * vec;
                    const long long element = which == 0   ? static_cast<long long>(h) * K
                                              : which == 1 ? static_cast<long long>(H + h) * K
                                                           : 2LL * H * K + static_cast<long long>(hv) * V;
                    src = row + element * esize + 16 * c;
                    dst = stage_base + QB + which * 512 + 16 * c;
                } else if (i < 64 + 3 * vec + 2) {
                    const int c = i - 64 - 3 * vec;
                    src = reinterpret_cast<const unsigned char*>(a.win_g + head * L) + 16 * c;
                    dst = stage_base + GB + 16 * c;
                } else {
                    const bool is_a = i == 64 + 3 * vec + 2;
                    const unsigned char* p = gate_address(is_a ? a.a : a.b, n, is_a ? a.stride_a : a.stride_b, hv, esize);
                    src = reinterpret_cast<const unsigned char*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(15));
                    dst = stage_base + (is_a ? AB : BB);
                }
                cp16(dst, src);
            }
        }
    }
    cp_commit();
}

// One item from its stage: warp w computes value rows 32w .. 32w+31.
__device__ __forceinline__ void compute(const Args& a, const unsigned char* s, unsigned char* x, long long item,
                                        int esize) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int HV = a.value_heads, R = a.compensators, dtype = a.in_dtype;
    const int n = static_cast<int>(item / HV);
    const int hv = static_cast<int>(item - static_cast<long long>(n) * HV);
    const int sl = reinterpret_cast<const int*>(s + IB)[0], wp = reinterpret_cast<const int*>(s + IB)[1];
    const long long out_base = (static_cast<long long>(n) * HV + hv) * V;
    const int v = tid;
    if (!(sl >= 0 && wp >= 0 && wp < L)) {    // a padded row (or a cursor out of range): zero, nothing appended
        store_out(a, out_base + v, 0.f);
        return;
    }
    const long long head = static_cast<long long>(sl) * HV + hv;

    // ---- gates (as the Triton step: beta rounded to b's dtype; the decay stored as an fp16 logarithm) ------------
    const int a_off = static_cast<int>(reinterpret_cast<uintptr_t>(gate_address(a.a, n, a.stride_a, hv, esize)) & 15);
    const int b_off = static_cast<int>(reinterpret_cast<uintptr_t>(gate_address(a.b, n, a.stride_b, hv, esize)) & 15);
    const float a_in = value_at(s + AB + a_off, 0, dtype), b_in = value_at(s + BB + b_off, 0, dtype);
    float alpha, beta, gl;
    if (a.pre_gated) {
        alpha = a_in;
        beta = b_in;
        gl = logf(alpha);
    } else {
        beta = round_to(1.f / (1.f + expf(-b_in)), dtype);
        const float xg = a_in + __ldg(a.dt_bias + hv);
        const float softplus = xg <= 20.f ? logf(1.f + expf(xg)) : xg;
        gl = -expf(__ldg(a.A_log + hv)) * softplus;
        alpha = expf(gl);
    }
    const __half g_stored = __float2half_rn(gl);
    const float alpha_stored = expf(__half2float(g_stored));

    // ---- decay each record received from the records after it (lane j < 16: record j) -------------------------
    float* const xf = reinterpret_cast<float*>(x + XF);
    const float gj = (lane < wp) ? __half2float(reinterpret_cast<const __half*>(s + GB)[lane]) : 0.f;
    float cum = gj;
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
        const float o = __shfl_up_sync(FULL, cum, off);
        if (lane >= off) cum += o;
    }
    const float g_total = __shfl_sync(FULL, cum, 15);
    if (lane < L) xf[lane] = (lane < wp) ? expf(g_total - cum) : 0.f;
    const float window_fade = expf(g_total);

    // ---- key side (every warp for itself): lane l has key channels 4l .. 4l+3 -----------------------------------
    float q4[4], k4[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        q4[i] = value_at(s + QB, 4 * lane + i, dtype);
        k4[i] = value_at(s + KB, 4 * lane + i, dtype);
    }
    const float4 sc4 = *reinterpret_cast<const float4*>(s + SC + 16 * lane);
    const float scs[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
    float sqq = 0.f, skk = 0.f, mq = 0.f, mk = 0.f, wq[4], wk[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sqq += q4[i] * q4[i];
        skk += k4[i] * k4[i];
        wq[i] = scs[i] * q4[i];
        wk[i] = scs[i] * k4[i];
        mq = fmaxf(mq, fabsf(wq[i]));
        mk = fmaxf(mk, fabsf(wk[i]));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sqq += __shfl_xor_sync(FULL, sqq, off);
        skk += __shfl_xor_sync(FULL, skk, off);
        mq = fmaxf(mq, __shfl_xor_sync(FULL, mq, off));
        mk = fmaxf(mk, __shfl_xor_sync(FULL, mk, off));
    }
    const float q_rnorm = a.l2norm ? 1.f / sqrtf(sqq + 1e-6f) : 1.f;
    const float k_rnorm = a.l2norm ? 1.f / sqrtf(skk + 1e-6f) : 1.f;
    const float qf = q_rnorm * a.scale;        // raw q -> the q of the step
    const float kf = k_rnorm;
    // The weights in four int8 pieces each (see above); row p of XW: piece p of q (p < 4) or of k (p >= 4).
    {
        const float inv_q = mq > 0.f ? 127.f / mq : 0.f, inv_k = mk > 0.f ? 127.f / mk : 0.f;
        uint32_t pq[4] = {0u, 0u, 0u, 0u}, pk[4] = {0u, 0u, 0u, 0u};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float r = wq[i] * inv_q;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float piece = rintf(r);
                pq[p] |= (static_cast<uint32_t>(static_cast<int>(piece)) & 0xffu) << (8 * i);
                r = (r - piece) * 128.f;
            }
            r = wk[i] * inv_k;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const float piece = rintf(r);
                pk[p] |= (static_cast<uint32_t>(static_cast<int>(piece)) & 0xffu) << (8 * i);
                r = (r - piece) * 128.f;
            }
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            *reinterpret_cast<uint32_t*>(x + XW + p * W_ROW + 4 * lane) = pq[p];
            *reinterpret_cast<uint32_t*>(x + XW + (4 + p) * W_ROW + 4 * lane) = pk[p];
        }
    }
    // The record of this token, key side: beta * key rounded to fp16; its dot with q uses the stored value.
    float new_kq = 0.f;
    {
        __half key[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            key[i] = __float2half_rn(beta * (k4[i] * k_rnorm));
            new_kq = fmaf(__half2float(key[i]), q4[i] * q_rnorm * a.scale, new_kq);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) new_kq += __shfl_xor_sync(FULL, new_kq, off);
        if (warp == 0) {
            const uint32_t w0 = static_cast<uint32_t>(__half_as_ushort(key[0])) |
                                (static_cast<uint32_t>(__half_as_ushort(key[1])) << 16);
            const uint32_t w1 = static_cast<uint32_t>(__half_as_ushort(key[2])) |
                                (static_cast<uint32_t>(__half_as_ushort(key[3])) << 16);
            *reinterpret_cast<uint2*>(a.win_k + (head * L + wp) * K + 4 * lane) = make_uint2(w0, w1);
        }
    }
    __syncwarp();                              // the pieces and the fades are visible to the warp

    // ---- record and compensator keys . q, . k on fp16 tensor cores (B columns: q hi, q lo, k hi, k lo, 0 ...) ---
    const int m = lane >> 3, lrow = ((m & 1) << 3) + (lane & 7), half8 = m >> 1;
    {
        float rec[4] = {0.f, 0.f, 0.f, 0.f}, cmp[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t rk_base = saddr(s + RK), ck_base = saddr(s + CK);
        const unsigned char* column = s + (g < 2 ? QB : KB);
        const int crow = lrow & 7;             // compensator rows 0..7 (the upper 8 repeat them)
#pragma unroll
        for (int ks = 0; ks < K / 16; ++ks) {
            uint32_t b0 = 0u, b1 = 0u;
            if (g < 4) {
                const int kk = 16 * ks + 2 * t;
                float x0 = value_at(column, kk, dtype), x1 = value_at(column, kk + 1, dtype);
                float x2 = value_at(column, kk + 8, dtype), x3 = value_at(column, kk + 9, dtype);
                if (g & 1) {
                    x0 = lo_part(x0);
                    x1 = lo_part(x1);
                    x2 = lo_part(x2);
                    x3 = lo_part(x3);
                }
                b0 = pack_half2(x0, x1);
                b1 = pack_half2(x2, x3);
            }
            const int c = 2 * ks + half8;
            uint32_t a0, a1, a2, a3;
            ldsm4(a0, a1, a2, a3, rk_base + lrow * 256 + ((c ^ (lrow & 7)) << 4));
            mma_f16(rec, a0, a1, a2, a3, b0, b1);
            ldsm4(a0, a1, a2, a3, ck_base + crow * 256 + ((c ^ crow) << 4));
            mma_f16(cmp, a0, a1, a2, a3, b0, b1);
        }
        float* const xr = reinterpret_cast<float*>(x + XR);
        float* const xc = reinterpret_cast<float*>(x + XC);
        if (t < 2) {                           // t = 0: columns q hi, q lo; t = 1: k hi, k lo
            const float f = t == 0 ? qf : kf;
            xr[t * L + g] = (rec[0] + rec[1]) * f;
            xr[t * L + g + 8] = (rec[2] + rec[3]) * f;
            xc[t * RMAX + g] = (cmp[0] + cmp[1]) * f;
        }
    }

    // ---- the int8 matrix: this warp's 32 value rows against the 8 piece columns (int8 tensor cores) -----------
    {
        int acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
        const uint32_t z_base = saddr(s + ZB);
        const unsigned char* xw = x + XW + g * W_ROW + 4 * t;
#pragma unroll
        for (int kt = 0; kt < K / 32; ++kt) {
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(xw + 32 * kt);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(xw + 32 * kt + 16);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int row = 32 * warp + 16 * mt + lrow;
                const int c = 2 * kt + half8;
                uint32_t a0, a1, a2, a3;
                ldsm4(a0, a1, a2, a3, z_base + row * 128 + ((c ^ (row & 7)) << 4));
                mma_s8(acc[mt], a0, a1, a2, a3, b0, b1);
            }
        }
        // Columns 2t, 2t+1: t = 0: q P0, P1; t = 1: q P2, P3; t = 2: k P0, P1; t = 3: k P2, P3.
        const float f0 = (t & 1) ? (1.f / 16384.f) : 1.f, f1 = (t & 1) ? (1.f / 2097152.f) : (1.f / 128.f);
        const float unit = (t >> 1) ? mk / 127.f : mq / 127.f;
        float* const xz = reinterpret_cast<float*>(x + XZ);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            float p0 = static_cast<float>(acc[mt][0]) * f0 + static_cast<float>(acc[mt][1]) * f1;
            float p1 = static_cast<float>(acc[mt][2]) * f0 + static_cast<float>(acc[mt][3]) * f1;
            p0 += __shfl_xor_sync(FULL, p0, 1);
            p1 += __shfl_xor_sync(FULL, p1, 1);
            if ((t & 1) == 0) {
                xz[(t >> 1) * 32 + 16 * mt + g] = p0 * unit;
                xz[(t >> 1) * 32 + 16 * mt + g + 8] = p1 * unit;
            }
        }
    }
    __syncwarp();                              // the dots of this warp are visible to its lanes

    // ---- value row v: state^T q and state^T k before the token, then the token ---------------------------------
    {
        const float* const xz = reinterpret_cast<const float*>(x + XZ);
        const float* const xr = reinterpret_cast<const float*>(x + XR);
        const float* const xc = reinterpret_cast<const float*>(x + XC);
        const float step = reinterpret_cast<const float*>(s + ST)[v];
        float sq = xz[lane] * qf * step, sk = xz[32 + lane] * kf * step;
        const __half* ccv = reinterpret_cast<const __half*>(s + CC);
        for (int r = 0; r < R; ++r) {
            const float c = __half2float(ccv[r * V + v]);
            sq = fmaf(c, xc[r], sq);
            sk = fmaf(c, xc[RMAX + r], sk);
        }
        sq *= window_fade;
        sk *= window_fade;
        const __half* rcv = reinterpret_cast<const __half*>(s + RC);
        for (int j = 0; j < wp; ++j) {
            const float cf = __half2float(rcv[j * V + v]) * xf[j];
            sq = fmaf(cf, xr[j], sq);
            sk = fmaf(cf, xr[L + j], sk);
        }
        const float value = value_at(s + VB, v, dtype);
        const __half correction = __float2half_rn(value - alpha * sk);
        a.win_c[(head * L + wp) * V + v] = correction;
        store_out(a, out_base + v, __half2float(correction) * new_kq + alpha_stored * sq);
        if (tid == 0) a.win_g[head * L + wp] = g_stored;
    }
}
}  // namespace v7

template <int S>
__global__ void __launch_bounds__(v7::THREADS) decode7_kernel(const Args a) {
    extern __shared__ __align__(128) unsigned char smem[];
    const long long total = static_cast<long long>(a.rows) * a.value_heads;
    const int esize = a.in_dtype == F32 ? 4 : 2;
    const long long first = blockIdx.x, stride = gridDim.x;
    unsigned char* const scratch = smem + S * v7::STAGE + (threadIdx.x >> 5) * v7::WSCR;
#pragma unroll
    for (int p = 0; p < S - 1; ++p) {         // prologue: the first S-1 items
        const long long item = first + p * stride;
        int sl = -1, wp = -1;
        if (item < total) {
            const int n = static_cast<int>(item / a.value_heads);
            sl = __ldg(a.slot + n);
            wp = __ldg(a.write_pos + n);
        }
        v7::issue(a, smem + p * v7::STAGE, item, total, sl, wp, esize);
    }
    long long next = first + (S - 1) * stride;   // the next item to copy; its slot and cursor loaded one item ahead
    int next_sl = -1, next_wp = -1;
    if (next < total) {
        const int n = static_cast<int>(next / a.value_heads);
        next_sl = __ldg(a.slot + n);
        next_wp = __ldg(a.write_pos + n);
    }
    int stage = 0;
    for (long long item = first; item < total; item += stride) {
        __syncthreads();                       // the stage refilled now was read by the previous item
        v7::issue(a, smem + ((stage + S - 1) % S) * v7::STAGE, next, total, next_sl, next_wp, esize);
        next += stride;
        if (next < total) {
            const int n = static_cast<int>(next / a.value_heads);
            next_sl = __ldg(a.slot + n);
            next_wp = __ldg(a.write_pos + n);
        }
        v7::cp_wait<S - 1>();                  // this item's copies (this thread's) have landed
        __syncthreads();                       // ... everyone's
        v7::compute(a, smem + stage * v7::STAGE, scratch, item, esize);
        stage = stage + 1 == S ? 0 : stage + 1;
    }
    v7::cp_wait<0>();
}

// =====================================================================================================================
// v8: one warp per item and no block-wide barrier. (v7's profile: ~6,300 warp-instructions per head, two block
// barriers per head, 8-12 warps per multiprocessor; each block took ~6.5 us per head.)
//   - The int8 levels go from global memory straight into registers, already in the order of the int8 tensor-core A
//     fragments: the key axis is permuted (a dot product does not care, and the weight pieces follow the same order),
//     so that each 16-byte load of a lane is part of a fully used 64-byte run of one row. A chunk of 32 rows (eight
//     16-byte loads per lane) is in flight while the chunk before it is multiplied.
//   - The small inputs of an item (record and compensator keys and corrections, steps, scales, q, k, v, decays,
//     gates) go into the warp's own shared memory by cp.async, issued at the end of the previous item.
//   - The input dtype is a template parameter; 32-bit indices except for addresses.
// The arithmetic is v7's (weights in four exact int8 pieces; records on fp16 tensor cores; the same step).
// =====================================================================================================================
namespace v8 {
constexpr int WARPS = 4;
constexpr int RK = 0;                          // fp16 [L][K] record keys (swizzled: chunk c of row r at c ^ (r & 7))
constexpr int CK = RK + L * K * 2;             // fp16 [RMAX][K] compensator keys (swizzled)
constexpr int RC = CK + RMAX * K * 2;          // fp16 [L][V] record corrections
constexpr int CC = RC + L * V * 2;             // fp16 [RMAX][V] compensator corrections
constexpr int ST = CC + RMAX * V * 2;          // fp32 [V] value channel steps
constexpr int SC = ST + V * 4;                 // fp32 [K] key row scales
constexpr int QB = SC + K * 4;                 // q | k | v of the row, input dtype (up to 4 bytes): 512 bytes each
constexpr int KB = QB + 512;
constexpr int VB = KB + 512;
constexpr int GB = VB + 512;                   // fp16 [L] log decays
constexpr int AB = GB + L * 2;                 // the 16-byte chunk that holds a[n, hv]
constexpr int BB = AB + 16;                    // the 16-byte chunk that holds b[n, hv]
constexpr int W_ROW = 144;
constexpr int XW = BB + 16;                    // int8 [8][W_ROW]: weight pieces by key channel (q: rows 0-3, k: 4-7)
constexpr int XZ = XW + 8 * W_ROW;             // fp32 [2][V]: the int8 part of every value row (q, k)
constexpr int XR = XZ + 2 * V * 4;             // fp32 [2][L]: record keys . q, . k
constexpr int XC = XR + 2 * L * 4;             // fp32 [2][RMAX]: compensator keys . q, . k
constexpr int XF = XC + 2 * RMAX * 4;          // fp32 [L]: fades
constexpr int WARP_BYTES = (XF + L * 4 + 127) / 128 * 128;
constexpr int BLOCK_BYTES = WARPS * WARP_BYTES;
static_assert(XW % 16 == 0 && XZ % 16 == 0 && QB % 16 == 0 && GB % 16 == 0 && AB % 16 == 0 && WARP_BYTES % 128 == 0,
              "aligned regions");

template <int IN>
__device__ __forceinline__ float in_at(const unsigned char* base, int i) {
    if constexpr (IN == BF16) return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[i]);
    else if constexpr (IN == F16) return __half2float(reinterpret_cast<const __half*>(base)[i]);
    else return reinterpret_cast<const float*>(base)[i];
}

// The small inputs of one item into the warp's shared memory (every lane its share), then one commit.
template <int IN>
__device__ __forceinline__ void issue(const Args& a, unsigned char* s, int item, int total, int sl, int wp) {
    constexpr int ESIZE = IN == F32 ? 4 : 2;
    constexpr int VEC = 8 * ESIZE;             // 16-byte chunks per q, k or v vector
    const int lane = threadIdx.x & 31;
    if (item < total && sl >= 0 && wp >= 0 && wp < L) {
        const int HV = a.value_heads, H = a.key_heads, R = a.compensators;
        const int n = item / HV, hv = item - n * HV;
        const int h = hv / (HV / H);
        const long long head = static_cast<long long>(sl) * HV + hv;
        const unsigned char* rk = reinterpret_cast<const unsigned char*>(a.win_k + head * (L * K));
        const unsigned char* rc = reinterpret_cast<const unsigned char*>(a.win_c + head * (L * V));
        for (int i = lane; i < wp * 16; i += 32) {
            const int r = i >> 4, c = i & 15;
            v7::cp16(s + RK + r * 256 + ((c ^ (r & 7)) << 4), rk + 16 * i);
            v7::cp16(s + RC + 16 * i, rc + 16 * i);
        }
        const unsigned char* ck = reinterpret_cast<const unsigned char*>(a.comp_k + head * (R * K));
        const unsigned char* cc = reinterpret_cast<const unsigned char*>(a.comp_c + head * (R * V));
        for (int i = lane; i < R * 16; i += 32) {
            const int r = i >> 4, c = i & 15;
            v7::cp16(s + CK + r * 256 + ((c ^ (r & 7)) << 4), ck + 16 * i);
            v7::cp16(s + CC + 16 * i, cc + 16 * i);
        }
        v7::cp16(s + ST + 16 * lane, reinterpret_cast<const unsigned char*>(a.channel_step + head * V) + 16 * lane);
        v7::cp16(s + SC + 16 * lane, reinterpret_cast<const unsigned char*>(a.row_scale + head * K) + 16 * lane);
        const unsigned char* row = static_cast<const unsigned char*>(a.mixed_qkv) + n * a.stride_qkv * ESIZE;
        for (int i = lane; i < 3 * VEC; i += 32) {
            const int which = i / VEC, c = i - which * VEC;
            const long long element = which == 0   ? static_cast<long long>(h) * K
                                      : which == 1 ? static_cast<long long>(H + h) * K
                                                   : 2LL * H * K + static_cast<long long>(hv) * V;
            v7::cp16(s + QB + which * 512 + 16 * c, row + element * ESIZE + 16 * c);
        }
        if (lane < 2) {
            v7::cp16(s + GB + 16 * lane, reinterpret_cast<const unsigned char*>(a.win_g + head * L) + 16 * lane);
        } else if (lane < 4) {
            const bool is_a = lane == 2;
            const unsigned char* p = v7::gate_address(is_a ? a.a : a.b, n, is_a ? a.stride_a : a.stride_b, hv, ESIZE);
            v7::cp16(s + (is_a ? AB : BB), reinterpret_cast<const unsigned char*>(reinterpret_cast<uintptr_t>(p) & ~uintptr_t(15)));
        }
    }
    v7::cp_commit();
}

// Rows 32c + 16mt + g and + 8 of a 32-row chunk, bytes 16t .. 16t+15 and 64+16t .. 64+16t+15 of each (streaming loads).
__device__ __forceinline__ void load_chunk(const int8_t* zg, int c, int g, int t, uint4 (&dst)[8]) {
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const uint4* r0 = reinterpret_cast<const uint4*>(zg + (32 * c + 16 * mt + g) * K + 16 * t);
        const uint4* r1 = reinterpret_cast<const uint4*>(zg + (32 * c + 16 * mt + g + 8) * K + 16 * t);
        dst[4 * mt + 0] = __ldcs(r0);
        dst[4 * mt + 1] = __ldcs(r0 + 4);
        dst[4 * mt + 2] = __ldcs(r1);
        dst[4 * mt + 3] = __ldcs(r1 + 4);
    }
}
}  // namespace v8

template <int IN, bool EPI4, bool EARLY>
__global__ void __launch_bounds__(v8::WARPS * 32, 3) decode8_kernel(const Args a) {
    using namespace v8;
    extern __shared__ __align__(128) unsigned char smem[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    unsigned char* const s = smem + warp * WARP_BYTES;
    const int HV = a.value_heads, R = a.compensators;
    const int total = a.rows * HV;
    const int wstride = gridDim.x * v8::WARPS;
    int item = blockIdx.x * v8::WARPS + warp;
    int sl = -1, wp = -1;
    if (item < total) {
        const int n = item / HV;
        sl = __ldg(a.slot + n);
        wp = __ldg(a.write_pos + n);
    }
    issue<IN>(a, s, item, total, sl, wp);
    int next = item + wstride, next_sl = -1, next_wp = -1;     // the next item's slot and cursor, one item ahead
    if (next < total) {
        const int n = next / HV;
        next_sl = __ldg(a.slot + n);
        next_wp = __ldg(a.write_pos + n);
    }
    uint4 zb[2][8];                            // two 32-row chunks of the levels in flight
    // EARLY: the first two chunks of the next item are loaded as soon as this item's matrix products are done, so that
    // their latency overlaps this item's last part (v9's profile: warps idle two thirds of the time, waiting on memory).
    auto first_chunks = [&](int it, int it_sl, int it_wp) {
        if (it < total && it_sl >= 0 && it_wp >= 0 && it_wp < L) {
            const int it_n = it / HV, it_hv = it - it_n * HV;
            const int8_t* z = a.residual + (static_cast<long long>(it_sl) * HV + it_hv) * (V * K);
            load_chunk(z, 0, g, t, zb[0]);
            load_chunk(z, 1, g, t, zb[1]);
        }
    };
    if constexpr (EARLY) first_chunks(item, sl, wp);
    for (; item < total; item += wstride) {
        const int n = item / HV, hv = item - n * HV;
        const long long out_base = (static_cast<long long>(n) * HV + hv) * V;
        const bool valid = sl >= 0 && wp >= 0 && wp < L;
        const long long head = static_cast<long long>(valid ? sl : 0) * HV + hv;
        const int8_t* zg = a.residual + head * (V * K);
        if (!EARLY && valid) {
            load_chunk(zg, 0, g, t, zb[0]);
            load_chunk(zg, 1, g, t, zb[1]);
        }
        v7::cp_wait<0>();
        __syncwarp();
        if (!valid) {                          // a padded row (or a cursor out of range): zero, nothing appended
#pragma unroll
            for (int c = 0; c < 4; ++c) v7::store_out(a, out_base + 32 * c + lane, 0.f);
            if constexpr (EARLY) first_chunks(next, next_sl, next_wp);
        } else {
            // ---- gates ------------------------------------------------------------------------------------------
            constexpr int ESIZE = IN == F32 ? 4 : 2;
            const int a_off = static_cast<int>(reinterpret_cast<uintptr_t>(v7::gate_address(a.a, n, a.stride_a, hv, ESIZE)) & 15);
            const int b_off = static_cast<int>(reinterpret_cast<uintptr_t>(v7::gate_address(a.b, n, a.stride_b, hv, ESIZE)) & 15);
            const float a_in = in_at<IN>(s + AB + a_off, 0), b_in = in_at<IN>(s + BB + b_off, 0);
            float alpha, beta, gl;
            if (a.pre_gated) {
                alpha = a_in;
                beta = b_in;
                gl = logf(alpha);
            } else {
                beta = round_to(1.f / (1.f + expf(-b_in)), IN);
                const float xg = a_in + __ldg(a.dt_bias + hv);
                const float softplus = xg <= 20.f ? logf(1.f + expf(xg)) : xg;
                gl = -expf(__ldg(a.A_log + hv)) * softplus;
                alpha = expf(gl);
            }
            const __half g_stored = __float2half_rn(gl);
            const float alpha_stored = expf(__half2float(g_stored));
            // ---- fades ------------------------------------------------------------------------------------------
            float* const xf = reinterpret_cast<float*>(s + XF);
            float cum = (lane < wp) ? __half2float(reinterpret_cast<const __half*>(s + GB)[lane]) : 0.f;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                const float o = __shfl_up_sync(FULL, cum, off);
                if (lane >= off) cum += o;
            }
            const float g_total = __shfl_sync(FULL, cum, 15);
            if (lane < L) xf[lane] = (lane < wp) ? expf(g_total - cum) : 0.f;
            const float window_fade = expf(g_total);
            // ---- key side: lane l has key channels 4l .. 4l+3 ----------------------------------------------------
            float q4[4], k4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                q4[i] = in_at<IN>(s + QB, 4 * lane + i);
                k4[i] = in_at<IN>(s + KB, 4 * lane + i);
            }
            const float4 sc4 = *reinterpret_cast<const float4*>(s + SC + 16 * lane);
            const float scs[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
            float sqq = 0.f, skk = 0.f, mq = 0.f, mk = 0.f, wq[4], wk[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sqq += q4[i] * q4[i];
                skk += k4[i] * k4[i];
                wq[i] = scs[i] * q4[i];
                wk[i] = scs[i] * k4[i];
                mq = fmaxf(mq, fabsf(wq[i]));
                mk = fmaxf(mk, fabsf(wk[i]));
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sqq += __shfl_xor_sync(FULL, sqq, off);
                skk += __shfl_xor_sync(FULL, skk, off);
                mq = fmaxf(mq, __shfl_xor_sync(FULL, mq, off));
                mk = fmaxf(mk, __shfl_xor_sync(FULL, mk, off));
            }
            const float q_rnorm = a.l2norm ? 1.f / sqrtf(sqq + 1e-6f) : 1.f;
            const float k_rnorm = a.l2norm ? 1.f / sqrtf(skk + 1e-6f) : 1.f;
            const float qf = q_rnorm * a.scale, kf = k_rnorm;
            {
                const float inv_q = mq > 0.f ? 127.f / mq : 0.f, inv_k = mk > 0.f ? 127.f / mk : 0.f;
                uint32_t pq[4] = {0u, 0u, 0u, 0u}, pk[4] = {0u, 0u, 0u, 0u};
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float r = wq[i] * inv_q;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float piece = rintf(r);
                        pq[p] |= (static_cast<uint32_t>(static_cast<int>(piece)) & 0xffu) << (8 * i);
                        r = (r - piece) * 128.f;
                    }
                    r = wk[i] * inv_k;
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float piece = rintf(r);
                        pk[p] |= (static_cast<uint32_t>(static_cast<int>(piece)) & 0xffu) << (8 * i);
                        r = (r - piece) * 128.f;
                    }
                }
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    *reinterpret_cast<uint32_t*>(s + XW + p * W_ROW + 4 * lane) = pq[p];
                    *reinterpret_cast<uint32_t*>(s + XW + (4 + p) * W_ROW + 4 * lane) = pk[p];
                }
            }
            float new_kq = 0.f;
            {
                __half key[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    key[i] = __float2half_rn(beta * (k4[i] * k_rnorm));
                    new_kq = fmaf(__half2float(key[i]), q4[i] * q_rnorm * a.scale, new_kq);
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) new_kq += __shfl_xor_sync(FULL, new_kq, off);
                const uint32_t w0 = static_cast<uint32_t>(__half_as_ushort(key[0])) |
                                    (static_cast<uint32_t>(__half_as_ushort(key[1])) << 16);
                const uint32_t w1 = static_cast<uint32_t>(__half_as_ushort(key[2])) |
                                    (static_cast<uint32_t>(__half_as_ushort(key[3])) << 16);
                *reinterpret_cast<uint2*>(a.win_k + (head * L + wp) * K + 4 * lane) = make_uint2(w0, w1);
            }
            __syncwarp();                      // the pieces and the fades are visible to the warp
            // ---- record and compensator keys . q, . k (fp16 tensor cores; B columns q hi, q lo, k hi, k lo) ---------
            const int m = lane >> 3, lrow = ((m & 1) << 3) + (lane & 7), half8 = m >> 1;
            {
                float rec[4] = {0.f, 0.f, 0.f, 0.f}, cmp[4] = {0.f, 0.f, 0.f, 0.f};
                const uint32_t rk_base = v7::saddr(s + RK), ck_base = v7::saddr(s + CK);
                const unsigned char* column = s + (g < 2 ? QB : KB);
                const int crow = lrow & 7;
#pragma unroll
                for (int ks = 0; ks < K / 16; ++ks) {
                    uint32_t b0 = 0u, b1 = 0u;
                    if (g < 4) {
                        const int kk = 16 * ks + 2 * t;
                        float x0 = in_at<IN>(column, kk), x1 = in_at<IN>(column, kk + 1);
                        float x2 = in_at<IN>(column, kk + 8), x3 = in_at<IN>(column, kk + 9);
                        if (g & 1) {
                            x0 = v7::lo_part(x0);
                            x1 = v7::lo_part(x1);
                            x2 = v7::lo_part(x2);
                            x3 = v7::lo_part(x3);
                        }
                        b0 = v7::pack_half2(x0, x1);
                        b1 = v7::pack_half2(x2, x3);
                    }
                    const int c = 2 * ks + half8;
                    uint32_t a0, a1, a2, a3;
                    if (wp > 0) {              // no record in the window: nothing to read
                        v7::ldsm4(a0, a1, a2, a3, rk_base + lrow * 256 + ((c ^ (lrow & 7)) << 4));
                        v7::mma_f16(rec, a0, a1, a2, a3, b0, b1);
                    }
                    v7::ldsm4(a0, a1, a2, a3, ck_base + crow * 256 + ((c ^ crow) << 4));
                    v7::mma_f16(cmp, a0, a1, a2, a3, b0, b1);
                }
                float* const xr = reinterpret_cast<float*>(s + XR);
                float* const xc = reinterpret_cast<float*>(s + XC);
                if (t < 2) {
                    const float f = t == 0 ? qf : kf;
                    xr[t * L + g] = (rec[0] + rec[1]) * f;
                    xr[t * L + g + 8] = (rec[2] + rec[3]) * f;
                    xc[t * RMAX + g] = (cmp[0] + cmp[1]) * f;
                }
            }
            // ---- the int8 matrix, chunk by chunk (int8 tensor cores; the key axis in load order) ----------------
            {
                uint32_t bw0[4], bw1[4];       // B fragments of the 4 k-tiles: piece column g, key channels in load order
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int base = 64 * (j >> 1) + 16 * t + 8 * (j & 1);
                    bw0[j] = *reinterpret_cast<const uint32_t*>(s + XW + g * W_ROW + base);
                    bw1[j] = *reinterpret_cast<const uint32_t*>(s + XW + g * W_ROW + base + 4);
                }
                const float f0 = (t & 1) ? (1.f / 16384.f) : 1.f, f1 = (t & 1) ? (1.f / 2097152.f) : (1.f / 128.f);
                const float unit = (t >> 1) ? mk / 127.f : mq / 127.f;
                float* const xz = reinterpret_cast<float*>(s + XZ);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    uint4 (&zc)[8] = zb[c & 1];
                    int acc[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        const uint4 lo_a = zc[4 * mt + 0], lo_b = zc[4 * mt + 1];
                        const uint4 hi_a = zc[4 * mt + 2], hi_b = zc[4 * mt + 3];
                        v7::mma_s8(acc[mt], lo_a.x, hi_a.x, lo_a.y, hi_a.y, bw0[0], bw1[0]);
                        v7::mma_s8(acc[mt], lo_a.z, hi_a.z, lo_a.w, hi_a.w, bw0[1], bw1[1]);
                        v7::mma_s8(acc[mt], lo_b.x, hi_b.x, lo_b.y, hi_b.y, bw0[2], bw1[2]);
                        v7::mma_s8(acc[mt], lo_b.z, hi_b.z, lo_b.w, hi_b.w, bw0[3], bw1[3]);
                    }
                    if (c + 2 < 4) load_chunk(zg, c + 2, g, t, zb[c & 1]);    // this buffer is free again
#pragma unroll
                    for (int mt = 0; mt < 2; ++mt) {
                        float p0 = static_cast<float>(acc[mt][0]) * f0 + static_cast<float>(acc[mt][1]) * f1;
                        float p1 = static_cast<float>(acc[mt][2]) * f0 + static_cast<float>(acc[mt][3]) * f1;
                        p0 += __shfl_xor_sync(FULL, p0, 1);
                        p1 += __shfl_xor_sync(FULL, p1, 1);
                        if ((t & 1) == 0) {
                            xz[(t >> 1) * V + 32 * c + 16 * mt + g] = p0 * unit;
                            xz[(t >> 1) * V + 32 * c + 16 * mt + g + 8] = p1 * unit;
                        }
                    }
                }
            }
            if constexpr (EARLY) first_chunks(next, next_sl, next_wp);     // both chunk buffers are free now
            __syncwarp();                      // the int8 parts and the record dots are visible to the warp
            if constexpr (EPI4) {
                // ---- value rows 4l .. 4l+3 of lane l: one 8-byte load per record or compensator covers them, and
                // each record's coefficient (fade * key . q) is formed once, not once per row --------------------------
                const float* const xz = reinterpret_cast<const float*>(s + XZ);
                const float* const xr = reinterpret_cast<const float*>(s + XR);
                const float* const xc = reinterpret_cast<const float*>(s + XC);
                const int v0 = 4 * lane;
                const float4 st4 = *reinterpret_cast<const float4*>(s + ST + 16 * lane);
                const float4 zq4 = *reinterpret_cast<const float4*>(xz + v0);
                const float4 zk4 = *reinterpret_cast<const float4*>(xz + V + v0);
                float sq[4] = {zq4.x * qf * st4.x, zq4.y * qf * st4.y, zq4.z * qf * st4.z, zq4.w * qf * st4.w};
                float sk[4] = {zk4.x * kf * st4.x, zk4.y * kf * st4.y, zk4.z * kf * st4.z, zk4.w * kf * st4.w};
                for (int r = 0; r < R; ++r) {
                    const uint2 w = *reinterpret_cast<const uint2*>(s + CC + (r * V + v0) * 2);
                    const float cv[4] = {half_lo(w.x), half_hi(w.x), half_lo(w.y), half_hi(w.y)};
                    const float cq = xc[r], ck = xc[RMAX + r];
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sq[i] = fmaf(cv[i], cq, sq[i]);
                        sk[i] = fmaf(cv[i], ck, sk[i]);
                    }
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sq[i] *= window_fade;
                    sk[i] *= window_fade;
                }
                for (int j = 0; j < wp; ++j) {
                    const float fj = xf[j];
                    const float cq = fj * xr[j], ck = fj * xr[L + j];
                    const uint2 w = *reinterpret_cast<const uint2*>(s + RC + (j * V + v0) * 2);
                    const float rv[4] = {half_lo(w.x), half_hi(w.x), half_lo(w.y), half_hi(w.y)};
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        sq[i] = fmaf(rv[i], cq, sq[i]);
                        sk[i] = fmaf(rv[i], ck, sk[i]);
                    }
                }
                __half corr[4];
                float o[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    corr[i] = __float2half_rn(in_at<IN>(s + VB, v0 + i) - alpha * sk[i]);
                    o[i] = __half2float(corr[i]) * new_kq + alpha_stored * sq[i];
                }
                *reinterpret_cast<uint2*>(a.win_c + (head * L + wp) * V + v0) =
                    make_uint2(static_cast<uint32_t>(__half_as_ushort(corr[0])) | (static_cast<uint32_t>(__half_as_ushort(corr[1])) << 16),
                               static_cast<uint32_t>(__half_as_ushort(corr[2])) | (static_cast<uint32_t>(__half_as_ushort(corr[3])) << 16));
                if (a.out_dtype == BF16) {
                    const __nv_bfloat162 p0 = __floats2bfloat162_rn(o[0], o[1]), p1 = __floats2bfloat162_rn(o[2], o[3]);
                    *reinterpret_cast<uint2*>(static_cast<__nv_bfloat16*>(a.out) + out_base + v0) =
                        make_uint2(*reinterpret_cast<const uint32_t*>(&p0), *reinterpret_cast<const uint32_t*>(&p1));
                } else if (a.out_dtype == F16) {
                    const __half2 p0 = __floats2half2_rn(o[0], o[1]), p1 = __floats2half2_rn(o[2], o[3]);
                    *reinterpret_cast<uint2*>(static_cast<__half*>(a.out) + out_base + v0) =
                        make_uint2(*reinterpret_cast<const uint32_t*>(&p0), *reinterpret_cast<const uint32_t*>(&p1));
                } else {
                    *reinterpret_cast<float4*>(static_cast<float*>(a.out) + out_base + v0) = make_float4(o[0], o[1], o[2], o[3]);
                }
                if (lane == 0) a.win_g[head * L + wp] = g_stored;
            } else {
            // ---- value rows v = 32c + lane: state^T q and state^T k before the token, then the token -------------
            {
                const float* const xz = reinterpret_cast<const float*>(s + XZ);
                const float* const xr = reinterpret_cast<const float*>(s + XR);
                const float* const xc = reinterpret_cast<const float*>(s + XC);
                const __half* ccv = reinterpret_cast<const __half*>(s + CC);
                const __half* rcv = reinterpret_cast<const __half*>(s + RC);
#pragma unroll
                for (int c = 0; c < 4; ++c) {
                    const int v = 32 * c + lane;
                    const float step = reinterpret_cast<const float*>(s + ST)[v];
                    float sq = xz[v] * qf * step, sk = xz[V + v] * kf * step;
                    for (int r = 0; r < R; ++r) {
                        const float cv = __half2float(ccv[r * V + v]);
                        sq = fmaf(cv, xc[r], sq);
                        sk = fmaf(cv, xc[RMAX + r], sk);
                    }
                    sq *= window_fade;
                    sk *= window_fade;
                    for (int j = 0; j < wp; ++j) {
                        const float cf = __half2float(rcv[j * V + v]) * xf[j];
                        sq = fmaf(cf, xr[j], sq);
                        sk = fmaf(cf, xr[L + j], sk);
                    }
                    const float value = in_at<IN>(s + VB, v);
                    const __half correction = __float2half_rn(value - alpha * sk);
                    a.win_c[(head * L + wp) * V + v] = correction;
                    v7::store_out(a, out_base + v, __half2float(correction) * new_kq + alpha_stored * sq);
                }
                if (lane == 0) a.win_g[head * L + wp] = g_stored;
            }
            }                                  // EPI4
        }
        // ---- the next item's small inputs (this item's shared memory is read by now) -----------------------------
        __syncwarp();
        sl = next_sl;
        wp = next_wp;
        issue<IN>(a, s, next, total, sl, wp);
        next += wstride;
        if (next < total) {
            const int n2 = next / HV;
            next_sl = __ldg(a.slot + n2);
            next_wp = __ldg(a.write_pos + n2);
        }
    }
    v7::cp_wait<0>();
}

}  // namespace lqfd

namespace {

// v7 launch: as many blocks as fit at once (persistent, grid stride), at most one per item.
template <int S>
int launch_v7(const lqfd::Args* args, long long items, cudaStream_t stream) {
    static int blocks_per_sm = 0, sms = 0;
    constexpr int bytes = lqfd::v7::smem_bytes<S>();
    if (blocks_per_sm == 0) {
        cudaError_t e = cudaFuncSetAttribute(lqfd::decode7_kernel<S>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return static_cast<int>(e);
        int device = 0;
        e = cudaGetDevice(&device);
        if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        int fit = 0;
        if (e == cudaSuccess)
            e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, lqfd::decode7_kernel<S>, lqfd::v7::THREADS, bytes);
        if (e != cudaSuccess) return static_cast<int>(e);
        if (fit < 1) return -2;
        blocks_per_sm = fit;
    }
    const long long most = static_cast<long long>(blocks_per_sm) * sms;
    const int grid = static_cast<int>(items < most ? items : most);
    lqfd::decode7_kernel<S><<<grid, lqfd::v7::THREADS, bytes, stream>>>(*args);
    return static_cast<int>(cudaGetLastError());
}

// v8 launch: one warp per item, as many blocks as fit at once (grid stride over the warps).
template <int IN, bool EPI4, bool EARLY>
int launch_v8(const lqfd::Args* args, long long items, cudaStream_t stream) {
    static int blocks_per_sm = 0, sms = 0;
    constexpr int bytes = lqfd::v8::BLOCK_BYTES;
    if (blocks_per_sm == 0) {
        cudaError_t e = cudaFuncSetAttribute(lqfd::decode8_kernel<IN, EPI4, EARLY>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (e != cudaSuccess) return static_cast<int>(e);
        int device = 0;
        e = cudaGetDevice(&device);
        if (e == cudaSuccess) e = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        int fit = 0;
        if (e == cudaSuccess)
            e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, lqfd::decode8_kernel<IN, EPI4, EARLY>, lqfd::v8::WARPS * 32, bytes);
        if (e != cudaSuccess) return static_cast<int>(e);
        if (fit < 1) return -2;
        blocks_per_sm = fit;
    }
    const long long warps_needed = items;
    const long long blocks_needed = (warps_needed + lqfd::v8::WARPS - 1) / lqfd::v8::WARPS;
    const long long most = static_cast<long long>(blocks_per_sm) * sms;
    const int grid = static_cast<int>(blocks_needed < most ? blocks_needed : most);
    lqfd::decode8_kernel<IN, EPI4, EARLY><<<grid, lqfd::v8::WARPS * 32, bytes, stream>>>(*args);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace

extern "C" {

int lq_fast_decode_args_size(void) { return static_cast<int>(sizeof(lqfd::Args)); }

// Blocks of the v7 decode that fit on one multiprocessor (0 if the variant is unknown), for the record.
int lq_fast_decode_blocks_per_sm(int variant) {
    int fit = 0;
    if (variant == 2)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, lqfd::decode7_kernel<2>, lqfd::v7::THREADS, lqfd::v7::smem_bytes<2>());
    else if (variant == 3)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, lqfd::decode7_kernel<3>, lqfd::v7::THREADS, lqfd::v7::smem_bytes<3>());
    else if (variant == 4)
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, lqfd::decode7_kernel<4>, lqfd::v7::THREADS, lqfd::v7::smem_bytes<4>());
    else if (variant == 8) {
        cudaFuncSetAttribute(lqfd::decode8_kernel<lqfd::BF16, false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, lqfd::v8::BLOCK_BYTES);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, lqfd::decode8_kernel<lqfd::BF16, false, false>, lqfd::v8::WARPS * 32,
                                                      lqfd::v8::BLOCK_BYTES);
    }
    return fit;
}

// One decode token for every row (as leapquant_triton.decode_step_raw / decode_step). Returns 0 or a CUDA error;
// -1 if the pool is outside the kernel's limits (v7 also needs a window of exactly 16 records).
int lq_fast_decode(const lqfd::Args* args, void* stream) {
    if (args->key_channels != lqfd::K || args->value_channels != lqfd::V || args->window_length < 1 ||
        args->window_length > lqfd::L || args->compensators < 1 || args->compensators > lqfd::RMAX ||
        args->key_heads < 1 || args->value_heads % args->key_heads != 0 || args->rows < 0)
        return -1;
    const long long items = static_cast<long long>(args->rows) * args->value_heads;
    if (items == 0) return 0;
    if (args->variant != 0) {
        if (args->window_length != lqfd::L) return -1;
        const cudaStream_t s = static_cast<cudaStream_t>(stream);
        if (args->variant == 2) return launch_v7<2>(args, items, s);
        if (args->variant == 3) return launch_v7<3>(args, items, s);
        if (args->variant == 4) return launch_v7<4>(args, items, s);
        if (args->variant >= 8 && args->variant <= 10) {   // 9: 4 consecutive value rows per lane; 10: 9 + early loads
            const bool epi4 = args->variant >= 9, early = args->variant == 10;
            if (args->in_dtype == lqfd::BF16) {
                if (early) return launch_v8<lqfd::BF16, true, true>(args, items, s);
                return epi4 ? launch_v8<lqfd::BF16, true, false>(args, items, s) : launch_v8<lqfd::BF16, false, false>(args, items, s);
            }
            if (args->in_dtype == lqfd::F16) {
                if (early) return launch_v8<lqfd::F16, true, true>(args, items, s);
                return epi4 ? launch_v8<lqfd::F16, true, false>(args, items, s) : launch_v8<lqfd::F16, false, false>(args, items, s);
            }
            if (early) return launch_v8<lqfd::F32, true, true>(args, items, s);
            return epi4 ? launch_v8<lqfd::F32, true, false>(args, items, s) : launch_v8<lqfd::F32, false, false>(args, items, s);
        }
        return -1;
    }
    const int grid = static_cast<int>((items + lqfd::WARPS - 1) / lqfd::WARPS);
    lqfd::fast_decode_kernel<<<grid, lqfd::WARPS * 32, 0, static_cast<cudaStream_t>(stream)>>>(*args);
    return static_cast<int>(cudaGetLastError());
}

}  // extern "C"
