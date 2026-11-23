// SPDX-License-Identifier: Apache-2.0
//
// leapquant_fast_fold.cu
//
// The fold of the serving path, written by hand for Hopper (sm_90). It does what the Triton fold of
// kernel/leapquant_triton.py does (same pool, same work list, same numerical contract) with fewer
// instructions and fewer live registers, so that more heads are in flight on each multiprocessor.
//
// One block of 256 threads (8 warps) folds one head at a time and walks the work list, so the
// launch has a fixed size and can be captured in a CUDA graph. Warp w owns value channels
// 16w..16w+15 (all 128 key channels of them).
//
//   0. Load. The int8 levels are staged in shared memory; the records and the old compensators are
//      laid out as the two factors of one low-rank product (records and old compensators each as two
//      fp16 parts, hi + lo, so their products lose nothing against fp32). Up to 4 compensators, the
//      next head's levels, records, compensators, steps and scales are fetched with cp.async while
//      this head is fitted (two blocks of about 111 KB still fit on a multiprocessor).
//   1. M, the full-precision state at the end of the window, is built ONCE, in registers, in the
//      accumulator layout of the tensor cores:  levels * (fade_all * step) * scale  plus the
//      low-rank product (one pass of mma.sync over 48 terms). A bf16 copy of M goes to shared memory.
//   2. The fit (block iteration, as the Triton fold: U = M^T X, U orthonormal (Gram-Schmidt), X = M U)
//      reads the bf16 copy on the tensor cores: 8 small products per warp and pass, the last X = M U too
//      (-DLQFF_FINAL_X_FROM_COPY=0: from the fp32 M in registers, split into bf16 hi + lo; no better).
//   3. New compensators as stored (fp16), then  remainder = M - new_c new_k^T  with one more product
//      into the same registers; smoothing scales (column sums: reduce-scatter in the warp, then across
//      warps), one step per value channel (a row lies in one warp: no block-wide reduction), rounding
//      to the nearest level (ties to even, by adding 1.5 * 2^23), and coalesced 16-byte stores.
//
// Differences from the Triton fold, all inside the fast path's tolerance (kernel/verify_kernel.py
// --fast, kernel/bench_triton.py, kernel/fast_fold_test.py): the fit reads M in bf16 (the Triton fit
// reads the levels in fp16 and the vectors rescaled to fp16), the sums run in another order, and the
// remainder is multiplied by reciprocals of the scales and steps instead of divided by them.
//
// Limits: key and value channels 128, window at most 16 records, 1 to 8 compensators, one step per
// value column. The caller (kernel/leapquant_fast_cuda.py) falls back to the Triton fold outside them.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// 1 (default since 14 Nov 2026, results/fast_fold_v3): the last X = M U from the bf16 copy of M, like the other passes.
// Measured: fold 1.11 against 1.16 ms at batch 512 (0.264 against 0.273 at 128), fold loss unchanged (1.0060 of the
// exact kernel's); the value sides are stored in fp16 anyway. 0: from the fp32 registers (bf16 hi + lo).
#ifndef LQFF_FINAL_X_FROM_COPY
#define LQFF_FINAL_X_FROM_COPY 1
#endif

// Gram-Schmidt with each step's sums in one butterfly (1) or one warp sum after another (0). Same arithmetic up to
// rounding; RP butterflies instead of RP (RP + 3) / 2 sums in series, on one warp while the block waits.
#ifndef LQFF_BATCHED_GS
#define LQFF_BATCHED_GS 1
#endif

namespace lqff {

constexpr int V = 128;          // value channels (rows of a head's state)
constexpr int K = 128;          // key channels (columns)
constexpr int L = 16;           // most records in a window
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int ZROW = 144;       // bytes per row of an int8 tile in shared memory (16-byte rows, no bank conflicts)
constexpr int MROW = 136;       // 16-bit elements per row of a 128-wide tile (272 bytes: ldmatrix without conflicts)
constexpr int AROW = 56;        // 16-bit elements per row of the low-rank factor A (48 used)
constexpr int NROW = 24;        // 16-bit elements per row of the new compensators' value sides (16 used)
constexpr unsigned FULL = 0xffffffffu;

struct Args {
    int8_t* residual;           // [slots, HV, V, K]   read, then rewritten
    float* channel_step;        // [slots, HV, V]      read, then rewritten
    float* row_scale;           // [slots, HV, K]      read, then rewritten
    __half* comp_c;             // [slots, HV, R, V]   read, then rewritten
    __half* comp_k;             // [slots, HV, R, K]   read, then rewritten
    const __half* win_c;        // [slots, HV, WL, V]
    const __half* win_k;        // [slots, HV, WL, K]
    const __half* win_g;        // [slots, HV, WL]
    const int32_t* slot;        // [B]
    const int32_t* records;     // [B]  records to fold; the work list holds only rows with records > 0
    const int32_t* order;       // [B]  rows that fold, packed to the front
    const int32_t* count;       // [1]  how many rows fold
    int32_t value_heads;
    int32_t key_channels;
    int32_t value_channels;
    int32_t window_length;
    int32_t compensators;
    int32_t sweeps;             // block-iteration sweeps; 0 = half sweep (key sides of the old compensators kept)
    int32_t use_smoothing;
    float levels;               // largest integer level (127 for 8 bits)
    // Store (adoption) only: the full-precision states to compress.
    const float* source;        // fp32 [entries, HV, V, K], entries source_stride floats apart
    long long source_stride;
    const int32_t* source_index;  // [B] entry per row, or null: the entry of the row's slot
    int32_t has_index;
    int32_t reserved;
};

// Byte offsets of the shared-memory regions. RP = compensators rounded up to 4 or 8.
template <int RP>
struct Layout {
    static constexpr bool prefetch = RP <= 4;
    static constexpr int m_tile = 0;                                  // M bf16 [V][MROW]; int8 staging [V][ZROW] before (no prefetch) and after
    static constexpr int a_lr = m_tile + V * MROW * 2;                // fp16 [V][AROW]: rec hi | rec lo | old hi | old lo
    static constexpr int b_lr = a_lr + V * AROW * 2;                  // fp16 [32][MROW]: record keys | old keys, old keys, 0
    static constexpr int a_new = b_lr + 32 * MROW * 2;                // fp16 [V][NROW]: -new value sides
    static constexpr int b_new = a_new + V * NROW * 2;                // fp16 [16][MROW]: new key sides
    static constexpr int w_x = b_new + 16 * MROW * 2;                 // bf16 [2RP][MROW]: X hi, lo per compensator, by value channel
    static constexpr int w_u = w_x + 2 * RP * MROW * 2;               // bf16 [2RP][MROW]: U hi, lo per compensator, by key channel
    static constexpr int u_f = w_u + 2 * RP * MROW * 2;               // fp32 [RP][128]
    static constexpr int colsum = u_f + RP * 128 * 4;                 // fp32 [WARPS][K]
    static constexpr int step = colsum + WARPS * K * 4;               // fp32 [V]   (no prefetch)
    static constexpr int scale = step + V * 4;                        // fp32 [K]   (no prefetch)
    static constexpr int fstep = scale + K * 4;                       // fp32 [V]: fade_all * step
    static constexpr int inv_scale = fstep + V * 4;                   // fp32 [K]
    static constexpr int fade = inv_scale + K * 4;                    // fp32 [L]
    static constexpr int xsq = fade + L * 4;                          // fp32 [WARPS][RP]
    static constexpr int balance = xsq + WARPS * RP * 4;              // fp32 [RP]
    static constexpr int flags = balance + RP * 4;                    // int  [RP]: old value side is not zero
    static constexpr int scalars = flags + RP * 4;                    // fp32 [4]: fade_all
    static constexpr int core_end = (scalars + 16 + 15) / 16 * 16;
    // Prefetched inputs of the head being folded (written by cp.async during the previous head).
    static constexpr int p_levels = core_end;                         // int8 [V][ZROW]
    static constexpr int p_win_c = p_levels + V * ZROW;               // fp16 [L][V]
    static constexpr int p_win_k = p_win_c + L * V * 2;               // fp16 [L][K]
    static constexpr int p_comp_c = p_win_k + L * K * 2;              // fp16 [RP][V]
    static constexpr int p_comp_k = p_comp_c + RP * V * 2;            // fp16 [RP][K]
    static constexpr int p_step = p_comp_k + RP * K * 2;              // fp32 [V]
    static constexpr int p_scale = p_step + V * 4;                    // fp32 [K]
    static constexpr int bytes = prefetch ? p_scale + K * 4 : core_end;
    // Store only (no prefetch): the pivoted start vectors.
    static constexpr int s_x = core_end;                              // fp32 [RP][V]: chosen directions, orthonormal
    static constexpr int s_len = s_x + RP * V * 4;                    // fp32 [K]: squared length left per key row
    static constexpr int s_col = s_len + K * 4;                       // fp32 [V]: the column being chosen
    static constexpr int s_part = s_col + V * 4;                      // fp32 [4][RP + 1]: partial sums of 4 warps
    static constexpr int s_pick = s_part + 4 * (RP + 1) * 4;          // int [1] key row; fp32 [1] its length
    static constexpr int store_bytes = (s_pick + 16 + 15) / 16 * 16;
    static_assert(a_lr % 16 == 0 && b_lr % 16 == 0 && a_new % 16 == 0 && b_new % 16 == 0 && w_x % 16 == 0 &&
                  w_u % 16 == 0 && u_f % 16 == 0 && colsum % 16 == 0 && p_levels % 16 == 0 && p_win_c % 16 == 0 &&
                  p_win_k % 16 == 0 && p_comp_c % 16 == 0 && p_comp_k % 16 == 0 && p_step % 16 == 0 &&
                  p_scale % 16 == 0, "16-byte aligned regions");
    static_assert(V * ZROW <= V * MROW * 2, "the int8 staging fits in the M tile");
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

__device__ __forceinline__ void ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1) : "r"(addr) : "memory");
}

// 16 bytes from global to shared memory without passing through registers (L2 only).
__device__ __forceinline__ void cp_async16(void* shared, const void* global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr(shared)), "l"(global) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// D += A B: m16n8k16, fp32 accumulation. A row-major (4 registers), B column-major (2 registers).
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t as_u32(__nv_bfloat162 v) { return *reinterpret_cast<uint32_t*>(&v); }

__device__ __forceinline__ uint32_t pack_bf16(float first, float second) {
    return as_u32(__floats2bfloat162_rn(first, second));
}

// Two values as bf16 hi and lo parts (hi + lo carries about 16 bits).
__device__ __forceinline__ void split_bf16(float first, float second, uint32_t& hi, uint32_t& lo) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(first, second);
    const float2 back = __bfloat1622float2(h);
    hi = as_u32(h);
    lo = as_u32(__floats2bfloat162_rn(first - back.x, second - back.y));
}

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(FULL, v, 16);
    v += __shfl_xor_sync(FULL, v, 8);
    v += __shfl_xor_sync(FULL, v, 4);
    v += __shfl_xor_sync(FULL, v, 2);
    v += __shfl_xor_sync(FULL, v, 1);
    return v;
}

// STORE = false: the fold (boundary state + window records -> new boundary state).
// STORE = true: the store, or adoption (a full-precision state from `source` -> boundary state; the fit starts
// from the key rows chosen by a pivoted QR and runs a.sweeps sweeps; the caller passes the Triton store's
// SWEEPS + 2). Everything after M is built is shared.
template <int RP, bool STORE>
__global__ void __launch_bounds__(THREADS, 2) fast_fold_kernel(const Args a) {
    using Lay = Layout<RP>;
    constexpr bool PREFETCH = Lay::prefetch && !STORE;
    constexpr int NTW = RP / 4;                 // n-tiles of the fit vectors (hi and lo of 4 compensators each)
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* const zst = smem + Lay::m_tile;
    __nv_bfloat16* const m_tile = reinterpret_cast<__nv_bfloat16*>(smem + Lay::m_tile);
    __half* const a_lr = reinterpret_cast<__half*>(smem + Lay::a_lr);
    __half* const b_lr = reinterpret_cast<__half*>(smem + Lay::b_lr);
    __half* const a_new = reinterpret_cast<__half*>(smem + Lay::a_new);
    __half* const b_new = reinterpret_cast<__half*>(smem + Lay::b_new);
    __nv_bfloat16* const w_x = reinterpret_cast<__nv_bfloat16*>(smem + Lay::w_x);
    __nv_bfloat16* const w_u = reinterpret_cast<__nv_bfloat16*>(smem + Lay::w_u);
    float* const u_f = reinterpret_cast<float*>(smem + Lay::u_f);
    float* const colsum = reinterpret_cast<float*>(smem + Lay::colsum);
    float* const fstep_s = reinterpret_cast<float*>(smem + Lay::fstep);
    float* const inv_scale_s = reinterpret_cast<float*>(smem + Lay::inv_scale);
    float* const fade_s = reinterpret_cast<float*>(smem + Lay::fade);
    float* const xsq_s = reinterpret_cast<float*>(smem + Lay::xsq);
    float* const bal_s = reinterpret_cast<float*>(smem + Lay::balance);
    int* const flag_s = reinterpret_cast<int*>(smem + Lay::flags);
    float* const scal_s = reinterpret_cast<float*>(smem + Lay::scalars);
    // Inputs of the head being folded: prefetched regions, or (no prefetch) the staging in the M tile
    // and the step / scale arrays, filled with ordinary loads.
    unsigned char* const levels_in = PREFETCH ? smem + Lay::p_levels : zst;
    float* const step_in = reinterpret_cast<float*>(smem + (PREFETCH ? Lay::p_step : Lay::step));
    float* const scale_in = reinterpret_cast<float*>(smem + (PREFETCH ? Lay::p_scale : Lay::scale));
    const __half* const p_win_c = reinterpret_cast<const __half*>(smem + Lay::p_win_c);
    const __half* const p_win_k = reinterpret_cast<const __half*>(smem + Lay::p_win_k);
    const __half* const p_comp_c = reinterpret_cast<const __half*>(smem + Lay::p_comp_c);
    const __half* const p_comp_k = reinterpret_cast<const __half*>(smem + Lay::p_comp_k);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;                    // accumulator row group
    const int t = lane & 3;                     // accumulator column pair
    const int HV = a.value_heads;
    const int R = a.compensators;
    const int WL = a.window_length;
    const float LEVELS = a.levels;

    // Parts of the factor tiles that no head writes stay zero.
    const __half zero_h = __float2half_rn(0.f);
    for (int i = tid; i < V * AROW; i += THREADS) a_lr[i] = zero_h;
    for (int i = tid; i < 32 * MROW; i += THREADS) b_lr[i] = zero_h;
    for (int i = tid; i < V * NROW; i += THREADS) a_new[i] = zero_h;
    for (int i = tid; i < 16 * MROW; i += THREADS) b_new[i] = zero_h;
    if (tid < RP) flag_s[tid] = 0;

    // cp.async of one head's inputs into the prefetch regions (every thread issues its pieces).
    auto prefetch = [&](int item) {
        const int row = a.order[item / HV];
        const long long head = static_cast<long long>(a.slot[row]) * HV + (item % HV);
        const int8_t* zg = a.residual + head * (V * K);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int q = tid + THREADS * i;
            cp_async16(smem + Lay::p_levels + (q >> 3) * ZROW + ((q & 7) << 4), zg + q * 16);
        }
        const int j = tid >> 4, c8 = (tid & 15) << 3;
        if (j < WL) {
            cp_async16(smem + Lay::p_win_c + (j * V + c8) * 2, a.win_c + (head * WL + j) * V + c8);
            cp_async16(smem + Lay::p_win_k + (j * K + c8) * 2, a.win_k + (head * WL + j) * K + c8);
        }
        if (j < R) {
            cp_async16(smem + Lay::p_comp_c + (j * V + c8) * 2, a.comp_c + (head * R + j) * V + c8);
            cp_async16(smem + Lay::p_comp_k + (j * K + c8) * 2, a.comp_k + (head * R + j) * K + c8);
        }
        if (tid < 32) cp_async16(smem + Lay::p_step + tid * 16, a.channel_step + head * V + tid * 4);
        else if (tid < 64) cp_async16(smem + Lay::p_scale + (tid - 32) * 16, a.row_scale + head * K + (tid - 32) * 4);
        cp_async_commit();
    };

    const int total = *a.count * HV;
    if constexpr (PREFETCH) {
        if (static_cast<int>(blockIdx.x) < total) prefetch(blockIdx.x);
    }
    for (int item = blockIdx.x; item < total; item += gridDim.x) {
        const int row = a.order[item / HV];
        const int hv = item - (item / HV) * HV;
        const long long head = static_cast<long long>(a.slot[row]) * HV + hv;
        if constexpr (PREFETCH) cp_async_wait_all();
        __syncthreads();                        // the previous head is done with shared memory; prefetched inputs landed
        float acc[16][4];                       // M, then the remainder, in the accumulator layout
        const int v0 = 16 * warp + g;
        const int v1 = v0 + 8;
        if constexpr (STORE) {
        // ---- 0/1 (store). M is given: the full-precision state, read straight into the registers ----------
        {
            const long long entry = a.has_index ? a.source_index[row] : a.slot[row];
            const float* src = a.source + entry * a.source_stride + static_cast<long long>(hv) * (V * K);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int col = 8 * nt + 2 * t;
                const float2 r0 = __ldg(reinterpret_cast<const float2*>(src + v0 * K + col));
                const float2 r1 = __ldg(reinterpret_cast<const float2*>(src + v1 * K + col));
                acc[nt][0] = r0.x;
                acc[nt][1] = r0.y;
                acc[nt][2] = r1.x;
                acc[nt][3] = r1.y;
            }
            const __nv_bfloat16 zero_b = __float2bfloat16_rn(0.f);
            for (int i = tid; i < 2 * RP * MROW; i += THREADS) w_x[i] = zero_b;   // the start vectors are chosen below
        }
        } else {
        const int n = min(a.records[row], WL);

        // ---- 0. load --------------------------------------------------------------------------
        if (warp == 0) {                        // decay of the window: fade[j] = exp(sum of the gates after j)
            const float gj = (lane < n) ? __half2float(a.win_g[head * WL + lane]) : 0.f;
            float cum = gj;
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) {
                const float o = __shfl_up_sync(FULL, cum, off);
                if (lane >= off) cum += o;
            }
            const float g_total = __shfl_sync(FULL, cum, 15);
            if (lane < L) fade_s[lane] = (lane < n) ? expf(g_total - cum) : 0.f;
            if (lane == 0) scal_s[0] = expf(g_total);
        }
        const int v_own = tid & (V - 1);              // two threads per value channel below
        const int half_own = tid >> 7;
        float co[RP / 2];                             // old value sides of this thread's compensators
        {
            const int j = tid >> 4;                    // record keys: 16 rows x 16 pieces of 8
            const int c8 = (tid & 15) << 3;
            int4 kv = make_int4(0, 0, 0, 0);
            if constexpr (PREFETCH) {
                if (j < n) kv = *reinterpret_cast<const int4*>(p_win_k + j * K + c8);
            } else {
                const int8_t* zg = a.residual + head * (V * K);
                int4 zv[4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int q = tid + THREADS * i;
                    zv[i] = __ldg(reinterpret_cast<const int4*>(zg + (q >> 3) * K + ((q & 7) << 4)));
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int q = tid + THREADS * i;
                    *reinterpret_cast<int4*>(zst + (q >> 3) * ZROW + ((q & 7) << 4)) = zv[i];
                }
                if (tid < V) step_in[tid] = a.channel_step[head * V + tid];
                else scale_in[tid - V] = a.row_scale[head * K + (tid - V)];
                if (j < n) kv = __ldg(reinterpret_cast<const int4*>(a.win_k + (head * WL + j) * K + c8));
            }
            *reinterpret_cast<int4*>(b_lr + j * MROW + c8) = kv;
            if (j < RP) {                              // old key sides, twice (paired with hi and with lo)
                int4 ov = make_int4(0, 0, 0, 0);
                if (j < R) {
                    if constexpr (PREFETCH) ov = *reinterpret_cast<const int4*>(p_comp_k + j * K + c8);
                    else ov = __ldg(reinterpret_cast<const int4*>(a.comp_k + (head * R + j) * K + c8));
                }
                *reinterpret_cast<int4*>(b_lr + (16 + j) * MROW + c8) = ov;
                *reinterpret_cast<int4*>(b_lr + (16 + RP + j) * MROW + c8) = ov;
            }
#pragma unroll
            for (int rr = 0; rr < RP / 2; ++rr) {
                const int r = half_own * (RP / 2) + rr;
                if constexpr (PREFETCH) co[rr] = (r < R) ? __half2float(p_comp_c[r * V + v_own]) : 0.f;
                else co[rr] = (r < R) ? __half2float(a.comp_c[(head * R + r) * V + v_own]) : 0.f;
                if (co[rr] != 0.f) flag_s[r] = 1;
            }
        }
        __syncthreads();

        auto record_c = [&](int j) -> float {           // correction of record j for this thread's value channel
            if constexpr (PREFETCH) return __half2float(p_win_c[j * V + v_own]);
            else return __half2float(a.win_c[(head * WL + j) * V + v_own]);
        };
        const float fade_all = scal_s[0];
#pragma unroll
        for (int jj = 0; jj < 8; jj += 2) {           // records: fade * correction, as hi + lo
            const int j = 8 * half_own + jj;
            const float c0 = (j < n) ? record_c(j) * fade_s[j] : 0.f;
            const float c1 = (j + 1 < n) ? record_c(j + 1) * fade_s[j + 1] : 0.f;
            const __half h0 = __float2half_rn(c0), h1 = __float2half_rn(c1);
            *reinterpret_cast<__half2*>(a_lr + v_own * AROW + j) = __halves2half2(h0, h1);
            *reinterpret_cast<__half2*>(a_lr + v_own * AROW + 16 + j) =
                __halves2half2(__float2half_rn(c0 - __half2float(h0)), __float2half_rn(c1 - __half2float(h1)));
        }
#pragma unroll
        for (int rr = 0; rr < RP / 2; ++rr) {         // old compensators: fade_all * value side, as hi + lo; start X
            const int r = half_own * (RP / 2) + rr;
            const float y = co[rr] * fade_all;
            const __half yh = __float2half_rn(y);
            a_lr[v_own * AROW + 32 + r] = yh;
            a_lr[v_own * AROW + 32 + RP + r] = __float2half_rn(y - __half2float(yh));
            float x = co[rr];
            const int alt = n - 1 - r;                // a missing compensator starts from one of the newest records
            if (r < R && flag_s[r] == 0 && alt >= 0) x += record_c(alt);
            const __nv_bfloat16 xh = __float2bfloat16_rn(x);
            w_x[(2 * r) * MROW + v_own] = xh;
            w_x[(2 * r + 1) * MROW + v_own] = __float2bfloat16_rn(x - __bfloat162float(xh));
        }
        if (half_own == 0) fstep_s[v_own] = fade_all * step_in[v_own];
        __syncthreads();
        if (tid < RP) flag_s[tid] = 0;                // for the next head (read only above)

        // ---- 1. M in registers: levels term, then the low-rank product ------------------------------
        {
            const float f0 = fstep_s[v0], f1 = fstep_s[v1];
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int col = 8 * nt + 2 * t;
                const uint16_t z0 = *reinterpret_cast<const uint16_t*>(levels_in + v0 * ZROW + col);
                const uint16_t z1 = *reinterpret_cast<const uint16_t*>(levels_in + v1 * ZROW + col);
                const float2 sc = *reinterpret_cast<const float2*>(scale_in + col);
                acc[nt][0] = static_cast<float>(static_cast<int8_t>(z0 & 0xff)) * f0 * sc.x;
                acc[nt][1] = static_cast<float>(static_cast<int8_t>(z0 >> 8)) * f0 * sc.y;
                acc[nt][2] = static_cast<float>(static_cast<int8_t>(z1 & 0xff)) * f1 * sc.x;
                acc[nt][3] = static_cast<float>(static_cast<int8_t>(z1 >> 8)) * f1 * sc.y;
            }
        }
        {
            uint32_t af[3][4];
            const int ar = 16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int s = 0; s < 3; ++s) ldsm_x4(af[s], smem_addr(a_lr + ar * AROW + 16 * s + (lane >> 4) * 8));
            const int br = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int np = 0; np < 8; ++np) {
                const int bc = 16 * np + (lane >> 4) * 8;
                uint32_t b[4];
                ldsm_x4_trans(b, smem_addr(b_lr + br * MROW + bc));          // record keys (for hi and lo)
                mma_f16(acc[2 * np], af[0], b[0], b[1]);
                mma_f16(acc[2 * np], af[1], b[0], b[1]);
                mma_f16(acc[2 * np + 1], af[0], b[2], b[3]);
                mma_f16(acc[2 * np + 1], af[1], b[2], b[3]);
                ldsm_x4_trans(b, smem_addr(b_lr + (16 + br) * MROW + bc));   // old key sides
                mma_f16(acc[2 * np], af[2], b[0], b[1]);
                mma_f16(acc[2 * np + 1], af[2], b[2], b[3]);
            }
        }
        }                                       // fold: M built
        __syncthreads();                        // every warp has read the inputs: the tile becomes M
        if constexpr (PREFETCH) {               // the next head's inputs land while this one is fitted
            if (item + static_cast<int>(gridDim.x) < total) prefetch(item + gridDim.x);
        }
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const int col = 8 * nt + 2 * t;
            *reinterpret_cast<uint32_t*>(m_tile + v0 * MROW + col) = pack_bf16(acc[nt][0], acc[nt][1]);
            *reinterpret_cast<uint32_t*>(m_tile + v1 * MROW + col) = pack_bf16(acc[nt][2], acc[nt][3]);
        }
        __syncthreads();

        // ---- 2. the fit ------------------------------------------------------------------------------
        // U = M^T X for the 16 key channels of this warp (bf16 copy of M), into u_f.
        auto u_pass = [&]() {
            float d[NTW][4];
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                uint32_t af[4];
                ldsm_x4_trans(af, smem_addr(m_tile + (16 * c + (lane & 7) + (lane >> 4) * 8) * MROW +
                                            16 * warp + ((lane >> 3) & 1) * 8));
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    uint32_t b0, b1;
                    ldsm_x2(b0, b1, smem_addr(w_x + (8 * nt + (lane & 7)) * MROW + 16 * c + ((lane >> 3) & 1) * 8));
                    mma_bf16(d[nt], af, b0, b1);
                }
            }
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                const int r = 4 * nt + t;
                u_f[r * 128 + 16 * warp + g] = d[nt][0] + d[nt][1];
                u_f[r * 128 + 16 * warp + g + 8] = d[nt][2] + d[nt][3];
            }
        };
        // Gram-Schmidt on the columns of U (warp 0), as the Triton fold: a column that is (nearly) a
        // combination of the ones before it becomes zero. Writes U (fp32) and its bf16 hi/lo parts.
        // (A Cholesky QR by 128 threads was tried: no faster, 14 Nov 2026; the other block on the
        // multiprocessor fills the issue slots of the waiting warps.)
        auto orthonormalize = [&]() {
            if (warp == 0) {
                float u[RP][4];
#pragma unroll
                for (int r = 0; r < RP; ++r)
#pragma unroll
                    for (int j = 0; j < 4; ++j) u[r][j] = u_f[r * 128 + lane + 32 * j];
                float start[RP];
#if LQFF_BATCHED_GS
                // Step p needs |u_p|^2 and u_p . u_q (q > p) of the current columns; u_q -= (u_p . u_q) / |u_p| *
                // (u_p / |u_p|) is the update below. Step 0 also gives every column's starting length.
#pragma unroll
                for (int p = 0; p < RP; ++p) {
                    float s[2 * RP - 1];        // [0] |u_p|^2; [q - p] u_p . u_q; step 0: [RP - 1 + q] |u_q|^2
#pragma unroll
                    for (int i = 0; i < 2 * RP - 1; ++i) s[i] = 0.f;
                    s[0] = u[p][0] * u[p][0] + u[p][1] * u[p][1] + u[p][2] * u[p][2] + u[p][3] * u[p][3];
#pragma unroll
                    for (int q = p + 1; q < RP; ++q)
                        s[q - p] = u[p][0] * u[q][0] + u[p][1] * u[q][1] + u[p][2] * u[q][2] + u[p][3] * u[q][3];
                    if (p == 0) {
#pragma unroll
                        for (int q = 1; q < RP; ++q)
                            s[RP - 1 + q] = u[q][0] * u[q][0] + u[q][1] * u[q][1] + u[q][2] * u[q][2] + u[q][3] * u[q][3];
                    }
                    const int count = p == 0 ? 2 * RP - 1 : RP - p;
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
#pragma unroll
                        for (int i = 0; i < 2 * RP - 1; ++i)
                            if (i < count) s[i] += __shfl_xor_sync(FULL, s[i], off);
                    }
                    if (p == 0) {
                        start[0] = s[0];
#pragma unroll
                        for (int q = 1; q < RP; ++q) start[q] = s[RP - 1 + q];
                    }
                    const bool keep = s[0] > 1e-6f * start[p];
                    const float inv = keep ? rsqrtf(s[0]) : 0.f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) u[p][j] *= inv;
#pragma unroll
                    for (int q = p + 1; q < RP; ++q) {
                        const float along = s[q - p] * inv;
#pragma unroll
                        for (int j = 0; j < 4; ++j) u[q][j] -= along * u[p][j];
                    }
                }
#else
#pragma unroll
                for (int r = 0; r < RP; ++r)
                    start[r] = warp_sum(u[r][0] * u[r][0] + u[r][1] * u[r][1] + u[r][2] * u[r][2] + u[r][3] * u[r][3]);
#pragma unroll
                for (int p = 0; p < RP; ++p) {
                    const float usq = warp_sum(u[p][0] * u[p][0] + u[p][1] * u[p][1] + u[p][2] * u[p][2] + u[p][3] * u[p][3]);
                    const bool keep = usq > 1e-6f * start[p];
                    const float inv = keep ? rsqrtf(usq) : 0.f;
#pragma unroll
                    for (int j = 0; j < 4; ++j) u[p][j] *= inv;
#pragma unroll
                    for (int q = p + 1; q < RP; ++q) {
                        const float along = warp_sum(u[p][0] * u[q][0] + u[p][1] * u[q][1] + u[p][2] * u[q][2] + u[p][3] * u[q][3]);
#pragma unroll
                        for (int j = 0; j < 4; ++j) u[q][j] -= along * u[p][j];
                    }
                }
#endif
#pragma unroll
                for (int r = 0; r < RP; ++r)
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int k = lane + 32 * j;
                        u_f[r * 128 + k] = u[r][j];
                        const __nv_bfloat16 h = __float2bfloat16_rn(u[r][j]);
                        w_u[(2 * r) * MROW + k] = h;
                        w_u[(2 * r + 1) * MROW + k] = __float2bfloat16_rn(u[r][j] - __bfloat162float(h));
                    }
            }
        };
        // X = M U for the 16 value channels of this warp, from the bf16 copy of M.
        auto x_pass_copy = [&](float (&d)[NTW][4]) {
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) d[nt][0] = d[nt][1] = d[nt][2] = d[nt][3] = 0.f;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                uint32_t af[4];
                ldsm_x4(af, smem_addr(m_tile + (16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8) * MROW +
                                      16 * c + (lane >> 4) * 8));
#pragma unroll
                for (int nt = 0; nt < NTW; ++nt) {
                    uint32_t b0, b1;
                    ldsm_x2(b0, b1, smem_addr(w_u + (8 * nt + (lane & 7)) * MROW + 16 * c + ((lane >> 3) & 1) * 8));
                    mma_bf16(d[nt], af, b0, b1);
                }
            }
        };
        // ... written as the next fit vectors.
        auto x_pass_next = [&]() {
            float d[NTW][4];
            x_pass_copy(d);
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                const int r = 4 * nt + t;
                const float x0 = d[nt][0] + d[nt][1], x1 = d[nt][2] + d[nt][3];
                const __nv_bfloat16 h0 = __float2bfloat16_rn(x0), h1 = __float2bfloat16_rn(x1);
                w_x[(2 * r) * MROW + v0] = h0;
                w_x[(2 * r + 1) * MROW + v0] = __float2bfloat16_rn(x0 - __bfloat162float(h0));
                w_x[(2 * r) * MROW + v1] = h1;
                w_x[(2 * r + 1) * MROW + v1] = __float2bfloat16_rn(x1 - __bfloat162float(h1));
            }
        };

        if constexpr (STORE) {
            // Start vectors as the Triton store: key rows (columns of the tile) chosen greedily, the longest, then
            // the one with the most length left outside the directions already chosen (a pivoted QR); plain
            // "the R longest" can pick R near-copies of the strongest pattern.
            float* const s_x = reinterpret_cast<float*>(smem + Lay::s_x);
            float* const s_len = reinterpret_cast<float*>(smem + Lay::s_len);
            float* const s_col = reinterpret_cast<float*>(smem + Lay::s_col);
            float* const s_part = reinterpret_cast<float*>(smem + Lay::s_part);
            int* const s_pick = reinterpret_cast<int*>(smem + Lay::s_pick);
            float* const s_pick_len = reinterpret_cast<float*>(smem + Lay::s_pick + 4);
            {                                    // squared length of every key row: reduce-scatter, then across warps
                float cs[32];
#pragma unroll
                for (int nt = 0; nt < 16; ++nt) {
                    cs[2 * nt] = acc[nt][0] * acc[nt][0] + acc[nt][2] * acc[nt][2];
                    cs[2 * nt + 1] = acc[nt][1] * acc[nt][1] + acc[nt][3] * acc[nt][3];
                }
                const bool b4 = (lane >> 4) & 1, b3 = (lane >> 3) & 1, b2 = (lane >> 2) & 1;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float keep = b4 ? cs[16 + i] : cs[i];
                    const float give = b4 ? cs[i] : cs[16 + i];
                    cs[i] = keep + __shfl_xor_sync(FULL, give, 16);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float keep = b3 ? cs[8 + i] : cs[i];
                    const float give = b3 ? cs[i] : cs[8 + i];
                    cs[i] = keep + __shfl_xor_sync(FULL, give, 8);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float keep = b2 ? cs[4 + i] : cs[i];
                    const float give = b2 ? cs[i] : cs[4 + i];
                    cs[i] = keep + __shfl_xor_sync(FULL, give, 4);
                }
                *reinterpret_cast<float2*>(colsum + warp * K + 16 * g + 2 * t) = make_float2(cs[0], cs[1]);
                *reinterpret_cast<float2*>(colsum + warp * K + 16 * g + 8 + 2 * t) = make_float2(cs[2], cs[3]);
            }
            __syncthreads();
            if (tid < K) {
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) s += colsum[w * K + tid];
                s_len[tid] = s;
            }
            __syncthreads();
            for (int p = 0; p < R; ++p) {
                if (warp == 0) {                 // the key row with the most length left (lowest index on ties)
                    float best = -2.f;
                    int index = 0;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int k = lane + 32 * j;
                        const float l = s_len[k];
                        if (l > best) {
                            best = l;
                            index = k;
                        }
                    }
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        const float other_best = __shfl_xor_sync(FULL, best, off);
                        const int other_index = __shfl_xor_sync(FULL, index, off);
                        if (other_best > best || (other_best == best && other_index < index)) {
                            best = other_best;
                            index = other_index;
                        }
                    }
                    if (lane == 0) {
                        s_pick[0] = index;
                        s_pick_len[0] = best;
                    }
                }
                __syncthreads();
                const int index = s_pick[0];
                // That column of M (from the bf16 copy: a start vector needs only its direction), minus its parts
                // along the directions chosen before, normalized. Threads 0..127: one value channel each.
                float c = 0.f;
                // s_part: [4 warps][RP] partial coefficients, then [4 warps] partial squared norms, in separate blocks
                // (racecheck: a widened read of the coefficients must not touch a norm being written).
                float* const s_norm = s_part + 4 * RP;
                if (tid < V) {
                    c = __bfloat162float(m_tile[tid * MROW + index]);
                    for (int q = 0; q < p; ++q) {
                        const float part = warp_sum(c * s_x[q * V + tid]);
                        if (lane == 0) s_part[warp * RP + q] = part;
                    }
                }
                __syncthreads();
                if (tid < V) {
                    for (int q = 0; q < p; ++q) {
                        const float coefficient = s_part[q] + s_part[RP + q] + s_part[2 * RP + q] + s_part[3 * RP + q];
                        c -= coefficient * s_x[q * V + tid];
                    }
                    const float part = warp_sum(c * c);
                    if (lane == 0) s_norm[warp] = part;
                }
                __syncthreads();
                if (tid < V) {
                    const float c2 = s_norm[0] + s_norm[1] + s_norm[2] + s_norm[3];
                    const bool use = s_pick_len[0] > 0.f && c2 > 0.f;
                    const float x = use ? c * rsqrtf(c2) : 0.f;
                    s_x[p * V + tid] = x;
                    const __nv_bfloat16 h = __float2bfloat16_rn(x);
                    w_x[(2 * p) * MROW + tid] = h;
                    w_x[(2 * p + 1) * MROW + tid] = __float2bfloat16_rn(x - __bfloat162float(h));
                }
                __syncthreads();
                u_pass();                        // u_f[p][k]: each key row along the new direction
                __syncthreads();
                if (tid < K) s_len[tid] = (tid == index) ? -1.f : s_len[tid] - u_f[p * 128 + tid] * u_f[p * 128 + tid];
                __syncthreads();
            }
        }

        const int sweeps = a.sweeps;
        if (sweeps <= 0) {                      // half sweep: U = the old key sides, made orthonormal
            for (int i = tid; i < RP * 128; i += THREADS) {
                const int r = i >> 7, k = i & 127;
                u_f[i] = __half2float(b_lr[(16 + r) * MROW + k]);
            }
            __syncthreads();
            orthonormalize();
            __syncthreads();
        } else {
            for (int s = 0; s < sweeps; ++s) {
                u_pass();
                __syncthreads();
                orthonormalize();
                __syncthreads();
                if (s + 1 < sweeps) {
                    x_pass_next();
                    __syncthreads();
                }
            }
        }
        float xd[NTW][4];
#if LQFF_FINAL_X_FROM_COPY
        x_pass_copy(xd);                        // the last X = M U from the bf16 copy as well
#else
        // The last X = M U from the fp32 M in registers (bf16 hi + lo of M times hi + lo of U).
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) xd[nt][0] = xd[nt][1] = xd[nt][2] = xd[nt][3] = 0.f;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            uint32_t ah[4], al[4];
            split_bf16(acc[2 * c][0], acc[2 * c][1], ah[0], al[0]);
            split_bf16(acc[2 * c][2], acc[2 * c][3], ah[1], al[1]);
            split_bf16(acc[2 * c + 1][0], acc[2 * c + 1][1], ah[2], al[2]);
            split_bf16(acc[2 * c + 1][2], acc[2 * c + 1][3], ah[3], al[3]);
#pragma unroll
            for (int nt = 0; nt < NTW; ++nt) {
                uint32_t b0, b1;
                ldsm_x2(b0, b1, smem_addr(w_u + (8 * nt + (lane & 7)) * MROW + 16 * c + ((lane >> 3) & 1) * 8));
                mma_bf16(xd[nt], ah, b0, b1);
                mma_bf16(xd[nt], al, b0, b1);
            }
        }
#endif

        // ---- 3. new compensators, remainder, smoothing, steps, rounding ---------------------------------
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {
            const float x0 = xd[nt][0] + xd[nt][1], x1 = xd[nt][2] + xd[nt][3];
            float p = x0 * x0 + x1 * x1;
            p += __shfl_xor_sync(FULL, p, 4);
            p += __shfl_xor_sync(FULL, p, 8);
            p += __shfl_xor_sync(FULL, p, 16);
            if (g == 0) xsq_s[warp * RP + 4 * nt + t] = p;
        }
        __syncthreads();
        if (tid < RP) {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < WARPS; ++w) s += xsq_s[w * RP + tid];
            bal_s[tid] = (s > 0.f && tid < R) ? sqrtf(sqrtf(s)) : 0.f;    // 0: no compensator in this slot
        }
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < NTW; ++nt) {      // value sides: X / balance, as stored (fp16)
            const int r = 4 * nt + t;
            const float b = bal_s[r];
            const float x0 = xd[nt][0] + xd[nt][1], x1 = xd[nt][2] + xd[nt][3];
            const __half c0 = __float2half_rn(b > 0.f ? x0 / b : 0.f);
            const __half c1 = __float2half_rn(b > 0.f ? x1 / b : 0.f);
            if (r < R) {
                a.comp_c[(head * R + r) * V + v0] = c0;
                a.comp_c[(head * R + r) * V + v1] = c1;
            }
            a_new[v0 * NROW + r] = __hneg(c0);
            a_new[v1 * NROW + r] = __hneg(c1);
        }
        for (int i = tid; i < RP * 128; i += THREADS) {   // key sides: U * balance, as stored (fp16)
            const int r = i >> 7, k = i & 127;
            const float b = bal_s[r];
            const __half kh = __float2half_rn(b > 0.f ? u_f[i] * b : 0.f);
            if (r < R) a.comp_k[(head * R + r) * K + k] = kh;
            b_new[r * MROW + k] = kh;
        }
        __syncthreads();
        {                                        // remainder = M - new_c new_k^T (products of fp16 values: exact)
            uint32_t af[4];
            ldsm_x4(af, smem_addr(a_new + (16 * warp + (lane & 7) + ((lane >> 3) & 1) * 8) * NROW + (lane >> 4) * 8));
            const int br = (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int np = 0; np < 8; ++np) {
                uint32_t b[4];
                ldsm_x4_trans(b, smem_addr(b_new + br * MROW + 16 * np + (lane >> 4) * 8));
                mma_f16(acc[2 * np], af, b[0], b[1]);
                mma_f16(acc[2 * np + 1], af, b[2], b[3]);
            }
        }
        if (a.use_smoothing) {                   // column sums of |remainder|: reduce-scatter over the 8 row groups
            float cs[32];
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                cs[2 * nt] = fabsf(acc[nt][0]) + fabsf(acc[nt][2]);
                cs[2 * nt + 1] = fabsf(acc[nt][1]) + fabsf(acc[nt][3]);
            }
            const bool b4 = (lane >> 4) & 1, b3 = (lane >> 3) & 1, b2 = (lane >> 2) & 1;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float keep = b4 ? cs[16 + i] : cs[i];
                const float give = b4 ? cs[i] : cs[16 + i];
                cs[i] = keep + __shfl_xor_sync(FULL, give, 16);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float keep = b3 ? cs[8 + i] : cs[i];
                const float give = b3 ? cs[i] : cs[8 + i];
                cs[i] = keep + __shfl_xor_sync(FULL, give, 8);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float keep = b2 ? cs[4 + i] : cs[i];
                const float give = b2 ? cs[i] : cs[4 + i];
                cs[i] = keep + __shfl_xor_sync(FULL, give, 4);
            }
            // This lane now holds the sums over the warp's 16 rows of n-tiles 2g and 2g+1, columns 2t, 2t+1.
            *reinterpret_cast<float2*>(colsum + warp * K + 16 * g + 2 * t) = make_float2(cs[0], cs[1]);
            *reinterpret_cast<float2*>(colsum + warp * K + 16 * g + 8 + 2 * t) = make_float2(cs[2], cs[3]);
        }
        __syncthreads();
        if (tid < K) {
            float sc = 1.f;
            if (a.use_smoothing) {
                float s = 0.f;
#pragma unroll
                for (int w = 0; w < WARPS; ++w) s += colsum[w * K + tid];
                sc = fmaxf(sqrtf(s / V), 1e-6f);
            }
            a.row_scale[head * K + tid] = sc;
            inv_scale_s[tid] = 1.f / sc;
        }
        __syncthreads();
        float m0 = 0.f, m1 = 0.f;               // one step per value channel: the row lies in this warp
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const float2 is = *reinterpret_cast<const float2*>(inv_scale_s + 8 * nt + 2 * t);
            acc[nt][0] *= is.x;
            acc[nt][1] *= is.y;
            acc[nt][2] *= is.x;
            acc[nt][3] *= is.y;
            m0 = fmaxf(m0, fmaxf(fabsf(acc[nt][0]), fabsf(acc[nt][1])));
            m1 = fmaxf(m1, fmaxf(fabsf(acc[nt][2]), fabsf(acc[nt][3])));
        }
        m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 1));
        m0 = fmaxf(m0, __shfl_xor_sync(FULL, m0, 2));
        m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 1));
        m1 = fmaxf(m1, __shfl_xor_sync(FULL, m1, 2));
        const float st0 = m0 / LEVELS, st1 = m1 / LEVELS;
        if (t == 0) {
            a.channel_step[head * V + v0] = st0;
            a.channel_step[head * V + v1] = st1;
        }
        const float q0 = st0 > 0.f ? 1.f / st0 : 0.f;
        const float q1 = st1 > 0.f ? 1.f / st1 : 0.f;
        // Round to the nearest level, ties to even: x * q + 1.5 * 2^23 in fp32 rounds x * q to an integer,
        // which sits in the low bits of the sum. |x * q| <= LEVELS by the choice of the step, so no clamp;
        // the low byte is the two's-complement int8 level.
        const float MAGIC = 12582912.0f;
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            const int col = 8 * nt + 2 * t;
            const uint32_t e0 = __float_as_uint(fmaf(acc[nt][0], q0, MAGIC));
            const uint32_t e1 = __float_as_uint(fmaf(acc[nt][1], q0, MAGIC));
            const uint32_t e2 = __float_as_uint(fmaf(acc[nt][2], q1, MAGIC));
            const uint32_t e3 = __float_as_uint(fmaf(acc[nt][3], q1, MAGIC));
            *reinterpret_cast<uint16_t*>(zst + v0 * ZROW + col) = static_cast<uint16_t>(__byte_perm(e0, e1, 0x0040));
            *reinterpret_cast<uint16_t*>(zst + v1 * ZROW + col) = static_cast<uint16_t>(__byte_perm(e2, e3, 0x0040));
        }
        __syncthreads();
        {
            int8_t* zg = a.residual + head * (V * K);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int q = tid + THREADS * i;
                *reinterpret_cast<int4*>(zg + (q >> 3) * K + ((q & 7) << 4)) =
                    *reinterpret_cast<const int4*>(zst + (q >> 3) * ZROW + ((q & 7) << 4));
            }
        }
    }
    if constexpr (PREFETCH) cp_async_wait_all();   // nothing in flight when the block ends
}

template <int RP, bool STORE>
constexpr int shared_bytes() { return STORE ? Layout<RP>::store_bytes : Layout<RP>::bytes; }

template <int RP, bool STORE>
int configure() {
    return static_cast<int>(cudaFuncSetAttribute(fast_fold_kernel<RP, STORE>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                 shared_bytes<RP, STORE>()));
}

template <int RP, bool STORE>
int blocks_per_sm() {
    int blocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fast_fold_kernel<RP, STORE>, THREADS,
                                                      shared_bytes<RP, STORE>()) != cudaSuccess)
        return 0;
    return blocks;
}

template <bool STORE>
int launch(const Args* args, int grid, void* stream) {
    if (args->key_channels != K || args->value_channels != V || args->compensators < 1 || args->compensators > 8 ||
        grid < 1 || (!STORE && (args->window_length < 1 || args->window_length > L)) ||
        (STORE && (args->source == nullptr || args->sweeps < 1)))
        return -1;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (args->compensators <= 4)
        fast_fold_kernel<4, STORE><<<grid, THREADS, shared_bytes<4, STORE>(), s>>>(*args);
    else
        fast_fold_kernel<8, STORE><<<grid, THREADS, shared_bytes<8, STORE>(), s>>>(*args);
    return static_cast<int>(cudaGetLastError());
}

}  // namespace lqff

extern "C" {

// Call once, outside CUDA-graph capture: allows the shared memory the kernels need.
int lq_fast_fold_init(void) {
    int e = lqff::configure<4, false>();
    if (!e) e = lqff::configure<8, false>();
    if (!e) e = lqff::configure<4, true>();
    if (!e) e = lqff::configure<8, true>();
    return e;
}

int lq_fast_fold_blocks_per_sm(int compensators) {
    return compensators <= 4 ? lqff::blocks_per_sm<4, false>() : lqff::blocks_per_sm<8, false>();
}

int lq_fast_store_blocks_per_sm(int compensators) {
    return compensators <= 4 ? lqff::blocks_per_sm<4, true>() : lqff::blocks_per_sm<8, true>();
}

int lq_fast_fold_shared_bytes(int compensators) {
    return compensators <= 4 ? lqff::shared_bytes<4, false>() : lqff::shared_bytes<8, false>();
}

int lq_fast_fold_args_size(void) { return static_cast<int>(sizeof(lqff::Args)); }

// Folds every row of the work list (order, count) on `stream`. Returns 0 or a CUDA error code;
// -1 if the pool is outside the kernel's limits.
int lq_fast_fold(const lqff::Args* args, int grid, void* stream) { return lqff::launch<false>(args, grid, stream); }

// Stores (adopts) the full-precision state of every row of the work list into the pool: args->source,
// source_stride, source_index / has_index; args->sweeps = the sweeps of the fit (the Triton store's SWEEPS + 2).
int lq_fast_store(const lqff::Args* args, int grid, void* stream) { return lqff::launch<true>(args, grid, stream); }

}  // extern "C"
