// =============================================================================
// leapquant_kernel_blocks.h
//
// The arithmetic of the LeapQuant kernel, written once.
//
// Every function here is the work of ONE thread block for ONE attention head of
// ONE request. The CUDA build runs it with 128 GPU threads per block. The CPU
// build runs it with one thread, or with several real threads for race checks.
//
// How the code is made to run on a thread block
//   A block has `thread.count` threads, numbered thread.index = 0 .. count-1.
//   All of them execute the same function. Three building blocks are used:
//
//     LQ_FOR(i, n) { ... }   a parallel loop. Thread t runs i = t, t + count,
//                            t + 2 * count, ... so the threads share 0 .. n-1.
//     LQ_MASTER { ... }      only thread 0 runs the body.
//     lq_sync(thread);       a barrier: every thread waits here for the others.
//                            (__syncthreads() on a GPU.)
//
//   With one thread, LQ_FOR is a plain loop, LQ_MASTER always runs and lq_sync
//   does nothing, so the function reads as ordinary sequential code.
//
// The three rules that keep it correct on a GPU
//   1. A value computed by one thread and needed by another goes through the
//      block's shared struct (LqDecodeShared / LqFoldShared) or through global
//      memory, and there is an lq_sync between the write and the read.
//   2. A local variable never carries a result across an lq_sync, unless every
//      thread computes the same value from the same inputs.
//   3. Every thread passes the same lq_sync calls: branches around an lq_sync
//      depend only on values that are the same for the whole block.
//
// Every individual sum is computed by ONE thread, front to back. So the result
// does not depend on the number of threads: the CPU build, the emulated block
// and the GPU add the same numbers in the same order. (A GPU can still differ
// in the last bit, because its exp() and its fused multiply-add are not the
// CPU's.)
//
// The steps and their names follow ../leapquant.cpp. Read that file first.
// =============================================================================

#ifndef LEAPQUANT_KERNEL_BLOCKS_H
#define LEAPQUANT_KERNEL_BLOCKS_H

#include "leapquant_kernel.h"
#include "../leapquant_small_matrix.h"

#include <math.h>
#include <stddef.h>

#include <string.h>
#if defined(__CUDACC__)
#include <cuda_fp16.h>
#define LQ_DEVICE __device__ __forceinline__
#else
#define LQ_DEVICE inline
#endif

// One thread of a block.
struct LqThread {
    int index;       // 0 .. count-1
    int count;       // threads in the block
    void* barrier;   // CPU emulation only: the block's barrier. Null when count is 1.
};

#if !defined(__CUDACC__)
// Defined in leapquant_kernel_cpu.cpp.
void lq_host_barrier_wait(void* barrier);
#endif

LQ_DEVICE void lq_sync(const LqThread& thread) {
#if defined(__CUDACC__)
    (void)thread;
    __syncthreads();
#else
    if (thread.barrier != nullptr) {
        lq_host_barrier_wait(thread.barrier);
    }
#endif
}

#define LQ_FOR(variable, count_) \
    for (int variable = thread.index; variable < (count_); variable += thread.count)
#define LQ_MASTER if (thread.index == 0)

// Profiling stops (LqPool::profile_stop_phase). Every thread of a block returns
// at the named point, right after a barrier, so the work before it is complete.
#define LQ_PHASE_DECODE_DECAYS 11       // decode: after section 1
#define LQ_PHASE_DECODE_FADE 12         // decode: after section 2
#define LQ_PHASE_DECODE_MATCHES 13      // decode: after section 3 (section 4 is the rest)
#define LQ_PHASE_FOLD_REBUILD 1         // fold: after the matrix is rebuilt
#define LQ_PHASE_FOLD_START 2           // store: after the starting guess of the first compensator
#define LQ_PHASE_FOLD_SWEEPS 3          // store: after the sweeps of the first compensator
#define LQ_PHASE_FOLD_FIRST 4           // store: after the first compensator is stored and subtracted
#define LQ_PHASE_FOLD_COMPENSATORS 5    // store: after all compensators
#define LQ_PHASE_FOLD_SMOOTHING 6       // store: after the row scales (rounding is the rest)
// Inside the first sweep of the first compensator:
#define LQ_PHASE_SWEEP_KEY_SIDE 21      // after key_side = matrix * value_side
#define LQ_PHASE_SWEEP_KEY_LENGTH 22    // after the length of key_side (one thread)
#define LQ_PHASE_SWEEP_NORMALISE 23     // after key_side is normalised
#define LQ_PHASE_SWEEP_VALUE_SIDE 24    // after value_side = matrix^T * key_side
#define LQ_PHASE_SWEEP_STRENGTH 25      // after the length of value_side (one thread)
#define LQ_STOP_AT(phase) (pool.profile_stop_phase == (phase))
#define LQ_STOP_IN_FIRST_SWEEP(phase) (compensator == 0 && sweep == 0 && pool.profile_stop_phase == (phase))

// Sums of products are added up in four chunks, exactly as dot_product() in
// ../leapquant.cpp defines it: chunk c covers the indices [c*n/4, (c+1)*n/4),
// each chunk left to right, then the four totals left to right.
#define LQ_SUM_CHUNKS 4
LQ_DEVICE int lq_chunk_begin(int chunk, int count) {
    return (chunk * count) / LQ_SUM_CHUNKS;
}
LQ_DEVICE float lq_combine_chunks(const float* chunk_total) {
    return ((chunk_total[0] + chunk_total[1]) + chunk_total[2]) + chunk_total[3];
}

// Smallest allowed smoothing scale; the same constant as in ../leapquant.cpp.
#define LQ_SMALLEST_KEY_ROW_SCALE 1e-6f

// -----------------------------------------------------------------------------
// fp16 conversion.
//
// A GPU converts in hardware. The CPU build repeats the bit arithmetic of
// ../leapquant.cpp; kernel_test.cpp checks that the two agree on every value.
// As in the reference, a value too large for fp16 becomes +-65504, not infinity.
// -----------------------------------------------------------------------------
LQ_DEVICE float lq_half_to_float(uint16_t half_bits) {
#if defined(__CUDACC__)
    return __half2float(__ushort_as_half(half_bits));
#else
    const uint32_t sign_bit_of_float = static_cast<uint32_t>(half_bits & 0x8000u) << 16;
    const uint32_t half_exponent = (half_bits >> 10) & 0x1Fu;
    const uint32_t half_fraction = half_bits & 0x3FFu;
    uint32_t float_bits = 0;
    if (half_exponent == 0u) {
        const float magnitude = static_cast<float>(half_fraction) * (1.0f / 16777216.0f);
        memcpy(&float_bits, &magnitude, sizeof(float_bits));
        float_bits = float_bits | sign_bit_of_float;
    } else if (half_exponent == 31u) {
        float_bits = sign_bit_of_float | 0x7F800000u | (half_fraction << 13);
    } else {
        float_bits = sign_bit_of_float | ((half_exponent + 112u) << 23) | (half_fraction << 13);
    }
    float result = 0.0f;
    memcpy(&result, &float_bits, sizeof(result));
    return result;
#endif
}

LQ_DEVICE uint16_t lq_float_to_half(float value) {
#if defined(__CUDACC__)
    // 65520 is the first value that would round up to infinity in fp16.
    // (A not-a-number fails the comparison and stays a not-a-number.)
    if (fabsf(value) >= 65520.0f) {
        value = copysignf(65504.0f, value);
    }
    return __half_as_ushort(__float2half(value));
#else
    uint32_t float_bits = 0;
    memcpy(&float_bits, &value, sizeof(float_bits));
    const uint32_t sign_bit_of_half = (float_bits >> 16) & 0x8000u;
    const uint32_t magnitude_bits = float_bits & 0x7FFFFFFFu;
    if (magnitude_bits > 0x7F800000u) {
        return static_cast<uint16_t>(sign_bit_of_half | 0x7E00u);      // not-a-number
    }
    if (magnitude_bits >= 0x477FF000u) {
        return static_cast<uint16_t>(sign_bit_of_half | 0x7BFFu);      // too large: 65504
    }
    if (magnitude_bits < 0x38800000u) {                                 // below 2^-14
        float magnitude = 0.0f;
        memcpy(&magnitude, &magnitude_bits, sizeof(magnitude));
        const float number_of_steps = nearbyintf(magnitude * 16777216.0f);
        return static_cast<uint16_t>(sign_bit_of_half | static_cast<uint32_t>(number_of_steps));
    }
    uint32_t half_bits = (((magnitude_bits >> 23) - 112u) << 10) | ((magnitude_bits >> 13) & 0x3FFu);
    const uint32_t dropped_bits = magnitude_bits & 0x1FFFu;
    if (dropped_bits > 0x1000u || (dropped_bits == 0x1000u && (half_bits & 1u) == 1u)) {
        half_bits = half_bits + 1u;
    }
    return static_cast<uint16_t>(sign_bit_of_half | half_bits);
#endif
}

// The bits of a float as an integer, and back.
LQ_DEVICE float lq_bits_to_float(uint32_t bits) {
#if defined(__CUDACC__)
    return __uint_as_float(bits);
#else
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}
LQ_DEVICE uint32_t lq_float_to_bits(float value) {
#if defined(__CUDACC__)
    return __float_as_uint(value);
#else
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}

// One element of a raw input tensor (LqStep::raw_*), as a float.
LQ_DEVICE float lq_load_raw(const void* data, size_t index, int dtype) {
    if (dtype == LQ_RAW_FP32) {
        return static_cast<const float*>(data)[index];
    }
    const uint16_t bits = static_cast<const uint16_t*>(data)[index];
    if (dtype == LQ_RAW_FP16) {
        return lq_half_to_float(bits);
    }
    // bfloat16 is the top 16 bits of a float.
    return lq_bits_to_float(static_cast<uint32_t>(bits) << 16);
}

// Rounds a float to the nearest value of the raw dtype (ties to even) and back.
LQ_DEVICE float lq_round_to_raw_dtype(float value, int dtype) {
    if (dtype == LQ_RAW_FP32) {
        return value;
    }
    if (dtype == LQ_RAW_FP16) {
        return lq_half_to_float(lq_float_to_half(value));
    }
    uint32_t bits = lq_float_to_bits(value);
    bits = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    return lq_bits_to_float(bits);
}

// How many key rows share one value-column step, and how many such groups a column has.
// 0, or a group at least as long as the column, is one step for the whole column.
LQ_DEVICE int lq_scale_group_length(const LqPool& pool) {
    if (pool.scale_group <= 0 || pool.scale_group >= pool.key_channels) {
        return pool.key_channels;
    }
    return pool.scale_group;
}

LQ_DEVICE int lq_scale_group_count(const LqPool& pool) {
    const int length = lq_scale_group_length(pool);
    return (pool.key_channels + length - 1) / length;
}

// -----------------------------------------------------------------------------
// Where one head's data starts inside the pool.
// -----------------------------------------------------------------------------
struct LqHeadPointers {
    int8_t* rounded_residual;             // [V][K]
    float* value_channel_step;            // [V][G], G = 1 when the column has one step
    float* key_row_scale;                 // [K]
    uint16_t* compensator_keys;           // [R][K]
    uint16_t* compensator_corrections;    // [R][V]
    uint16_t* window_keys;                // [L][K]
    uint16_t* window_corrections;         // [L][V]
    uint16_t* window_log_decays;          // [L]
};

LQ_DEVICE LqHeadPointers lq_locate_head(const LqPool& pool, int slot, int head) {
    const size_t head_index = static_cast<size_t>(slot) * pool.value_heads + static_cast<size_t>(head);
    const size_t key_channels = static_cast<size_t>(pool.key_channels);
    const size_t value_channels = static_cast<size_t>(pool.value_channels);
    const size_t compensator_count = static_cast<size_t>(pool.compensator_count);
    const size_t window_length = static_cast<size_t>(pool.window_length);

    LqHeadPointers pointers;
    const size_t scale_groups = static_cast<size_t>(lq_scale_group_count(pool));
    pointers.rounded_residual = pool.rounded_residual + head_index * value_channels * key_channels;
    pointers.value_channel_step = pool.value_channel_step + head_index * value_channels * scale_groups;
    pointers.key_row_scale = pool.key_row_scale + head_index * key_channels;
    pointers.compensator_keys = pool.compensator_keys + head_index * compensator_count * key_channels;
    pointers.compensator_corrections = pool.compensator_corrections + head_index * compensator_count * value_channels;
    pointers.window_keys = pool.window_keys + head_index * window_length * key_channels;
    pointers.window_corrections = pool.window_corrections + head_index * window_length * value_channels;
    pointers.window_log_decays = pool.window_log_decays + head_index * window_length;
    return pointers;
}

// =============================================================================
// Block 1. The decode step (Head::process_one_token for Gated DeltaNet).
// =============================================================================
//
// What the reference does in two walks over the int8 matrix, this block does
// in one. Both reads are of the state BEFORE this token:
//
//     held   = state_before^T * (decay * key)
//     output = correction * (stored_key . query) + state_before^T * (decay_as_stored * query)
//
// The second line is the reference's "read the state after the token" with the
// new record split off. It uses the record AS STORED in fp16, like the reference.
//
// Every sum below adds the same numbers in the same order as Head::read_state.
// Built without fused multiply-add, the CPU build therefore reproduces the
// reference core bit for bit.

// Bytes between two rows of the staged int8 matrix.
//
// Shared memory is 32 banks of 4 bytes. A warp that reads one byte from each
// of 32 consecutive rows hits 32 different banks when this stride, divided by
// 4, is 1 modulo 32. The largest stride for a key width up to LQ_MAX_CHANNELS
// is 132, which is what the staging buffer below is sized for.
LQ_DEVICE int lq_staged_int8_stride(int key_channels) {
    int words = (key_channels + 3) >> 2;
    const int remainder = words & 31;
    if (remainder != 1) {
        words += (1 - remainder + 32) & 31;
    }
    return words << 2;
}

struct LqDecodeShared {
    // The two input vectors faded by the whole window: what the boundary state sees.  [K]
    float read_input[LQ_MAX_CHANNELS];
    float query_input[LQ_MAX_CHANNELS];
    // Row weights for the int8 matrix: faded input * key_row_scale.  [K]
    float read_weight[LQ_MAX_CHANNELS];
    float query_weight[LQ_MAX_CHANNELS];
    // Decay of every record in the window.  [L]
    // Section 3 rebuilds a record's faded input from these and the raw key,
    // with the same multiplies section 2 used to. Keeping every faded copy
    // (16 x 129 floats, twice) made this block about 16 KiB larger for no
    // change in the numbers.
    float record_decay[LQ_MAX_WINDOW];
    // key_match of every record, every compensator and the new record.  [L + R + 1]
    float read_match[LQ_MAX_WINDOW + LQ_MAX_COMPENSATORS + 1];
    float query_match[LQ_MAX_WINDOW + LQ_MAX_COMPENSATORS + 1];
    // The same, per chunk of the sum, before the chunks are combined.  [L + R + 1][4]
    float read_part[LQ_MAX_WINDOW + LQ_MAX_COMPENSATORS + 1][LQ_SUM_CHUNKS];
    float query_part[LQ_MAX_WINDOW + LQ_MAX_COMPENSATORS + 1][LQ_SUM_CHUNKS];
    // Raw-input mode only: the inputs of the recurrence, computed by the block.
    float raw_key[LQ_MAX_CHANNELS];       // [K]  normalised key
    float raw_query[LQ_MAX_CHANNELS];     // [K]  normalised, scaled query
    float raw_norm_part[2][LQ_SUM_CHUNKS];   // squared lengths of key and query, per chunk
    float raw_decay;
    float raw_beta;
    // One head's int8 residual, copied in once with consecutive threads reading
    // consecutive bytes, then read back per value channel. Rows are padded
    // (lq_staged_int8_stride) so that read is free of shared-memory bank conflicts.
    alignas(16) uint8_t staged_residual[LQ_MAX_CHANNELS * 132];
};

LQ_DEVICE void lq_decode_block(const LqPool& pool,
                               const LqStep& step,
                               int row,
                               int head,
                               const LqThread& thread,
                               LqDecodeShared* shared) {
    const int key_channels = pool.key_channels;
    const int value_channels = pool.value_channels;
    const int value_heads = pool.value_heads;
    const int compensator_count = pool.compensator_count;

    float* output = step.output + (static_cast<size_t>(row) * value_heads + head) * value_channels;

    // A padded row, or a cursor outside the window: write zeros, store nothing.
    const int slot = step.slot[row];
    const int records = step.tokens_in_window[row];
    if (slot < 0 || records < 0 || records >= pool.window_length) {
        LQ_FOR(value_index, value_channels) {
            output[value_index] = 0.0f;
        }
        return;
    }

    const LqHeadPointers stored = lq_locate_head(pool, slot, head);
    const int key_head = head / (value_heads / step.key_heads);
    const bool raw = step.raw_qkv != nullptr;
    const float* key = nullptr;
    const float* query = nullptr;
    const float* value = nullptr;
    float decay = 0.0f;
    float beta = 0.0f;
    // Raw mode: where this row's packed (q | k | v) starts, and this head's value inside it.
    const size_t raw_row = static_cast<size_t>(row) * static_cast<size_t>(step.raw_stride);
    const size_t raw_value_start =
        raw_row + 2u * static_cast<size_t>(step.key_heads) * key_channels + static_cast<size_t>(head) * value_channels;
    if (!raw) {
        key = step.key + (static_cast<size_t>(row) * step.key_heads + key_head) * key_channels;
        query = step.query + (static_cast<size_t>(row) * step.key_heads + key_head) * key_channels;
        value = step.value + (static_cast<size_t>(row) * value_heads + head) * value_channels;
        decay = step.decay[static_cast<size_t>(row) * value_heads + head];
        beta = step.beta[static_cast<size_t>(row) * value_heads + head];
    } else {
        // ---- Section 0 (raw mode) -------------------------------------------------
        // The gates and the normalised key and query of this head.
        const size_t query_start = raw_row + static_cast<size_t>(key_head) * key_channels;
        const size_t key_start = query_start + static_cast<size_t>(step.key_heads) * key_channels;
        LQ_FOR(key_index, key_channels) {
            shared->raw_key[key_index] = lq_load_raw(step.raw_qkv, key_start + key_index, step.raw_dtype);
            shared->raw_query[key_index] = lq_load_raw(step.raw_qkv, query_start + key_index, step.raw_dtype);
        }
        lq_sync(thread);
        LQ_FOR(job, 2 * LQ_SUM_CHUNKS) {
            const float* vector = (job < LQ_SUM_CHUNKS) ? shared->raw_key : shared->raw_query;
            const int chunk = job % LQ_SUM_CHUNKS;
            float sum = 0.0f;
            for (int key_index = lq_chunk_begin(chunk, key_channels); key_index < lq_chunk_begin(chunk + 1, key_channels);
                 ++key_index) {
                sum += vector[key_index] * vector[key_index];
            }
            shared->raw_norm_part[job / LQ_SUM_CHUNKS][chunk] = sum;
        }
        LQ_MASTER {
            const size_t gate_index = static_cast<size_t>(row) * value_heads + head;
            const float x = lq_load_raw(step.raw_a, gate_index, step.raw_dtype) + step.dt_bias[head];
            const float softplus = (x <= 20.0f) ? logf(1.0f + expf(x)) : x;
            shared->raw_decay = expf(-expf(step.A_log[head]) * softplus);
            const float b_value = lq_load_raw(step.raw_b, gate_index, step.raw_dtype);
            shared->raw_beta = lq_round_to_raw_dtype(1.0f / (1.0f + expf(-b_value)), step.raw_dtype);
        }
        lq_sync(thread);
        const float key_factor = 1.0f / sqrtf(lq_combine_chunks(shared->raw_norm_part[0]) + 1e-6f);
        const float query_factor = step.scale / sqrtf(lq_combine_chunks(shared->raw_norm_part[1]) + 1e-6f);
        LQ_FOR(key_index, key_channels) {
            shared->raw_key[key_index] = shared->raw_key[key_index] * key_factor;
            shared->raw_query[key_index] = shared->raw_query[key_index] * query_factor;
        }
        lq_sync(thread);
        key = shared->raw_key;
        query = shared->raw_query;
        decay = shared->raw_decay;
        beta = shared->raw_beta;
    }

    // The decay is stored as its logarithm in fp16 (see ../leapquant.h).
    // The reference reads the new state with the decay AS STORED, so we do too.
    const uint16_t stored_log_decay = lq_float_to_half(logf(decay));
    const float decay_as_stored = expf(lq_half_to_float(stored_log_decay));

    uint16_t* new_record_key = stored.window_keys + static_cast<size_t>(records) * key_channels;
    uint16_t* new_record_correction = stored.window_corrections + static_cast<size_t>(records) * value_channels;

    // ---- Section 1 ------------------------------------------------------------
    // The decay of every record in the window, and the decay of the new record.
    LQ_FOR(record, records) {
        shared->record_decay[record] = expf(lq_half_to_float(stored.window_log_decays[record]));
    }
    LQ_MASTER {
        stored.window_log_decays[records] = stored_log_decay;
    }
    lq_sync(thread);
    if (LQ_STOP_AT(LQ_PHASE_DECODE_DECAYS)) {
        return;
    }

    // ---- Section 2 ------------------------------------------------------------
    // One thread per key channel. Append the key of the new record, then fade
    // the two input vectors while walking from the newest record to the oldest:
    //     write_key   = beta  * key     (stored)
    //     read_vector = decay * key
    // The per-record faded copies are not stored. Section 3 rebuilds the one
    // it needs from `key`, `query` and `record_decay`, with these same multiplies.
    LQ_FOR(key_index, key_channels) {
        new_record_key[key_index] = lq_float_to_half(beta * key[key_index]);

        float read_value = decay * key[key_index];
        float query_value = query[key_index] * decay_as_stored;
        for (int record = records - 1; record >= 0; --record) {
            // Everything older than this record was faded by this record's decay.
            read_value = read_value * shared->record_decay[record];
            query_value = query_value * shared->record_decay[record];
        }
        // Here both carry the decay of the whole window.
        shared->read_input[key_index] = read_value;
        shared->query_input[key_index] = query_value;
        shared->read_weight[key_index] = read_value * stored.key_row_scale[key_index];
        shared->query_weight[key_index] = query_value * stored.key_row_scale[key_index];
    }
    lq_sync(thread);
    if (LQ_STOP_AT(LQ_PHASE_DECODE_FADE)) {
        return;
    }

    // ---- Section 3 ------------------------------------------------------------
    // key_match of every record and every compensator with both input vectors,
    // and of the new record with the query. Each sum is four chunks (see
    // lq_chunk_begin); one thread computes one chunk of one item.
    const int item_count = records + compensator_count + 1;
    LQ_FOR(job, item_count * LQ_SUM_CHUNKS) {
        const int item = job / LQ_SUM_CHUNKS;
        const int chunk = job % LQ_SUM_CHUNKS;
        const int begin = lq_chunk_begin(chunk, key_channels);
        const int end = lq_chunk_begin(chunk + 1, key_channels);
        float read_sum = 0.0f;
        float query_sum = 0.0f;
        if (item < records) {
            const uint16_t* item_key = stored.window_keys + static_cast<size_t>(item) * key_channels;
            for (int key_index = begin; key_index < end; ++key_index) {
                const float key_value = lq_half_to_float(item_key[key_index]);
                // The input as this record sees it: faded by every newer record,
                // not yet by its own. Same multiplies as section 2, so the same bits.
                float read_value = decay * key[key_index];
                float query_value = query[key_index] * decay_as_stored;
                for (int newer = records - 1; newer > item; --newer) {
                    read_value = read_value * shared->record_decay[newer];
                    query_value = query_value * shared->record_decay[newer];
                }
                read_sum += key_value * read_value;
                query_sum += key_value * query_value;
            }
        } else if (item < records + compensator_count) {
            const uint16_t* item_key = stored.compensator_keys + static_cast<size_t>(item - records) * key_channels;
            for (int key_index = begin; key_index < end; ++key_index) {
                const float key_value = lq_half_to_float(item_key[key_index]);
                read_sum += key_value * shared->read_input[key_index];
                query_sum += key_value * shared->query_input[key_index];
            }
        } else {
            // The new record (its key was stored in section 2). It is not faded:
            // it is read with the query itself.
            for (int key_index = begin; key_index < end; ++key_index) {
                query_sum += lq_half_to_float(new_record_key[key_index]) * query[key_index];
            }
        }
        shared->read_part[item][chunk] = read_sum;
        shared->query_part[item][chunk] = query_sum;
    }
    lq_sync(thread);
    LQ_FOR(item, item_count) {
        shared->read_match[item] = lq_combine_chunks(shared->read_part[item]);
        shared->query_match[item] = lq_combine_chunks(shared->query_part[item]);
    }
    lq_sync(thread);
    if (LQ_STOP_AT(LQ_PHASE_DECODE_MATCHES)) {
        return;
    }

    // ---- Section 4 ------------------------------------------------------------
    // One thread per value channel: add up records, compensators and the int8
    // matrix; form the correction; store it; write the output.
    //
    // The int8 matrix is [value][key], so a thread's own row is contiguous, but
    // a warp of threads reading "the same key of the next row" used to touch
    // addresses K bytes apart. The copy below gives consecutive threads
    // consecutive source bytes. The sum itself is unchanged: one thread, left
    // to right, the order read_state uses.
    {
        const int8_t* residual = stored.rounded_residual;
        const int stride = lq_staged_int8_stride(key_channels);
        const int total = value_channels * key_channels;
        if ((reinterpret_cast<uintptr_t>(residual) & 15u) == 0u && (key_channels & 15) == 0) {
            // Every row is a whole number of 16-byte vectors, so a vector never straddles two
            // rows: one division per 16 bytes (not per byte), one 16-byte load from memory and
            // four 4-byte stores into the staged row. The row stride is a multiple of 4, so the
            // stores are aligned, and with 33 words per row the lanes of a warp hit 32 banks.
            const int vectors_per_row = key_channels >> 4;
            const int vectors = value_channels * vectors_per_row;
            LQ_FOR(vector, vectors) {
                const int value_row = vector / vectors_per_row;
                const int key_index = (vector - value_row * vectors_per_row) << 4;
                uint32_t words[4];
                memcpy(words, __builtin_assume_aligned(residual + (static_cast<size_t>(vector) << 4), 16), sizeof(words));
                memcpy(__builtin_assume_aligned(shared->staged_residual + static_cast<size_t>(value_row) * stride + key_index, 4),
                       words, sizeof(words));
            }
        } else {
            LQ_FOR(linear, total) {
                const int value_row = linear / key_channels;
                const int key_index = linear - value_row * key_channels;
                shared->staged_residual[static_cast<size_t>(value_row) * stride + key_index] =
                    static_cast<uint8_t>(residual[linear]);
            }
        }
    }
    lq_sync(thread);
    const int residual_stride = lq_staged_int8_stride(key_channels);
    const int scale_group_count = lq_scale_group_count(pool);
    LQ_FOR(value_index, value_channels) {
        // The value side of every record and compensator, for this value channel.
        float item_correction[LQ_MAX_WINDOW + LQ_MAX_COMPENSATORS];
        for (int record = 0; record < records; ++record) {
            item_correction[record] =
                lq_half_to_float(stored.window_corrections[static_cast<size_t>(record) * value_channels + value_index]);
        }
        for (int compensator = 0; compensator < compensator_count; ++compensator) {
            item_correction[records + compensator] = lq_half_to_float(
                stored.compensator_corrections[static_cast<size_t>(compensator) * value_channels + value_index]);
        }

        // The int8 matrix, for both input vectors in one walk.
        // One group (every 8-bit state): levels come from the staged row, 4 bytes
        // at a time, added left to right, then multiplied by the one column step.
        // Several groups: each band of key rows is summed left to right and
        // multiplied by that band's step. A word load stays on the one-group path,
        // so a word never crosses two steps.
        const uint8_t* residual_of_channel =
            shared->staged_residual + static_cast<size_t>(value_index) * residual_stride;
        // One step per column: finish the key sum, then one multiply. The product
        // is added after the records and compensators, which is the 8-bit order.
        float read_from_residual = 0.0f;
        float query_from_residual = 0.0f;
        if (scale_group_count == 1) {
            float read_column_sum = 0.0f;
            float query_column_sum = 0.0f;
            int key_index = 0;
            for (; key_index + 4 <= key_channels; key_index += 4) {
                uint32_t word = 0;
                // Rows start on a multiple of 4 bytes (stride 4 * 33) and key_index steps by 4: one 4-byte load.
                memcpy(&word, __builtin_assume_aligned(residual_of_channel + key_index, 4), sizeof(word));
                for (int byte = 0; byte < 4; ++byte) {
                    const float level = static_cast<float>(static_cast<int8_t>(word & 0xFFu));
                    word = word >> 8;
                    read_column_sum += shared->read_weight[key_index + byte] * level;
                    query_column_sum += shared->query_weight[key_index + byte] * level;
                }
            }
            for (; key_index < key_channels; ++key_index) {
                const float level = static_cast<float>(static_cast<int8_t>(residual_of_channel[key_index]));
                read_column_sum += shared->read_weight[key_index] * level;
                query_column_sum += shared->query_weight[key_index] * level;
            }
            const float channel_step = stored.value_channel_step[value_index];
            read_from_residual = read_column_sum * channel_step;
            query_from_residual = query_column_sum * channel_step;
        }

        // held = state_before^T * read_vector: (a) records, newest first,
        // (b) compensators, (c) the int8 matrix.
        // Several groups add each group's product into this running total, in
        // group order. Summing the products first, then adding once, is a
        // different rounding.
        float held = 0.0f;
        for (int record = records - 1; record >= 0; --record) {
            held += shared->read_match[record] * item_correction[record];
        }
        for (int compensator = 0; compensator < compensator_count; ++compensator) {
            held += shared->read_match[records + compensator] * item_correction[records + compensator];
        }
        if (scale_group_count == 1) {
            held += read_from_residual;
        } else {
            const int scale_group_length = lq_scale_group_length(pool);
            for (int group = 0; group < scale_group_count; ++group) {
                float read_column_sum = 0.0f;
                const int begin = group * scale_group_length;
                int end = begin + scale_group_length;
                if (end > key_channels) {
                    end = key_channels;
                }
                for (int key_index = begin; key_index < end; ++key_index) {
                    const float level = static_cast<float>(static_cast<int8_t>(residual_of_channel[key_index]));
                    read_column_sum += shared->read_weight[key_index] * level;
                }
                const float channel_step =
                    stored.value_channel_step[static_cast<size_t>(value_index) * static_cast<size_t>(scale_group_count) +
                                              static_cast<size_t>(group)];
                held += read_column_sum * channel_step;
            }
        }

        // correction = value - held (equation 3), stored in fp16.
        const float value_entry =
            raw ? lq_load_raw(step.raw_qkv, raw_value_start + value_index, step.raw_dtype) : value[value_index];
        const uint16_t stored_correction = lq_float_to_half(value_entry - held);
        new_record_correction[value_index] = stored_correction;

        // output = state_after^T * query: the new record first, then the same walk.
        float answer = 0.0f;
        answer += shared->query_match[records + compensator_count] * lq_half_to_float(stored_correction);
        for (int record = records - 1; record >= 0; --record) {
            answer += shared->query_match[record] * item_correction[record];
        }
        for (int compensator = 0; compensator < compensator_count; ++compensator) {
            answer += shared->query_match[records + compensator] * item_correction[records + compensator];
        }
        if (scale_group_count == 1) {
            answer += query_from_residual;
        } else {
            const int scale_group_length = lq_scale_group_length(pool);
            for (int group = 0; group < scale_group_count; ++group) {
                float query_column_sum = 0.0f;
                const int begin = group * scale_group_length;
                int end = begin + scale_group_length;
                if (end > key_channels) {
                    end = key_channels;
                }
                for (int key_index = begin; key_index < end; ++key_index) {
                    const float level = static_cast<float>(static_cast<int8_t>(residual_of_channel[key_index]));
                    query_column_sum += shared->query_weight[key_index] * level;
                }
                const float channel_step =
                    stored.value_channel_step[static_cast<size_t>(value_index) * static_cast<size_t>(scale_group_count) +
                                              static_cast<size_t>(group)];
                answer += query_column_sum * channel_step;
            }
        }
        output[value_index] = answer;
    }
}

// =============================================================================
// Block 2. Rebuild, store, fold, export.
// =============================================================================

// How far apart two value channels (rows) of the working matrix are, in floats.
//
// In the caller's scratch memory a row is K floats. In a block's own matrix it is
// K + 1: one float of padding per row. The reason is a property of GPU shared
// memory, which is split into 32 "banks" by address; threads of a warp that touch
// the same bank in the same instruction are served one after the other. Several
// loops here give each thread one row and walk all rows in step. With rows K = 128
// floats apart, the 32 threads of a warp would always hit the same bank. Measured
// on an H100, batch 512: the matrix-transpose product took 1.225 ms that way, the
// same amount of arithmetic in the other direction 0.085 ms. One float of padding
// moves each row one bank along.
LQ_DEVICE size_t lq_matrix_stride(const LqPool& pool, const float* block_matrix) {
    return static_cast<size_t>(pool.key_channels) + (block_matrix != nullptr ? 1u : 0u);
}

// The joint fit works on compensator_count + 1 vectors (one rescue vector).
#define LQ_FIT_VECTORS (LQ_MAX_COMPENSATORS + 1)
#define LQ_FIT_PAIRS ((LQ_FIT_VECTORS * (LQ_FIT_VECTORS + 1)) / 2)

struct LqFoldShared {
    float key_side[LQ_MAX_CHANNELS];      // [K]  also reused for row lengths and row scales
    float value_side[LQ_MAX_CHANNELS];    // [V]
    float record_decay[LQ_MAX_WINDOW];    // [L]
    // For the rebuild: the key side of every compensator and window record, and the
    // row scales, fetched once per block. (Without this, every entry of the matrix
    // fetched them again from GPU memory: 16,384 x 21 fetches per head.)
    // Kept as fp16 on purpose: as floats they push a block's shared memory from 71 to 77 KiB,
    // and then only 2 blocks instead of 3 fit on a GPU core at once (measured: fold 11.9 -> 13.4 ms).
    // After the rebuild the same memory holds the vectors of the joint compensator fit
    // (a union, so that the block does not grow: its size decides how many blocks fit on a GPU core).
    union {
        uint16_t item_keys[LQ_MAX_COMPENSATORS + LQ_MAX_WINDOW][LQ_MAX_CHANNELS];   // fp16
        struct {
            float fit_u[LQ_FIT_VECTORS][LQ_MAX_CHANNELS];   // key sides    [W][K]
            float fit_v[LQ_FIT_VECTORS][LQ_MAX_CHANNELS];   // value sides  [W][V]
        };
    };
    float fit_part[LQ_FIT_PAIRS][LQ_SUM_CHUNKS];   // chunks of the dot products between the fit vectors
    float fit_gram[LQ_SMALL * LQ_SMALL];
    float fit_factor[LQ_SMALL * LQ_SMALL];         // Cholesky factor, then the unmixing matrix
    float fit_strength[LQ_SMALL];
    int fit_alive[LQ_SMALL];
    int fit_has_guess[LQ_SMALL];
    int fit_chosen_row[LQ_SMALL];
    int fit_some_guess_is_missing;
    float row_scale[LQ_MAX_CHANNELS];
    float length_part[LQ_SUM_CHUNKS];     // the chunks of a vector's squared length
    float length;                         // a vector length, or its reciprocal
    float strength;                       // the singular value found by the last sweep
    int longest_row;
    int no_pattern;                       // 1 when the matrix has no pattern (it is all zeros)
};

// One entry of the rebuilt matrix. The record replay is oldest first:
//     entry = decay * entry + key * correction
// and that order is the whole of the bit-identical contract for this entry.
// Which thread computes the entry does not matter.
LQ_DEVICE float lq_rebuild_entry(const LqFoldShared* shared,
                                 const float* compensator_correction,
                                 const float* record_correction,
                                 int compensator_count,
                                 int records,
                                 int key_index,
                                 float channel_step,
                                 float level) {
    float entry = shared->row_scale[key_index] * level * channel_step;
    for (int compensator = 0; compensator < compensator_count; ++compensator) {
        const float key_value = lq_half_to_float(shared->item_keys[compensator][key_index]);
        entry += key_value * compensator_correction[compensator];
    }
    for (int record = 0; record < records; ++record) {
        const float key_value = lq_half_to_float(shared->item_keys[compensator_count + record][key_index]);
        entry = entry * shared->record_decay[record];
        entry += key_value * record_correction[record];
    }
    return entry;
}

// Head::copy_state_to_full_precision. Builds the full matrix of one head into
// `matrix`, stored [value][key]. `records` window records are replayed.
LQ_DEVICE void lq_rebuild_block(const LqPool& pool,
                                const LqHeadPointers& stored,
                                int records,
                                float* matrix,
                                size_t matrix_stride,
                                const LqThread& thread,
                                LqFoldShared* shared) {
    const int key_channels = pool.key_channels;
    const int value_channels = pool.value_channels;
    const int compensator_count = pool.compensator_count;
    const int scale_groups = lq_scale_group_count(pool);
    const int scale_length = lq_scale_group_length(pool);

    LQ_FOR(record, records) {
        shared->record_decay[record] = expf(lq_half_to_float(stored.window_log_decays[record]));
    }
    LQ_FOR(key_index, key_channels) {
        for (int compensator = 0; compensator < compensator_count; ++compensator) {
            shared->item_keys[compensator][key_index] =
                stored.compensator_keys[static_cast<size_t>(compensator) * key_channels + key_index];
        }
        for (int record = 0; record < records; ++record) {
            shared->item_keys[compensator_count + record][key_index] =
                stored.window_keys[static_cast<size_t>(record) * key_channels + key_index];
        }
        shared->row_scale[key_index] = stored.key_row_scale[key_index];
    }
    lq_sync(thread);

    // Each entry is independent. With a full warp of threads, one warp owns a
    // value channel and its lanes take consecutive keys, so a GPU warp's load
    // of the int8 row is one contiguous run of bytes. Fewer than 32 threads
    // (the one-thread CPU path, and the 5-thread race check) keep one thread
    // on a whole row. The entry math is lq_rebuild_entry either way.
    if (thread.count >= 32 && (thread.count % 32) == 0) {
        const int lane = thread.index & 31;
        const int group = thread.index >> 5;
        const int group_count = thread.count >> 5;
        for (int value_index = group; value_index < value_channels; value_index += group_count) {
            float compensator_correction[LQ_MAX_COMPENSATORS];
            float record_correction[LQ_MAX_WINDOW];
            for (int compensator = 0; compensator < compensator_count; ++compensator) {
                compensator_correction[compensator] = lq_half_to_float(
                    stored.compensator_corrections[static_cast<size_t>(compensator) * value_channels + value_index]);
            }
            for (int record = 0; record < records; ++record) {
                record_correction[record] = lq_half_to_float(
                    stored.window_corrections[static_cast<size_t>(record) * value_channels + value_index]);
            }
            const int8_t* residual_of_channel =
                stored.rounded_residual + static_cast<size_t>(value_index) * key_channels;
            float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
            if (scale_groups == 1) {
                const float channel_step = stored.value_channel_step[value_index];
                for (int key_index = lane; key_index < key_channels; key_index += 32) {
                    const float level = static_cast<float>(residual_of_channel[key_index]);
                    matrix_of_channel[key_index] =
                        lq_rebuild_entry(shared, compensator_correction, record_correction, compensator_count, records,
                                         key_index, channel_step, level);
                }
            } else {
                for (int key_index = lane; key_index < key_channels; key_index += 32) {
                    const int band = key_index / scale_length;
                    const float channel_step =
                        stored.value_channel_step[static_cast<size_t>(value_index) * static_cast<size_t>(scale_groups) +
                                                  static_cast<size_t>(band)];
                    const float level = static_cast<float>(residual_of_channel[key_index]);
                    matrix_of_channel[key_index] =
                        lq_rebuild_entry(shared, compensator_correction, record_correction, compensator_count, records,
                                         key_index, channel_step, level);
                }
            }
        }
    } else {
        LQ_FOR(value_index, value_channels) {
            float compensator_correction[LQ_MAX_COMPENSATORS];
            float record_correction[LQ_MAX_WINDOW];
            for (int compensator = 0; compensator < compensator_count; ++compensator) {
                compensator_correction[compensator] = lq_half_to_float(
                    stored.compensator_corrections[static_cast<size_t>(compensator) * value_channels + value_index]);
            }
            for (int record = 0; record < records; ++record) {
                record_correction[record] = lq_half_to_float(
                    stored.window_corrections[static_cast<size_t>(record) * value_channels + value_index]);
            }
            const int8_t* residual_of_channel =
                stored.rounded_residual + static_cast<size_t>(value_index) * key_channels;
            float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
            if (scale_groups == 1) {
            const float channel_step = stored.value_channel_step[value_index];
#if defined(LQ_WORD_LOADS)
            // One thread owns the whole row here, so 8-byte loads cut its
            // instruction count. The levels are still consumed left to right.
            const bool by_words =
                (reinterpret_cast<uintptr_t>(residual_of_channel) & 7u) == 0u && (key_channels & 7) == 0;
            const uint64_t* residual_words = reinterpret_cast<const uint64_t*>(residual_of_channel);
            uint64_t word = 0;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                float level = 0.0f;
                if (by_words) {
                    if ((key_index & 7) == 0) {
                        word = residual_words[key_index >> 3];
                    }
                    level = static_cast<float>(static_cast<int8_t>(word & 0xFFu));
                    word = word >> 8;
                } else {
                    level = static_cast<float>(residual_of_channel[key_index]);
                }
                matrix_of_channel[key_index] =
                    lq_rebuild_entry(shared, compensator_correction, record_correction, compensator_count, records,
                                     key_index, channel_step, level);
            }
#else
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                const float level = static_cast<float>(residual_of_channel[key_index]);
                matrix_of_channel[key_index] =
                    lq_rebuild_entry(shared, compensator_correction, record_correction, compensator_count, records,
                                     key_index, channel_step, level);
            }
#endif
            } else {
                for (int key_index = 0; key_index < key_channels; ++key_index) {
                    const int band = key_index / scale_length;
                    const float channel_step =
                        stored.value_channel_step[static_cast<size_t>(value_index) * static_cast<size_t>(scale_groups) +
                                                  static_cast<size_t>(band)];
                    const float level = static_cast<float>(residual_of_channel[key_index]);
                    matrix_of_channel[key_index] =
                        lq_rebuild_entry(shared, compensator_correction, record_correction, compensator_count, records,
                                         key_index, channel_step, level);
                }
            }
        }
    }
    lq_sync(thread);
}

// The table of dot products of `count` vectors of length `length`, in shared->fit_gram.
// Every dot product is four chunks, one thread per chunk (dot_product() of the reference).
LQ_DEVICE void lq_fit_gram(const float (*vectors)[LQ_MAX_CHANNELS], int count, int length, const LqThread& thread,
                           LqFoldShared* shared) {
    const int pair_count = (count * (count + 1)) / 2;
    LQ_FOR(job, pair_count * LQ_SUM_CHUNKS) {
        const int pair = job / LQ_SUM_CHUNKS;
        const int chunk = job % LQ_SUM_CHUNKS;
        // Pair number -> (first, second) with first <= second, in the order of the reference's loops.
        int first = 0;
        int remaining = pair;
        while (remaining >= count - first) {
            remaining -= count - first;
            first += 1;
        }
        const int second = first + remaining;
        float sum = 0.0f;
        for (int index = lq_chunk_begin(chunk, length); index < lq_chunk_begin(chunk + 1, length); ++index) {
            sum += vectors[first][index] * vectors[second][index];
        }
        shared->fit_part[pair][chunk] = sum;
    }
    lq_sync(thread);
    LQ_MASTER {
        int pair = 0;
        for (int first = 0; first < count; ++first) {
            for (int second = first; second < count; ++second) {
                const float product = lq_combine_chunks(shared->fit_part[pair]);
                shared->fit_gram[first * LQ_SMALL + second] = product;
                shared->fit_gram[second * LQ_SMALL + first] = product;
                pair += 1;
            }
        }
    }
    lq_sync(thread);
}

// find_strongest_patterns_together() of ../leapquant.cpp, step for step.
// On return shared->fit_u[p], fit_v[p] and fit_strength[p] describe pattern p (before balancing).
LQ_DEVICE void lq_fit_together_block(const LqPool& pool,
                                     const LqHeadPointers& stored,
                                     const float* matrix,
                                     size_t matrix_stride,
                                     bool continue_from_previous,
                                     const LqThread& thread,
                                     LqFoldShared* shared) {
    const int key_channels = pool.key_channels;
    const int value_channels = pool.value_channels;
    const int pattern_count = pool.compensator_count;
    const int vector_count = pattern_count + 1;

    // Step 1a. The guesses: the stored value sides of the previous compensators, or none.
    LQ_FOR(value_index, value_channels) {
        for (int pattern = 0; pattern < pattern_count; ++pattern) {
            float guess = 0.0f;
            if (continue_from_previous) {
                guess = lq_half_to_float(
                    stored.compensator_corrections[static_cast<size_t>(pattern) * value_channels + value_index]);
            }
            shared->fit_v[pattern][value_index] = guess;
        }
    }
    lq_sync(thread);
    LQ_FOR(job, pattern_count * LQ_SUM_CHUNKS) {
        const int pattern = job / LQ_SUM_CHUNKS;
        const int chunk = job % LQ_SUM_CHUNKS;
        float sum = 0.0f;
        for (int index = lq_chunk_begin(chunk, value_channels); index < lq_chunk_begin(chunk + 1, value_channels); ++index) {
            sum += shared->fit_v[pattern][index] * shared->fit_v[pattern][index];
        }
        shared->fit_part[pattern][chunk] = sum;
    }
    lq_sync(thread);
    LQ_MASTER {
        shared->fit_some_guess_is_missing = 0;
        for (int pattern = 0; pattern < pattern_count; ++pattern) {
            shared->fit_has_guess[pattern] = lq_combine_chunks(shared->fit_part[pattern]) > 0.0f ? 1 : 0;
            if (shared->fit_has_guess[pattern] == 0) {
                shared->fit_some_guess_is_missing = 1;
            }
        }
    }
    lq_sync(thread);

    // Step 1b. Missing guesses are replaced by the longest rows of the matrix.
    if (shared->fit_some_guess_is_missing != 0) {
        LQ_FOR(key_index, key_channels) {
            float chunk_total[LQ_SUM_CHUNKS];
            for (int chunk = 0; chunk < LQ_SUM_CHUNKS; ++chunk) {
                float sum = 0.0f;
                for (int value_index = lq_chunk_begin(chunk, value_channels);
                     value_index < lq_chunk_begin(chunk + 1, value_channels); ++value_index) {
                    const float entry = matrix[static_cast<size_t>(value_index) * matrix_stride + key_index];
                    sum += entry * entry;
                }
                chunk_total[chunk] = sum;
            }
            shared->key_side[key_index] = lq_combine_chunks(chunk_total);
        }
        lq_sync(thread);
        LQ_MASTER {
            bool rows_left = true;
            for (int pattern = 0; pattern < pattern_count; ++pattern) {
                shared->fit_chosen_row[pattern] = -1;
                if (shared->fit_has_guess[pattern] != 0 || !rows_left) {
                    continue;
                }
                int longest_row = -1;
                float longest = 0.0f;
                for (int key_index = 0; key_index < key_channels; ++key_index) {
                    if (shared->key_side[key_index] > longest) {
                        longest = shared->key_side[key_index];
                        longest_row = key_index;
                    }
                }
                if (longest_row < 0) {
                    rows_left = false;       // no row with content is left
                    continue;
                }
                shared->key_side[longest_row] = 0.0f;      // each row is used once
                shared->fit_chosen_row[pattern] = longest_row;
            }
        }
        lq_sync(thread);
        LQ_FOR(value_index, value_channels) {
            for (int pattern = 0; pattern < pattern_count; ++pattern) {
                if (shared->fit_chosen_row[pattern] >= 0) {
                    shared->fit_v[pattern][value_index] =
                        matrix[static_cast<size_t>(value_index) * matrix_stride + shared->fit_chosen_row[pattern]];
                }
            }
        }
        lq_sync(thread);
    }

    // Step 1c. The rescue vector: the energy of every column.
    LQ_FOR(value_index, value_channels) {
        const float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
        float chunk_total[LQ_SUM_CHUNKS];
        for (int chunk = 0; chunk < LQ_SUM_CHUNKS; ++chunk) {
            float sum = 0.0f;
            for (int key_index = lq_chunk_begin(chunk, key_channels); key_index < lq_chunk_begin(chunk + 1, key_channels);
                 ++key_index) {
                const float value = matrix_of_channel[key_index];
                sum += value * value;
            }
            chunk_total[chunk] = sum;
        }
        shared->fit_v[pattern_count][value_index] = lq_combine_chunks(chunk_total);
    }
    lq_sync(thread);

    // Step 2. The sweeps.
    const int sweeps_used = shared->fit_some_guess_is_missing != 0 ? pool.power_iteration_sweeps + 2
                                                                   : pool.power_iteration_sweeps;
    for (int sweep = 0; sweep < sweeps_used; ++sweep) {
        // U = matrix * V.
        // One thread owns one key column and still adds each vector's products
        // in the four chunks of dot_product(). The vectors share that column, so
        // the matrix value is loaded once and applied to every vector before
        // the next row. The chunk totals stay per vector, so the bits match a
        // separate dot product for each vector.
        LQ_FOR(key_index, key_channels) {
            float chunk_total[LQ_FIT_VECTORS][LQ_SUM_CHUNKS];
            for (int chunk = 0; chunk < LQ_SUM_CHUNKS; ++chunk) {
                float partial[LQ_FIT_VECTORS];
                for (int vector = 0; vector < vector_count; ++vector) {
                    partial[vector] = 0.0f;
                }
                for (int value_index = lq_chunk_begin(chunk, value_channels);
                     value_index < lq_chunk_begin(chunk + 1, value_channels); ++value_index) {
                    const float entry = matrix[static_cast<size_t>(value_index) * matrix_stride + key_index];
                    for (int vector = 0; vector < vector_count; ++vector) {
                        partial[vector] += entry * shared->fit_v[vector][value_index];
                    }
                }
                for (int vector = 0; vector < vector_count; ++vector) {
                    chunk_total[vector][chunk] = partial[vector];
                }
            }
            for (int vector = 0; vector < vector_count; ++vector) {
                shared->fit_u[vector][key_index] = lq_combine_chunks(chunk_total[vector]);
            }
        }
        lq_sync(thread);
        // Make U orthonormal ("Cholesky QR", twice).
        for (int pass = 0; pass < 2; ++pass) {
            lq_fit_gram(shared->fit_u, vector_count, key_channels, thread, shared);
            LQ_MASTER {
                lq_small_cholesky(shared->fit_gram, vector_count, shared->fit_factor, shared->fit_alive);
            }
            lq_sync(thread);
            LQ_FOR(key_index, key_channels) {
                float values[LQ_SMALL];
                for (int vector = 0; vector < vector_count; ++vector) {
                    values[vector] = shared->fit_u[vector][key_index];
                }
                lq_small_apply_inverse(shared->fit_factor, shared->fit_alive, vector_count, values);
                for (int vector = 0; vector < vector_count; ++vector) {
                    shared->fit_u[vector][key_index] = values[vector];
                }
            }
            lq_sync(thread);
        }
        // V = matrix^T * U. Same chunk order as above, one load of the row per key.
        LQ_FOR(value_index, value_channels) {
            const float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
            float chunk_total[LQ_FIT_VECTORS][LQ_SUM_CHUNKS];
            for (int chunk = 0; chunk < LQ_SUM_CHUNKS; ++chunk) {
                float partial[LQ_FIT_VECTORS];
                for (int vector = 0; vector < vector_count; ++vector) {
                    partial[vector] = 0.0f;
                }
                for (int key_index = lq_chunk_begin(chunk, key_channels);
                     key_index < lq_chunk_begin(chunk + 1, key_channels); ++key_index) {
                    const float entry = matrix_of_channel[key_index];
                    for (int vector = 0; vector < vector_count; ++vector) {
                        partial[vector] += shared->fit_u[vector][key_index] * entry;
                    }
                }
                for (int vector = 0; vector < vector_count; ++vector) {
                    chunk_total[vector][chunk] = partial[vector];
                }
            }
            for (int vector = 0; vector < vector_count; ++vector) {
                shared->fit_v[vector][value_index] = lq_combine_chunks(chunk_total[vector]);
            }
        }
        lq_sync(thread);
    }

    // Step 3. Unmix and order by strength; the weakest vector is dropped.
    lq_fit_gram(shared->fit_v, vector_count, value_channels, thread, shared);
    LQ_MASTER {
        lq_small_eigenvectors(shared->fit_gram, vector_count, shared->fit_factor, shared->fit_strength);
    }
    lq_sync(thread);
    LQ_FOR(key_index, key_channels) {
        float values[LQ_SMALL];
        for (int vector = 0; vector < vector_count; ++vector) {
            values[vector] = shared->fit_u[vector][key_index];
        }
        for (int pattern = 0; pattern < pattern_count; ++pattern) {
            float sum = 0.0f;
            for (int vector = 0; vector < vector_count; ++vector) {
                sum += values[vector] * shared->fit_factor[vector * LQ_SMALL + pattern];
            }
            shared->fit_u[pattern][key_index] = sum;
        }
    }
    LQ_FOR(value_index, value_channels) {
        float values[LQ_SMALL];
        for (int vector = 0; vector < vector_count; ++vector) {
            values[vector] = shared->fit_v[vector][value_index];
        }
        for (int pattern = 0; pattern < pattern_count; ++pattern) {
            float sum = 0.0f;
            for (int vector = 0; vector < vector_count; ++vector) {
                sum += values[vector] * shared->fit_factor[vector * LQ_SMALL + pattern];
            }
            shared->fit_v[pattern][value_index] = sum;
        }
    }
    lq_sync(thread);
    LQ_FOR(job, pattern_count * LQ_SUM_CHUNKS) {
        const int pattern = job / LQ_SUM_CHUNKS;
        const int chunk = job % LQ_SUM_CHUNKS;
        float sum = 0.0f;
        for (int index = lq_chunk_begin(chunk, value_channels); index < lq_chunk_begin(chunk + 1, value_channels); ++index) {
            sum += shared->fit_v[pattern][index] * shared->fit_v[pattern][index];
        }
        shared->fit_part[pattern][chunk] = sum;
    }
    lq_sync(thread);
    LQ_MASTER {
        for (int pattern = 0; pattern < pattern_count; ++pattern) {
            shared->fit_strength[pattern] = sqrtf(lq_combine_chunks(shared->fit_part[pattern]));
        }
    }
    lq_sync(thread);
}

// Step D of the store: one entry to its integer level.
LQ_DEVICE int8_t lq_round_to_level(float entry, float channel_step, float largest_level) {
    float integer_level = 0.0f;
    if (channel_step > 0.0f) {
        integer_level = nearbyintf(entry / channel_step);
    }
    // Guard against a level of 128 caused by floating-point rounding.
    if (integer_level > largest_level) {
        integer_level = largest_level;
    }
    if (integer_level < -largest_level) {
        integer_level = -largest_level;
    }
    return static_cast<int8_t>(integer_level);
}

// Head::store_boundary_state. Turns `matrix` (stored [value][key], destroyed)
// into compensators + key_row_scale + value_channel_step + rounded_residual.
LQ_DEVICE void lq_store_block(const LqPool& pool,
                              const LqHeadPointers& stored,
                              float* matrix,
                              size_t matrix_stride,
                              bool continue_from_previous,
                              const LqThread& thread,
                              LqFoldShared* shared) {
    const int key_channels = pool.key_channels;
    const int value_channels = pool.value_channels;
    const int sweeps = pool.power_iteration_sweeps;

    // -------------------------------------------------------------------------
    // Step A. Compensators (equation 6): peel off the strongest pattern, then
    // the strongest pattern of what is left, and so on.
    // -------------------------------------------------------------------------
    const bool together = pool.fit_together != 0 && pool.compensator_count > 0;
    if (together) {
        lq_fit_together_block(pool, stored, matrix, matrix_stride, continue_from_previous, thread, shared);
    }
    for (int compensator = 0; compensator < pool.compensator_count; ++compensator) {
        if (together) {
            // Pattern `compensator` of the joint fit, in the places the one-at-a-time code leaves its result.
            LQ_FOR(key_index, key_channels) {
                shared->key_side[key_index] = shared->fit_u[compensator][key_index];
            }
            LQ_FOR(value_index, value_channels) {
                shared->value_side[value_index] = shared->fit_v[compensator][value_index];
            }
            LQ_MASTER {
                shared->strength = shared->fit_strength[compensator];
                shared->no_pattern = shared->fit_strength[compensator] > 0.0f ? 0 : 1;
            }
            lq_sync(thread);
        } else {
        // A.1 Starting guess: the key row with the largest length.
        LQ_FOR(key_index, key_channels) {
            float chunk_total[LQ_SUM_CHUNKS];
            for (int chunk = 0; chunk < LQ_SUM_CHUNKS; ++chunk) {
                float sum = 0.0f;
                for (int value_index = lq_chunk_begin(chunk, value_channels);
                     value_index < lq_chunk_begin(chunk + 1, value_channels); ++value_index) {
                    const float entry = matrix[static_cast<size_t>(value_index) * matrix_stride + key_index];
                    sum += entry * entry;
                }
                chunk_total[chunk] = sum;
            }
            shared->key_side[key_index] = lq_combine_chunks(chunk_total);
        }
        lq_sync(thread);
        LQ_MASTER {
            int longest_row = 0;
            float longest_row_squared_length = -1.0f;
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                if (shared->key_side[key_index] > longest_row_squared_length) {
                    longest_row_squared_length = shared->key_side[key_index];
                    longest_row = key_index;
                }
            }
            shared->longest_row = longest_row;
            if (longest_row_squared_length > 0.0f) {
                shared->no_pattern = 0;
                shared->length = 1.0f / sqrtf(longest_row_squared_length);
            } else {
                shared->no_pattern = 1;   // the matrix is all zeros
                shared->length = 0.0f;
            }
            shared->strength = 0.0f;
        }
        lq_sync(thread);
        LQ_FOR(value_index, value_channels) {
            float start = 0.0f;
            if (shared->no_pattern == 0) {
                start = matrix[static_cast<size_t>(value_index) * matrix_stride + shared->longest_row] * shared->length;
            }
            shared->value_side[value_index] = start;
        }
        lq_sync(thread);
        if (compensator == 0 && LQ_STOP_AT(LQ_PHASE_FOLD_START)) {
            return;
        }

        // A.2 The power-iteration sweeps.
        for (int sweep = 0; sweep < sweeps; ++sweep) {
            // key_side = matrix * value_side
            LQ_FOR(key_index, key_channels) {
                if (shared->no_pattern == 0) {
                    float chunk_total[LQ_SUM_CHUNKS];
                    for (int chunk = 0; chunk < LQ_SUM_CHUNKS; ++chunk) {
                        float sum = 0.0f;
                        for (int value_index = lq_chunk_begin(chunk, value_channels);
                             value_index < lq_chunk_begin(chunk + 1, value_channels); ++value_index) {
                            sum += matrix[static_cast<size_t>(value_index) * matrix_stride + key_index] *
                                   shared->value_side[value_index];
                        }
                        chunk_total[chunk] = sum;
                    }
                    shared->key_side[key_index] = lq_combine_chunks(chunk_total);
                }
            }
            lq_sync(thread);
            if (LQ_STOP_IN_FIRST_SWEEP(LQ_PHASE_SWEEP_KEY_SIDE)) {
                return;
            }
            // The length of key_side: four threads add one chunk each, then thread 0 combines.
            LQ_FOR(chunk, LQ_SUM_CHUNKS) {
                float sum = 0.0f;
                for (int key_index = lq_chunk_begin(chunk, key_channels);
                     key_index < lq_chunk_begin(chunk + 1, key_channels); ++key_index) {
                    sum += shared->key_side[key_index] * shared->key_side[key_index];
                }
                shared->length_part[chunk] = sum;
            }
            lq_sync(thread);
            LQ_MASTER {
                if (shared->no_pattern == 0) {
                    const float key_side_length = sqrtf(lq_combine_chunks(shared->length_part));
                    if (key_side_length > 0.0f) {
                        shared->length = key_side_length;
                    } else {
                        shared->no_pattern = 1;
                    }
                }
            }
            lq_sync(thread);
            if (LQ_STOP_IN_FIRST_SWEEP(LQ_PHASE_SWEEP_KEY_LENGTH)) {
                return;
            }
            // Normalise key_side to length 1.
            LQ_FOR(key_index, key_channels) {
                if (shared->no_pattern == 0) {
                    shared->key_side[key_index] = shared->key_side[key_index] / shared->length;
                }
            }
            lq_sync(thread);
            if (LQ_STOP_IN_FIRST_SWEEP(LQ_PHASE_SWEEP_NORMALISE)) {
                return;
            }
            // value_side = matrix^T * key_side
            LQ_FOR(value_index, value_channels) {
                if (shared->no_pattern == 0) {
                    const float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
                    float sum = 0.0f;
                    for (int key_index = 0; key_index < key_channels; ++key_index) {
                        sum += shared->key_side[key_index] * matrix_of_channel[key_index];
                    }
                    shared->value_side[value_index] = sum;
                }
            }
            lq_sync(thread);
            if (LQ_STOP_IN_FIRST_SWEEP(LQ_PHASE_SWEEP_VALUE_SIDE)) {
                return;
            }
            LQ_FOR(chunk, LQ_SUM_CHUNKS) {
                float sum = 0.0f;
                for (int value_index = lq_chunk_begin(chunk, value_channels);
                     value_index < lq_chunk_begin(chunk + 1, value_channels); ++value_index) {
                    sum += shared->value_side[value_index] * shared->value_side[value_index];
                }
                shared->length_part[chunk] = sum;
            }
            lq_sync(thread);
            LQ_MASTER {
                if (shared->no_pattern == 0) {
                    const float strength = sqrtf(lq_combine_chunks(shared->length_part));
                    if (strength > 0.0f) {
                        shared->strength = strength;
                    } else {
                        shared->no_pattern = 1;
                    }
                }
            }
            lq_sync(thread);
            if (LQ_STOP_IN_FIRST_SWEEP(LQ_PHASE_SWEEP_STRENGTH)) {
                return;
            }
            // Normalise value_side for the next sweep, but not after the last one.
            if (sweep != sweeps - 1) {
                LQ_FOR(value_index, value_channels) {
                    if (shared->no_pattern == 0) {
                        shared->value_side[value_index] = shared->value_side[value_index] / shared->strength;
                    }
                }
                lq_sync(thread);
            }
        }

        }   // one-at-a-time fit
        if (compensator == 0 && LQ_STOP_AT(LQ_PHASE_FOLD_SWEEPS)) {
            return;
        }

        // A.3 Balance the two vectors, store them in fp16, and keep the values
        //     AS STORED, so the fp16 rounding error stays in the residual.
        uint16_t* key_slot = stored.compensator_keys + static_cast<size_t>(compensator) * key_channels;
        uint16_t* correction_slot = stored.compensator_corrections + static_cast<size_t>(compensator) * value_channels;
        LQ_FOR(key_index, key_channels) {
            float balanced = 0.0f;
            if (shared->no_pattern == 0) {
                balanced = shared->key_side[key_index] * sqrtf(shared->strength);
            }
            key_slot[key_index] = lq_float_to_half(balanced);
            shared->key_side[key_index] = lq_half_to_float(key_slot[key_index]);
        }
        LQ_FOR(value_index, value_channels) {
            float balanced = 0.0f;
            if (shared->no_pattern == 0) {
                balanced = shared->value_side[value_index] / sqrtf(shared->strength);
            }
            correction_slot[value_index] = lq_float_to_half(balanced);
            shared->value_side[value_index] = lq_half_to_float(correction_slot[value_index]);
        }
        lq_sync(thread);

        // A.4 matrix -= stored_key_side * stored_value_side^T
        LQ_FOR(value_index, value_channels) {
            float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
            const float value_side_entry = shared->value_side[value_index];
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                const float row_factor = -1.0f * shared->key_side[key_index];
                matrix_of_channel[key_index] += row_factor * value_side_entry;
            }
        }
        lq_sync(thread);
        if (compensator == 0 && LQ_STOP_AT(LQ_PHASE_FOLD_FIRST)) {
            return;
        }
    }
    if (LQ_STOP_AT(LQ_PHASE_FOLD_COMPENSATORS)) {
        return;
    }
    // From here on, `matrix` holds the RESIDUAL.

    // -------------------------------------------------------------------------
    // Step B. Smoothing (section 3.4): key_row_scale = sqrt(mean |row|).
    // -------------------------------------------------------------------------
    LQ_FOR(key_index, key_channels) {
        float scale = 1.0f;
        if (pool.use_smoothing != 0) {
            float sum_of_absolute_values = 0.0f;
            for (int value_index = 0; value_index < value_channels; ++value_index) {
                sum_of_absolute_values += fabsf(matrix[static_cast<size_t>(value_index) * matrix_stride + key_index]);
            }
            scale = sqrtf(sum_of_absolute_values / static_cast<float>(value_channels));
            if (scale < LQ_SMALLEST_KEY_ROW_SCALE) {
                scale = LQ_SMALLEST_KEY_ROW_SCALE;
            }
        }
        stored.key_row_scale[key_index] = scale;
        shared->key_side[key_index] = scale;
    }
    lq_sync(thread);
    if (LQ_STOP_AT(LQ_PHASE_FOLD_SMOOTHING)) {
        return;
    }

    // -------------------------------------------------------------------------
    // Steps C and D. One step size per value channel (appendix B.3), then round.
    // One thread owns one value channel from here to the end.
    // Several groups: each band of key rows has its own absmax. The one-group
    // path below is the 8-bit store, including its word stores.
    // -------------------------------------------------------------------------
    const float largest_level = static_cast<float>(pool.largest_integer_level);
    const int scale_groups = lq_scale_group_count(pool);
    const int scale_length = lq_scale_group_length(pool);
    LQ_FOR(value_index, value_channels) {
        float* matrix_of_channel = matrix + static_cast<size_t>(value_index) * matrix_stride;
        int8_t* residual_of_channel = stored.rounded_residual + static_cast<size_t>(value_index) * key_channels;

        if (scale_groups == 1) {
        float largest_absolute_entry = 0.0f;
        for (int key_index = 0; key_index < key_channels; ++key_index) {
            const float entry = matrix_of_channel[key_index] / shared->key_side[key_index];
            matrix_of_channel[key_index] = entry;
            if (fabsf(entry) > largest_absolute_entry) {
                largest_absolute_entry = fabsf(entry);
            }
        }
        const float channel_step = largest_absolute_entry / largest_level;
        stored.value_channel_step[value_index] = channel_step;

        int key_index = 0;
#if defined(LQ_WORD_LOADS)
        // Tuning option: write the levels 8 at a time (see the same option in the decode block).
        if ((reinterpret_cast<size_t>(residual_of_channel) & 7u) == 0u) {
            uint64_t* residual_words = reinterpret_cast<uint64_t*>(residual_of_channel);
            for (; key_index + 8 <= key_channels; key_index += 8) {
                uint64_t word = 0;
                for (int byte = 0; byte < 8; ++byte) {
                    const int8_t level = lq_round_to_level(matrix_of_channel[key_index + byte], channel_step, largest_level);
                    word |= static_cast<uint64_t>(static_cast<uint8_t>(level)) << (8 * byte);   // little-endian
                }
                residual_words[key_index >> 3] = word;
            }
        }
#endif
        for (; key_index < key_channels; ++key_index) {
            residual_of_channel[key_index] = lq_round_to_level(matrix_of_channel[key_index], channel_step, largest_level);
        }
        } else {
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                matrix_of_channel[key_index] = matrix_of_channel[key_index] / shared->key_side[key_index];
            }
            for (int group = 0; group < scale_groups; ++group) {
                const int begin = group * scale_length;
                int end = begin + scale_length;
                if (end > key_channels) {
                    end = key_channels;
                }
                float largest_absolute_entry = 0.0f;
                for (int key_index = begin; key_index < end; ++key_index) {
                    const float entry = matrix_of_channel[key_index];
                    if (fabsf(entry) > largest_absolute_entry) {
                        largest_absolute_entry = fabsf(entry);
                    }
                }
                stored.value_channel_step[static_cast<size_t>(value_index) * static_cast<size_t>(scale_groups) +
                                          static_cast<size_t>(group)] = largest_absolute_entry / largest_level;
            }
            for (int key_index = 0; key_index < key_channels; ++key_index) {
                const int band = key_index / scale_length;
                const float channel_step =
                    stored.value_channel_step[static_cast<size_t>(value_index) * static_cast<size_t>(scale_groups) +
                                              static_cast<size_t>(band)];
                residual_of_channel[key_index] = lq_round_to_level(matrix_of_channel[key_index], channel_step, largest_level);
            }
        }
    }
}

// Copy one head's [value][key] matrix. Consecutive threads take consecutive
// entries of a row, so on a GPU a warp reads and writes a contiguous run.
// Either stride may be longer than key_channels (the padded block matrix).
LQ_DEVICE void lq_copy_matrix(const float* source,
                              size_t source_stride,
                              float* destination,
                              size_t destination_stride,
                              int value_channels,
                              int key_channels,
                              const LqThread& thread) {
    const int total = value_channels * key_channels;
    for (int linear = thread.index; linear < total; linear += thread.count) {
        const int value_index = linear / key_channels;
        const int key_index = linear - value_index * key_channels;
        destination[static_cast<size_t>(value_index) * destination_stride + key_index] =
            source[static_cast<size_t>(value_index) * source_stride + key_index];
    }
}

// Head::fold_window_into_boundary_state, for the rows that need it.
// `block_matrix`: null, or V * K floats that belong to this block alone (a GPU
// block's shared memory). The fold then keeps the full-precision matrix there
// instead of in fold.scratch. The arithmetic is the same; what changes is that
// the 64 walks over the matrix that the power iteration makes stay on the chip
// instead of going to GPU memory each time. Measured on an H100 at batch 512,
// with the matrix in fold.scratch the fold moved about 64 GB per step.
LQ_DEVICE void lq_fold_block(const LqPool& pool,
                             const LqFold& fold,
                             int row,
                             int head,
                             const LqThread& thread,
                             LqFoldShared* shared,
                             float* block_matrix) {
    const int slot = fold.slot[row];
    if (slot < 0) {
        return;
    }
    int records = fold.tokens_in_window[row] + fold.tokens_added;
    const bool window_is_full = records >= pool.window_length;
    const bool forced = fold.force != nullptr && fold.force[row] != 0 && records > 0;
    if (!window_is_full && !forced) {
        return;
    }
    if (records > pool.window_length) {
        records = pool.window_length;
    }
    const LqHeadPointers stored = lq_locate_head(pool, slot, head);
    float* matrix = block_matrix != nullptr
                        ? block_matrix
                        : fold.scratch + (static_cast<size_t>(row) * pool.value_heads + head) *
                                             pool.value_channels * pool.key_channels;
    const size_t matrix_stride = lq_matrix_stride(pool, block_matrix);
    lq_rebuild_block(pool, stored, records, matrix, matrix_stride, thread, shared);
    if (LQ_STOP_AT(LQ_PHASE_FOLD_REBUILD)) {
        return;
    }
    if (fold.export_by_slot != nullptr) {
        // Hand the full-precision state to the engine before it is rounded.
        // The store writes `matrix` later, so every thread finishes the copy first.
        float* destination = fold.export_by_slot + (static_cast<size_t>(slot) * pool.value_heads + head) *
                                                       pool.value_channels * pool.key_channels;
        lq_copy_matrix(matrix, matrix_stride, destination, static_cast<size_t>(pool.key_channels), pool.value_channels,
                       pool.key_channels, thread);
        lq_sync(thread);
    }
    lq_store_block(pool, stored, matrix, matrix_stride, true, thread, shared);
}

// Head::load_full_precision_state.
LQ_DEVICE void lq_store_entry_block(const LqPool& pool,
                                    const LqStore& store,
                                    int row,
                                    int head,
                                    const LqThread& thread,
                                    LqFoldShared* shared,
                                    float* block_matrix) {
    const int slot = store.slot[row];
    if (slot < 0) {
        return;
    }
    if (store.skip_slot != nullptr && store.skip_slot[slot] != 0) {
        return;
    }
    const LqHeadPointers stored = lq_locate_head(pool, slot, head);
    const size_t head_floats = static_cast<size_t>(pool.value_channels) * pool.key_channels;
    float* state_of_row = store.state + (static_cast<size_t>(row) * pool.value_heads + head) * head_floats;
    float* matrix = block_matrix != nullptr ? block_matrix : state_of_row;
    const size_t matrix_stride = lq_matrix_stride(pool, block_matrix);
    if (store.source_by_slot != nullptr || block_matrix != nullptr) {
        // Copy the input into the working matrix: from the engine's tensor, or,
        // when the block has its own matrix, from store.state.
        const float* source = state_of_row;
        if (store.source_by_slot != nullptr) {
            const size_t entry = static_cast<size_t>(store.source_index != nullptr ? store.source_index[row] : slot);
            const size_t entry_stride = store.source_stride > 0 ? static_cast<size_t>(store.source_stride)
                                                                : head_floats * pool.value_heads;
            source = store.source_by_slot + entry * entry_stride + static_cast<size_t>(head) * head_floats;
        }
        lq_copy_matrix(source, static_cast<size_t>(pool.key_channels), matrix, matrix_stride, pool.value_channels,
                       pool.key_channels, thread);
        lq_sync(thread);
    }
    lq_store_block(pool, stored, matrix, matrix_stride, false, thread, shared);
}

// Head::copy_state_to_full_precision.
LQ_DEVICE void lq_export_block(const LqPool& pool,
                               const LqExport& request,
                               int row,
                               int head,
                               const LqThread& thread,
                               LqFoldShared* shared) {
    const int slot = request.slot[row];
    if (slot < 0) {
        return;
    }
    int records = request.tokens_in_window != nullptr ? request.tokens_in_window[row] : 0;
    if (records < 0) {
        records = 0;
    }
    if (records > pool.window_length) {
        records = pool.window_length;
    }
    const LqHeadPointers stored = lq_locate_head(pool, slot, head);
    float* matrix = request.state +
                    (static_cast<size_t>(row) * pool.value_heads + head) * pool.value_channels * pool.key_channels;
    lq_rebuild_block(pool, stored, records, matrix, static_cast<size_t>(pool.key_channels), thread, shared);
}

// =============================================================================
// Argument checks, compiled into exactly one file of each backend.
// =============================================================================
#if defined(LQ_DEFINE_CHECKS)

int lq_check_pool(const LqPool* pool) {
    if (pool == nullptr) {
        return LQ_ERROR_BAD_POOL;
    }
    const bool sizes_ok = pool->slot_count >= 1 && pool->value_heads >= 1 &&
                          pool->key_channels >= 1 && pool->key_channels <= LQ_MAX_CHANNELS &&
                          pool->value_channels >= 1 && pool->value_channels <= LQ_MAX_CHANNELS &&
                          pool->window_length >= 1 && pool->window_length <= LQ_MAX_WINDOW &&
                          pool->compensator_count >= 0 && pool->compensator_count <= LQ_MAX_COMPENSATORS &&
                          pool->largest_integer_level >= 1 && pool->largest_integer_level <= 127 &&
                          pool->power_iteration_sweeps >= 1 && pool->scale_group >= 0;
    if (!sizes_ok) {
        return LQ_ERROR_BAD_POOL;
    }
    const bool compensators_ok =
        pool->compensator_count == 0 || (pool->compensator_keys != nullptr && pool->compensator_corrections != nullptr);
    const bool pointers_ok = pool->rounded_residual != nullptr && pool->value_channel_step != nullptr &&
                             pool->key_row_scale != nullptr && compensators_ok;
    return pointers_ok ? 0 : LQ_ERROR_BAD_POOL;
}

// A pool that only stores boundary states (a checkpoint store) has no window
// buffer. The calls that read or write the window require one.
static int lq_check_window(const LqPool* pool) {
    const bool window_ok = pool->window_keys != nullptr && pool->window_corrections != nullptr &&
                           pool->window_log_decays != nullptr;
    return window_ok ? 0 : LQ_ERROR_BAD_POOL;
}

int lq_check_step(const LqPool* pool, const LqStep* step) {
    const int pool_status = lq_check_pool(pool);
    if (pool_status != 0) {
        return pool_status;
    }
    if (lq_check_window(pool) != 0) {
        return LQ_ERROR_BAD_POOL;
    }
    if (step == nullptr || step->rows < 0 || step->key_heads < 1 || pool->value_heads % step->key_heads != 0) {
        return LQ_ERROR_BAD_ARGUMENT;
    }
    if (step->slot == nullptr || step->tokens_in_window == nullptr || step->output == nullptr) {
        return LQ_ERROR_BAD_ARGUMENT;
    }
    if (step->raw_qkv != nullptr) {
        const bool raw_ok = step->raw_a != nullptr && step->raw_b != nullptr && step->A_log != nullptr &&
                            step->dt_bias != nullptr && step->raw_dtype >= LQ_RAW_FP32 && step->raw_dtype <= LQ_RAW_FP16 &&
                            step->raw_stride >= 2 * static_cast<int64_t>(step->key_heads) * pool->key_channels +
                                                    static_cast<int64_t>(pool->value_heads) * pool->value_channels;
        return raw_ok ? 0 : LQ_ERROR_BAD_ARGUMENT;
    }
    const bool pointers_ok = step->decay != nullptr && step->beta != nullptr && step->key != nullptr &&
                             step->query != nullptr && step->value != nullptr;
    return pointers_ok ? 0 : LQ_ERROR_BAD_ARGUMENT;
}

int lq_check_fold(const LqPool* pool, const LqFold* fold) {
    const int pool_status = lq_check_pool(pool);
    if (pool_status != 0) {
        return pool_status;
    }
    if (lq_check_window(pool) != 0) {
        return LQ_ERROR_BAD_POOL;
    }
    if (fold == nullptr || fold->rows < 0 || fold->tokens_added < 0 || fold->tokens_added > 1) {
        return LQ_ERROR_BAD_ARGUMENT;
    }
    const bool pointers_ok = fold->slot != nullptr && fold->tokens_in_window != nullptr && fold->scratch != nullptr;
    return pointers_ok ? 0 : LQ_ERROR_BAD_ARGUMENT;
}

int lq_check_store(const LqPool* pool, const LqStore* store) {
    const int pool_status = lq_check_pool(pool);
    if (pool_status != 0) {
        return pool_status;
    }
    if (store == nullptr || store->rows < 0 || store->slot == nullptr || store->state == nullptr) {
        return LQ_ERROR_BAD_ARGUMENT;
    }
    return 0;
}

int lq_check_export(const LqPool* pool, const LqExport* request) {
    const int pool_status = lq_check_pool(pool);
    if (pool_status != 0) {
        return pool_status;
    }
    if (request == nullptr || request->rows < 0 || request->slot == nullptr || request->state == nullptr) {
        return LQ_ERROR_BAD_ARGUMENT;
    }
    if (request->tokens_in_window != nullptr && lq_check_window(pool) != 0) {
        return LQ_ERROR_BAD_POOL;
    }
    return 0;
}

#endif  // LQ_DEFINE_CHECKS

#endif  // LEAPQUANT_KERNEL_BLOCKS_H
