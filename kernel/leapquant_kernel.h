// =============================================================================
// leapquant_kernel.h
//
// The batched LeapQuant kernel: the same arithmetic as the reference core
// (../leapquant.h), laid out the way a serving engine needs it.
//
// How it differs from the reference core
//   Reference core                       This kernel
//   -----------------------------------  ------------------------------------------
//   one Head object per attention head   one state POOL for all requests and heads
//   one token of one head per call       one decode step of a whole batch per call
//   folds by itself when the window      the fold is a second entry point; the
//     is full                              caller launches it after every step and
//                                          it decides per row, on the device
//   Gated DeltaNet and KDA               Gated DeltaNet only (one decay per token)
//   int8 matrix stored [key][value]      int8 matrix stored [value][key], the
//                                          orientation SGLang uses for its state
//
// One source, three ways to run it
//   The arithmetic lives in leapquant_kernel_blocks.h and is written once, as the
//   work of ONE thread block for ONE attention head of ONE request.
//     - leapquant_kernel_cuda.cu   launches one CUDA thread block per head.
//     - leapquant_kernel_cpu.cpp   runs the same code on a CPU, either as one
//                                  thread per block (fast, for trace replay) or
//                                  as an emulated thread block with real threads
//                                  and a real barrier (slow, for race checking).
//
// The state pool. `slots` requests, `HV` value heads per layer.
//   BOUNDARY STATE, 19,456 bytes per head at the default sizes:
//     rounded_residual         int8  [slots][HV][V][K]
//     value_channel_step       fp32  [slots][HV][V][G]   G = 1 at 8 bits (one step per column)
//     key_row_scale            fp32  [slots][HV][K]
//     compensator_keys         fp16  [slots][HV][R][K]
//     compensator_corrections  fp16  [slots][HV][R][V]
//   WINDOW BUFFER, 8,224 bytes per head at the default sizes:
//     window_keys              fp16  [slots][HV][L][K]
//     window_corrections       fp16  [slots][HV][L][V]
//     window_log_decays        fp16  [slots][HV][L]
//   K = key_channels, V = value_channels, R = compensator_count, L = window_length.
//
// The window cursor
//   The number of records in a request's window is NOT stored in the pool. The
//   engine owns it (one int32 per request, shared by all layers) and passes it
//   to every call as `tokens_in_window`. One decode step of one layer is:
//
//       lq_*_decode_step(pool, step)     appends record number tokens_in_window[row]
//       lq_*_fold(pool, fold)            with tokens_added = 1; folds the rows whose
//                                        window is now full, or that are forced
//
//   After all layers ran, the engine advances the cursor once:
//       folded  ->  0          otherwise  ->  tokens_in_window + 1
//
// Every function returns 0 on success and a negative LQ_ERROR_* code otherwise.
// =============================================================================

#ifndef LEAPQUANT_KERNEL_H
#define LEAPQUANT_KERNEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Compile-time limits. They size the per-block scratch arrays of the kernel.
#define LQ_MAX_CHANNELS 128      // largest key_channels and value_channels
#ifndef LQ_MAX_WINDOW
#define LQ_MAX_WINDOW 16         // largest window_length (the decode block keeps 2 * L * K floats)
#endif
#ifndef LQ_MAX_COMPENSATORS
// Largest compensator_count the exact kernel can store. 6 is the paper's rank at
// 6 and 4 bits. The macro sizes every fold block, including an 8-bit run that
// uses rank 4. Rank 6 grows the joint-fit vectors from 5 to 7, about 2.2 KiB,
// so a fold block is about 74 KiB. Three of those are near an H100's 228 KiB
// shared-memory limit; that occupancy was not measured here. 8 would fit the
// reference (its small matrix is 9) and is not the default: -DLQ_MAX_COMPENSATORS=8.
#define LQ_MAX_COMPENSATORS 6
#endif

#define LQ_ERROR_BAD_POOL (-1)       // a pool size is out of range or a pool pointer is null
#define LQ_ERROR_BAD_ARGUMENT (-2)   // a call argument is out of range or a pointer is null
#define LQ_ERROR_CUDA (-3)           // the CUDA runtime reported an error at launch

// The state pool of one layer. The kernel does not own the memory.
typedef struct LqPool {
    int8_t* rounded_residual;             // [slots][HV][V][K]
    float* value_channel_step;            // [slots][HV][V]
    float* key_row_scale;                 // [slots][HV][K]
    uint16_t* compensator_keys;           // [slots][HV][R][K]  fp16
    uint16_t* compensator_corrections;    // [slots][HV][R][V]  fp16
    // The three window pointers may be null in a pool that only stores boundary
    // states (a checkpoint store). Decode and fold then return LQ_ERROR_BAD_POOL.
    uint16_t* window_keys;                // [slots][HV][L][K]  fp16
    uint16_t* window_corrections;         // [slots][HV][L][V]  fp16
    uint16_t* window_log_decays;          // [slots][HV][L]     fp16, natural log of the decay
    int32_t slot_count;
    int32_t value_heads;                  // HV
    int32_t key_channels;                 // K
    int32_t value_channels;               // V
    int32_t window_length;                // L
    int32_t compensator_count;            // R
    int32_t largest_integer_level;        // 127 for 8 bits
    int32_t power_iteration_sweeps;       // 8 in the reference core
    int32_t use_smoothing;                // 0 or 1
    int32_t fit_together;                 // 1 = all compensators in one joint fit (Settings::fit_compensators_together)
    // For profiling only. 0 = normal. Any other value makes every block stop at
    // that point of its work (see LQ_PHASE_* in leapquant_kernel_blocks.h), so that
    // the time of each phase can be measured as a difference. The stored state is
    // then incomplete: never set it outside kernel/profile_kernel.py.
    int32_t profile_stop_phase;
    // Key rows that share one value-column step. 0, or a length >= key_channels,
    // is one step for the whole column (every 8-bit state). A smaller length stores
    // value_channel_step as [V][G], G = ceil(key_channels / scale_group), group g
    // covering key rows [g * scale_group, g * scale_group + scale_group).
    // Sits at the end so the fields above keep their offsets.
    int32_t scale_group;
} LqPool;

// One decode step: one token for each of `rows` requests.
// key and query arrive exactly as the recurrence uses them: normalised, and
// the query already multiplied by 1 / sqrt(key_channels).
typedef struct LqStep {
    int32_t rows;                         // B
    int32_t key_heads;                    // H. value_heads must be a multiple of it.
                                          // Value head h reads key head h / (HV / H).
    const int32_t* slot;                  // [B]  pool slot of each row. Negative = padded row:
                                          //      its output is zero and nothing is stored.
    const int32_t* tokens_in_window;      // [B]  records already in the window: 0 .. L-1
    const float* decay;                   // [B][HV]     alpha, a number in (0, 1]
    const float* beta;                    // [B][HV]
    const float* key;                     // [B][H][K]
    const float* query;                   // [B][H][K]
    const float* value;                   // [B][HV][V]
    float* output;                        // [B][HV][V]  filled by the step

    // RAW INPUTS (optional). If raw_qkv is not null, the five input arrays above
    // (decay, beta, key, query, value) are ignored and the kernel computes them
    // itself from what the model's projections give, with the formulas of
    // SGLang's and vLLM's packed decode kernels:
    //     g = -exp(A_log) * softplus(a + dt_bias)     decay = exp(g)
    //     beta = sigmoid(b), rounded to the dtype of b
    //     q, k L2-normalised with eps 1e-6;  q also times `scale`
    // This saves about ten small GPU kernels per layer in front of the step.
    const void* raw_qkv;                  // [B][raw_stride], packed (q | k | v): 2*H*K + HV*V used
    const void* raw_a;                    // [B][HV]
    const void* raw_b;                    // [B][HV]
    const float* A_log;                   // [HV]
    const float* dt_bias;                 // [HV]
    float scale;                          // usually 1 / sqrt(K)
    int32_t raw_dtype;                    // dtype of raw_qkv, raw_a, raw_b: LQ_RAW_FP32, LQ_RAW_BF16 or LQ_RAW_FP16
    int64_t raw_stride;                   // elements per row of raw_qkv
} LqStep;

#define LQ_RAW_FP32 0
#define LQ_RAW_BF16 1
#define LQ_RAW_FP16 2

// Fold: closes the window of every row for which
//       tokens_in_window[row] + tokens_added >= window_length
//   or  force[row] != 0 and the window holds at least one record.
// Other rows are left untouched.
typedef struct LqFold {
    int32_t rows;                         // B
    int32_t tokens_added;                 // 1 when called right after lq_*_decode_step, else 0
    const int32_t* slot;                  // [B]
    const int32_t* tokens_in_window;      // [B]
    const int32_t* force;                 // [B], or null for "no forced folds"
    float* scratch;                       // [B][HV][V][K] fp32 working memory, contents destroyed
    float* export_by_slot;                // null, or [slots][HV][V][K] fp32: every row that folds also
                                          // writes its full-precision state, as it is BEFORE rounding,
                                          // to the entry of its slot. An engine that keeps an fp32 copy
                                          // of the state for checkpoints uses this to keep it current.
} LqFold;

// Store: turns full-precision states into boundary states, for example after
// prefill. The window of the slot must be treated as empty afterwards.
typedef struct LqStore {
    int32_t rows;                         // B
    const int32_t* slot;                  // [B]  negative = skip this row
    float* state;                         // [B][HV][V][K] fp32. Used as working memory and destroyed.
                                          // It is the input, unless source_by_slot is set.
    const float* source_by_slot;          // null, or [slots][HV][V][K] fp32: the input of a row is the
                                          // entry of its slot. It is copied into `state` and not changed.
    const int32_t* skip_slot;             // null, or [slots]: a row whose slot has a nonzero entry is skipped
    // For engines whose fp32 state is not laid out by pool slot (vLLM keeps it in
    // KV-cache blocks, with padding between blocks):
    const int32_t* source_index;          // null, or [B]: the entry of source_by_slot a row reads,
                                          // instead of its pool slot
    int64_t source_stride;                // 0, or the distance in floats between two entries of
                                          // source_by_slot (0 means HV * V * K: no padding)
} LqStore;

// Export: writes the current state (boundary state plus window) in full precision.
typedef struct LqExport {
    int32_t rows;                         // B
    const int32_t* slot;                  // [B]  negative = skip this row
    const int32_t* tokens_in_window;      // [B], or null for "every window is empty"
    float* state;                         // [B][HV][V][K] fp32, filled
} LqExport;

// ---- CPU backend (leapquant_kernel_cpu.cpp) ---------------------------------
int lq_cpu_decode_step(const LqPool* pool, const LqStep* step);
int lq_cpu_fold(const LqPool* pool, const LqFold* fold);
int lq_cpu_store(const LqPool* pool, const LqStore* store);
int lq_cpu_export(const LqPool* pool, const LqExport* request);

// 1 (the default): every block runs in the calling thread.
// More than 1: every block is run by that many real threads that meet at a
// barrier wherever the CUDA build calls __syncthreads(). The results are the
// same bit for bit; the point is to let a race detector see the block as a
// GPU would run it.
void lq_cpu_set_threads_per_block(int thread_count);

// 0 (the default): fold and store work in the caller's scratch memory.
// 1: every block gets its own V * K matrix, as the GPU build gives each block in
// its shared memory. The results are the same bit for bit.
void lq_cpu_set_block_matrix(int enabled);

// ---- CUDA backend (leapquant_kernel_cuda.cu) --------------------------------
// All pointers are device pointers. `cuda_stream` is a cudaStream_t (null = the
// default stream). The calls only launch the kernel; they do not wait for it.
int lq_cuda_decode_step(const LqPool* pool, const LqStep* step, void* cuda_stream);
int lq_cuda_fold(const LqPool* pool, const LqFold* fold, void* cuda_stream);
int lq_cuda_store(const LqPool* pool, const LqStore* store, void* cuda_stream);
int lq_cuda_export(const LqPool* pool, const LqExport* request, void* cuda_stream);

// ---- Shared helpers ---------------------------------------------------------
// The checks every backend runs before it touches memory. They look at sizes
// and null pointers only, so they are safe to call with device pointers.
int lq_check_pool(const LqPool* pool);
int lq_check_step(const LqPool* pool, const LqStep* step);
int lq_check_fold(const LqPool* pool, const LqFold* fold);
int lq_check_store(const LqPool* pool, const LqStore* store);
int lq_check_export(const LqPool* pool, const LqExport* request);

// The fp16 conversion the CPU build uses, exported so that a test can compare
// it with the reference core.
uint16_t lq_cpu_float_to_half(float value);
float lq_cpu_half_to_float(uint16_t half_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LEAPQUANT_KERNEL_H
