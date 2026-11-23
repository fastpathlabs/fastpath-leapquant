// =============================================================================
// leapquant_kernel_cuda.cu
//
// Launches the kernel blocks of leapquant_kernel_blocks.h on a GPU.
// One CUDA thread block per (request row, value head).
//
// Build (see ../Makefile, target `cuda`):
//     nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC \
//          kernel/leapquant_kernel_cuda.cu -o build/libleapquant_cuda.so
//
// The launches are the same every step. Which rows fold is decided on the
// device from `tokens_in_window` and `force`, so a captured CUDA graph stays
// valid. Skipping the fold launch from the host when the window is not full
// would record whichever path the warmup step happened to take.
// =============================================================================

#include <cuda_runtime.h>

#define LQ_DEFINE_CHECKS
#include "leapquant_kernel_blocks.h"

// Threads per block. 128 = one thread per key channel and per value channel at
// the default sizes. The blocks are correct for any thread count.
#ifndef LQ_CUDA_THREADS_PER_BLOCK
#define LQ_CUDA_THREADS_PER_BLOCK 128
#endif

namespace {

__device__ __forceinline__ LqThread lq_this_thread() {
    LqThread thread;
    thread.index = static_cast<int>(threadIdx.x);
    thread.count = static_cast<int>(blockDim.x);
    thread.barrier = nullptr;
    return thread;
}

__global__ void lq_decode_kernel(LqPool pool, LqStep step) {
    __shared__ LqDecodeShared shared;
    const int row = static_cast<int>(blockIdx.x) / pool.value_heads;
    const int head = static_cast<int>(blockIdx.x) % pool.value_heads;
    lq_decode_block(pool, step, row, head, lq_this_thread(), &shared);
}

// LQ_FOLD_IN_SHARED (default 1): fold and store keep the head's full-precision
// matrix (V * K floats, 64 KiB at 128 x 128) in the block's shared memory. That
// is more than the 48 KiB a kernel gets by default, so the limit is raised once
// per kernel with cudaFuncSetAttribute. With 0 the matrix stays in the caller's
// scratch memory, as in the first version.
#ifndef LQ_FOLD_IN_SHARED
#define LQ_FOLD_IN_SHARED 1
#endif

#if LQ_FOLD_IN_SHARED
extern __shared__ float lq_block_matrix[];
#define LQ_BLOCK_MATRIX lq_block_matrix
#else
#define LQ_BLOCK_MATRIX nullptr
#endif

__global__ void lq_fold_kernel(LqPool pool, LqFold fold) {
    __shared__ LqFoldShared shared;
    const int row = static_cast<int>(blockIdx.x) / pool.value_heads;
    const int head = static_cast<int>(blockIdx.x) % pool.value_heads;
    lq_fold_block(pool, fold, row, head, lq_this_thread(), &shared, LQ_BLOCK_MATRIX);
}

__global__ void lq_store_kernel(LqPool pool, LqStore store) {
    __shared__ LqFoldShared shared;
    const int row = static_cast<int>(blockIdx.x) / pool.value_heads;
    const int head = static_cast<int>(blockIdx.x) % pool.value_heads;
    lq_store_entry_block(pool, store, row, head, lq_this_thread(), &shared, LQ_BLOCK_MATRIX);
}

__global__ void lq_export_kernel(LqPool pool, LqExport request) {
    __shared__ LqFoldShared shared;
    const int row = static_cast<int>(blockIdx.x) / pool.value_heads;
    const int head = static_cast<int>(blockIdx.x) % pool.value_heads;
    lq_export_block(pool, request, row, head, lq_this_thread(), &shared);
}

// Number of thread blocks for `rows` request rows, or 0 if there is nothing to do.
unsigned int lq_block_count(const LqPool* pool, int rows) {
    return static_cast<unsigned int>(rows) * static_cast<unsigned int>(pool->value_heads);
}

// Bytes of dynamic shared memory the fold and store kernels are launched with.
size_t lq_block_matrix_bytes(const LqPool* pool) {
#if LQ_FOLD_IN_SHARED
    return static_cast<size_t>(pool->value_channels) * (pool->key_channels + 1) * sizeof(float);   // lq_matrix_stride
#else
    (void)pool;
    return 0;
#endif
}

// Raises the kernels' shared-memory limit to what the pool needs.
// The limit is a property of the kernel, not of one launch. cudaFuncSetAttribute
// runs once per kernel per byte count; later launches with the same count skip it.
template <typename Kernel>
bool lq_allow_shared_memory(Kernel kernel, size_t bytes) {
    if (bytes == 0) {
        return true;
    }
    static size_t applied_bytes = static_cast<size_t>(-1);
    if (applied_bytes == bytes) {
        return true;
    }
    const int limit = static_cast<int>(bytes + sizeof(LqFoldShared));
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, limit) != cudaSuccess) {
        return false;
    }
    applied_bytes = bytes;
    return true;
}

int lq_launch_status() {
    return cudaGetLastError() == cudaSuccess ? 0 : LQ_ERROR_CUDA;
}

}  // unnamed namespace

extern "C" {

int lq_cuda_decode_step(const LqPool* pool, const LqStep* step, void* cuda_stream) {
    const int status = lq_check_step(pool, step);
    if (status != 0) {
        return status;
    }
    const unsigned int blocks = lq_block_count(pool, step->rows);
    if (blocks == 0) {
        return 0;
    }
    lq_decode_kernel<<<blocks, LQ_CUDA_THREADS_PER_BLOCK, 0, static_cast<cudaStream_t>(cuda_stream)>>>(*pool, *step);
    return lq_launch_status();
}

int lq_cuda_fold(const LqPool* pool, const LqFold* fold, void* cuda_stream) {
    const int status = lq_check_fold(pool, fold);
    if (status != 0) {
        return status;
    }
    const unsigned int blocks = lq_block_count(pool, fold->rows);
    if (blocks == 0) {
        return 0;
    }
    const size_t shared_bytes = lq_block_matrix_bytes(pool);
    if (!lq_allow_shared_memory(lq_fold_kernel, shared_bytes)) {
        return LQ_ERROR_CUDA;
    }
    lq_fold_kernel<<<blocks, LQ_CUDA_THREADS_PER_BLOCK, shared_bytes, static_cast<cudaStream_t>(cuda_stream)>>>(*pool, *fold);
    return lq_launch_status();
}

int lq_cuda_store(const LqPool* pool, const LqStore* store, void* cuda_stream) {
    const int status = lq_check_store(pool, store);
    if (status != 0) {
        return status;
    }
    const unsigned int blocks = lq_block_count(pool, store->rows);
    if (blocks == 0) {
        return 0;
    }
    const size_t shared_bytes = lq_block_matrix_bytes(pool);
    if (!lq_allow_shared_memory(lq_store_kernel, shared_bytes)) {
        return LQ_ERROR_CUDA;
    }
    lq_store_kernel<<<blocks, LQ_CUDA_THREADS_PER_BLOCK, shared_bytes, static_cast<cudaStream_t>(cuda_stream)>>>(*pool, *store);
    return lq_launch_status();
}

int lq_cuda_export(const LqPool* pool, const LqExport* request, void* cuda_stream) {
    const int status = lq_check_export(pool, request);
    if (status != 0) {
        return status;
    }
    const unsigned int blocks = lq_block_count(pool, request->rows);
    if (blocks == 0) {
        return 0;
    }
    lq_export_kernel<<<blocks, LQ_CUDA_THREADS_PER_BLOCK, 0, static_cast<cudaStream_t>(cuda_stream)>>>(*pool, *request);
    return lq_launch_status();
}

}  // extern "C"
