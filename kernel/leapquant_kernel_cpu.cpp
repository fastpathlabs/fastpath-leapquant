// =============================================================================
// leapquant_kernel_cpu.cpp
//
// Runs the kernel blocks of leapquant_kernel_blocks.h on a CPU.
//
// Two modes, chosen with lq_cpu_set_threads_per_block():
//   1 thread per block (default)
//       Every block runs in the calling thread. This is what trace replay and
//       the Python tools use.
//   N threads per block
//       Every block is run by N real threads. They meet at a real barrier
//       wherever the CUDA build calls __syncthreads(). This is slow. It exists
//       so that ThreadSanitizer can check the block for data races the way a
//       GPU would expose them, without a GPU.
// =============================================================================

#define LQ_DEFINE_CHECKS
#include "leapquant_kernel_blocks.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace {

// A reusable barrier: wait() returns in all threads once all of them called it.
struct HostBarrier {
    std::mutex mutex;
    std::condition_variable all_arrived;
    int thread_count = 1;
    int waiting = 0;
    long generation = 0;
};

int g_threads_per_block = 1;
bool g_block_matrix = false;

enum Operation { kDecode, kFold, kStore, kExport };

struct Job {
    Operation operation;
    const LqPool* pool;
    const LqStep* step;
    const LqFold* fold;
    const LqStore* store;
    const LqExport* request;
};

// The memory a GPU block would hold in its shared memory.
struct BlockShared {
    LqDecodeShared decode;
    LqFoldShared fold;
    float* block_matrix = nullptr;   // the block's own V * K matrix, or null (see lq_cpu_set_block_matrix)
};

void run_one_thread_of_block(const Job& job, int row, int head, LqThread thread, BlockShared* shared) {
    switch (job.operation) {
        case kDecode:
            lq_decode_block(*job.pool, *job.step, row, head, thread, &shared->decode);
            break;
        case kFold:
            lq_fold_block(*job.pool, *job.fold, row, head, thread, &shared->fold, shared->block_matrix);
            break;
        case kStore:
            lq_store_entry_block(*job.pool, *job.store, row, head, thread, &shared->fold, shared->block_matrix);
            break;
        case kExport:
            lq_export_block(*job.pool, *job.request, row, head, thread, &shared->fold);
            break;
    }
}

void run_all_blocks(const Job& job, int rows) {
    const int value_heads = job.pool->value_heads;
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < value_heads; ++head) {
            BlockShared shared;
            std::vector<float> block_matrix;
            if (g_block_matrix) {
                // Rows are K + 1 floats apart in a block's own matrix (lq_matrix_stride).
                block_matrix.resize(static_cast<std::size_t>(job.pool->value_channels) * (job.pool->key_channels + 1));
                shared.block_matrix = block_matrix.data();
            }
            if (g_threads_per_block <= 1) {
                LqThread thread;
                thread.index = 0;
                thread.count = 1;
                thread.barrier = nullptr;
                run_one_thread_of_block(job, row, head, thread, &shared);
            } else {
                HostBarrier barrier;
                barrier.thread_count = g_threads_per_block;
                std::vector<std::thread> threads;
                for (int index = 0; index < g_threads_per_block; ++index) {
                    LqThread thread;
                    thread.index = index;
                    thread.count = g_threads_per_block;
                    thread.barrier = &barrier;
                    threads.push_back(std::thread(run_one_thread_of_block, std::cref(job), row, head, thread, &shared));
                }
                for (std::size_t index = 0; index < threads.size(); ++index) {
                    threads[index].join();
                }
            }
        }
    }
}

}  // unnamed namespace

void lq_host_barrier_wait(void* barrier_pointer) {
    HostBarrier* barrier = static_cast<HostBarrier*>(barrier_pointer);
    std::unique_lock<std::mutex> lock(barrier->mutex);
    const long my_generation = barrier->generation;
    barrier->waiting = barrier->waiting + 1;
    if (barrier->waiting == barrier->thread_count) {
        barrier->waiting = 0;
        barrier->generation = barrier->generation + 1;
        barrier->all_arrived.notify_all();
    } else {
        while (barrier->generation == my_generation) {
            barrier->all_arrived.wait(lock);
        }
    }
}

extern "C" {

void lq_cpu_set_block_matrix(int enabled) {
    g_block_matrix = enabled != 0;
}

void lq_cpu_set_threads_per_block(int thread_count) {
    g_threads_per_block = thread_count < 1 ? 1 : thread_count;
}

int lq_cpu_decode_step(const LqPool* pool, const LqStep* step) {
    const int status = lq_check_step(pool, step);
    if (status != 0) {
        return status;
    }
    Job job = {kDecode, pool, step, nullptr, nullptr, nullptr};
    run_all_blocks(job, step->rows);
    return 0;
}

int lq_cpu_fold(const LqPool* pool, const LqFold* fold) {
    const int status = lq_check_fold(pool, fold);
    if (status != 0) {
        return status;
    }
    Job job = {kFold, pool, nullptr, fold, nullptr, nullptr};
    run_all_blocks(job, fold->rows);
    return 0;
}

int lq_cpu_store(const LqPool* pool, const LqStore* store) {
    const int status = lq_check_store(pool, store);
    if (status != 0) {
        return status;
    }
    Job job = {kStore, pool, nullptr, nullptr, store, nullptr};
    run_all_blocks(job, store->rows);
    return 0;
}

int lq_cpu_export(const LqPool* pool, const LqExport* request) {
    const int status = lq_check_export(pool, request);
    if (status != 0) {
        return status;
    }
    Job job = {kExport, pool, nullptr, nullptr, nullptr, request};
    run_all_blocks(job, request->rows);
    return 0;
}

uint16_t lq_cpu_float_to_half(float value) {
    return lq_float_to_half(value);
}

float lq_cpu_half_to_float(uint16_t half_bits) {
    return lq_half_to_float(half_bits);
}

}  // extern "C"
