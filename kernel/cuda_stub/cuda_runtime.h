// Minimal stand-in for the CUDA headers, for a syntax check only.
#pragma once
struct __cuda_dim3 { unsigned x, y, z; };
__attribute__((device)) extern const __cuda_dim3 threadIdx, blockIdx, blockDim, gridDim;
__attribute__((device)) void __syncthreads();
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
enum { cudaSuccess = 0 };
cudaError_t cudaGetLastError();
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
int cudaConfigureCall(dim3 grid, dim3 block, unsigned long shared = 0, cudaStream_t stream = 0);
#define __shared__ __attribute__((shared))
#define __forceinline__ inline __attribute__((always_inline))
// Device versions of the math functions the kernel uses (the real toolkit provides these).
__attribute__((device)) float expf(float);
__attribute__((device)) float logf(float);
__attribute__((device)) float sqrtf(float);
__attribute__((device)) float fabsf(float);
__attribute__((device)) float nearbyintf(float);
__attribute__((device)) float copysignf(float, float);
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8 };
template <typename Kernel> cudaError_t cudaFuncSetAttribute(Kernel, cudaFuncAttribute, int);
__attribute__((device)) float __uint_as_float(unsigned int);
__attribute__((device)) unsigned int __float_as_uint(float);
// memcpy is available in device code in CUDA (the real headers declare it for both sides).
extern "C" __attribute__((device)) void* memcpy(void*, const void*, unsigned long);
