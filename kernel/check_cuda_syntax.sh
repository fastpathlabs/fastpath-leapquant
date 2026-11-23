#!/usr/bin/env bash
# Parses leapquant_kernel_cuda.cu with clang's CUDA front end against the
# stand-in headers in kernel/cuda_stub/. It checks syntax, types and the
# host/device call rules. It does not generate GPU code and proves nothing
# about the real toolkit's headers. Use it on a machine without nvcc.
set -euo pipefail
cd "$(dirname "$0")"
FLAGS=(-x cuda -fsyntax-only -nocudainc -nocudalib -std=c++17 -Wall -Wextra -D__CUDACC__
       -D__device__='__attribute__((device))' -D__global__='__attribute__((global))'
       -D__host__='__attribute__((host))' -Icuda_stub -I.)
# Both builds: the default one and the one with 8-byte int8 loads. A failure stops the script
# with a non-zero status (it used to print "no errors" after a failing first build).
clang++ "${FLAGS[@]}" leapquant_kernel_cuda.cu
clang++ "${FLAGS[@]}" -DLQ_WORD_LOADS leapquant_kernel_cuda.cu
echo "leapquant_kernel_cuda.cu: no errors from clang's CUDA front end"
