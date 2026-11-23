#!/usr/bin/env bash
# Writes what a result was produced with: source and binary hashes, build options, GPU, driver,
# compiler and package versions. Called by run_on_cuda.sh before every step.
#   manifest.sh <output file>
# A result whose manifest differs from the current source was not produced by the current source.
set -uo pipefail
HOME_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: manifest.sh <output file>}"
cd "$HOME_DIR"
hash_of() { cat "$@" 2> /dev/null | sha256sum | cut -c1-16; }
{
    echo "time_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host: $(hostname -s)"
    echo "kernel_source_hash: $(hash_of kernel/leapquant_kernel.h kernel/leapquant_kernel_blocks.h kernel/leapquant_kernel_cuda.cu kernel/leapquant_kernel_cpu.cpp)"
    echo "reference_source_hash: $(hash_of leapquant.h leapquant.cpp reference_heads.h reference_heads.cpp)"
    echo "engine_glue_hash: $(hash_of kernel/leapquant_engine.py kernel/leapquant_kernel.py)"
    echo "fast_kernels_hash: $(hash_of kernel/leapquant_triton.py)"
    # The hand-written CUDA fold / store / decode: sources, and the library the run loads (LEAPQUANT_BUILD_DIR).
    echo "fast_cuda_source_hash: $(hash_of kernel/leapquant_fast_fold.cu kernel/leapquant_fast_decode.cu kernel/leapquant_fast_cuda.py)"
    echo "run_scripts_hash: $(hash_of runbook/*.sh)"
    echo "environment: LEAPQUANT_STATE=${LEAPQUANT_STATE:-} LEAPQUANT_FAST=${LEAPQUANT_FAST:-} LEAPQUANT_ALIGN_FOLDS=${LEAPQUANT_ALIGN_FOLDS:-} LEAPQUANT_COMPENSATORS=${LEAPQUANT_COMPENSATORS:-} LEAPQUANT_FOLD_BACKEND=${LEAPQUANT_FOLD_BACKEND:-} LEAPQUANT_DECODE_BACKEND=${LEAPQUANT_DECODE_BACKEND:-} LEAPQUANT_CUDA_DECODE_VARIANT=${LEAPQUANT_CUDA_DECODE_VARIANT:-} LEAPQUANT_STEP_REUSE=${LEAPQUANT_STEP_REUSE:-} LEAPQUANT_LAYERS=${LEAPQUANT_LAYERS:-} LEAPQUANT_FOLD_POLICY=${LEAPQUANT_FOLD_POLICY:-} LEAPQUANT_TRACK_FLUSH=${LEAPQUANT_TRACK_FLUSH:-} LEAPQUANT_FUSED_SLOT_COPY=${LEAPQUANT_FUSED_SLOT_COPY:-}"
    echo "sglang_patch_hash: $(hash_of integrations/sglang/leapquant_phase1.patch)"
    BUILD_DIR="${LEAPQUANT_BUILD_DIR:-$HOME_DIR/build}"
    echo "build_dir: $BUILD_DIR"
    for lib in "$BUILD_DIR"/libleapquant_cuda.so "$BUILD_DIR"/libleapquant_cpu.so "$BUILD_DIR"/libleapquant_reference.so \
               "$BUILD_DIR"/libleapquant_fastfold.so; do
        [ -f "$lib" ] && echo "$(basename "$lib"): $(sha256sum "$lib" | cut -c1-16) built $(date -u -r "$lib" +%Y-%m-%dT%H:%M:%SZ)"
    done
    [ -f build/tuned_options.txt ] && echo "tuned_options: $(cat build/tuned_options.txt)"
    echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2> /dev/null | head -1)"
    NVCC="$(command -v nvcc || echo "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc")"
    echo "nvcc: $("$NVCC" --version 2> /dev/null | tail -1)"
    for venv in "${VENV_SGLANG:-$HOME/venv-sglang}" "${VENV_TRACE:-$HOME/venv-trace}"; do
        [ -x "$venv/bin/python" ] && echo "$(basename "$venv"): $("$venv/bin/python" -c "
import importlib.metadata as m, torch
names = ['torch', 'sglang', 'sglang-kernel', 'triton', 'transformers', 'flashinfer-python']
found = []
for n in names:
    try: found.append(n + ' ' + m.version(n))
    except Exception: pass
print('; '.join(found), '; cuda', torch.version.cuda)" 2> /dev/null)"
    done
    [ -d "${SGLANG_DIR:-$HOME/sglang}/.git" ] && echo "sglang_commit: $(git -C "${SGLANG_DIR:-$HOME/sglang}" rev-parse --short HEAD) with $(git -C "${SGLANG_DIR:-$HOME/sglang}" status --short | wc -l | tr -d ' ') changed files"
    # The checkout's own edits (the integration patch plus later edits such as the layer switch and the fold policy).
    [ -d "${SGLANG_DIR:-$HOME/sglang}/.git" ] && echo "sglang_diff_hash: $(git -C "${SGLANG_DIR:-$HOME/sglang}" diff | sha256sum | cut -c1-16)"
    echo "model: ${MODEL:-Qwen/Qwen3.5-9B}"
} > "$OUT"
