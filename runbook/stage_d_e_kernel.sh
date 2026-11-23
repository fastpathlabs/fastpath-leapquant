#!/usr/bin/env bash
# Stage D. The GPU kernel against the reference, on the traces of stage C.
# Stage E. Kernel speed.
#
#   ./stage_d_e_kernel.sh d      verify the GPU kernel on every recorded trace
#   ./stage_d_e_kernel.sh e      per-step latency at batch 32, 128, 512, next to SGLang's kernels
#
# Stage D pass criterion: every report ends with "RESULT: PASS". Read the header
# of kernel/verify_kernel.py for what is compared and why it is not the plan's
# "1e-4 on every token" (no GPU kernel can meet that; the script's criteria can
# tell a correct kernel from a wrong one).
#
# This is the first time the CUDA build runs. If verify fails, first run
#     python3 kernel/engine_test.py --device cuda
# and try the build with fused multiply-add as nvcc would do it by default:
#     make clean-cuda cuda FMAD=true
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEP="${1:?usage: stage_d_e_kernel.sh d|e}"
cd "$LEAPQUANT_HOME"
TRACES="${TRACES:-$LEAPQUANT_HOME/traces}"

case "$STEP" in
d)
    source "$VENV_TRACE/bin/activate"
    python3 kernel/engine_test.py --device cuda 2>&1 | tee "$RESULTS/d_engine_test_cuda.log"
    status=0
    for trace in "$TRACES"/*/trace_layer_*.bin; do
        label="$(basename "$(dirname "$trace")")_$(basename "$trace" .bin)"
        python3 kernel/verify_kernel.py "$trace" --device cuda --rows 2 2>&1 | tee "$RESULTS/d_verify_${label}.txt" || status=1
    done
    grep -H "RESULT" "$RESULTS"/d_verify_*.txt
    exit $status
    ;;
e)
    # Runs in the SGLang environment so that SGLang's kernels can be timed in the same harness.
    source "$VENV_SGLANG/bin/activate"
    for dtype in fp32 bf16; do
        python3 kernel/bench_kernel.py --device cuda --batch-sizes 32 128 512 --dtype "$dtype" --with-sglang \
            2>&1 | tee "$RESULTS/e_kernel_speed_${dtype}.log"
    done
    ;;
*)
    echo "unknown step $STEP"; exit 2 ;;
esac
