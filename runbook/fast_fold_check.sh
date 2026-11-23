#!/usr/bin/env bash
# fast_fold_check.sh: the hand-written CUDA fold (kernel/leapquant_fast_fold.cu) on the GPU.
#   F1  fast_fold_test.py under compute-sanitizer memcheck and racecheck (2 heads per row)
#   F2  fast_fold_test.py, 32 heads: CUDA fold against the Triton fold, 9 cases
#   F3  bench_triton.py, batch 32/128/512, fold backend triton and cuda alternated twice (ABAB):
#       fold loss against the exact kernel, decode and fold times
#   F4  verify_kernel.py --fast on the 9 prompt traces with the CUDA fold; set verdict (summarize_verify.py)
#   F5  engine glue with the CUDA fold (engine_test.py --fast, compressed_test.py)
#   OUT=results/fast_fold_v1 runbook/fast_fold_check.sh        (from the repository root, with make fast-fold done)
set -u
cd "$(dirname "$0")/.."
source "${VENV_SGLANG:-$HOME/venv-sglang}/bin/activate"
export LEAPQUANT_BUILD_DIR="$PWD/build"
OUT="${OUT:-$PWD/results/fast_fold_v1}"
mkdir -p "$OUT"
runbook/manifest.sh "$OUT/manifest.txt"
cp build/fast_fold_resources.txt "$OUT/" 2>/dev/null
SANITIZER="${CUDA_HOME:-/usr/local/cuda}/bin/compute-sanitizer"
TRACES="${TRACES:-$HOME/leapquant/traces}"
step() {           # step ID description command...
    local id="$1" what="$2"; shift 2
    local start=$SECONDS
    if "$@" > "$OUT/$id.log" 2>&1; then
        echo "$id PASS $((SECONDS - start)) s  $what" >> "$OUT/STATUS.txt"
    else
        echo "$id FAIL $((SECONDS - start)) s  $what" >> "$OUT/STATUS.txt"
    fi
}
step F1a "compute-sanitizer memcheck, fast_fold_test --quick" \
    "$SANITIZER" --tool memcheck --error-exitcode 1 python3 kernel/fast_fold_test.py --quick
step F1b "compute-sanitizer racecheck (shared memory) of the CUDA fold, fast_fold_test --quick" \
    "$SANITIZER" --tool racecheck --racecheck-report all --kernel-name kns=fast_fold_kernel --error-exitcode 1 \
    python3 kernel/fast_fold_test.py --quick
step F2 "CUDA fold against Triton fold, 9 cases, 32 heads" python3 kernel/fast_fold_test.py
for round in 1 2; do
    for backend in triton cuda; do
        step "F3_${backend}_$round" "bench_triton batch 32/128/512, fold backend $backend, round $round" \
            python3 kernel/bench_triton.py --batch-sizes 32 128 512 --windows 2 --no-exact-times --fold-backend "$backend"
    done
done
step F3x "bench_triton with exact times, fold backend cuda" \
    python3 kernel/bench_triton.py --batch-sizes 128 512 --windows 2 --fold-backend cuda
verify_one() {     # trace label
    LEAPQUANT_FOLD_BACKEND=cuda python3 kernel/verify_kernel.py "$1" --device cuda --rows 1 --fast > "$OUT/verify_$2.txt" 2>&1
}
for trace in "$TRACES"/{explain,code,reasoning}/trace_layer_{00,13,30}.bin; do
    label="$(basename "$(dirname "$trace")")_$(basename "$trace" .bin)"
    verify_one "$trace" "$label" &
    while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 2; done
done
wait
step F4 "verify --fast with the CUDA fold, 9 traces, set verdict" python3 kernel/summarize_verify.py "$OUT"/verify_*.txt
step F5a "engine glue, fast kernels, CUDA fold" env LEAPQUANT_FOLD_BACKEND=cuda python3 kernel/engine_test.py --device cuda --fast
step F5b "compressed state, CUDA fold" env LEAPQUANT_FOLD_BACKEND=cuda python3 kernel/compressed_test.py
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/STATUS.txt"
