#!/usr/bin/env bash
# fast_checks.sh: checks of the fast (Triton) kernels on the GPU machine. Restartable, meant
# to run detached (tmux). Writes $OUT/STATUS.txt, one line per check, and a log per check.
#
#   runbook/fast_checks.sh [results-folder]          (from the repository root)
#
#   F01-F03  engine glue with the fast kernels (engine_test, block_engine_test), and the exact
#            kernels on the GPU for comparison
#   F04      kernel/bench_triton.py at batch 32, 128, 512 with SGLang's kernels for comparison
#   F05...   kernel/verify_kernel.py --fast on every recorded trace (criteria A, B, C)
set -u
cd "$(dirname "$0")/.."
OUT="${1:-results/fast_checks}"
TRACES="${TRACES:-$HOME/leapquant/traces}"
VENV="${VENV_SGLANG:-$HOME/venv-sglang}"
PARALLEL="${PARALLEL:-4}"
mkdir -p "$OUT"
STATUS="$OUT/STATUS.txt"
touch "$STATUS"
export LEAPQUANT_BUILD_DIR="$PWD/build"
source "$VENV/bin/activate"
runbook/manifest.sh "$OUT/manifest.txt"

check() {          # check ID "description" command...
    local id="$1" description="$2"; shift 2
    if grep -q "^$id PASS" "$STATUS"; then return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$OUT/$id.log" 2>&1; then verdict=PASS; fi
    sed -i "/^$id /d" "$STATUS"
    echo "$id $verdict $(( $(date +%s) - start )) s  $description" >> "$STATUS"
}

check F01 "engine glue, fast kernels"           python3 kernel/engine_test.py --device cuda --fast
check F02 "block engine glue, fast kernels"     python3 kernel/block_engine_test.py --device cuda --fast
check F03 "engine glue, exact kernels on GPU"   python3 kernel/engine_test.py --device cuda --fused
check F03c "engine glue, compressed state (phase 2)" python3 kernel/compressed_test.py
check F04 "speed and agreement, batch 32/128/512" python3 kernel/bench_triton.py --batch-sizes 32 128 512 --windows 2 --with-sglang

# Every trace, PARALLEL at a time (the reference core runs on the CPU).
verify() {         # verify ID trace
    local id="$1" trace="$2" name
    name="$(basename "$(dirname "$trace")")_$(basename "$trace" .bin)"
    check "$id" "verify --fast $name" bash -c "python3 kernel/verify_kernel.py '$trace' --device cuda --rows 1 --fast > '$OUT/verify_$name.txt' 2>&1; tail -1 '$OUT/verify_$name.txt'; grep -q 'RESULT: PASS' '$OUT/verify_$name.txt'"
}
number=5
running=0
for trace in "$TRACES"/explain/trace_layer_*.bin "$TRACES"/code/trace_layer_*.bin "$TRACES"/reasoning/trace_layer_*.bin \
             "$TRACES"/all_layers/trace_layer_*.bin; do
    case "$trace" in *.outputs.bin) continue ;; esac
    [ -f "$trace" ] || continue
    verify "$(printf 'F%02d' "$number")" "$trace" &
    number=$((number + 1))
    running=$((running + 1))
    if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running - 1)); fi
done
wait
# Criterion C over the whole set of traces (see kernel/summarize_verify.py).
check F99 "criteria A, B per trace and C over all traces" python3 kernel/summarize_verify.py "$OUT"/verify_*.txt
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
