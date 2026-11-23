#!/usr/bin/env bash
# harness_sanitizers.sh: compute-sanitizer on the FINAL kernels of the serving candidate (lqdec8 build_v9: CUDA fold,
# CUDA store, CUDA decode variant 9) and on the full-cycle graph path (prompt rev. 3, section 14; backlog B15).
# memcheck, initcheck and racecheck; batches 1, 32 and 512 (decode, full cycle) and the fold/store test cases.
# Every log is recorded in sanitizer_results.json with its command, exit status and error classification
# (python -m lqh.sanitizer): an exit code alone is not taken as a memory violation, nor its absence as a pass.
#   RESULTS=$HOME/lqexp/results/harness/sanitizers runbook/harness_sanitizers.sh     (from the tree root)
set -u
cd "$(dirname "$0")/.."
source "${VENV_SGLANG:-$HOME/venv-sglang}/bin/activate"
mkdir -p "$RESULTS"
runbook/manifest.sh "$RESULTS/manifest.txt"
SAN=/usr/local/cuda/bin/compute-sanitizer
"$SAN" --version > "$RESULTS/tool_version.txt" 2>&1
export LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda LEAPQUANT_CUDA_DECODE_VARIANT=9
OUT="$RESULTS/sanitizer_results.json"
STARTED=$(date +%s)
check() {          # check NAME TOOL "coverage" sanitizer-options... -- program...
    local name="$1" tool="$2" coverage="$3"; shift 3
    local command="$SAN --tool $tool $*"
    # BUDGET_S: checks not started after this many seconds are recorded as not run (never as passed)
    if [ $(( $(date +%s) - STARTED )) -ge "${BUDGET_S:-1000000}" ]; then
        echo "$name not_run (budget of ${BUDGET_S} s used up) $(date -u +%H:%M)" >> "$RESULTS/STATUS.txt"; return 0
    fi
    timeout "${SAN_TIMEOUT:-2400}" "$SAN" --tool "$tool" "$@" > "$RESULTS/$name.txt" 2>&1
    local status=$?
    echo "$name exit $status $(date -u +%H:%M)" >> "$RESULTS/STATUS.txt"
    PYTHONPATH=harness python3 -m lqh.sanitizer record "$RESULTS/$name.txt" --tool "$tool" --command "$command" \
        --exit "$status" --manifest "$RESULTS/manifest.txt" --coverage "$coverage" --append "$OUT"
}
FOLDTEST="python3 kernel/fast_fold_test.py --quick"
for batch in ${SAN_BATCHES:-1 32 512}; do
    SMALL="python3 kernel/bench_triton.py --batch-sizes $batch --windows 1 --no-exact-times --repeats 2 --fold-backend cuda --decode-backend cuda --full-cycle"
    check "memcheck_decode_cycle_b$batch" memcheck "decode v9 at every window position, fold, store, export and the full-cycle graph, batch $batch" \
        --report-api-errors no --error-exitcode 1 $SMALL
    check "racecheck_decode_b$batch" racecheck "decode v9 shared memory, batch $batch" \
        --racecheck-report all --kernel-name kns=decode8_kernel --error-exitcode 1 $SMALL
done
check memcheck_fold_store memcheck "CUDA fold and store, fast_fold_test quick cases" --report-api-errors no --error-exitcode 1 $FOLDTEST
check initcheck_fold_store initcheck "CUDA fold and store, uninitialised global reads" --error-exitcode 1 $FOLDTEST
check racecheck_fold racecheck "CUDA fold shared memory" --racecheck-report all --kernel-name kns=fast_fold_kernel --error-exitcode 1 $FOLDTEST
PYTHONPATH=harness python3 - "$OUT" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1]))
for r in doc["records"]:
    print(f"{r['log'].split('/')[-1]:34s} {r['tool']:9s} exit {r['exit_status']}  {r['classification']:16s} {r.get('summary')}")
EOF
