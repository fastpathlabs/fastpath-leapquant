#!/usr/bin/env bash
# Round 10: concurrency sweeps in the style of the public inference benchmarks (system throughput against speed per
# user), LeapQuant (compressed state, fast decode, CUDA fold and store) against bf16 + window, paired:
#   aa1k     1,000 in / 1,000 out, concurrency 1 .. 512: Artificial Analysis's system load test shape
#   ix8k1k   8,192 in / 1,024 out, concurrency 1 .. 256: SemiAnalysis InferenceX's single-turn 8k/1k shape
# Radix cache off and 560 state slots for both, so every point really runs its concurrency. Labels
# off_<arm>_r<rep>, read by reports/generate_reports.py (paired section and sweep charts).
#   RESULTS=results/round10 REPS=1 runbook/serving_round10.sh       (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
REPS="${REPS:-1}"
export SERVING_POINTS="${SERVING_POINTS:-aa1k:1 aa1k:4 aa1k:16 aa1k:64 aa1k:128 aa1k:256 aa1k:512 ix8k1k:1 ix8k1k:16 ix8k1k:64 ix8k1k:128 ix8k1k:256}"
FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
step() {
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
serve() { local label="$1"; shift; start_server "$RESULTS/server_$label.log" "$@" && run_serving_client "$label"; }
bf16() { serve "off_bf16_window_r$1" --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $FLAGS; }
fp32() { serve "off_fp32_packed_r$1" --mamba-ssm-dtype float32 $COMMON_FLAGS $FLAGS; }   # SGLang's default state: the paper's baseline
lq() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    serve "off_leapquant_cuda_r$1" --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS
}
for rep in $(seq 1 "$REPS"); do
    step "S$rep" "sweep: bf16 state, window kernel, repetition $rep" bf16 "$rep"
    step "T$rep" "sweep: LeapQuant, compressed state, CUDA fold and store, repetition $rep" lq "$rep"
done
step U1 "sweep: fp32 state, SGLang's packed kernel (the default; the paper compares against fp32)" fp32 1
if grep -l "does not load" "$RESULTS"/server_*leapquant*.log 2> /dev/null; then
    echo "WARNING: a LeapQuant server fell back to the Triton fold" >> "$STATUS"
fi
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
