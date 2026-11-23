#!/usr/bin/env bash
# Round 9: LeapQuant (compressed state, fast decode, hand-written CUDA fold) against bf16 + window (SGLang's 16-bit
# state with its window decode kernel: the bar of plan 5's G3 and of the state-cache plan's decision N3).
# Paired and interleaved: for each repetition bf16 window, then LeapQuant, radix cache off (round 7's shapes, 560
# slots each) and on (round 8's shapes, 1,100 bf16 slots against 1,560 LeapQuant slots: what each fits).
#   RESULTS=results/round9 REPS=2 runbook/serving_round9.sh        (from the repository root, make fast-fold done)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
REPS="${REPS:-2}"
GRAPHS="--cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
RADIX="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
OFF_POINTS="${OFF_POINTS:-decode:128 decode:512 short:512 decode2k:512 long:128}"
ON_POINTS="${ON_POINTS:-decode:128 decode:512 short:512 decode2k:512}"

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
serve() {          # serve LABEL server-flags...
    local label="$1"; shift
    start_server "$RESULTS/server_$label.log" "$@" && run_serving_client "$label"
    local status=$?
    grep -E "max_total_num_tokens|LeapQuant decode state|KV Cache is allocated|Mamba Cache is allocated" \
        "$RESULTS/server_$label.log" | grep -v server_args | head -8 > "$RESULTS/server_${label}_memory.txt"
    return $status
}
off_bf16() { SERVING_POINTS="$OFF_POINTS" serve "off_bf16_window_r$1" --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS --disable-radix-cache $GRAPHS; }
off_lq() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}" SERVING_POINTS="$OFF_POINTS"
    serve "off_leapquant_cuda_r$1" --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant --disable-radix-cache $GRAPHS
}
on_bf16() {
    SERVING_POINTS="$ON_POINTS" serve "on_bf16_window_r$1" --mamba-ssm-dtype bfloat16 $RADIX $WINDOW $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${BF16_SLOTS:-1100}"
}
on_lq() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}" SERVING_POINTS="$ON_POINTS"
    serve "on_leapquant_cuda_r$1" --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${LQ_SLOTS:-1560}"
}
for rep in $(seq 1 "$REPS"); do
    if [ -z "${ONLY_ON:-}" ]; then       # ONLY_ON=1: the radix-on half only (e.g. with LQ_SLOTS set to equal memory)
        step "A$rep" "radix off: bf16 window, repetition $rep" off_bf16 "$rep"
        step "B$rep" "radix off: LeapQuant compressed, CUDA fold, repetition $rep" off_lq "$rep"
    fi
    step "C$rep" "radix on: bf16 window, ${BF16_SLOTS:-1100} slots, repetition $rep" on_bf16 "$rep"
    step "D$rep" "radix on: LeapQuant compressed, CUDA fold, ${LQ_SLOTS:-1560} slots, repetition $rep" on_lq "$rep"
done
# The server must really have run the CUDA fold: its import of leapquant_fast_cuda would have warned otherwise.
if grep -l "does not load" "$RESULTS"/server_*leapquant_cuda*.log 2> /dev/null; then
    echo "WARNING: a LeapQuant server fell back to the Triton fold" >> "$STATUS"
fi
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
