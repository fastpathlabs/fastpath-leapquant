#!/usr/bin/env bash
# Open-loop service sweep (plan v6 E02, backlog B28): requests arrive at a fixed Poisson rate (so a slower server
# does not get a lighter load), at rising rates up to the capacity boundary; goodput counts only requests that meet
# the SLO (time to first token <= OL_TTFT ms, time per output token <= OL_TPOT ms). Shape: AA's 1,000 in / 1,000 out.
# Arms: bf16 + window; LeapQuant (compressed state, CUDA fold and store) with folds aligned (default) and with
# folds on each request's own step (LEAPQUANT_ALIGN_FOLDS=0: with natural arrivals, no shared fold step).
# No client-side concurrency cap: requests beyond the server's 512 running wait in the server's queue, and that wait
# is part of their time to first token (a client cap would start the clock only after the wait).
#   RESULTS=results/openloop runbook/serving_openloop.sh            (from the repository root)
# Output: openloop_<arm>_rate<R>.log per point (the client's report, goodput included).
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
RATES="${OL_RATES:-2 4 6 8 10 12}"
PROMPTS="${OL_PROMPTS:-600}"
SLO="ttft:${OL_TTFT:-2000} tpot:${OL_TPOT:-50}"
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
sweep() {          # sweep ARM server-flags...
    local arm="$1"; shift
    start_server "$RESULTS/server_openloop_$arm.log" "$@" || return 1
    for rate in $RATES; do
        python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port "$PORT" \
            --dataset-name random --random-input-len 1000 --random-output-len 1000 --random-range-ratio 1.0 \
            --num-prompts "$PROMPTS" --request-rate "$rate" --warmup-requests 3 --flush-cache \
            --seed 1 --goodput $SLO 2>&1 | tee "$RESULTS/openloop_${arm}_rate${rate}.log"
    done
}
bf16() { sweep bf16_window --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $FLAGS; }
lq_aligned() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}" LEAPQUANT_ALIGN_FOLDS=1
    sweep leapquant_aligned --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS
}
lq_own_step() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}" LEAPQUANT_ALIGN_FOLDS=0
    sweep leapquant_own_step --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS
}
step O1 "open loop: bf16 + window" bf16
step O2 "open loop: LeapQuant, folds aligned" lq_aligned
step O3 "open loop: LeapQuant, folds on each request's own step" lq_own_step
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
