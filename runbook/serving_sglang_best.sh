#!/usr/bin/env bash
# serving_sglang_best.sh (backlog B34; state-cache plan TC.6, decision N3): LeapQuant against SGLang's OWN best state
# settings with the radix cache on: bf16 state, lazy extra-buffer slots (--mamba-radix-cache-strategy
# extra_buffer_lazy) and the int8 checkpoint pool (--enable-int8-mamba-checkpoint), with the window kernel if SGLang
# accepts it with that strategy, else its packed kernel (the log says which). Shapes of round 8 / round 9 radix on.
#   RESULTS=results/sglang-best runbook/serving_sglang_best.sh        (from the repository root)
# BEST_EXTRA: more server flags for SGLang's arm (14 Nov: its packed fallback ran out of memory at start with
# SGLang's own slot count; e.g. BEST_EXTRA="--max-mamba-cache-size 1100 --mem-fraction-static 0.82").
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
export SERVING_POINTS="${SERVING_POINTS:-decode:128 decode:512 short:512 decode2k:512}"
GRAPHS="--cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
BEST="--mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy --enable-int8-mamba-checkpoint"
step() {
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
sglang_best() {
    if start_server "$RESULTS/server_best_window.log" $BEST $WINDOW $GRAPHS --max-running-requests 512 ${BEST_EXTRA:-}; then
        echo "SGLang best settings WITH the window kernel"
        run_serving_client best_bf16_lazy_int8_window
    else
        pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
        echo "the window kernel refused these settings (see server_best_window.log); packed kernel instead"
        start_server "$RESULTS/server_best_packed.log" $BEST $GRAPHS --max-running-requests 512 ${BEST_EXTRA:-} || return 1
        run_serving_client best_bf16_lazy_int8_packed
    fi
}
leapquant() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LQ_DECODE:-cuda}"
    start_server "$RESULTS/server_best_leapquant.log" --mamba-ssm-dtype float32 \
        --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule $WINDOW --enable-linear-leapquant $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${LQ_SLOTS:-1560}" || return 1
    run_serving_client best_leapquant_cuda
}
step G1 "radix on: SGLang's best state settings (bf16, lazy slots, int8 checkpoints)" sglang_best
step G2 "radix on: LeapQuant compressed, CUDA kernels, ${LQ_SLOTS:-1560} slots" leapquant
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
