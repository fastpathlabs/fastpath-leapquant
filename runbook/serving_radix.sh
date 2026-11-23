#!/usr/bin/env bash
# serving_radix.sh: serving with SGLang's radix (prefix) cache ON, the default of a production
# server. Restartable, meant to run detached; finished steps are recorded in $RESULTS/STATUS.txt.
#
#   RESULTS=results/run-xyz runbook/serving_radix.sh          (from the repository root)
#
# With the radix cache on, SGLang runs at most one request per three state slots (a working
# slot, a prefix checkpoint, the slot copied from). 512 running requests then need about
# 1,536 slots: 77 GB with the fp32 state (more than the card), 33 GB with LeapQuant's
# compressed state. Each configuration gets the most slots it can hold:
#   R1  fp32 state + window, 560 slots (at most 186 running)            -- as in rounds 2-5
#   R2  bf16 state + window, BF16_SLOTS slots
#   R3  LeapQuant compressed state, fast kernels, 1,560 slots (512 running)
# Step R0 first checks prefix-cache hits on the compressed state: the smoke prompts twice on
# one server (the second time from the cache), against the fp32 window server's answers.
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
SERVING_POINTS="${SERVING_POINTS:-decode:128 decode:512 short:512 decode2k:512}"
export SERVING_POINTS
GRAPHS="--cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
RADIX="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"

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
cache_hits() {
    start_server "$RESULTS/server_radix_window.log" --mamba-ssm-dtype float32 $RADIX $WINDOW \
        --max-running-requests 128 --max-mamba-cache-size 560 &&
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_radix_window.json" || return 1
    stop_server
    export LEAPQUANT_STATE=compressed
    start_server "$RESULTS/server_radix_leapquant.log" --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant \
        --max-running-requests 128 --max-mamba-cache-size 560 || return 1
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_radix_lq_first.json" \
        --compare "$RESULTS/smoke_radix_window.json" --max-early-differences 2 || return 1
    # Same prompts again: the prefixes come from the radix cache (copied compressed states).
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_radix_lq_cached.json" \
        --compare "$RESULTS/smoke_radix_window.json" --max-early-differences 2 || return 1
    grep -iE "cache hit|#cached-token: [1-9]" "$RESULTS/server_radix_leapquant.log" | tail -3
}
lq_compressed() {
    export LEAPQUANT_STATE=compressed
    serve radix_leapquant --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${LQ_SLOTS:-1560}"
}

step R0 "radix cache on: compressed state from cache hits (smoke)" cache_hits
step R1 "serving, radix on: fp32 window, 560 slots" serve radix_fp32_window --mamba-ssm-dtype float32 $RADIX $WINDOW $GRAPHS \
    --max-running-requests 512 --max-mamba-cache-size 560
step R2 "serving, radix on: bf16 window, ${BF16_SLOTS:-1100} slots" serve radix_bf16_window --mamba-ssm-dtype bfloat16 $RADIX $WINDOW \
    $GRAPHS --max-running-requests 512 --max-mamba-cache-size "${BF16_SLOTS:-1100}"
step R3 "serving, radix on: LeapQuant compressed, ${LQ_SLOTS:-1560} slots" lq_compressed
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
