#!/usr/bin/env bash
# serving_fast.sh: end-to-end round with the fast (Triton) LeapQuant kernels. Restartable, meant
# to run detached (tmux): every finished step is recorded in $RESULTS/STATUS.txt and skipped
# on a rerun.
#
#   RESULTS=results/run-xyz runbook/serving_fast.sh          (from the repository root)
#
# 1. Smoke: the fp32 window server, then LeapQuant with the fp32 state mirrored (phase 1), then
#    LeapQuant with the compressed state only (phase 2): same prompts at temperature 0.
# 2. Serving, all servers with the radix cache off (so the server runs as many requests as it
#    has state slots: SGLang caps running requests at slots / 3 with the radix cache on), state
#    slots for 512 running requests and CUDA graphs up to batch 512. Configurations:
#      config1  fp32 state, SGLang's packed decode kernel
#      config2  fp32 state, SGLang's window (ReplaySSM) decode kernel
#      config3  bf16 state, SGLang's window decode kernel
#      config4  LeapQuant, compressed state only, fast kernels
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
# Shapes: see run_serving_client in runbook/env.sh. Round 6 used "short:64 short:512 long:128";
# round 7 adds the decode-heavy shapes, where the decode kernel is most of the time.
SERVING_POINTS="${SERVING_POINTS:-decode:128 decode:512 short:512 decode2k:512 long:128}"
export SERVING_POINTS
SERVE_FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}

smoke_window() {
    start_server "$RESULTS/server_smoke_window.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS &&
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_window.json"
}
smoke_mirror() {
    export LEAPQUANT_STATE=mirror
    start_server "$RESULTS/server_smoke_lq_mirror.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS \
        --enable-linear-leapquant --mem-fraction-static "${LQ_MEM_FRACTION:-0.70}" &&
    grep "LeapQuant decode state allocated" "$RESULTS/server_smoke_lq_mirror.log" &&
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_lq_mirror.json" \
        --compare "$RESULTS/smoke_window.json" --max-early-differences 2
}
smoke_compressed() {
    export LEAPQUANT_STATE=compressed
    start_server "$RESULTS/server_smoke_lq_compressed.log" --mamba-ssm-dtype float32 \
        $WINDOW_FLAGS --enable-linear-leapquant --disable-radix-cache &&
    grep "LeapQuant decode state allocated" "$RESULTS/server_smoke_lq_compressed.log" &&
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_lq_compressed.json" \
        --compare "$RESULTS/smoke_window.json" --max-early-differences 2
}
serve() {          # serve LABEL server-flags...
    local label="$1"; shift
    start_server "$RESULTS/server_$label.log" "$@" && run_serving_client "$label"
    local status=$?
    grep -E "max_total_num_tokens|max_running_requests|LeapQuant decode state|KV Cache is allocated|Mamba Cache is allocated" \
        "$RESULTS/server_$label.log" | head -8 > "$RESULTS/server_${label}_memory.txt"
    return $status
}

serve_leapquant_compressed() {
    export LEAPQUANT_STATE=compressed
    serve config4_leapquant --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $SERVE_FLAGS
}

step S1 "smoke: fp32 window server"                      smoke_window
step S2 "smoke: LeapQuant, fp32 state mirrored (phase 1)" smoke_mirror
step S3 "smoke: LeapQuant, compressed state (phase 2)"    smoke_compressed
step P1 "serving config1: fp32, packed kernel"      serve config1_fp32 --mamba-ssm-dtype float32 $COMMON_FLAGS $SERVE_FLAGS
step P2 "serving config2: fp32, window kernel"      serve config2_fp32_window --mamba-ssm-dtype float32 $WINDOW_FLAGS $SERVE_FLAGS
step P3 "serving config3: bf16, window kernel"      serve config3_bf16_window --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $SERVE_FLAGS
step P4 "serving config4: LeapQuant compressed, fast kernels" serve_leapquant_compressed
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
