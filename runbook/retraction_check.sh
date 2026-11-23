#!/usr/bin/env bash
# retraction_check.sh (backlog B12; plan v6 S04/E04): LeapQuant's compressed state under KV-cache pressure. SGLang
# retracts (preempts) running requests when the KV cache runs out and prefills them again later; a retracted request
# must restart from a clean slot, never from a state another request left behind.
#   K0  fp32 + window server: reference answers of the smoke prompts (temperature 0)
#   K1  LeapQuant (compressed state, CUDA kernels), KV cache capped at RETRACT_TOKENS tokens; a background load of
#       long outputs (128 in / 2,048 out, 256 at once) forces retractions while the smoke prompts are answered.
#       PASS: the smoke answers agree with K0 (first 15 words, as every smoke test), the background load completes,
#       and the server log shows retractions.
#   RESULTS=results/retraction runbook/retraction_check.sh        (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
FLAGS="--mamba-ssm-dtype float32 $WINDOW_FLAGS --disable-radix-cache --cuda-graph-max-bs-decode 512"
step() {
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
reference() {
    start_server "$RESULTS/server_reference.log" $FLAGS || return 1
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_reference.json"
}
under_pressure() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    start_server "$RESULTS/server_pressure.log" $FLAGS --enable-linear-leapquant \
        --max-total-tokens "${RETRACT_TOKENS:-200000}" || return 1
    python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port "$PORT" --dataset-name random \
        --random-input-len 128 --random-output-len 2048 --random-range-ratio 1.0 --num-prompts 512 \
        --max-concurrency 256 > "$RESULTS/background_load.log" 2>&1 &
    local load=$!
    sleep 60                      # the load has filled the KV cache
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_pressure.json" \
        --compare "$RESULTS/smoke_reference.json" --max-early-differences 2
    local smoke=$?
    wait "$load"
    local load_status=$?
    grep -E "Successful requests|Output token throughput" "$RESULTS/background_load.log"
    local retracted
    retracted=$(grep -ciE "retract" "$RESULTS/server_pressure.log")
    echo "retraction lines in the server log: $retracted"
    [ "$smoke" -eq 0 ] && [ "$load_status" -eq 0 ] && [ "$retracted" -gt 0 ]
}
step K0 "fp32 + window: reference answers" reference
step K1 "LeapQuant under KV pressure: answers agree, retractions happened" under_pressure
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
