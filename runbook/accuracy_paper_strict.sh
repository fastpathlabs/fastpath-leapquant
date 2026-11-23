#!/usr/bin/env bash
# accuracy_paper_strict.sh: the paper lane's LeapQuant and bf16-window arms with no retraction possible (backlog B58).
# In results/accuracy-paper the KV cache (495,468 tokens) ran full and SGLang retracted requests 516 times in the fp32
# arm's first 3 hours (~2 per answer). A retracted request's KV and state are dropped and rebuilt by prefill when it is
# readmitted, which resets a bf16-window or LeapQuant request's accumulated decode error: long answers would be tested
# in stretches instead of over up to 81,920 tokens. (The fp32 reference is unaffected: an fp32 rebuild is the same
# state up to rounding.) Here:
#   - the state pool holds only the running requests (MAMBA_SLOTS 24, RUNNING_REQUESTS 16; it was 560 slots, 26 GB of
#     fp32 state at 2-3% use), which leaves the memory to the KV cache;
#   - the client never runs more requests than fit at full length: concurrency = min(RUNNING_REQUESTS,
#     KV tokens / (PROMPT_BUDGET + max tokens)), the KV size read from the server log (PROMPT_BUDGET 1,024 tokens; the
#     AIME prompts are a few hundred);
#   - after each arm the server log is checked: 0 retractions, otherwise a WARNING line in STATUS.txt.
# Same questions, sampling and judging as runbook/accuracy_paper.sh; the reference is that run's fp32 window answers.
# Restartable like it: finished steps are recorded in $RESULTS/STATUS.txt and skipped on a rerun.
#
#   RESULTS=results/accuracy-strict REFERENCE=results/accuracy-paper/paper_fp32_window.json \
#       runbook/accuracy_paper_strict.sh                                  (from the repository root)
set -u
cd "$(dirname "$0")/.."
export RUNNING_REQUESTS="${RUNNING_REQUESTS:-16}" MAMBA_SLOTS="${MAMBA_SLOTS:-24}"
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
REFERENCE="${REFERENCE:-results/accuracy-paper/paper_fp32_window.json}"
if [ ! -s "$REFERENCE" ]; then
    echo "no reference run: $REFERENCE" | tee -a "$STATUS"
    exit 1
fi
TASKS="${PAPER_TASKS:-aime2025,aime2024}"
MAX_TOKENS="${PAPER_MAX_TOKENS:-81920}"
PROMPT_BUDGET="${PROMPT_BUDGET:-1024}"
BENCH="python3 eval/accuracy_bench.py --port $PORT --tasks $TASKS --limit 100000 --max-tokens $MAX_TOKENS
    --samples ${PAPER_SAMPLES:-4} --temperature ${PAPER_TEMPERATURE:-1.0} --top-p ${PAPER_TOP_P:-1.0}
    --seed ${PAPER_SEED:-1234} --thinking on --keep-text --timeout 14400"
BENCH="$(echo $BENCH)"
SERVE_FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-$RUNNING_REQUESTS}"

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
kv_tokens() { grep -o "max_total_num_tokens=[0-9]*" "$1" | tail -1 | cut -d= -f2; }
arm() {            # arm NAME SERVER-FLAGS...: server, tasks at the retraction-free concurrency, retraction count
    local name="$1"; shift
    local log="$RESULTS/server_strict_$name.log"
    start_server "$log" "$@" || return 1
    if [ -n "${ARM_REQUIRE:-}" ] && ! grep -q "$ARM_REQUIRE" "$log"; then
        echo "not in the server log: $ARM_REQUIRE"
        return 1
    fi
    if grep -q "does not load" "$log"; then
        echo "WARNING: $name: a CUDA library did not load (Triton fallback); see $log" | tee -a "$STATUS"
    fi
    local kv concurrency cap
    kv=$(kv_tokens "$log")
    if [ -z "$kv" ]; then echo "no max_total_num_tokens in $log"; return 1; fi
    concurrency=$(( kv / (PROMPT_BUDGET + MAX_TOKENS) ))
    [ "$concurrency" -le "$RUNNING_REQUESTS" ] || concurrency=$RUNNING_REQUESTS
    cap=$(grep -o "max_running_requests=[0-9]*" "$log" | tail -1 | cut -d= -f2)   # the server's own cap
    if [ -n "$cap" ] && [ "$concurrency" -gt "$cap" ]; then concurrency=$cap; fi
    echo "KV cache $kv tokens, at most $((PROMPT_BUDGET + MAX_TOKENS)) per request: concurrency $concurrency" \
        | tee "$RESULTS/strict_${name}_concurrency.txt"
    if [ "$concurrency" -lt "${MIN_CONCURRENCY:-8}" ]; then
        echo "concurrency $concurrency is below ${MIN_CONCURRENCY:-8}"
        return 1
    fi
    $BENCH --concurrency "$concurrency" --out "$RESULTS/paper_strict_$name.json" --compare "$REFERENCE" \
        | tee "$RESULTS/paper_strict_${name}_vs_fp32.txt"
    local retractions
    retractions=$(grep -c "#retracted_reqs" "$log")
    echo "$retractions" > "$RESULTS/strict_${name}_retractions.txt"
    echo "retractions: $retractions"
    if [ "$retractions" != "0" ]; then
        echo "WARNING: $name: $retractions retractions, the arm is not retraction-free" >> "$STATUS"
    fi
    test -s "$RESULTS/paper_strict_$name.json"
}
leapquant() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    export ARM_REQUIRE="LeapQuant decode state allocated"
    arm leapquant --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $SERVE_FLAGS
}
bf16_window() {
    arm bf16_window --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $SERVE_FLAGS
}
compare_arms() {
    python3 eval/accuracy_bench.py --report "$RESULTS/paper_strict_leapquant.json" \
        --compare "$RESULTS/paper_strict_bf16_window.json" | tee "$RESULTS/paper_strict_leapquant_vs_bf16_window.txt"
    true
}
step S1 "retraction-free: LeapQuant, compressed state, CUDA fold, against the fp32 window ($REFERENCE)" leapquant
step S2 "retraction-free: bf16 window against the fp32 window" bf16_window
step S3 "retraction-free: LeapQuant against bf16 window" compare_arms
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
