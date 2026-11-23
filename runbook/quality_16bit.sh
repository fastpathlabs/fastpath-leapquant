#!/usr/bin/env bash
# quality_16bit.sh: test-harness order of work, step 2 (prompt rev. 3, section 13): qualify the 16-bit control on the
# paper's tasks before any other quality work. Four arms on the same questions, the same seeds and sampling:
#   fp32          FP32 state, SGLang's packed kernel (the reference)
#   bf16_rounded  BF16 state rounded every token (packed kernel): the paper's BF16 baseline (Table 1, "per-step")
#   bf16_window   BF16 state + 16-token window (SGLang replay kernel): our performance control
#   leapquant     the product candidate lq_final (15 Nov 2026): CUDA fold, CUDA decode v9, step inputs built once,
#                 fused slot copy, staggered folds (the new default; with the radix cache off the track-boundary
#                 flushes are dropped), ONE fit sweep (LEAPQUANT_SWEEPS=1, backlog B52)
# Sampling: the paper's Table 5 for Qwen3.5-9B: thinking on, temperature 1.0, top-p 0.95, top-k 20, presence penalty
# 1.5. Output budget: 131,072 tokens (Asif, 15 Nov 2026: raised from the paper's 81,920, which cut 11-12% of the
# strict lane's answers; target under 5%). QUALITY_SAMPLES seeds per question (sample j uses seed 1234 + j; the
# 12-hour schedule uses 1). The paper's early stop on repetitive output is not reproduced (its method is not
# published). Capped answers count as wrong; the cap rate is reported per arm and task.
# Retraction-free like runbook/accuracy_paper_strict.sh (backlog B58: no retraction may reset a compressed state):
# radix cache off, a state pool sized to the running requests, concurrency = min(RUNNING_REQUESTS, KV tokens /
# (PROMPT_BUDGET + max tokens)); 0 retractions is checked per arm. QUALITY_MEM_FRACTION raises SGLang's static memory
# (its automatic choice, ~0.78 here, leaves ~16 GB unused during these runs): at 0.90 the KV pool holds ~1.71M tokens,
# 12 requests at the 131,072-token budget instead of 10. QUALITY_CONCURRENCY (if set) overrides the formula.
# All tasks go through ONE pool per arm (eval/accuracy_bench.py), long-answer tasks first.
# Output per arm: quality_<arm>.json (accuracy_bench run), quality_items_<arm>.jsonl (one line per answer, joinable by
# sample id), server log; then quality_decision.json from `python -m lqh.quality` with the decision rule frozen in
# harness/config/campaign.json (bf16_window must be noninferior to FP32 at 1 point on EVERY task).
#   RESULTS=$HOME/lqexp/results/harness/quality-12h QUALITY_TASKS=beyond_aime:20,aime2026,gpqa_diamond:30 runbook/quality_16bit.sh
set -u
cd "$(dirname "$0")/.."
export RUNNING_REQUESTS="${RUNNING_REQUESTS:-16}" MAMBA_SLOTS="${MAMBA_SLOTS:-24}"
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
TASKS="${QUALITY_TASKS:-aime2026}"
MAX_TOKENS="${QUALITY_MAX_TOKENS:-131072}"
PROMPT_BUDGET="${PROMPT_BUDGET:-1024}"
SAMPLES="${QUALITY_SAMPLES:-1}"
ARMS="${QUALITY_ARMS:-fp32 bf16_window leapquant}"
BENCH="python3 eval/accuracy_bench.py --port $PORT --tasks $TASKS --limit 100000 --max-tokens $MAX_TOKENS
    --samples $SAMPLES --seed 1234 --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5
    --thinking on --keep-text --timeout 21600"
BENCH="$(echo $BENCH)"
NOBUF="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
LIMITS="--max-running-requests $RUNNING_REQUESTS --max-mamba-cache-size $MAMBA_SLOTS"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
SERVE_FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-$RUNNING_REQUESTS} ${QUALITY_MEM_FRACTION:+--mem-fraction-static $QUALITY_MEM_FRACTION}"
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
arm() {            # arm NAME SERVER-FLAGS...
    local name="$1"; shift
    local log="$RESULTS/server_quality_$name.log"
    start_server "$log" "$@" || return 1
    if [ -n "${ARM_REQUIRE:-}" ] && ! grep -q "$ARM_REQUIRE" "$log"; then echo "not in the server log: $ARM_REQUIRE"; return 1; fi
    if grep -q "does not load" "$log"; then
        echo "WARNING: $name: a CUDA library did not load (Triton fallback); see $log" | tee -a "$STATUS"
    fi
    local kv concurrency cap guaranteed
    kv=$(kv_tokens "$log")
    if [ -z "$kv" ]; then echo "no max_total_num_tokens in $log"; return 1; fi
    guaranteed=$(( kv / (PROMPT_BUDGET + MAX_TOKENS) ))
    concurrency=${QUALITY_CONCURRENCY:-$guaranteed}
    [ "$concurrency" -le "$RUNNING_REQUESTS" ] || concurrency=$RUNNING_REQUESTS
    cap=$(grep -o "max_running_requests=[0-9]*" "$log" | tail -1 | cut -d= -f2)
    if [ -n "$cap" ] && [ "$concurrency" -gt "$cap" ]; then concurrency=$cap; fi
    echo "KV cache $kv tokens, at most $((PROMPT_BUDGET + MAX_TOKENS)) per request: retraction-free up to $guaranteed; concurrency $concurrency" \
        | tee "$RESULTS/quality_${name}_concurrency.txt"
    if [ "$concurrency" -lt "${MIN_CONCURRENCY:-8}" ]; then echo "concurrency $concurrency is below ${MIN_CONCURRENCY:-8}"; return 1; fi
    local compare=""
    [ "$name" != fp32 ] && [ -s "$RESULTS/quality_fp32.json" ] && compare="--compare $RESULTS/quality_fp32.json"
    $BENCH --concurrency "$concurrency" --arm "$name" --out "$RESULTS/quality_$name.json" \
        --items-out "$RESULTS/quality_items_$name.jsonl" $compare | tee "$RESULTS/quality_${name}.txt"
    local retractions
    retractions=$(grep -c "#retracted_reqs" "$log")
    echo "$retractions" > "$RESULTS/quality_${name}_retractions.txt"
    [ "$retractions" != "0" ] && echo "WARNING: $name: $retractions retractions, the arm is not retraction-free" >> "$STATUS"
    # Job 118 (15 Nov 2026): every request failed (the server died compiling FlashInfer's sampling kernel) and each arm
    # was still marked PASS because its items file existed. An arm with failed requests is not a pass.
    if grep -q "EXECUTION INVALID" "$RESULTS/quality_${name}.txt"; then
        echo "$name: $(grep -o "EXECUTION INVALID: [0-9]* requests failed" "$RESULTS/quality_${name}.txt")" | tee -a "$STATUS"
        return 1
    fi
    test -s "$RESULTS/quality_items_$name.jsonl"
}
unset_lq() { unset LEAPQUANT_STATE LEAPQUANT_FOLD_BACKEND LEAPQUANT_DECODE_BACKEND LEAPQUANT_CUDA_DECODE_VARIANT \
                   LEAPQUANT_STEP_REUSE LEAPQUANT_FOLD_POLICY LEAPQUANT_FUSED_SLOT_COPY LEAPQUANT_TRACK_FLUSH \
                   LEAPQUANT_LAYERS LEAPQUANT_SWEEPS ARM_REQUIRE; }
fp32() { unset_lq; arm fp32 --mamba-ssm-dtype float32 $NOBUF $LIMITS $SERVE_FLAGS; }
bf16_rounded() { unset_lq; arm bf16_rounded --mamba-ssm-dtype bfloat16 $NOBUF $LIMITS $SERVE_FLAGS; }
bf16_window() { unset_lq; arm bf16_window --mamba-ssm-dtype bfloat16 $NOBUF $WINDOW $LIMITS $SERVE_FLAGS; }
leapquant() {
    unset_lq
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda \
           LEAPQUANT_CUDA_DECODE_VARIANT=9 LEAPQUANT_STEP_REUSE=1 LEAPQUANT_FUSED_SLOT_COPY=1 \
           LEAPQUANT_FOLD_POLICY=stagger LEAPQUANT_SWEEPS=1
    export ARM_REQUIRE="LeapQuant decode state allocated"
    arm leapquant --mamba-ssm-dtype float32 $NOBUF $WINDOW --enable-linear-leapquant $LIMITS $SERVE_FLAGS
}
decide() {
    PYTHONPATH=harness python3 -m lqh.quality "$RESULTS" --campaign harness/config/campaign.json \
        | tee "$RESULTS/quality_decision.txt"
}
for name in $ARMS; do
    step "Q_$name" "$TASKS: $name (paper sampling, budget $MAX_TOKENS, $SAMPLES seed(s), retraction-free)" "$name"
done
step Q_decision "decision rule (campaign.json quality_qualification)" decide
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
