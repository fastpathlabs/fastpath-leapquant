#!/usr/bin/env bash
# accuracy_paper.sh: task accuracy in the paper's lane (backlog B09; plan v6 N08/E03): thinking mode, sampling at
# temperature 1.0, answers up to 81,920 tokens, several samples per question, judged per question (question-clustered
# paired interval, eval/accuracy_bench.py). Three arms on the same questions: fp32 window (the reference), bf16 window
# (the speed bar, which must itself be quality-qualified) and LeapQuant (compressed state, fast decode, CUDA fold).
# Restartable, meant to run detached: finished steps are recorded in $RESULTS/STATUS.txt and skipped on a rerun.
#
#   RESULTS=results/accuracy-paper runbook/accuracy_paper.sh          (from the repository root)
#
# PAPER_TASKS (default aime2025,aime2024; add beyond_aime for 100 more), PAPER_SAMPLES (4), PAPER_MAX_TOKENS (81920),
# PAPER_TEMPERATURE (1.0), PAPER_TOP_P (1.0), PAPER_CONCURRENCY (64), PAPER_SEED (1234). Raw answers are kept.
set -u
cd "$(dirname "$0")/.."
# A queue can defer this run (it takes hours): a file DEFER in the results folder makes it a no-op that says why.
if [ -n "${RESULTS:-}" ] && [ -f "$RESULTS/DEFER" ]; then
    echo "deferred: $(cat "$RESULTS/DEFER")"
    exit 0
fi
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
TASKS="${PAPER_TASKS:-aime2025,aime2024}"
BENCH="python3 eval/accuracy_bench.py --port $PORT --tasks $TASKS --limit 100000 --max-tokens ${PAPER_MAX_TOKENS:-81920}
    --concurrency ${PAPER_CONCURRENCY:-64} --samples ${PAPER_SAMPLES:-4} --temperature ${PAPER_TEMPERATURE:-1.0}
    --top-p ${PAPER_TOP_P:-1.0} --seed ${PAPER_SEED:-1234} --thinking on --keep-text --timeout 14400"
BENCH="$(echo $BENCH)"
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
fp32_window() {
    start_server "$RESULTS/server_paper_fp32_window.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS $SERVE_FLAGS || return 1
    $BENCH --out "$RESULTS/paper_fp32_window.json" | tee "$RESULTS/paper_fp32_window.txt"
    test -s "$RESULTS/paper_fp32_window.json"
}
bf16_window() {
    start_server "$RESULTS/server_paper_bf16_window.log" --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $SERVE_FLAGS || return 1
    $BENCH --out "$RESULTS/paper_bf16_window.json" --compare "$RESULTS/paper_fp32_window.json" \
        | tee "$RESULTS/paper_bf16_window_vs_fp32.txt"
    test -s "$RESULTS/paper_bf16_window.json"
}
leapquant() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    start_server "$RESULTS/server_paper_leapquant.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS \
        --enable-linear-leapquant $SERVE_FLAGS || return 1
    grep "LeapQuant decode state allocated" "$RESULTS/server_paper_leapquant.log" || return 1
    $BENCH --out "$RESULTS/paper_leapquant.json" --compare "$RESULTS/paper_fp32_window.json" \
        | tee "$RESULTS/paper_leapquant_vs_fp32.txt"
    test -s "$RESULTS/paper_leapquant.json"
}
compare_arms() {
    python3 eval/accuracy_bench.py --report "$RESULTS/paper_leapquant.json" --compare "$RESULTS/paper_bf16_window.json" \
        | tee "$RESULTS/paper_leapquant_vs_bf16_window.txt"
    true
}
step P1 "paper lane: fp32 window (reference)" fp32_window
step P2 "paper lane: bf16 window" bf16_window
step P3 "paper lane: LeapQuant, compressed state, CUDA fold" leapquant
step P4 "paper lane: LeapQuant against bf16 window" compare_arms
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
