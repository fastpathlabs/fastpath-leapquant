#!/usr/bin/env bash
# accuracy_fast.sh: task accuracy (plan 5, step S24) with the fast LeapQuant kernels and the
# compressed state, against fp32 (twice: the server's own run-to-run noise) and against bf16
# with the window (the fastest configuration SGLang ships). Restartable, meant to run detached
# (tmux): finished steps are recorded in $RESULTS/STATUS.txt and skipped on a rerun.
#
#   RESULTS=results/run-xyz runbook/accuracy_fast.sh          (from the repository root)
#
# Every server: radix cache off (the compressed state needs it; the others get the same),
# temperature 0. ACC_LIMIT questions per task (GSM8K has 1,319, MMLU-Pro 12,032, AIME 30),
# answers up to ACC_MAX_TOKENS tokens, ACC_CONCURRENCY requests at once.
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
python3 -c "import datasets" 2> /dev/null || uv pip install datasets
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
LIMIT="${ACC_LIMIT:-1000}"
MAX_TOKENS="${ACC_MAX_TOKENS:-8192}"
CONCURRENCY="${ACC_CONCURRENCY:-256}"
BENCH="python3 eval/accuracy_bench.py --port $PORT --limit $LIMIT --max-tokens $MAX_TOKENS --concurrency $CONCURRENCY"
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

fp32_twice() {
    start_server "$RESULTS/server_accuracy_fp32.log" --mamba-ssm-dtype float32 $COMMON_FLAGS $SERVE_FLAGS || return 1
    $BENCH --out "$RESULTS/accuracy_fp32_run1.json" | tee "$RESULTS/accuracy_fp32_run1.txt"
    $BENCH --out "$RESULTS/accuracy_fp32_run2.json" --compare "$RESULTS/accuracy_fp32_run1.json" \
        | tee "$RESULTS/accuracy_fp32_noise.txt"
    test -s "$RESULTS/accuracy_fp32_run2.json"
}
leapquant() {
    export LEAPQUANT_STATE=compressed
    start_server "$RESULTS/server_accuracy_leapquant.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS \
        --enable-linear-leapquant $SERVE_FLAGS || return 1
    grep "LeapQuant decode state allocated" "$RESULTS/server_accuracy_leapquant.log" || return 1
    $BENCH --out "$RESULTS/accuracy_leapquant.json" --compare "$RESULTS/accuracy_fp32_run1.json" \
        | tee "$RESULTS/accuracy_leapquant_vs_fp32.txt"
    test -s "$RESULTS/accuracy_leapquant.json"
}
bf16_window() {
    start_server "$RESULTS/server_accuracy_bf16_window.log" --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $SERVE_FLAGS || return 1
    $BENCH --out "$RESULTS/accuracy_bf16_window.json" --compare "$RESULTS/accuracy_fp32_run1.json" \
        | tee "$RESULTS/accuracy_bf16_window_vs_fp32.txt"
    test -s "$RESULTS/accuracy_bf16_window.json"
}
leapquant_rank8_late() {   # 8 compensators on the model's layers 20-31 (plan 5, step S21)
    export LEAPQUANT_STATE=compressed LEAPQUANT_COMPENSATORS="4,20:8"
    start_server "$RESULTS/server_accuracy_leapquant_r8.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS \
        --enable-linear-leapquant $SERVE_FLAGS || return 1
    grep "LeapQuant decode state allocated" "$RESULTS/server_accuracy_leapquant_r8.log" || return 1
    $BENCH --out "$RESULTS/accuracy_leapquant_r8.json" --compare "$RESULTS/accuracy_fp32_run1.json" \
        | tee "$RESULTS/accuracy_leapquant_r8_vs_fp32.txt"
    test -s "$RESULTS/accuracy_leapquant_r8.json"
}
versus() {         # LeapQuant against bf16 + window, paired
    python3 eval/accuracy_bench.py --report "$RESULTS/accuracy_leapquant.json" \
        --compare "$RESULTS/accuracy_bf16_window.json" | tee "$RESULTS/accuracy_leapquant_vs_bf16_window.txt"
}

step A1 "accuracy: fp32 server, twice (run-to-run noise)" fp32_twice
step A2 "accuracy: LeapQuant, compressed state, fast kernels" leapquant
step A3 "accuracy: bf16 state with the window" bf16_window
step A4 "accuracy: LeapQuant against bf16 + window" versus
step A5 "accuracy: LeapQuant with 8 compensators on layers 20-31" leapquant_rank8_late
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
