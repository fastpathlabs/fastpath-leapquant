#!/usr/bin/env bash
# prefill_rate.sh: prefill throughput (input tokens per second) of fp32 and LeapQuant (compressed state):
# 4,096-token prompts, 1-token answers, 64 at a time, so the run is almost all prefill.
#   RESULTS=results/run-xyz runbook/prefill_rate.sh          (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode 512"
measure() {        # measure LABEL server-flags...
    local label="$1"; shift
    start_server "$RESULTS/server_prefill_$label.log" "$@" || return 1
    python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port "$PORT" --dataset-name random \
        --random-input-len 4096 --random-output-len 1 --random-range-ratio 1.0 --num-prompts 256 --max-concurrency 64 \
        --warmup-requests 3 --flush-cache > "$RESULTS/prefill_$label.log" 2>&1
    stop_server
    grep -E "Input token throughput|Median TTFT|Successful requests" "$RESULTS/prefill_$label.log"
}
measure fp32 --mamba-ssm-dtype float32 $COMMON_FLAGS $FLAGS
( export LEAPQUANT_STATE=compressed; measure leapquant --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS )
