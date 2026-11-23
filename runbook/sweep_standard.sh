#!/usr/bin/env bash
# sweep_standard.sh: concurrency sweeps in the style of the public inference benchmarks, so that our
# numbers are read the way theirs are (system throughput against speed per user):
#   aa1k     1,000 in / 1,000 out, concurrency 1 .. 512: Artificial Analysis's system load test shape
#   ix8k1k   8,192 in / 1,024 out, concurrency 1 .. 256: SemiAnalysis InferenceX's single-turn 8k/1k shape
# Configurations: fp32 state (SGLang's packed kernel), bf16 state + window, LeapQuant (compressed state,
# fast kernels). Radix cache off and 560 state slots for all, so every point really runs its concurrency.
# Restartable; meant to run detached.
#   RESULTS=results/sweep-xyz runbook/sweep_standard.sh          (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
export SERVING_POINTS="${SERVING_POINTS:-aa1k:1 aa1k:4 aa1k:16 aa1k:64 aa1k:128 aa1k:256 aa1k:512 ix8k1k:1 ix8k1k:16 ix8k1k:64 ix8k1k:128 ix8k1k:256}"
FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode 512"
step() {
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
serve() { local label="$1"; shift; start_server "$RESULTS/server_$label.log" "$@" && run_serving_client "$label"; }
leapquant() { export LEAPQUANT_STATE=compressed; serve config4_leapquant --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS; }
step W1 "sweep: fp32 state, packed kernel" serve config1_fp32 --mamba-ssm-dtype float32 $COMMON_FLAGS $FLAGS
step W3 "sweep: bf16 state, window kernel" serve config3_bf16_window --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $FLAGS
step W4 "sweep: LeapQuant, compressed state, fast kernels" leapquant
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
