# Settings shared by the stage scripts. Source this file; do not run it.
# Override any of them in the environment before running a stage.

# nvcc and uv are often installed but not on PATH.
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
# FlashInfer JIT-compiles its sampling kernel on the first top-k/top-p request, and /usr/local/cuda (13.0, a minimal
# toolkit) has no curand headers: job 118 (15 Nov 2026) lost every quality request to that compile. This folder holds
# only the curand headers of the venv's CUDA 13 package (symlinks), so no other CUDA header can be shadowed.
if [ -d "$HOME/lqexp/cuda_extra_include/curand_only" ]; then
    export FLASHINFER_EXTRA_CUDAFLAGS="${FLASHINFER_EXTRA_CUDAFLAGS:--isystem $HOME/lqexp/cuda_extra_include/curand_only}"
fi

export LEAPQUANT_HOME="${LEAPQUANT_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export LEAPQUANT_BUILD_DIR="${LEAPQUANT_BUILD_DIR:-$LEAPQUANT_HOME/build}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-30000}"

# One environment per engine and one for recording (stage A creates them).
VENV_SGLANG="${VENV_SGLANG:-$HOME/venv-sglang}"
VENV_VLLM="${VENV_VLLM:-$HOME/venv-vllm}"
VENV_TRACE="${VENV_TRACE:-$HOME/venv-trace}"

# SGLang source. The commit is the one the implementation plan was written
# against and the one integrations/sglang/leapquant_phase1.patch applies to.
SGLANG_DIR="${SGLANG_DIR:-$HOME/sglang}"
SGLANG_COMMIT="${SGLANG_COMMIT:-c8ea70797157cafab46b3807f850d1c5c061a8d1}"

# Where a stage writes its logs. Check the directory in when the run is done.
RESULTS="${RESULTS:-$LEAPQUANT_HOME/results/$(date +%Y-%m-%d)-$(hostname -s)}"
mkdir -p "$RESULTS"

# Flags every server of the comparison gets, so that the configurations differ in ONE thing.
# Found on the first run (H100, 13 Nov 2026):
#   - with SGLang's defaults the server capped itself at 83 running requests ("5 state slots per
#     request"), so a client at concurrency 512 measured a queue, not 512 requests. Hence the
#     no_buffer strategy and explicit limits.
#   - the no_buffer strategy refuses to start without --disable-overlap-schedule.
# MAMBA_SLOTS: state slots. 560 leaves room for the LeapQuant state on an 80 GB card.
RUNNING_REQUESTS="${RUNNING_REQUESTS:-512}"
MAMBA_SLOTS="${MAMBA_SLOTS:-560}"
COMMON_FLAGS="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --max-running-requests $RUNNING_REQUESTS --max-mamba-cache-size $MAMBA_SLOTS"
WINDOW_FLAGS="$COMMON_FLAGS --enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"

# The LeapQuant state (about 12 GB for 560 slots) is allocated outside SGLang's memory accounting.
# First start on an 80 GB H100: out of memory. So the LeapQuant server reserves less for the KV cache.
# That changes KV capacity (unused in these runs), not speed.
LEAPQUANT_FLAGS="--enable-linear-leapquant --mem-fraction-static ${LQ_MEM_FRACTION:-0.60}"

# Starts an SGLang server with the given extra flags, waits until it answers,
# and leaves its process id in SERVER_PID.
start_server() {
    local log="$1"; shift
    python3 -m sglang.launch_server --model-path "$MODEL" --port "$PORT" "$@" > "$log" 2>&1 &
    SERVER_PID=$!
    echo "server starting (pid $SERVER_PID), log: $log"
    for _ in $(seq 1 360); do
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then echo "server is up"; return 0; fi
        if ! kill -0 "$SERVER_PID" 2> /dev/null; then echo "server exited; see $log"; return 1; fi
        sleep 5
    done
    echo "server did not come up in 30 minutes; see $log"; return 1
}

stop_server() {
    kill "$SERVER_PID" 2> /dev/null || true
    wait "$SERVER_PID" 2> /dev/null || true
    sleep 5
}

# The benchmark client of stage B2. By default: long shape (4,096 in, 1,024 out) at
# concurrency 32 and 128, short shape (512 in, 512 out) at concurrency 512, four
# prompts per concurrent request. $1 is a label used in the log file names.
# Shapes (input / output tokens):
#   long     4,096 / 1,024   random lengths up to these (SGLang's default range)
#   short      512 /   512   random lengths up to these
#   decode     128 /   512   exactly these: decode-heavy, fits every configuration at 512
#   decode2k   128 / 2,048   exactly these: long outputs (the method's case); memory-bound at 512
#   aa1k     1,000 / 1,000   exactly these: Artificial Analysis's system load test shape (AA-SLT)
#   ix8k1k   8,192 / 1,024   exactly these: SemiAnalysis InferenceX's single-turn 8k/1k shape
#   ix1k8k   1,024 / 8,192   exactly these: SemiAnalysis InferenceX's 1k/8k shape (long outputs)
#   (aa1k and ix8k1k use max(2 x concurrency, 8) prompts per point, ix1k8k max(2 x concurrency, 4): concurrency sweeps)
# Random lengths average about half the maximum, so "short" spends much of its time in prefill.
run_serving_client() {
    local label="$1"
    # SERVING_POINTS: which runs to do, as "shape:concurrency" items.
    for point in ${SERVING_POINTS:-long:32 long:128 short:512}; do
        local shape="${point%%:*}"
        local c="${point##*:}"
        local input=4096 output=1024 extra=""
        local prompts=$((c * 4))
        case "$shape" in
            short) input=512; output=512 ;;
            decode) input=128; output=512; extra="--random-range-ratio 1.0"; prompts=$((c * 2)) ;;
            decode2k) input=128; output=2048; extra="--random-range-ratio 1.0"; prompts=$c ;;
            aa1k) input=1000; output=1000; extra="--random-range-ratio 1.0"; prompts=$(( c * 2 > 8 ? c * 2 : 8 )) ;;
            ix8k1k) input=8192; output=1024; extra="--random-range-ratio 1.0"; prompts=$(( c * 2 > 8 ? c * 2 : 8 )) ;;
            ix1k8k) input=1024; output=8192; extra="--random-range-ratio 1.0"; prompts=$(( c * 2 > 4 ? c * 2 : 4 )) ;;
        esac
        # GPU memory while the point runs (plan v6 S09 / backlog B17): sampled every 0.5 s, the peak kept.
        local memfile="$RESULTS/serving_${label}_${shape}_c${c}.mem"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -lms 500 > "$memfile.samples" 2> /dev/null &
        local sampler=$!
        # The result also as JSON (every metric at full precision, the client's settings), for publishing;
        # SERVING_DETAILS=1 adds every request's TTFT and token gaps to it.
        python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port "$PORT" \
            --dataset-name random --random-input-len "$input" --random-output-len "$output" $extra \
            --num-prompts "$prompts" --max-concurrency "$c" --warmup-requests 3 --flush-cache \
            --output-file "$RESULTS/serving_${label}_${shape}_c${c}.jsonl" ${SERVING_DETAILS:+--output-details} \
            2>&1 | tee "$RESULTS/serving_${label}_${shape}_c${c}.log"
        kill "$sampler" 2> /dev/null; wait "$sampler" 2> /dev/null
        awk -F', *' 'NF >= 2 { if ($1 > peak) peak = $1; total = $2 } END { printf "peak_memory_used_mib %d\nmemory_total_mib %d\n", peak, total }' \
            "$memfile.samples" > "$memfile" 2> /dev/null
        rm -f "$memfile.samples"
    done
}
