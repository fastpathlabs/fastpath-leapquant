#!/usr/bin/env bash
# public_bench.sh: the public benchmark suite (backlog B60; Asif, 14 Nov 2026 evening: "yes to all"). Concurrency
# sweeps in the shapes public inference benchmarks use, so that our numbers read as theirs do: system output
# throughput against speed per user, with TTFT, TPOT, ITL and end-to-end percentiles (reports/generate_reports.py):
#   aa1k     1,000 in / 1,000 out, exact   Artificial Analysis's system load test         users 1, 4, 16, 64, 128, 256, 512
#   ix8k1k   8,192 in / 1,024 out, exact   SemiAnalysis InferenceX 8k/1k                  users 1, 4, 16, 64, 128, 256
#   ix1k8k   1,024 in / 8,192 out, exact   SemiAnalysis InferenceX 1k/8k (long outputs)   users 1, 4, 16, 64, 128, 256
# RADIX=on (default; SGLang's default serving mode). Servers, with the settings of our latest runs:
#   bf16_window     bf16 state + 16-token window, radix no_buffer, 1,100 slots (equal state memory with LeapQuant)
#   sglang_default  SGLang started with no options at all (its own state type, radix strategy and pool)
#   sglang_best     SGLang's best state settings: bf16, lazy extra buffers, int8 checkpoints; its packed kernel (its
#                   window kernel refuses that strategy), 600 slots, static memory fraction 0.60 (results/sglang-best-3)
#   leapquant       LeapQuant's final configuration: compressed state, CUDA fold, CUDA decode (variant from the
#                   environment), step inputs built once, fused slot copy, folds 'auto', radix no_buffer, 1,310 slots
# RADIX=off (--disable-radix-cache, as round 13): bf16_window and leapquant (aligned folds) with 1,100 slots,
# fp32_packed (SGLang's default kernel, fp32 state) with 560 slots, up to 1,024 running.
# REPS full passes over the servers (default 2), the order reversed on even passes so that drift favours no server.
# Labels <on|off>_<server>_r<pass>; every point writes serving_<label>_<shape>_c<N>.log, .mem and .jsonl (env.sh).
# Restartable: finished server passes are recorded in $RESULTS/STATUS.txt and skipped on a rerun.
#   RESULTS=results/public-bench-on RADIX=on REPS=2 runbook/public_bench.sh      (from the repository root, in the
#   LeapQuant environment of the final configuration: LEAPQUANT_BUILD_DIR, LEAPQUANT_CUDA_DECODE_VARIANT=9, ...)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
MODE="${RADIX:-on}"
REPS="${REPS:-2}"
export SERVING_POINTS="${SERVING_POINTS:-aa1k:1 aa1k:4 aa1k:16 aa1k:64 aa1k:128 aa1k:256 aa1k:512 ix8k1k:1 ix8k1k:4 ix8k1k:16 ix8k1k:64 ix8k1k:128 ix8k1k:256 ix1k8k:1 ix1k8k:4 ix1k8k:16 ix1k8k:64 ix1k8k:128 ix1k8k:256}"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
NO_BUFFER="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
if [ "$MODE" = on ]; then
    LIMITS="--max-running-requests 512 --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
    SERVERS="${SERVERS:-bf16_window sglang_default sglang_best leapquant}"
    LQ_DEFAULT_SLOTS=1310
else
    LIMITS="--max-running-requests 1024 --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-1024} --disable-radix-cache"
    SERVERS="${SERVERS:-bf16_window fp32_packed leapquant}"
    LQ_DEFAULT_SLOTS=1100
fi

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
serve() {          # serve LABEL server-flags...: every point of SERVING_POINTS on one server
    local label="$1"; shift
    start_server "$RESULTS/server_$label.log" "$@" || return 1
    run_serving_client "$label"
    grep -E "max_total_num_tokens|LeapQuant decode state|KV Cache is allocated|Mamba Cache is allocated|capped" \
        "$RESULTS/server_$label.log" | grep -v server_args | head -8 > "$RESULTS/server_${label}_memory.txt"
    local missing=0 point
    for point in $SERVING_POINTS; do
        grep -q "Output token throughput" "$RESULTS/serving_${label}_${point%%:*}_c${point##*:}.log" 2> /dev/null \
            || missing=$((missing + 1))
    done
    echo "points without a result: $missing"
    [ "$missing" -eq 0 ]
}
server() {         # server NAME PASS
    local name="$1" label="${MODE}_$1_r$2"
    case "$name" in
        bf16_window)
            serve "$label" --mamba-ssm-dtype bfloat16 $NO_BUFFER $WINDOW $LIMITS --max-mamba-cache-size "${BF16_SLOTS:-1100}" ;;
        sglang_default)
            serve "$label" ;;
        sglang_best)
            serve "$label" --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy \
                --enable-int8-mamba-checkpoint $LIMITS --max-mamba-cache-size "${BEST_SLOTS:-600}" --mem-fraction-static 0.60 ;;
        fp32_packed)
            serve "$label" --mamba-ssm-dtype float32 $NO_BUFFER $LIMITS --max-mamba-cache-size "${FP32_SLOTS:-560}" ;;
        leapquant)
            export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda
            export LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
            if [ "$MODE" = on ]; then export LEAPQUANT_FOLD_POLICY=auto; else unset LEAPQUANT_FOLD_POLICY; fi
            serve "$label" --mamba-ssm-dtype float32 $NO_BUFFER $WINDOW --enable-linear-leapquant $LIMITS \
                --max-mamba-cache-size "${LQ_SLOTS:-$LQ_DEFAULT_SLOTS}" || return 1
            if grep -q "does not load" "$RESULTS/server_$label.log"; then
                echo "WARNING: $label fell back to the Triton fold" | tee -a "$STATUS"
            fi ;;
        *) echo "unknown server $name"; return 1 ;;
    esac
}
for pass in $(seq 1 "$REPS"); do
    order="$SERVERS"
    if [ $((pass % 2)) -eq 0 ]; then order=$(echo $SERVERS | tr ' ' '\n' | tac | tr '\n' ' '); fi
    for name in $order; do
        step "${MODE}_${name}_r${pass}" "radix $MODE, $name, pass $pass of $REPS: $(echo $SERVING_POINTS | wc -w) points" \
            server "$name" "$pass"
    done
done
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
