#!/usr/bin/env bash
# Round 13: the capacity regime: the whole 80 GB GPU, configurations differ only in how many state
# slots and KV tokens they fit. At 1,024 users fp32 state fits ~560 slots (50 MB each) where bf16 window and LeapQuant
# fit 1,100 (25 / 21 MB): the regime where a compressed state is meant to pay. Decode-heavy cell (128 in / 512 out) at
# 512 and 1,024 users, radix cache off, CUDA graphs to 1,024. Arms (labels as round 10): fp32 packed (SGLang's default,
# the paper's baseline) with FP32_SLOTS, bf16 window and LeapQuant with SLOTS. LeapQuant's kernels come from the
# environment (LEAPQUANT_BUILD_DIR, LEAPQUANT_DECODE_BACKEND, ...).
#   RESULTS=results/round13 runbook/serving_round13.sh          (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
export SERVING_POINTS="${SERVING_POINTS:-decode:512 decode:1024}"
RUN="--max-running-requests ${RUNNING:-1024} --disable-radix-cache --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-1024}"
COMMON="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
WIN="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
step() {
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
serve() {
    local label="$1"; shift
    start_server "$RESULTS/server_$label.log" "$@" && run_serving_client "$label"
    local status=$?
    grep -E "max_total_num_tokens|LeapQuant decode state|Mamba Cache is allocated" "$RESULTS/server_$label.log" \
        | grep -v server_args | head -4 > "$RESULTS/server_${label}_memory.txt"
    return $status
}
bf16() { serve off_bf16_window_r1 --mamba-ssm-dtype bfloat16 $COMMON $WIN --max-mamba-cache-size "${SLOTS:-1100}" $RUN; }
lq() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    serve off_leapquant_cuda_r1 --mamba-ssm-dtype float32 $COMMON $WIN --enable-linear-leapquant \
        --max-mamba-cache-size "${SLOTS:-1100}" $RUN
}
fp32() { serve off_fp32_packed_r1 --mamba-ssm-dtype float32 $COMMON --max-mamba-cache-size "${FP32_SLOTS:-560}" $RUN; }
step S1 "bf16 window, ${SLOTS:-1100} slots, up to ${RUNNING:-1024} running" bf16
step T1 "LeapQuant, ${SLOTS:-1100} slots, up to ${RUNNING:-1024} running" lq
step U1 "fp32 packed (SGLang's default), ${FP32_SLOTS:-560} slots: what fits" fp32
if grep -l "does not load" "$RESULTS"/server_*leapquant*.log 2> /dev/null; then
    echo "WARNING: a LeapQuant server could not load the CUDA library" >> "$STATUS"
fi
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
