#!/usr/bin/env bash
# q38_probe.sh: first GPU contact with Qwen3.8-27B (FP8 checkpoint, revision c3a462c) for backlog B75 (15 Nov 2026).
#   K1  kernel agreement at the 27B's head layout (48 value heads over 16 key heads, 128x128 state per head):
#       kernel/bench_triton.py fast vs exact decode step and fold, CUDA decode v9 + CUDA fold, one fit sweep (lq_final),
#       batch 4/32/128; the 9B layout (32 over 16) in the same process run for comparison.
#   K2  compute-sanitizer memcheck of the same kernels at the 27B layout, batch 4, one window.
#   S*  one server per arm (fp32 packed, bf16 window, LeapQuant) at PROBE_SLOTS state slots with the harness flags of
#       the 9B arms: pools from the log, 8 greedy completions (agreement with fp32) and 2 sampled chat answers
#       (FlashInfer's top-k/top-p path). A failed start at CUDA-graph batch 512 is retried once at 256.
#   P   runbook/q38_probe.py plan: per-arm cost per slot and per KV token, pool budget, balanced slot counts for 1k/1k.
#   RESULTS=$HOME/lqexp/results/q38/probe runbook/q38_probe.sh     (from the tree root)
set -u
cd "$(dirname "$0")/.."
export MODEL="${MODEL:-Qwen/Qwen3.8-27B-FP8}"
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/hf/hub}" HF_HUB_OFFLINE=1
runbook/manifest.sh "$RESULTS/manifest.txt"
echo "model: $MODEL (HF_HUB_CACHE $HF_HUB_CACHE)" >> "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
PROBE_SLOTS="${PROBE_SLOTS:-120}"
step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
kernels() {        # kernels HV H BATCHES...
    local hv="$1" h="$2"; shift 2
    env LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda LEAPQUANT_CUDA_DECODE_VARIANT=9 LEAPQUANT_SWEEPS=1 \
        LEAPQUANT_STEP_REUSE=1 timeout 1800 python3 kernel/bench_triton.py --hv "$hv" --h "$h" --batch-sizes "$@" \
        --windows 2 --fold-backend cuda --decode-backend cuda --sweeps 1
}
memcheck() {
    env LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda LEAPQUANT_CUDA_DECODE_VARIANT=9 LEAPQUANT_SWEEPS=1 \
        LEAPQUANT_STEP_REUSE=1 timeout 2400 compute-sanitizer --tool memcheck --error-exitcode 9 \
        python3 kernel/bench_triton.py --hv 48 --h 16 --batch-sizes 4 --windows 1 --no-exact-times \
        --fold-backend cuda --decode-backend cuda --sweeps 1
}
NOBUF="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
unset_lq() { unset LEAPQUANT_STATE LEAPQUANT_FOLD_BACKEND LEAPQUANT_DECODE_BACKEND LEAPQUANT_CUDA_DECODE_VARIANT \
                   LEAPQUANT_STEP_REUSE LEAPQUANT_FOLD_POLICY LEAPQUANT_FUSED_SLOT_COPY LEAPQUANT_TRACK_FLUSH \
                   LEAPQUANT_LAYERS LEAPQUANT_SWEEPS; }
arm() {            # arm NAME server-flags... (the CUDA-graph limit is appended here)
    local name="$1"; shift
    local log="$RESULTS/server_$name.log" gbs
    for gbs in 512 256; do
        : > "$log"
        if start_server "$log" "$@" --max-running-requests 512 --cuda-graph-max-bs-decode "$gbs" \
                --max-mamba-cache-size "$PROBE_SLOTS"; then
            echo "$name: server up with CUDA graphs to batch $gbs"
            if [ "$name" = leapquant ] && ! grep -q "LeapQuant decode state allocated" "$log"; then
                echo "LeapQuant pool not in the log"; stop_server; return 1
            fi
            python3 runbook/q38_probe.py smoke --port "$PORT" --model "$MODEL" --arm "$name" --out "$RESULTS"
            local smoke_status=$?
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader > "$RESULTS/gpu_memory_$name.txt"
            stop_server
            return $smoke_status
        fi
        echo "$name: no start with CUDA graphs to batch $gbs (tail of the log follows)"; tail -5 "$log"
        cp "$log" "$RESULTS/server_${name}_failed_gbs$gbs.txt"
        stop_server 2> /dev/null; pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 10
    done
    return 1
}
fp32() { unset_lq; arm fp32 --mamba-ssm-dtype float32 $NOBUF; }
bf16_window() { unset_lq; arm bf16_window --mamba-ssm-dtype bfloat16 $NOBUF $WINDOW; }
leapquant() {
    unset_lq
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda \
           LEAPQUANT_CUDA_DECODE_VARIANT=9 LEAPQUANT_STEP_REUSE=1 LEAPQUANT_FUSED_SLOT_COPY=1 \
           LEAPQUANT_FOLD_POLICY=stagger LEAPQUANT_SWEEPS=1
    arm leapquant --mamba-ssm-dtype float32 $NOBUF $WINDOW --enable-linear-leapquant
}
both_layouts() { kernels 48 16 4 32 128 && echo "=== 9B layout (HV 32, H 16) ===" && kernels 32 16 4 32 128; }
step K1 "kernel agreement, 27B layout (HV 48, H 16) then 9B layout (HV 32, H 16), batch 4/32/128" both_layouts
step K2 "compute-sanitizer memcheck, 27B layout, batch 4" memcheck
step S1 "server fp32 packed, $PROBE_SLOTS slots: pools, greedy and sampled answers" fp32
step S2 "server bf16 window, $PROBE_SLOTS slots: pools, greedy and sampled answers" bf16_window
step S3 "server LeapQuant (lq_final settings), $PROBE_SLOTS slots: pools, greedy and sampled answers" leapquant
step P "pool budget and balanced slot counts for 1k/1k; greedy agreement with fp32" python3 runbook/q38_probe.py plan "$RESULTS"
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
