#!/usr/bin/env bash
# serving_ab_step.sh: serving A/B of two per-step changes, paired with bf16 window in one session (backlog B48, B51):
#   - the decode inputs that are the same for every layer (slots, records to fold, the fold's work list) built once
#     per step instead of once per layer (LEAPQUANT_STEP_REUSE 0 / 1; about ten small kernels fewer per layer);
#   - the decode kernel: Triton, or a CUDA decode (LEAPQUANT_DECODE_BACKEND=cuda, LEAPQUANT_CUDA_DECODE_VARIANT).
# Radix cache off, 560 slots each; AA 1k/1k at 1, 4, 64 and 512 users (the fixed cost per step shows at low load).
# Labels off_<arm>_r1; arms: bf16_window, lq_noreuse, lq_reuse, lq_reuse_cudadec (the CUDA decode,
# LEAPQUANT_CUDA_DECODE_VARIANT from the environment).
#   RESULTS=results/serving_ab_step ARMS="bf16 noreuse reuse cudadec" runbook/serving_ab_step.sh   (repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"   # what this run measured: sources, binaries, versions
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
export SERVING_POINTS="${SERVING_POINTS:-aa1k:1 aa1k:4 aa1k:64 aa1k:512}"
FLAGS="--disable-radix-cache --cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
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
bf16() { serve "off_bf16_window_r1" --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $FLAGS; }
lq() {             # lq ARM REUSE DECODE_BACKEND
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_STEP_REUSE="$2" LEAPQUANT_DECODE_BACKEND="$3"
    serve "off_$1_r1" --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS
}
for arm in ${ARMS:-bf16 noreuse reuse}; do
    case "$arm" in
        bf16) step A "bf16 window" bf16 ;;
        noreuse) step B "LeapQuant, decode inputs built per layer (as rounds 9 and 10)" lq lq_noreuse 0 triton ;;
        reuse) step C "LeapQuant, decode inputs built once per step" lq lq_reuse 1 triton ;;
        cudadec) step D "LeapQuant, once per step, CUDA decode variant ${LEAPQUANT_CUDA_DECODE_VARIANT:-3}" lq lq_reuse_cudadec 1 cuda ;;
    esac
done
if grep -l "does not load" "$RESULTS"/server_*lq*.log 2> /dev/null; then
    echo "WARNING: a LeapQuant server could not load the CUDA library" >> "$STATUS"
fi
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
