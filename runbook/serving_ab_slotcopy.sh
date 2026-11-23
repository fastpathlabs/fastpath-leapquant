#!/usr/bin/env bash
# serving_ab_slotcopy.sh: backlog B56. With the radix cache on SGLang copies each request's state slot once
# (MambaPool.copy_from), and LeapQuant's copy_slots did it field by field: ~245 small kernels per request, 62,643
# index kernels and 273 ms of GPU time per burst of 256 prompts (results/prefill-radix), so LeapQuant's prefill was
# ~18% slower than bf16 window's there. Now one launch (leapquant_triton.SlotRows, LEAPQUANT_FUSED_SLOT_COPY=1).
# One session, radix cache on, equal state memory (bf16 window 1,100 slots, LeapQuant 1,310), round 9's radix-on shapes:
#   C        bf16 window
#   T_field  LeapQuant, LEAPQUANT_FOLD_POLICY=auto, LEAPQUANT_FUSED_SLOT_COPY=0 (the copy field by field, as before)
#   T_fused  LeapQuant, LEAPQUANT_FOLD_POLICY=auto, LEAPQUANT_FUSED_SLOT_COPY=1
# Each server first runs prefill-only bursts (eval/prefill_burst.py, 256 x 128 tokens, 5 bursts) and the two-turn radix
# check (integrations/sglang/radix_turns_client.py: second turns start from radix-cached states, so they go through
# copy_slots), then the serving shapes. Labels on_<arm>_r1.
#   RESULTS=results/ab-slotcopy runbook/serving_ab_slotcopy.sh          (from the repository root, LEAPQUANT_HOME with
#   the fused copy)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
export SERVING_POINTS="${SERVING_POINTS:-decode:128 decode:512 short:512 decode2k:512}"
GRAPHS="--cuda-graph-max-bs-decode ${GRAPH_MAX_BS:-512}"
RADIX="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
grep -q "class SlotRows" kernel/leapquant_triton.py \
    || { echo "this tree has no fused slot copy (kernel/leapquant_triton.py)" | tee -a "$STATUS"; exit 1; }

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
checks() {         # checks LABEL: prefill bursts and the two-turn radix check on the running server
    local label="$1" compare=""
    python3 eval/prefill_burst.py --port "$PORT" --out "$RESULTS/burst_$label.json" > "$RESULTS/burst_$label.txt" 2>&1
    cat "$RESULTS/burst_$label.txt"
    if [ "$label" != "on_bf16_window_r1" ] && [ -f "$RESULTS/turns_on_bf16_window_r1.json" ]; then
        compare="--compare $RESULTS/turns_on_bf16_window_r1.json"
    fi
    python3 integrations/sglang/radix_turns_client.py --port "$PORT" --out "$RESULTS/turns_$label.json" $compare \
        > "$RESULTS/turns_$label.txt" 2>&1
    echo $? > "$RESULTS/turns_$label.status"
    cat "$RESULTS/turns_$label.txt"
}
serve() {          # serve LABEL server-flags...
    local label="$1"; shift
    start_server "$RESULTS/server_$label.log" "$@" || return 1
    checks "$label"
    run_serving_client "$label"
    local status=$?
    grep -E "max_total_num_tokens|LeapQuant decode state|KV Cache is allocated|Mamba Cache is allocated" \
        "$RESULTS/server_$label.log" | grep -v server_args | head -8 > "$RESULTS/server_${label}_memory.txt"
    return $status
}
check_turns() { cat "$RESULTS/turns_$1.txt"; [ "$(cat "$RESULTS/turns_$1.status" 2> /dev/null)" = "0" ]; }
bf16() {
    serve on_bf16_window_r1 --mamba-ssm-dtype bfloat16 $RADIX $WINDOW $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${BF16_SLOTS:-1100}"
}
lq() {             # lq ARM FUSED
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    export LEAPQUANT_FOLD_POLICY=auto LEAPQUANT_FUSED_SLOT_COPY="$2"
    serve "on_lq_$1_r1" --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${LQ_SLOTS:-1310}"
}
step C "radix on: bf16 window, ${BF16_SLOTS:-1100} slots" bf16
step K_bf16 "two-turn radix check: bf16 window (the reference)" check_turns on_bf16_window_r1
step T_field "radix on: LeapQuant ${LQ_SLOTS:-1310} slots, auto, slot copy field by field (LEAPQUANT_FUSED_SLOT_COPY=0)" lq field 0
step K_field "two-turn radix check: LeapQuant, copy field by field, against bf16 window" check_turns on_lq_field_r1
step T_fused "radix on: LeapQuant ${LQ_SLOTS:-1310} slots, auto, fused slot copy (LEAPQUANT_FUSED_SLOT_COPY=1)" lq fused 1
step K_fused "two-turn radix check: LeapQuant, fused copy, against bf16 window" check_turns on_lq_fused_r1
if grep -l "does not load" "$RESULTS"/server_on_lq_*.log 2> /dev/null; then
    echo "WARNING: a LeapQuant server fell back to the Triton fold" >> "$STATUS"
fi
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
