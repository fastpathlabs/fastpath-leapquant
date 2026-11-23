#!/usr/bin/env bash
# serving_ab_trackflush.sh: backlog B55. Radix cache ON at equal state memory (bf16 window 1,100 slots against
# LeapQuant 1,310: ~26 GB each, as results/radix-equalmem-final), one session, round 9's radix-on shapes:
#   C          bf16 window
#   T_auto     LeapQuant, LEAPQUANT_FOLD_POLICY=auto (radix on: each request folds on its own step, track flushes
#              kept: the radix-on setting of results/radix-equalmem-final)
#   T_aligned  LeapQuant, policy unset (aligned folds), LEAPQUANT_TRACK_FLUSH=0 (no track-boundary folds)
#   T_stagger  LeapQuant, LEAPQUANT_FOLD_POLICY=stagger, LEAPQUANT_TRACK_FLUSH=0
# Before its load each server answers two-turn prompts (integrations/sglang/radix_turns_client.py): the second turn
# starts from a state the radix cache kept when the first (600 tokens, crossing track boundaries) finished. LeapQuant's
# answers are compared with the bf16 window's; the verdict is step K_<arm>. Labels on_<arm>_r1.
# Needs integrations/sglang/apply_track_flush.py applied to the SGLang checkout.
#   RESULTS=results/ab-trackflush runbook/serving_ab_trackflush.sh          (from the repository root)
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
BACKEND_FILE="${SGLANG_DIR:-$HOME/sglang}/python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
grep -q "LEAPQUANT_TRACK_FLUSH" "$BACKEND_FILE" \
    || { echo "apply integrations/sglang/apply_track_flush.py first ($BACKEND_FILE)" | tee -a "$STATUS"; exit 1; }

step() {           # step NAME "description" command...
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
turns() {          # turns LABEL: the two-turn check on the running server; its exit status goes to turns_LABEL.status
    local label="$1" compare=""
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
    turns "$label"
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
lq() {             # lq ARM POLICY TRACK_FLUSH
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    export LEAPQUANT_FOLD_POLICY="$2" LEAPQUANT_TRACK_FLUSH="$3"
    serve "on_lq_$1_r1" --mamba-ssm-dtype float32 $RADIX $WINDOW --enable-linear-leapquant $GRAPHS \
        --max-running-requests 512 --max-mamba-cache-size "${LQ_SLOTS:-1310}"
}
step C "radix on: bf16 window, ${BF16_SLOTS:-1100} slots" bf16
step K_bf16 "two-turn radix check: bf16 window (the reference)" check_turns on_bf16_window_r1
step T_auto "radix on: LeapQuant ${LQ_SLOTS:-1310} slots, LEAPQUANT_FOLD_POLICY=auto (own-step folds, track flushes kept)" lq auto auto ""
step K_auto "two-turn radix check: LeapQuant auto against bf16 window" check_turns on_lq_auto_r1
step T_aligned "radix on: LeapQuant ${LQ_SLOTS:-1310} slots, aligned folds, LEAPQUANT_TRACK_FLUSH=0" lq aligned "" 0
step K_aligned "two-turn radix check: LeapQuant aligned, no track flushes, against bf16 window" check_turns on_lq_aligned_r1
step T_stagger "radix on: LeapQuant ${LQ_SLOTS:-1310} slots, LEAPQUANT_FOLD_POLICY=stagger, LEAPQUANT_TRACK_FLUSH=0" lq stagger stagger 0
step K_stagger "two-turn radix check: LeapQuant stagger, no track flushes, against bf16 window" check_turns on_lq_stagger_r1
if grep -l "does not load" "$RESULTS"/server_on_lq_*.log 2> /dev/null; then
    echo "WARNING: a LeapQuant server fell back to the Triton fold" >> "$STATUS"
fi
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
