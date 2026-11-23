#!/usr/bin/env bash
# serving_ab_policy.sh: serving A/B of the fold policy (integrations/sglang/apply_fold_policy.py), radix cache off:
# bf16 window and LeapQuant with each fold policy of POLICIES (default "default auto"; "default" = unset:
# aligned folds, track-boundary flushes kept; "auto": flushes dropped with the radix cache off; "stagger": each request
# on its own phase). One session, 560 slots. Labels off_<arm>_r1: bf16_window, lq_policy_<policy>.
#   RESULTS=results/ab-policy runbook/serving_ab_policy.sh          (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
export SERVING_POINTS="${SERVING_POINTS:-decode:512 aa1k:512}"
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
bf16() { serve off_bf16_window_r1 --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS $FLAGS; }
lq() {             # lq LABEL POLICY
    export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND="${LEAPQUANT_DECODE_BACKEND:-cuda}"
    export LEAPQUANT_FOLD_POLICY="$2"
    serve "off_$1_r1" --mamba-ssm-dtype float32 $WINDOW_FLAGS --enable-linear-leapquant $FLAGS
}
step A "bf16 window" bf16
for policy in ${POLICIES:-default auto}; do
    value="$policy"; [ "$policy" = "default" ] && value=""
    step "P_$policy" "LeapQuant, LEAPQUANT_FOLD_POLICY=${value:-(unset)}" lq "lq_policy_$policy" "$value"
done
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
