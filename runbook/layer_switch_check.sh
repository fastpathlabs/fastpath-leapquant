#!/usr/bin/env bash
# layer_switch_check.sh: state-cache plan v2, task TQ.5 (backlog B30). Applies the LEAPQUANT_LAYERS switch to the
# SGLang checkout (integrations/sglang/apply_layer_switch.py), then:
#   L0  fp32 + window server: reference answers (smoke client)
#   L1  LeapQuant on all layers (mirror state): the answers agree with L0 as in the earlier smoke tests
#   L2  LeapQuant on layer 30 only, L3 on layers 0-13 only: they agree with L0 too, and the server log names the subset
#   L4  compressed state with a subset: the server must refuse to start
# With a subset the ReplaySSM ring keeps its full size (apply_layer_switch_ring.py, B30: with one slot the window
# kernel of the other layers wrote out of bounds), ~5 GB outside SGLang's budget: L2 and L3 reserve less for the KV
# cache (SUBSET_MEM_FRACTION, default 0.70).
#   RESULTS=results/layer-switch runbook/layer_switch_check.sh      (from the repository root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
STATUS="$RESULTS/STATUS.txt"
touch "$STATUS"
FLAGS="--mamba-ssm-dtype float32 $WINDOW_FLAGS --disable-radix-cache"
step() {
    local name="$1" description="$2"; shift 2
    if grep -q "^$name PASS" "$STATUS"; then echo "skip $name"; return 0; fi
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/$name.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null; sleep 5
    sed -i "/^$name /d" "$STATUS"
    echo "$name $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
reference() {
    start_server "$RESULTS/server_window.log" $FLAGS || return 1
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_window.json"
}
layers() {          # layers SPEC LABEL [server flags...]
    export LEAPQUANT_STATE=mirror LEAPQUANT_LAYERS="$1"
    local label="$2"; shift 2
    start_server "$RESULTS/server_layers_$label.log" $FLAGS --enable-linear-leapquant "$@" || return 1
    python3 integrations/sglang/smoke_client.py --port "$PORT" --out "$RESULTS/smoke_layers_$label.json" \
        --compare "$RESULTS/smoke_window.json" --max-early-differences 2
}
refuses() {
    export LEAPQUANT_STATE=compressed LEAPQUANT_LAYERS=30
    if start_server "$RESULTS/server_layers_compressed_subset.log" $FLAGS --enable-linear-leapquant; then
        echo "UNEXPECTED: the server started"; return 1
    fi
    grep -q "LEAPQUANT_LAYERS selects a subset" "$RESULTS/server_layers_compressed_subset.log"
}
python3 integrations/sglang/apply_layer_switch.py "${SGLANG_DIR:-$HOME/sglang}" || exit 1
python3 integrations/sglang/apply_layer_switch_ring.py "${SGLANG_DIR:-$HOME/sglang}" || exit 1
step L0 "fp32 + window: reference answers" reference
step L1 "LeapQuant on all layers (mirror)" layers all all
step L2 "LeapQuant on layer 30 only (mirror)" layers 30 l30 --mem-fraction-static "${SUBSET_MEM_FRACTION:-0.70}"
step L3 "LeapQuant on layers 0-13 only (mirror)" layers 0-13 l0_13 --mem-fraction-static "${SUBSET_MEM_FRACTION:-0.70}"
step L4 "compressed state with a subset of layers: refused at start" refuses
echo "done $(date -u +%FT%TZ)" >> "$STATUS"
