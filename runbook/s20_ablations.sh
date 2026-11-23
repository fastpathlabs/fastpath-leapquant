#!/usr/bin/env bash
# Plan 5, step S20: precision ablations of LeapQuant on the 9 recorded traces (CPU replays of the reference core),
# and backlog B41: the paper's low-bit settings on the same traces.
# Baseline: window 16, 4 compensators, 8 bits, 2 sweeps (the settings the server uses).
#
#   OUT=results/s20_ablations runbook/s20_ablations.sh [name:options ...]     (from the repository root, after make)
#
# Since 14 Nov 2026, --bits 6 or 4 WITHOUT --compensators and --scale-group follow the paper (rank 6, one step per
# group of 8 key rows). The original S20 runs (rank 4, one step per value column) are therefore pinned explicitly
# as b6 and b4; b6p and b4p are the paper's settings. Existing outputs are never overwritten.
set -u
cd "$(dirname "$0")/.."
OUT="${OUT:-$PWD/results/s20_ablations}"
TRACES="${TRACES:-$HOME/leapquant/traces}"
JOBS="${JOBS:-12}"
mkdir -p "$OUT"
CONFIGS=("base:" "w8:--window 8" "b6:--bits 6 --compensators 4 --scale-group 0"
         "b4:--bits 4 --compensators 4 --scale-group 0" "r2:--compensators 2" "r8:--compensators 8"
         "b6p:--bits 6" "b4p:--bits 4")
[ $# -gt 0 ] && CONFIGS=("$@")
run() {   # name trace options...
    local name="$1" trace="$2"; shift 2
    local label="$(basename "$(dirname "$trace")")_$(basename "$trace" .bin)"
    local file="$OUT/${name}_${label}.txt"
    [ -s "$file" ] && grep -q "fp32" "$file" && return 0
    nice -n 19 ./leapquant_replay "$trace" --sweeps 2 "$@" > "$file" 2>&1
}
for trace in "$TRACES"/{explain,code,reasoning}/trace_layer_{00,13,30}.bin; do
    for config in "${CONFIGS[@]}"; do
        name="${config%%:*}"; options="${config#*:}"
        # shellcheck disable=SC2086
        run "$name" "$trace" $options &
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 2; done
    done
done
wait
python3 kernel/ablation_table.py "$OUT" > "$OUT/TABLE.txt" 2>&1
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/DONE"
