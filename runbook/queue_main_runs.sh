#!/usr/bin/env bash
# queue_main_runs.sh: the main GPU runs of 14 Nov 2026, one after the other, with the kernel build chosen by the
# kernel checks. The SGLang layer-switch edit (it changes the SGLang checkout) runs last.
#   BUILD=build_v6 DECODE=triton runbook/queue_main_runs.sh            (from the repository root, detached)
# Order: the paired serving round against bf16 window first (the open question), then the open-loop run (tail
# latency and goodput), then the rest.
set -u
cd "$(dirname "$0")/.."
export LEAPQUANT_BUILD_DIR="$PWD/${BUILD:-build}"
export LEAPQUANT_DECODE_BACKEND="${DECODE:-triton}"
OUT="$HOME/lqexp/results"
echo "start $(date -u +%FT%TZ) build $LEAPQUANT_BUILD_DIR decode $LEAPQUANT_DECODE_BACKEND" >> "$OUT/QUEUE.txt"
insert() {          # a short GPU job dropped in between two runs: $OUT/INSERT.sh, run once, then kept renamed
    [ -f "$OUT/INSERT.sh" ] || return 0
    local stamp
    stamp="$(date -u +%H%M%S)"
    mv "$OUT/INSERT.sh" "$OUT/INSERT_$stamp.sh"
    echo "insert $stamp start" >> "$OUT/QUEUE.txt"
    bash "$OUT/INSERT_$stamp.sh" > "$OUT/INSERT_$stamp.out" 2>&1
    local status=$?
    echo "insert $stamp end $(date -u +%H:%M) exit $status" >> "$OUT/QUEUE.txt"
}
run() {             # run NAME script [env...]
    local name="$1" script="$2"; shift 2
    insert
    echo "$name start $(date -u +%H:%M)" >> "$OUT/QUEUE.txt"
    env RESULTS="$OUT/$name" "$@" bash "runbook/$script" > "$OUT/$name.out" 2>&1
    local status=$?
    echo "$name end $(date -u +%H:%M) exit $status" >> "$OUT/QUEUE.txt"
}
run round10 serving_round10.sh REPS=1
run openloop serving_openloop.sh
# Radix cache on with the state memory equal: bf16 window 1,100 slots (~26 GB) against LeapQuant 1,310 slots (~26 GB).
run round9-equalmem serving_round9.sh ONLY_ON=1 REPS=1 LQ_SLOTS=1310
run accuracy-paper accuracy_paper.sh
run sglang-best serving_sglang_best.sh LQ_DECODE="$LEAPQUANT_DECODE_BACKEND"
run retraction retraction_check.sh
run exact-requalify exact_requalify.sh
run layer-switch layer_switch_check.sh
insert
echo "done $(date -u +%FT%TZ)" >> "$OUT/QUEUE.txt"
