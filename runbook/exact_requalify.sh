#!/usr/bin/env bash
# exact_requalify.sh (backlog B40): the exact CUDA kernel after the low-bit change of 14 Nov (LQ_MAX_COMPENSATORS 6,
# grouped steps): every fold bit-identical to the reference core on the 9 recorded traces (8-bit, the server's
# settings), and a grouped 6-bit case (rank 6, groups of 8) through the CPU kernel against the reference.
#   RESULTS=results/exact-requalify runbook/exact_requalify.sh        (from the repository root, make cuda done)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
TRACES="${TRACES:-$HOME/leapquant/traces}"
for trace in "$TRACES"/{explain,code,reasoning}/trace_layer_{00,13,30}.bin; do
    label="$(basename "$(dirname "$trace")")_$(basename "$trace" .bin)"
    python3 kernel/verify_kernel.py "$trace" --device cuda --rows 1 > "$RESULTS/exact_$label.txt" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 2; done
done
wait
{
    echo "trace                                   bit-identical folds        criterion C (free run)   result"
    for f in "$RESULTS"/exact_*.txt; do
        printf "%-40s %-26s %-24s %s\n" "$(basename "$f" .txt)" \
            "$(grep -oE 'bit-identical to the reference: [0-9]+' "$f" | head -1 | grep -oE '[0-9]+$')/255" \
            "$(grep -E 'at most 10% worse' "$f" | grep -oE 'pass|FAIL' | head -1)" "$(grep -oE 'RESULT: [A-Z]+' "$f")"
    done
} > "$RESULTS/SUMMARY.txt"
# The grouped low-bit path on the CPU kernel against the reference (the exact GPU kernel shares its code).
make kernel-test > "$RESULTS/kernel_test.txt" 2>&1
tail -3 "$RESULTS/kernel_test.txt" >> "$RESULTS/SUMMARY.txt"
cat "$RESULTS/SUMMARY.txt"
