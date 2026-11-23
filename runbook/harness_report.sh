#!/usr/bin/env bash
# harness_report.sh: the campaign report, rebuilt after every harness job (and on demand), into a NEW folder.
#   1. carried evidence: every earlier client log of ~/lqexp/results (python -m lqh.carried serving) and the strict
#      paper-lane accuracy runs against the fp32 window reference (python -m lqh.carried accuracy);
#   2. python -m lqh report over ~/lqexp/results/harness (plan runs, kernel cycle, sanitizers, lifecycle, quality,
#      NLL, compounding, plateau, carried) into ~/lqexp/reports/harness-campaign_<UTC>;
#   3. python -m lqh check on it; the outcome is appended to ~/lqexp/results/harness/REPORTS.txt.
#   runbook/harness_report.sh "after job 100"          (from the tree root)
set -u
cd "$(dirname "$0")/.."
VENV="${VENV_SGLANG:-$HOME/venv-sglang}"
# HARNESS_RESULTS / CAMPAIGN / REPORT_PREFIX: another campaign (the Qwen3.8-27B one: ~/lqexp/results/q38/harness,
# harness/config/campaign_q38_27b.json, harness-q38). The carried 9B evidence is added only to the default campaign.
H="${HARNESS_RESULTS:-$HOME/lqexp/results/harness}"
mkdir -p "$H/carried" "$HOME/lqexp/reports"
PY="env PYTHONPATH=harness $VENV/bin/python"
if [ -z "${CAMPAIGN:-}" ]; then
$PY -m lqh.carried serving "$HOME/lqexp/results" --out "$H/carried/serving" > "$H/carried/serving.log" 2>&1
REF="$HOME/lqexp/results/accuracy-paper/paper_fp32_window.json"
ARMS=""
for arm in leapquant bf16_window; do
    f="$HOME/lqexp/results/accuracy-strict/paper_strict_$arm.json"
    [ -s "$f" ] && ARMS="$ARMS --arm strict_$arm=$f"
done
if [ -s "$REF" ] && [ -n "$ARMS" ]; then
    $PY -m lqh.carried accuracy --reference fp32_window=$REF $ARMS --out "$H/carried/accuracy-strict" \
        --label "carried: strict paper lane (B58), AIME 2024/2025, 4 samples, T 1.0, 81,920 tokens, vs fp32 window" \
        > "$H/carried/accuracy.log" 2>&1
fi
fi
STAMP=$(date -u +%Y%m%d-%H%M%S)
OUT="$HOME/lqexp/reports/${REPORT_PREFIX:-harness-campaign}_$STAMP"
$PY -m lqh report "$H" --out "$OUT" ${CAMPAIGN:+--campaign "$CAMPAIGN"} --title "LeapQuant H100 campaign, ${1:-on demand} ($STAMP UTC)" > "$H/report_$STAMP.log" 2>&1
$PY -m lqh check "$OUT" >> "$H/report_$STAMP.log" 2>&1
status=$?
echo "$STAMP ${1:-on demand}: $OUT check exit $status" >> "$H/REPORTS.txt"
echo "$OUT (check exit $status)"
