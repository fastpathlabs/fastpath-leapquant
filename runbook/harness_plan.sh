#!/usr/bin/env bash
# harness_plan.sh: runs one campaign plan (harness/config/campaign.json) through the test harness on the GPU host:
# servers per candidate and block (balanced order), every point's raw evidence, validation and metrics.
#   RESULTS=$HOME/lqexp/results/harness PLAN=pilot_128_512_c512 runbook/harness_plan.sh   (from the tree root)
# The bundle (HTML, CSV/JSON, sidecars, checker) is built afterwards, on any machine with the results:
#   python -m lqh report RESULTS/PLAN --out reports/harness_PLAN_<UTC> && python -m lqh check reports/harness_PLAN_<UTC>
set -u
cd "$(dirname "$0")/.."
VENV="${VENV_SGLANG:-$HOME/venv-sglang}"
export LQH_INFX_DIR="${LQH_INFX_DIR:-$HOME/lqharness/upstream/inferencex-e2e}"
export LQH_INFX_PYTHON="${LQH_INFX_PYTHON:-$VENV/bin/python}"
export PATH="/usr/local/cuda/bin:$VENV/bin:$PATH"
mkdir -p "$RESULTS"
# 15 Nov 2026: the AA-SLT run at 1,024 users hit the 1,024 open-file limit (OSError 24) and replaced every failed
# connection at once: 2.3 million failed attempts per arm. Raise the soft limit to the hard limit.
ulimit -n "$(ulimit -Hn)" 2> /dev/null || true
echo "open-file limit: $(ulimit -n)"
# CAMPAIGN: another campaign file (default harness/config/campaign.json), e.g. harness/config/campaign_q38_27b.json
# Exit status: the runner's (3 when any point crashed, was invalid or did not run); pipefail keeps it through tee.
set -o pipefail
PYTHONPATH=harness "$VENV/bin/python" -m lqh run "${PLAN:?PLAN is required}" --out "$RESULTS" --port "${PORT:-30000}" \
    ${BLOCKS:+--blocks $BLOCKS} ${CAMPAIGN:+--campaign $CAMPAIGN} 2>&1 | tee -a "$RESULTS/${PLAN}.log"
