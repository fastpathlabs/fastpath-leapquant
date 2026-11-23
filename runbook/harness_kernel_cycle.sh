#!/usr/bin/env bash
# harness_kernel_cycle.sh: test-harness order of work, step 1 (prompt rev. 3, sections 0, 6, 14): time the complete,
# state-advancing layer cycle with the CUDA fold on ONE build, with its log. Until this exists, 0.332 ms per
# layer-token (and the component-sum 0.3004 ms of results/g3-final) are estimates, not results.
# Method (kernel/bench_triton.py --full-cycle): one CUDA graph holds the 16 decode steps (cursor 0..15, each with its
# own inputs, appending its record) and the fold, replayed from the same valid start state restored outside the timed
# part; SGLang's fp32 and bf16 window kernels are timed the same way in the same process. Repetitions alternate which
# kernel is timed first (the agent's finding: a kernel's time can depend on what ran before it). Batch 512 is the
# gate cell; 128, 32, 8 and 1 are the small-batch diagnostics.
# Output: rep_<n>.txt (full bench output, one per repetition), gpu_<n>.csv (clocks, power, temperature during the
# repetition), manifest.txt (build identity), kernel_cycle.json (per-rep values, paired ratios, interval).
#   RESULTS=$HOME/lqexp/results/harness/kernel-cycle REPS=10 runbook/harness_kernel_cycle.sh   (from the tree root)
set -u
cd "$(dirname "$0")/.."
source "${VENV_SGLANG:-$HOME/venv-sglang}/bin/activate"
mkdir -p "$RESULTS"
runbook/manifest.sh "$RESULTS/manifest.txt"
export LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda LEAPQUANT_CUDA_DECODE_VARIANT="${LEAPQUANT_CUDA_DECODE_VARIANT:-9}"
B="python3 kernel/bench_triton.py --batch-sizes ${BATCHES:-512 128 32 8 1} --windows 2 --no-exact-times --fold-backend cuda --decode-backend cuda --with-sglang --full-cycle"
started=$(date +%s)
for rep in $(seq 1 "${REPS:-10}"); do
    # BUDGET_S: no new repetition after this many seconds (the analysis below uses the ones that finished)
    if [ $(( $(date +%s) - started )) -ge "${BUDGET_S:-1000000}" ]; then
        echo "rep $rep: not run (budget of ${BUDGET_S} s used up)" >> "$RESULTS/STATUS.txt"; break
    fi
    order=lq-first
    [ $((rep % 2)) -eq 0 ] && order=sglang-first
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu --format=csv -lms 500 \
        > "$RESULTS/gpu_$rep.csv" 2> /dev/null &
    sampler=$!
    timeout 1800 $B --cycle-order "$order" > "$RESULTS/rep_$rep.txt" 2>&1
    echo "rep $rep ($order): exit $?" >> "$RESULTS/STATUS.txt"
    kill "$sampler" 2> /dev/null; wait "$sampler" 2> /dev/null
done
PYTHONPATH=harness python3 -m lqh.kernel_cycle "$RESULTS" | tee "$RESULTS/kernel_cycle.txt"
