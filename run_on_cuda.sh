#!/usr/bin/env bash
# =============================================================================
# run_on_cuda.sh
#
# Runs the whole LeapQuant test plan on the GPU machine, step by step.
#
#   ./run_on_cuda.sh                 run every step that has not passed yet
#   ./run_on_cuda.sh --list          show the steps and their state
#   ./run_on_cuda.sh --only d        run one step (again)
#   ./run_on_cuda.sh --from c        run this step and everything after it
#   ./run_on_cuda.sh --to tune       stop after this step
#
# Needs: one NVIDIA GPU (the plan assumes an 80 GB H100), nvcc from a CUDA
# toolkit whose major version matches PyTorch's, git, curl, Python 3.12, and
# access to Hugging Face for Qwen/Qwen3.5-9B and the benchmark datasets.
#
# How it behaves
#   - A step that passed is skipped on the next run. A step that failed is
#     tried again. So the script can simply be started again after a fix.
#   - A failed step stops only the steps that need it. The rest still run.
#   - Each step logs to results/run-<host>/steps/<step>.log, with a manifest
#     (<step>.manifest.txt: source and binary hashes, versions) beside it.
#   - "passed" means the step's script exited with status 0. Read its log for
#     what it measured; the reports say what the numbers mean.
#   - The two GATES of the plan are not decided by this script. It prints the
#     numbers; you decide whether to go on.
#
# Honest status of what this runs for the first time
#   - `make cuda`: the GPU kernel has never been compiled by nvcc.
#   - Every SGLang and vLLM command: taken from the engine source, never run.
#   - The SGLang patch and the vLLM patch: byte-compiled only.
#   Expect to fix things at steps setup, gpu-check, b2 and f-smoke.
#
# Useful settings (environment variables)
#   MODEL=Qwen/Qwen3.5-9B   PORT=30000   TOKENS=4096   ACC_LIMIT=300
#   SKIP="b3 b4"            steps to leave out (the vLLM steps are g-patch g-smoke g-forward)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"
export LEAPQUANT_HOME="$PWD"
export RESULTS="${RESULTS:-$LEAPQUANT_HOME/results/run-$(hostname -s)}"
STEP_DIR="$RESULTS/steps"
mkdir -p "$STEP_DIR"
source runbook/env.sh

# step id | steps it needs | what it is
STEPS=(
"setup||environments, every CPU check, build of the GPU kernel (stage A)"
"gpu-check|setup|first run of the GPU kernel: engine test and reference comparison on a synthetic trace"
"tune|gpu-check|try the kernel's build options, keep the fastest correct build"
"b1|setup|SGLang kernel microbenchmark: packed decode against its window decode (stage B1)"
"b2|setup|serving baselines: fp32, fp32 + window, bf16 + window (stage B2)  -- GATE 1"
"c|setup|record Qwen3.5-9B traces and replay them on the CPU (stage C)  -- GATE 2"
"d|gpu-check c|GPU kernel against the reference on the recorded traces (stage D)"
"e|gpu-check|kernel speed next to SGLang's kernels (stage E)"
"f-patch|gpu-check|apply the LeapQuant patch to SGLang"
"f-smoke|f-patch|same prompts with the flag off and on; compare the answers"
"f-serving|f-smoke|serving benchmark with LeapQuant on (stage F)"
"f-accuracy|f-smoke|GSM8K, AIME 2025, MMLU-Pro: fp32 twice, then LeapQuant (task T12)"
"f-cache|f-smoke|prefix-cache capacity with LeapQuant checkpoints (task T11)"
"b4|setup|prefix-cache capacity today, fp32 and SGLang int8 (stage B4)"
"b3|setup|share of GPU time in the decode kernel, measured with vLLM (stage B3)"
"g-patch|gpu-check|vLLM from source with the LeapQuant first port (task T14)"
"g-smoke|g-patch|vLLM: same prompts with LeapQuant off and on"
"g-forward|g-smoke|vLLM: decode-step time with LeapQuant off and on"
)

run_step() {
    case "$1" in
    setup)      runbook/stage_a_setup.sh ;;
    gpu-check)
        ( source "$VENV_TRACE/bin/activate"
          set -e
          python3 kernel/engine_test.py --device cuda
          python3 -c "import sys; sys.path.insert(0, '.'); from record_trace import run_self_test; run_self_test('build/gpu_check_trace.bin', heads=32, key_channels=128, value_channels=128, tokens=400)"
          python3 kernel/verify_kernel.py build/gpu_check_trace.bin --device cuda --rows 2 ) ;;
    tune)       ( source "$VENV_TRACE/bin/activate" && python3 kernel/tune_kernel.py ) ;;
    b1|b2|b3|b4) runbook/stage_b_baselines.sh "$1" ;;
    c)          runbook/stage_c_traces.sh ;;
    d)          runbook/stage_d_e_kernel.sh d ;;
    e)          runbook/stage_d_e_kernel.sh e ;;
    f-patch)    runbook/stage_f_end_to_end.sh patch ;;
    f-smoke)    runbook/stage_f_end_to_end.sh smoke ;;
    f-serving)  runbook/stage_f_end_to_end.sh serving ;;
    f-accuracy) runbook/stage_f_end_to_end.sh accuracy ;;
    f-cache)    runbook/stage_f_end_to_end.sh cache ;;
    g-patch)    runbook/stage_g_vllm.sh patch ;;
    g-smoke)    runbook/stage_g_vllm.sh smoke ;;
    g-forward)  runbook/stage_g_vllm.sh forward ;;
    *) echo "no such step: $1"; return 2 ;;
    esac
}

state_of() {
    if [ -f "$STEP_DIR/$1.passed" ]; then echo passed
    elif [ -f "$STEP_DIR/$1.failed" ]; then echo FAILED
    else echo "not run"; fi
}

list_steps() {
    printf "%-11s %-8s %s\n" "step" "state" "what it does"
    for entry in "${STEPS[@]}"; do
        IFS='|' read -r id needs text <<< "$entry"
        printf "%-11s %-8s %s\n" "$id" "$(state_of "$id")" "$text"
    done
}

ONLY=""; FROM=""; TO=""
while [ $# -gt 0 ]; do
    case "$1" in
    --list) list_steps; exit 0 ;;
    --only) ONLY="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to)   TO="$2"; shift 2 ;;
    -h|--help) sed -n 2,36p "$0"; exit 0 ;;
    *) echo "unknown option $1 (try --help)"; exit 2 ;;
    esac
done

started=0; [ -z "$FROM" ] && started=1
for entry in "${STEPS[@]}"; do
    IFS='|' read -r id needs text <<< "$entry"
    [ "$id" = "$FROM" ] && started=1
    [ "$started" = 1 ] || continue
    if [ -n "$ONLY" ] && [ "$id" != "$ONLY" ]; then continue; fi
    if [[ " ${SKIP:-} " == *" $id "* ]]; then echo "== $id: skipped (SKIP)"; continue; fi
    if [ -z "$ONLY" ] && [ "$id" != "$FROM" ] && [ -f "$STEP_DIR/$id.passed" ]; then
        echo "== $id: passed earlier, skipped"
    else
        blocked=""
        for need in $needs; do [ -f "$STEP_DIR/$need.passed" ] || blocked="$need"; done
        if [ -n "$blocked" ]; then
            echo "== $id: not run, it needs step '$blocked' to pass first"
        else
            echo
            echo "================================================================================"
            echo "== $id: $text"
            echo "== log: $STEP_DIR/$id.log"
            echo "================================================================================"
            rm -f "${STEP_DIR:?}/${id:?}.passed" "${STEP_DIR:?}/${id:?}.failed"
            start_time=$(date +%s)
            runbook/manifest.sh "$STEP_DIR/$id.manifest.txt"
            run_step "$id" 2>&1 | tee "$STEP_DIR/$id.log"
            status=${PIPESTATUS[0]}
            minutes=$(( ($(date +%s) - start_time) / 60 ))
            if [ "$status" = 0 ]; then
                touch "$STEP_DIR/$id.passed"; echo "== $id: passed (${minutes} min)"
            else
                touch "$STEP_DIR/$id.failed"; echo "== $id: FAILED with status $status (${minutes} min)"
                # Never leave a server behind for the next step.
                pkill -f "sglang.launch_server.*--port $PORT" 2> /dev/null || true
                pkill -f "vllm serve.*--port $PORT" 2> /dev/null || true
            fi
            case "$id" in
            b2) echo; echo "GATE 1. Compare output token throughput of config2 (window) with config1 (no window):"
                grep -H -i "output token throughput" "$RESULTS"/serving_config[12]_*.log 2> /dev/null
                echo "If the window gives no gain at your target concurrency, LeapQuant will not either: stop here." ;;
            c)  echo; echo "GATE 2. In every table below, compare the LeapQuant row with 'bf16, rounded every window'."
                echo "If bf16 is as accurate, it is the simpler product (2 bytes per entry instead of 1.19)."
                grep -h -E "^trace:|self-check|bf16, rounded every window|= LeapQuant" "$RESULTS"/c_replay_*.txt 2> /dev/null ;;
            esac
        fi
    fi
    [ "$id" = "$TO" ] && break
done

echo
echo "================================ summary ================================"
list_steps
echo
echo "Logs and measurements: $RESULTS"
echo "Fill in results/README.md from them, and check the directory in."
[ -z "$(ls "$STEP_DIR"/*.failed 2> /dev/null)" ]
