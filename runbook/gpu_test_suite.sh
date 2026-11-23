#!/usr/bin/env bash
# =============================================================================
# gpu_test_suite.sh
#
# The GPU test suite, as ONE unattended job.
# It keeps running when the connection that started it goes away.
#
#   Start (on the GPU machine):
#       tmux new-session -d -s suite 'cd ~/leapquant && runbook/gpu_test_suite.sh'
#   Look at it later, from anywhere:
#       ssh <machine> 'cat ~/leapquant/results/gpu-suite/STATUS.txt'
#
# Every test writes one line to STATUS.txt: its number, PASS or FAIL, the minutes
# it took, and what it is. Its full output is in T<number>.log. A failed test
# does not stop the suite. Tests that already passed are skipped when the suite
# is started again (delete STATUS.txt to run everything again).
#
# PASS means: the command exited with status 0. For the verification tests that
# is `RESULT: PASS` of kernel/verify_kernel.py. For benchmarks it only means the
# benchmark ran; read the numbers.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
export LEAPQUANT_HOME="$PWD"
export RESULTS="${RESULTS:-$LEAPQUANT_HOME/results/gpu-suite}"
export MAMBA_SLOTS="${MAMBA_SLOTS:-384}" LQ_MEM_FRACTION="${LQ_MEM_FRACTION:-0.72}"
export SERVING_POINTS="${SERVING_POINTS:-long:32 short:64 short:128 short:512}"
source runbook/env.sh
STATUS="$RESULTS/STATUS.txt"
TRACES="${TRACES:-$LEAPQUANT_HOME/traces}"
touch "$STATUS"
echo "suite started $(date -u +%Y-%m-%dT%H:%MZ) on $(hostname -s)" >> "$RESULTS/suite.log"

# t <number> <description> <command ...>
t() {
    local number="$1" description="$2"; shift 2
    if grep -q "^T$number PASS" "$STATUS"; then return 0; fi
    sed -i "/^T$number /d" "$STATUS"
    echo "T$number RUNNING since $(date -u +%H:%MZ)  $description" >> "$STATUS"
    runbook/manifest.sh "$RESULTS/T$number.manifest.txt"
    local start=$(date +%s) verdict=FAIL
    if ( "$@" ) > "$RESULTS/T$number.log" 2>&1; then verdict=PASS; fi
    pkill -f "[s]glang.launch_server.*--port $PORT" 2> /dev/null
    sed -i "/^T$number /d" "$STATUS"
    echo "T$number $verdict $(( ($(date +%s) - start) / 60 )) min  $description" >> "$STATUS"
}
trace_py() { ( source "$VENV_TRACE/bin/activate" && "$@" ); }
sglang_py() { ( source "$VENV_SGLANG/bin/activate" && "$@" ); }

verify_three() {      # three traces at a time: the reference core runs on the CPU
    local layer="$1" status=0
    for prompt in explain code reasoning; do
        trace_py python3 kernel/verify_kernel.py "$TRACES/$prompt/trace_layer_$layer.bin" --device cuda --rows 1 \
            > "$RESULTS/d_verify_${prompt}_trace_layer_$layer.txt" 2>&1 &
    done
    wait
    for prompt in explain code reasoning; do
        tail -1 "$RESULTS/d_verify_${prompt}_trace_layer_$layer.txt"
        grep -q "RESULT: PASS" "$RESULTS/d_verify_${prompt}_trace_layer_$layer.txt" || status=1
    done
    return $status
}

smoke_graphs() {      # LeapQuant server with CUDA graphs off, then on: same prompts, compare
    sglang_py bash -c "source runbook/env.sh
        start_server '$RESULTS/server_lq_nograph.log' --mamba-ssm-dtype float32 \$WINDOW_FLAGS \$LEAPQUANT_FLAGS --disable-cuda-graph &&
        python3 integrations/sglang/smoke_client.py --port \$PORT --out '$RESULTS/smoke_lq_nograph.json'; stop_server
        start_server '$RESULTS/server_lq_graph.log' --mamba-ssm-dtype float32 \$WINDOW_FLAGS \$LEAPQUANT_FLAGS &&
        python3 integrations/sglang/smoke_client.py --port \$PORT --out '$RESULTS/smoke_lq_graph.json' --compare '$RESULTS/smoke_lq_nograph.json'; stop_server"
}

all_layers() {        # one prompt, every linear-attention layer: replay table and fit quality per layer
    trace_py python3 record_trace.py --model "$MODEL" --max-new-tokens 2048 --layers all \
        --out-dir "$TRACES/all_layers" --prompt-file runbook/prompts/prompt_reasoning.txt || return 1
    for trace in "$TRACES"/all_layers/trace_layer_*.bin; do
        ./leapquant_replay "$trace" --sweeps 2 > "$RESULTS/s21_replay_$(basename "$trace" .bin).txt" || return 1
    done
    trace_py python3 kernel/fit_quality.py "$TRACES"/all_layers/trace_layer_*.bin --every 512 --sweeps 2 > "$RESULTS/s21_fit_quality_all_layers.txt"
    grep -h -E "^trace:|bf16, rounded every window|= LeapQuant" "$RESULTS"/s21_replay_*.txt | paste - - - | awk '{print $2, $NF}' | head -0
    for f in "$RESULTS"/s21_replay_*.txt; do
        printf "%s  " "$(basename "$f" .txt | sed 's/s21_replay_trace_//')"
        grep -E "bf16, rounded every window|= LeapQuant" "$f" | awk -F'|' '{printf "%s ", $NF}'; echo
    done | tee "$RESULTS/s21_layer_table.txt"
}

sweeps() {            # S20: ranks 0, 2, 4, 8 and window 8 on the nine traces
    for trace in "$TRACES"/{explain,code,reasoning}/trace_layer_*.bin; do
        name="$(basename "$(dirname "$trace")")_$(basename "$trace" .bin)"
        for rank in 0 2 8; do ./leapquant_replay "$trace" --sweeps 2 --compensators $rank > "$RESULTS/s20_rank${rank}_$name.txt" & done
        ./leapquant_replay "$trace" --sweeps 2 --window 8 > "$RESULTS/s20_window8_$name.txt" &
        ./leapquant_replay "$trace" --sweeps 8 > "$RESULTS/s20_sweeps8_$name.txt" &
        wait
    done
    grep -c "self-check" "$RESULTS"/s20_*.txt | tail -1
}

# ---- build and CPU checks ---------------------------------------------------------------
t 01 "build: reference, CPU kernel, GPU kernel for sm_90"            bash -c "make && make clean-cuda cuda && sha256sum build/libleapquant_cuda.so"
t 02 "CPU check suite (make check)"                                  trace_py make check PYTHON=python3
# ---- kernel conformance on the GPU --------------------------------------------------------
t 03 "engine glue on the GPU, gates outside the kernel"               trace_py python3 kernel/engine_test.py --device cuda
t 04 "engine glue on the GPU, gates inside the kernel"                trace_py python3 kernel/engine_test.py --device cuda --fused
t 05 "vLLM-style glue on the GPU"                                     trace_py python3 kernel/block_engine_test.py --device cuda --fused
t 06 "verify: synthetic trace of the model's shape"                   trace_py bash -c "python3 -c \"import sys; sys.path.insert(0, '.'); from record_trace import run_self_test; run_self_test('build/gpu_check_trace.bin', heads=32, key_channels=128, value_channels=128, tokens=400)\" && python3 kernel/verify_kernel.py build/gpu_check_trace.bin --device cuda --rows 2"
t 07 "verify: layer 30, three prompts, all 4,095 tokens"              verify_three 30
t 08 "verify: layer 13, three prompts, all 4,095 tokens"              verify_three 13
t 09 "verify: layer 0, three prompts, all 4,095 tokens"               verify_three 00
# ---- kernel speed ----------------------------------------------------------------------------
t 10 "profile by phase, batch 32 / 128 / 512"                         trace_py bash -c "for b in 32 128 512; do python3 kernel/profile_kernel.py --device cuda --batch \$b --label suite | tee $RESULTS/profile_suite_b\$b.txt; done"
t 11 "kernel speed next to SGLang's kernels"                          sglang_py bash -c "for d in fp32 bf16; do python3 kernel/bench_kernel.py --device cuda --batch-sizes 32 128 512 --dtype \$d --with-sglang | tee $RESULTS/e_kernel_speed_\$d.log; done"
t 12 "complete windows, every state format"                           sglang_py bash -c "for d in fp32 bf16; do python3 kernel/bench_kernel.py --device cuda --batch-sizes 32 128 512 --dtype \$d --with-sglang --full-cycles | tee $RESULTS/e_full_cycles_\$d.log; done"
t 13 "the same benchmark a second time (repeatability, gate G0)"      sglang_py bash -c "python3 kernel/bench_kernel.py --device cuda --batch-sizes 32 128 512 --dtype bf16 --with-sglang --full-cycles | tee $RESULTS/e_full_cycles_bf16_repeat.log"
# ---- numerical quality (CPU; plan 5 steps S19 to S21) ------------------------------------------
t 14 "fit quality against the exact optimum, nine traces"             trace_py bash -c "python3 kernel/fit_quality.py $TRACES/{explain,code,reasoning}/trace_layer_*.bin --every 512 --sweeps 2 | tee $RESULTS/s19_fit_quality_2sweeps.txt"
t 15 "replay sweeps: rank 0/2/8, window 8, 8 sweeps, nine traces"      sweeps
t 16 "all 24 linear layers, one prompt: replay and fit quality"       all_layers
# ---- inside the engine -----------------------------------------------------------------------
t 17 "SGLang: apply the patch"                                        runbook/stage_f_end_to_end.sh patch
t 18 "SGLang smoke: window server against LeapQuant server"           runbook/stage_f_end_to_end.sh smoke
t 19 "SGLang smoke: LeapQuant with CUDA graphs off against on"        smoke_graphs
t 20 "serving baselines: fp32, fp32 + window, bf16 + window"          runbook/stage_b_baselines.sh b2
t 21 "serving with LeapQuant"                                         runbook/stage_f_end_to_end.sh serving
t 22 "serving with LeapQuant, second repetition"                      bash -c "RESULTS=$RESULTS/repeat2 && mkdir -p \$RESULTS && export RESULTS && runbook/stage_f_end_to_end.sh serving"
t 23 "task accuracy: fp32 twice, then LeapQuant (GSM8K, AIME, MMLU-Pro)"  runbook/stage_f_end_to_end.sh accuracy
echo "suite finished $(date -u +%Y-%m-%dT%H:%MZ)" | tee -a "$RESULTS/suite.log" >> "$STATUS"
