#!/usr/bin/env bash
# Stage C. Error on real states: record on the GPU, replay on the CPU.
#
#   ./stage_c_traces.sh            record 3 prompts x 3 layers, 4,096 tokens each, and replay them
#
# Each layer file is about 270 MB. Read three things in every report:
#   1. The self-check line must end in "(ok)". If it does not, the recorder is
#      wrong and the table means nothing.
#   2. The LeapQuant row against "int8, rounded every token": what the method buys.
#   3. The LeapQuant row against "bf16, rounded every window": which format to build.
#      GATE: if bf16 is as accurate, it is the simpler product at 2 bytes instead of 1.19.
#
# The recorder has run on Qwen3.5-0.8B on a CPU. This is its first run on the 9B model.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$VENV_TRACE/bin/activate"
cd "$LEAPQUANT_HOME"

LAYERS="${LAYERS:-0,13,30}"
TOKENS="${TOKENS:-4096}"
TRACES="${TRACES:-$LEAPQUANT_HOME/traces}"

for prompt in explain code reasoning; do
    out="$TRACES/$prompt"
    python3 record_trace.py --model "$MODEL" --max-new-tokens "$TOKENS" --layers "$LAYERS" \
        --out-dir "$out" --prompt-file "runbook/prompts/prompt_${prompt}.txt" 2>&1 | tee "$RESULTS/c_record_${prompt}.log"
    for trace in "$out"/trace_layer_*.bin; do
        name="$(basename "$trace" .bin)"
        # The replay exits with status 3 if the trace fails its self-check; pipefail makes that fail the stage.
        ./leapquant_replay "$trace" --sweeps 2 --write-outputs "$out/${name}.outputs.bin" | tee "$RESULTS/c_replay_${prompt}_${name}.txt"
    done
done

echo
echo "Self-check lines:"
grep -h "self-check" "$RESULTS"/c_replay_*.txt
