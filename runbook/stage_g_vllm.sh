#!/usr/bin/env bash
# Stage G. LeapQuant in vLLM, first port (task T14). Prefix caching off, no speculative decoding.
#
#   ./stage_g_vllm.sh patch     vLLM from source at the pinned commit, with the patch applied
#   ./stage_g_vllm.sh smoke     same prompts with LeapQuant off and on; compare the answers
#   ./stage_g_vllm.sh forward   decode-step time at batch 32 and 128, LeapQuant off and on
#
# Untested: the patch was written by reading the vLLM source and has only been
# byte-compiled. The install line (an editable install that reuses vLLM's
# precompiled binaries) and the `vllm bench forward` line are not verified either.
# The server log line "LeapQuant GDN decode" shows that the switch took effect.
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEP="${1:?usage: stage_g_vllm.sh patch|smoke|forward}"

VLLM_DIR="${VLLM_DIR:-$HOME/vllm}"
VLLM_COMMIT="${VLLM_COMMIT:-145cc6428b65db1c4b26ce8d28c7dec444563561}"
VENV_VLLM_SRC="${VENV_VLLM_SRC:-$HOME/venv-vllm-src}"
PATCH="$LEAPQUANT_HOME/integrations/vllm/leapquant_first_port.patch"
SERVE_FLAGS="--port $PORT --served-model-name default --mamba-ssm-cache-dtype float32"

start_vllm() {
    local log="$1"
    vllm serve "$MODEL" $SERVE_FLAGS > "$log" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 360); do
        if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then echo "server is up"; return 0; fi
        if ! kill -0 "$SERVER_PID" 2> /dev/null; then echo "server exited; see $log"; return 1; fi
        sleep 5
    done
    echo "server did not come up; see $log"; return 1
}

case "$STEP" in
patch)
    [ -d "$VLLM_DIR" ] || git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    git -C "$VLLM_DIR" checkout "$VLLM_COMMIT"
    if ! git -C "$VLLM_DIR" apply --reverse --check "$PATCH" 2> /dev/null; then
        git -C "$VLLM_DIR" apply --check "$PATCH"
        git -C "$VLLM_DIR" apply "$PATCH"
    fi
    if [ ! -d "$VENV_VLLM_SRC" ]; then
        uv venv "$VENV_VLLM_SRC" --python 3.12
        ( source "$VENV_VLLM_SRC/bin/activate" && cd "$VLLM_DIR" && VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto )
    fi
    git -C "$VLLM_DIR" status --short
    ;;
smoke)
    source "$VENV_VLLM_SRC/bin/activate"
    start_vllm "$RESULTS/g_vllm_server_fp32.log"
    python3 "$LEAPQUANT_HOME/integrations/sglang/smoke_client.py" --port "$PORT" --out "$RESULTS/g_smoke_fp32.json"
    stop_server
    VLLM_GDN_LEAPQUANT=1 start_vllm "$RESULTS/g_vllm_server_leapquant.log"
    grep -q "LeapQuant GDN decode" "$RESULTS/g_vllm_server_leapquant.log" || { echo "the switch did not take effect"; stop_server; exit 1; }
    python3 "$LEAPQUANT_HOME/integrations/sglang/smoke_client.py" --port "$PORT" --out "$RESULTS/g_smoke_leapquant.json" \
        --compare "$RESULTS/g_smoke_fp32.json"
    stop_server
    ;;
forward)
    source "$VENV_VLLM_SRC/bin/activate"
    cd "$RESULTS"
    for mode in 0 1; do
        VLLM_GDN_LEAPQUANT=$mode vllm bench forward --model "$MODEL" --trust-remote-code --mamba-ssm-cache-dtype float32 \
            --case 'd32=32*1@4096' --case 'd128=128*1@4096' --output-json "g_forward_leapquant${mode}.json" \
            2>&1 | tee "g_forward_leapquant${mode}.log"
    done
    echo "Compare the decode-step times in g_forward_leapquant0.json and g_forward_leapquant1.json."
    ;;
*)
    echo "unknown step $STEP"; exit 2 ;;
esac
