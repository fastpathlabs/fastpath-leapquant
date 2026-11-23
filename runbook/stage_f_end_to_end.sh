#!/usr/bin/env bash
# Stage F. End to end with LeapQuant in SGLang (integration phase 1).
#
#   ./stage_f_end_to_end.sh patch     apply the patch to the SGLang checkout
#   ./stage_f_end_to_end.sh smoke     flag off and flag on: same prompts, compare the answers
#   ./stage_f_end_to_end.sh serving   the stage B2 client against a server with the flag on
#   ./stage_f_end_to_end.sh accuracy  GSM8K, AIME 2025 and MMLU-Pro: fp32 twice, then LeapQuant
#   ./stage_f_end_to_end.sh cache     prefix-cache capacity with LeapQuant checkpoints
#
# Run stage D first: do not put a kernel in a server before it passes.
#
# The patch is untested: it was written by reading the SGLang source and has
# only been byte-compiled. Expect to fix things on the first start. The server
# log line "LeapQuant decode state allocated" shows that the flag took effect.
#
# Phase 1 measures decode speed (user story US1). It does not reduce memory:
# the fp32 state stays and the LeapQuant state comes on top, about 21 MB per
# request slot. If the server runs out of memory, lower --max-mamba-cache-size.
#
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEP="${1:?usage: stage_f_end_to_end.sh patch|smoke|serving|accuracy|cache}"


case "$STEP" in
patch)
    git -C "$SGLANG_DIR" checkout "$SGLANG_COMMIT"
    if git -C "$SGLANG_DIR" apply --reverse --check "$LEAPQUANT_HOME/integrations/sglang/leapquant_phase1.patch" 2> /dev/null; then
        echo "the patch is already applied"; exit 0
    fi
    git -C "$SGLANG_DIR" apply --check "$LEAPQUANT_HOME/integrations/sglang/leapquant_phase1.patch"
    git -C "$SGLANG_DIR" apply "$LEAPQUANT_HOME/integrations/sglang/leapquant_phase1.patch"
    git -C "$SGLANG_DIR" status --short
    echo "patched. The editable install of stage A picks the change up as is."
    ;;
smoke)
    source "$VENV_SGLANG/bin/activate"
    start_server "$RESULTS/f_server_window.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS
    python3 "$LEAPQUANT_HOME/integrations/sglang/smoke_client.py" --port "$PORT" --out "$RESULTS/f_smoke_window.json"
    stop_server
    start_server "$RESULTS/f_server_leapquant.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS $LEAPQUANT_FLAGS
    grep "LeapQuant decode state allocated" "$RESULTS/f_server_leapquant.log" || { echo "the flag did not take effect"; stop_server; exit 1; }
    python3 "$LEAPQUANT_HOME/integrations/sglang/smoke_client.py" --port "$PORT" --out "$RESULTS/f_smoke_leapquant.json" \
        --compare "$RESULTS/f_smoke_window.json"
    stop_server
    ;;
serving)
    source "$VENV_SGLANG/bin/activate"
    start_server "$RESULTS/f_server_leapquant_serving.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS $LEAPQUANT_FLAGS
    run_serving_client config4_leapquant; stop_server
    echo "Compare with the three baselines of stage B2 (results/README.md)."
    ;;
accuracy)
    # Task T12. The fp32 server is measured twice: the difference between its two runs is the
    # run-to-run noise that the LeapQuant run is judged against. ACC_LIMIT questions per task.
    source "$VENV_SGLANG/bin/activate"
    python3 -c "import datasets" 2> /dev/null || uv pip install datasets
    LIMIT="${ACC_LIMIT:-300}"
    BENCH="python3 $LEAPQUANT_HOME/eval/accuracy_bench.py --port $PORT --limit $LIMIT --concurrency ${ACC_CONCURRENCY:-64}"
    start_server "$RESULTS/f_server_accuracy_fp32.log" --mamba-ssm-dtype float32 $COMMON_FLAGS
    $BENCH --out "$RESULTS/accuracy_fp32_run1.json" | tee "$RESULTS/accuracy_fp32_run1.txt"
    $BENCH --out "$RESULTS/accuracy_fp32_run2.json" --compare "$RESULTS/accuracy_fp32_run1.json" | tee "$RESULTS/accuracy_fp32_noise.txt" || true
    stop_server
    start_server "$RESULTS/f_server_accuracy_leapquant.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS $LEAPQUANT_FLAGS
    $BENCH --out "$RESULTS/accuracy_leapquant.json" --compare "$RESULTS/accuracy_fp32_run1.json" | tee "$RESULTS/accuracy_leapquant_vs_fp32.txt"
    stop_server
    ;;
cache)
    # Prefix-cache capacity with checkpoints in the LeapQuant boundary format (task T11), for
    # comparison with stage B4. With the flag on, --enable-int8-mamba-checkpoint stores LeapQuant
    # boundary states. Unlike B4 this uses the no_buffer strategy, which the window needs.
    source "$VENV_SGLANG/bin/activate"
    start_server "$RESULTS/f_server_cache_leapquant.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS $LEAPQUANT_FLAGS \
        --enable-int8-mamba-checkpoint
    ( cd "$SGLANG_DIR" && python benchmark/bench_linear_attention/bench_int8_checkpoint_reuse.py --port "$PORT" \
        --prefix-tokens 1000 --num-prefixes 128 384 640 --parallel 8 2>&1 | tee "$RESULTS/f_prefix_cache_leapquant.log" )
    stop_server
    ;;
*)
    echo "unknown step $STEP"; exit 2 ;;
esac
