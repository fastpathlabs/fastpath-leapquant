#!/usr/bin/env bash
# Stage B. What the engines give you today. No LeapQuant code runs here.
#
#   ./stage_b_baselines.sh b1     kernel microbenchmark: packed decode against the window decode
#   ./stage_b_baselines.sh b2     end to end: three server configurations, nine client runs
#   ./stage_b_baselines.sh b3     share of GPU time in the linear-attention decode kernel (vLLM)
#   ./stage_b_baselines.sh b4     prefix-cache capacity with fp32 and with int8 checkpoints
#
# Every engine flag below comes from the implementation plan, which took it
# from the engine source at the pinned commits. None has been run yet.
#
# GATE (b2): configuration 2 against configuration 1 is the gain of the window
# alone. If it is not measurable at your target concurrency, stop: LeapQuant
# builds on the same effect.
set -euo pipefail
source "$(dirname "$0")/env.sh"
STEP="${1:?usage: stage_b_baselines.sh b1|b2|b3|b4}"


case "$STEP" in
b1)
    source "$VENV_SGLANG/bin/activate"
    cd "$SGLANG_DIR"
    for dtype in fp32 bf16; do
        python benchmark/kernels/attention/bench_gdn_replayssm_decode.py --hv 32 --h 16 --k 128 --v 128 \
            --batch-sizes 32 128 512 --ls 1 8 16 --dtype "$dtype" 2>&1 | tee "$RESULTS/b1_microbench_${dtype}.log"
    done
    ;;
b2)
    source "$VENV_SGLANG/bin/activate"
    # All three share COMMON_FLAGS (runbook/env.sh); read there why.
    start_server "$RESULTS/b2_server_config1.log" --mamba-ssm-dtype float32 $COMMON_FLAGS
    run_serving_client config1_fp32; stop_server

    start_server "$RESULTS/b2_server_config2.log" --mamba-ssm-dtype float32 $WINDOW_FLAGS
    run_serving_client config2_fp32_window; stop_server

    start_server "$RESULTS/b2_server_config3.log" --mamba-ssm-dtype bfloat16 $WINDOW_FLAGS
    run_serving_client config3_bf16_window; stop_server
    echo "Record output token throughput and inter-token latency of the nine runs in results/README.md."
    ;;
b3)
    if [ ! -d "$VENV_VLLM" ]; then
        uv venv "$VENV_VLLM" --python 3.12
        ( source "$VENV_VLLM/bin/activate" && uv pip install vllm --torch-backend=auto )
    fi
    source "$VENV_VLLM/bin/activate"
    cd "$RESULTS"
    vllm bench forward --model "$MODEL" --trust-remote-code --case 'd32=32*1@4096' --case 'd128=128*1@4096' \
        --profile-dir prof_fp32 --output-json forward_fp32.json
    python -m vllm.benchmarks.forward.profile prof_fp32/d128.rank0.kernels.json | tee b3_profile_fp32.log
    vllm bench forward --model "$MODEL" --trust-remote-code --mamba-ssm-cache-dtype bfloat16 \
        --case 'd128=128*1@4096' --profile-dir prof_bf16
    python -m vllm.benchmarks.forward.profile prof_fp32/d128.rank0.kernels.json prof_bf16/d128.rank0.kernels.json \
        | tee b3_profile_fp32_vs_bf16.log
    echo "Read s, the share of GPU time in the Gated DeltaNet decode kernel. A kernel k times"
    echo "faster speeds up the decode step by at most 1 / (1 - s + s / k)."
    ;;
b4)
    source "$VENV_SGLANG/bin/activate"
    for mode in fp32 int8; do
        extra=""; [ "$mode" = int8 ] && extra="--enable-int8-mamba-checkpoint"
        start_server "$RESULTS/b4_server_${mode}.log" --mamba-radix-cache-strategy extra_buffer --max-mamba-cache-size 256 $extra
        ( cd "$SGLANG_DIR" && python benchmark/bench_linear_attention/bench_int8_checkpoint_reuse.py --port "$PORT" \
            --prefix-tokens 1000 --num-prefixes 128 384 640 --parallel 8 2>&1 | tee "$RESULTS/b4_prefix_cache_${mode}.log" )
        stop_server
    done
    ;;
*)
    echo "unknown step $STEP"; exit 2 ;;
esac
