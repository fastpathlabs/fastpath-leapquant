#!/usr/bin/env bash
# harness_longcontext.sh: long-context diagnostics (prompt rev. 3, section 15). Teacher-forced NLL by position on the
# same held-out token sequences (PG-19 test books, public domain; 8 documents x 33k tokens; lengths 1k, 4k, 8k, 16k,
# 32k), read through the serving engine's own /generate API, for fp32 (reference), bf16 window and the LeapQuant
# candidate; radix cache off (no prefix reuse between lengths), each at the default prefill chunk (8,192) and at a
# chunk of 16 tokens (a compressed state every window, the closest prefill analogue of decode-time folding).
# Model context: 262,144 positions; lengths stay at or below 32k (declared grid). Output: nll_<arm>_<chunk>/ and
# nll_compare.json (paired by document, document-level bootstrap).
#   RESULTS=$HOME/lqexp/results/harness/longcontext runbook/harness_longcontext.sh     (from the tree root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
DOCS="$RESULTS/pg19_docs.jsonl"
[ -s "$DOCS" ] || PYTHONPATH=harness python3 -m lqh.nll make-docs --out "$DOCS" --count 8 --min-tokens 33000
NOBUF="--mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --disable-radix-cache --max-running-requests 8 --max-mamba-cache-size 16"
WINDOW="--enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton"
measure() {        # measure ARM CHUNK server-flags...
    local arm="$1" chunk="$2"; shift 2
    start_server "$RESULTS/server_${arm}_c$chunk.log" "$@" --chunked-prefill-size "$chunk" || return 1
    PYTHONPATH=harness python3 -m lqh.nll measure --base-url "http://127.0.0.1:$PORT" --docs "$DOCS" \
        --lengths 1024,4096,8192,16384,32768 --out "$RESULTS/nll_${arm}_c$chunk"
    stop_server
}
lq_env() { export LEAPQUANT_STATE=compressed LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda \
                  LEAPQUANT_CUDA_DECODE_VARIANT=9 LEAPQUANT_STEP_REUSE=1 LEAPQUANT_FUSED_SLOT_COPY=1 \
                  LEAPQUANT_FOLD_POLICY=stagger LEAPQUANT_SWEEPS=1; }   # lq_final (15 Nov 2026)
for chunk in 8192 16; do
    measure fp32 "$chunk" --mamba-ssm-dtype float32 $NOBUF
    measure bf16_window "$chunk" --mamba-ssm-dtype bfloat16 $NOBUF $WINDOW
    ( lq_env; measure leapquant "$chunk" --mamba-ssm-dtype float32 $NOBUF $WINDOW --enable-linear-leapquant )
    PYTHONPATH=harness python3 -m lqh.nll compare "$RESULTS/nll_fp32_c$chunk" "$RESULTS/nll_bf16_window_c$chunk" \
        "$RESULTS/nll_leapquant_c$chunk" --out "$RESULTS/nll_compare_c$chunk.json"
done
