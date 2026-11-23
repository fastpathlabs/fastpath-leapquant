#!/usr/bin/env bash
# harness_compounding.sh: compounding test (prompt rev. 3, section 13). The candidate state format on ONE linear layer
# at a time, then on all 24 at once, on the same inputs: teacher-forced NLL (lqh.nll, 4 PG-19 documents x 8,192
# tokens, prefill chunk 16 so the state is compressed every window) against the fp32 window reference. A change small
# in each layer alone can be larger when every layer carries it: the result lists each single-layer delta, their sum,
# and the all-layer delta. Caveat: a layer subset needs LEAPQUANT_STATE=mirror (backlog B30; the compressed-only
# build refuses subsets), so this test runs the LeapQuant kernels with the mirror state, all-layer run included.
#   RESULTS=$HOME/lqexp/results/harness/compounding runbook/harness_compounding.sh     (from the tree root)
set -u
cd "$(dirname "$0")/.."
source runbook/env.sh
source "$VENV_SGLANG/bin/activate"
runbook/manifest.sh "$RESULTS/manifest.txt"
DOCS="$RESULTS/pg19_docs.jsonl"
LONG="$HOME/lqexp/results/harness/longcontext/pg19_docs.jsonl"
[ -s "$DOCS" ] || { [ -s "$LONG" ] && head -4 "$LONG" > "$DOCS"; } || true
[ -s "$DOCS" ] || PYTHONPATH=harness python3 -m lqh.nll make-docs --out "$DOCS" --count 4 --min-tokens 8192
FLAGS="--mamba-ssm-dtype float32 --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --disable-radix-cache --max-running-requests 4 --max-mamba-cache-size 8 --enable-linear-replayssm --linear-replayssm-cache-len 16 --linear-attn-decode-backend triton --chunked-prefill-size 16 --mem-fraction-static ${SUBSET_MEM_FRACTION:-0.70}"
LINEAR="0 1 2 4 5 6 8 9 10 12 13 14 16 17 18 20 21 22 24 25 26 28 29 30"
run() {            # run NAME [LEAPQUANT_LAYERS]: one server, NLL on the documents
    local name="$1" layers="${2:-}"
    [ -s "$RESULTS/nll_$name/nll_summary.json" ] && return 0
    if [ -n "$layers" ]; then
        export LEAPQUANT_STATE=mirror LEAPQUANT_LAYERS="$layers" LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda \
               LEAPQUANT_CUDA_DECODE_VARIANT=9 LEAPQUANT_STEP_REUSE=1 LEAPQUANT_SWEEPS=1 LEAPQUANT_FOLD_POLICY=stagger
        start_server "$RESULTS/server_$name.log" $FLAGS --enable-linear-leapquant || return 1
    else
        unset LEAPQUANT_STATE LEAPQUANT_LAYERS
        start_server "$RESULTS/server_$name.log" $FLAGS || return 1
    fi
    PYTHONPATH=harness python3 -m lqh.nll measure --base-url "http://127.0.0.1:$PORT" --docs "$DOCS" --lengths 8192 \
        --max-docs 4 --out "$RESULTS/nll_$name"
    stop_server
}
# SINGLE_LAYERS: the layers run one at a time (default: all 24). With a subset, those layers also run TOGETHER
# (nll_subset_layers), so the sum of their single-layer deltas has a measured counterpart; the all-layer run stays.
SINGLE="${SINGLE_LAYERS:-$LINEAR}"
run reference
run all_layers "$(echo $LINEAR | tr ' ' ',')"
SUBSET_DIR=""
if [ "$SINGLE" != "$LINEAR" ]; then run subset_layers "$(echo $SINGLE | tr ' ' ',')"; SUBSET_DIR="$RESULTS/nll_subset_layers"; fi
for layer in $SINGLE; do run "layer_$layer" "$layer"; done
PYTHONPATH=harness python3 -m lqh.nll compare "$RESULTS/nll_reference" "$RESULTS/nll_all_layers" $SUBSET_DIR \
    $(for layer in $SINGLE; do echo "$RESULTS/nll_layer_$layer"; done) --out "$RESULTS/compounding_compare.json"
python3 - "$RESULTS/compounding_compare.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
key = "L8192:1-1024"
rows = {name.rsplit("/", 1)[-1]: data for name, data in doc["candidates"].items()}
singles = {n: r for n, r in rows.items() if n.startswith("nll_layer_")}
total = {}
for n, r in singles.items():
    for k, v in r.items():
        total[k] = total.get(k, 0.0) + v["delta_nll_mean"]
doc["compounding"] = {k: {"single_layers": sorted(singles), "sum_of_single_layer_deltas": total[k],
                          "same_layers_together_delta": rows.get("nll_subset_layers", {}).get(k, {}).get("delta_nll_mean"),
                          "all_layer_delta": rows.get("nll_all_layers", {}).get(k, {}).get("delta_nll_mean")}
                      for k in total}
json.dump(doc, open(sys.argv[1], "w"), indent=1)
for k, v in sorted(doc["compounding"].items()):
    print(k, v)
PY
