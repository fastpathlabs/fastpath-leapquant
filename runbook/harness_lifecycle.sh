#!/usr/bin/env bash
# harness_lifecycle.sh: the final binary's numerical and lifecycle qualification at engine level (prompt rev. 3,
# section 13; v6 G1/G3): the serving candidate's kernels (CUDA fold, CUDA decode v9, step inputs built once, fused
# slot copy) on
#   restore_test.py    save at a window boundary, restore into fresh buffers and into buffers moved by other
#                      allocations (another slot, another batch position), decode 64 tokens: every layer output
#                      bit for bit equal to the uninterrupted run (restore on OUTPUTS, not only stored bytes)
#   slot_copy_test.py  fused slot copy = field-by-field copy, bit for bit
#   fast_fold_test.py  CUDA fold and store against the Triton fold (9 + 4 cases)
#   engine_test.py, compressed_test.py   engine glue and the compressed-only state against the reference
# Output: <test>.txt and lifecycle_results.json (test, command, exit status, log hash).
#   RESULTS=$HOME/lqexp/results/harness/lifecycle runbook/harness_lifecycle.sh      (from the tree root)
set -u
cd "$(dirname "$0")/.."
source "${VENV_SGLANG:-$HOME/venv-sglang}/bin/activate"
mkdir -p "$RESULTS"
runbook/manifest.sh "$RESULTS/manifest.txt"
export LEAPQUANT_FOLD_BACKEND=cuda LEAPQUANT_DECODE_BACKEND=cuda LEAPQUANT_CUDA_DECODE_VARIANT=9 \
       LEAPQUANT_STEP_REUSE=1 LEAPQUANT_FUSED_SLOT_COPY=1 LEAPQUANT_SWEEPS="${LEAPQUANT_SWEEPS:-1}"
for test in restore_test slot_copy_test fast_fold_test engine_test compressed_test; do
    timeout "${TEST_TIMEOUT:-1800}" python3 "kernel/$test.py" > "$RESULTS/$test.txt" 2>&1
    echo "$test $? $(sha256sum "$RESULTS/$test.txt" | cut -c1-64)" >> "$RESULTS/exit_codes.txt"
done
python3 - "$RESULTS" <<'PY'
import json, sys
folder = sys.argv[1]
rows = []
for line in open(f"{folder}/exit_codes.txt"):
    name, code, digest = line.split()
    rows.append({"test": name, "command": f"python3 kernel/{name}.py", "exit_status": int(code),
                 "status": "PASS" if code == "0" else "FAIL", "log": f"{name}.txt", "log_sha256": digest})
json.dump({"schema": "lqh.lifecycle.v1", "environment": "CUDA fold, CUDA decode v9, step reuse, fused slot copy, one fit sweep",
           "tests": rows, "all_pass": all(r["status"] == "PASS" for r in rows)}, open(f"{folder}/lifecycle_results.json", "w"), indent=1)
for r in rows: print(r["test"], r["status"])
PY
