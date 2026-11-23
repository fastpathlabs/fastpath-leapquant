#!/usr/bin/env python3
"""
gates_test.py

One test per false-pass path that plan 5 (workstream 1) found in the verification tools.
Each of these would have passed on the code as it was on 12 November 2026.

    python3 kernel/gates_test.py        (needs numpy, and ./leapquant_replay built)

    1. A nonzero output against a zero reference is not "no difference".
    2. Not-a-number never passes.
    3. A zero reference loss does not make a bad kernel pass.
    4. A kernel that is MORE accurate than the reference passes.
    5. A trace that fails its self-check makes leapquant_replay exit with an error.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from verify_kernel import not_worse, relative_difference  # noqa: E402
from record_trace import run_self_test  # noqa: E402

failures = []


def check(condition, text):
    print(f"  {'ok  ' if condition else 'FAIL'}  {text}")
    if not condition:
        failures.append(text)


zero = np.zeros((2, 8), dtype=np.float32)
ones = np.ones((2, 8), dtype=np.float32)
bad = ones.copy()
bad[0, 3] = np.nan

check(np.isinf(relative_difference(ones, zero)).all(), "1. nonzero output, zero reference: difference is infinite")
check((relative_difference(zero, zero) == 0).all(), "1. zero output, zero reference: difference is zero")
check(np.isinf(relative_difference(bad, ones)[0]), "2. not-a-number in the output: difference is infinite")
check(not not_worse(float("nan"), 0.5, 0.05, 1e-7), "2. a not-a-number loss fails the limit")
check(not not_worse(0.2, 0.0, 0.05, 1e-7), "3. reference loss zero, kernel loss 0.2: fails")
check(not_worse(0.0, 0.0, 0.05, 1e-7), "3. both losses zero: passes")
check(not_worse(0.001, 0.5, 0.05, 1e-7), "4. a kernel far more accurate than the reference passes")
check(not not_worse(0.6, 0.5, 0.05, 1e-7), "4. a kernel 20% worse than the reference fails")

replay = os.path.join(ROOT, "leapquant_replay")
if not os.path.exists(replay):
    check(False, "5. ./leapquant_replay is not built (run make)")
else:
    with tempfile.TemporaryDirectory() as folder:
        good = os.path.join(folder, "good.bin")
        run_self_test(good, tokens=40)
        status_good = subprocess.run([replay, good], capture_output=True).returncode
        # Corrupt the recorded kernel outputs of a few tokens (the last floats of each head's record).
        data = bytearray(open(good, "rb").read())
        tail = np.frombuffer(data, dtype="<f4", offset=len(data) - 4 * 4000).copy()
        tail *= 1.5
        data[len(data) - 4 * 4000:] = tail.tobytes()
        corrupt = os.path.join(folder, "corrupt.bin")
        open(corrupt, "wb").write(data)
        result = subprocess.run([replay, corrupt], capture_output=True, text=True)
        check(status_good == 0, "5. a valid trace: leapquant_replay exits with 0")
        check(result.returncode == 3 and "INVALID TRACE" in result.stdout,
              "5. a corrupted trace: leapquant_replay exits with 3 and says INVALID TRACE")

print(f"\n{'PASS' if not failures else f'FAIL ({len(failures)})'}")
sys.exit(0 if not failures else 1)
