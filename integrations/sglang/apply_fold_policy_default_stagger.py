#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
apply_fold_policy_default_stagger.py   (after apply_fold_policy.py and apply_fold_policy_stagger.py)

15 Nov 2026, Asif's decision: staggered folds are the DEFAULT fold policy. With LEAPQUANT_FOLD_POLICY unset or empty
the server now folds each request on its own phase of the window (the "stagger" policy of backlog B53: p99 between
tokens 79 -> 62 ms at AA c512, 70 -> 49 ms at decode c512, for 1.3-1.9% throughput; results/ab-stagger).
    LEAPQUANT_FOLD_POLICY=aligned   the earlier default: when one request's window is full, every request folds
    LEAPQUANT_FOLD_POLICY=auto      unchanged (radix on: own-step folds; radix off: track flushes dropped)
    LEAPQUANT_FOLD_POLICY=stagger   unchanged (now also what unset means)

Edits an SGLang checkout that carries leapquant_phase1.patch, apply_fold_policy.py and apply_fold_policy_stagger.py,
idempotently and atomically (the edited file must parse before it replaces the original):
    python3 integrations/sglang/apply_fold_policy_default_stagger.py ~/sglang
"""

import ast
import os
import sys

BACKEND = "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"
MARKER = "fold_policy = os.environ.get(\"LEAPQUANT_FOLD_POLICY\", \"\").strip() or \"stagger\""

EDITS = [
    ("default policy",
     """        if os.environ.get("LEAPQUANT_FOLD_POLICY", "").strip() == "auto":
""",
     """        # Default fold policy (15 Nov 2026, Asif's decision): "stagger" when LEAPQUANT_FOLD_POLICY is unset or
        # empty; LEAPQUANT_FOLD_POLICY=aligned gives the earlier default (aligned folds).
        fold_policy = os.environ.get("LEAPQUANT_FOLD_POLICY", "").strip() or "stagger"
        if fold_policy == "auto":
"""),
    ("stagger test",
     """            os.environ.get("LEAPQUANT_FOLD_POLICY", "").strip() == "stagger"
""",
     """            fold_policy == "stagger"
"""),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = os.path.join(sys.argv[1], BACKEND)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    if MARKER in text:
        print(f"already applied: {path}")
        return
    for name, old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"error: edit '{name}': expected the original text exactly once in {path} "
                     f"(found {text.count(old)}); apply apply_fold_policy.py and apply_fold_policy_stagger.py first")
        text = text.replace(old, new)
    ast.parse(text)
    temporary = path + ".tmp_default_stagger"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(temporary, path)
    print(f"applied: {path} (fold policy default: stagger; LEAPQUANT_FOLD_POLICY=aligned for the earlier default)")


if __name__ == "__main__":
    main()
