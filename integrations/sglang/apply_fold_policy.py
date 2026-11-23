#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
apply_fold_policy.py

LEAPQUANT_FOLD_POLICY=auto (14 Nov 2026): when LeapQuant's windows are folded, chosen by the radix-cache mode.
  - Radix cache off: nothing ever reads a radix snapshot, so the track-boundary flushes (seq_len % 256 == 0) are
    dropped for LeapQuant. Each was a single-request fold that put the request out of phase; aligned folds then
    pulled every request into an early fold when that request's window filled.
  - Radix cache on: every request folds on its own step (results/radix-align0: 2-3% faster than aligned there).
Unset (the default): unchanged (aligned folds, track flushes kept; LEAPQUANT_ALIGN_FOLDS=0 for own-step folds).

Edits an SGLang checkout that carries integrations/sglang/leapquant_phase1.patch, idempotently:
    python3 integrations/sglang/apply_fold_policy.py ~/sglang
"""

import os
import sys

BACKEND = "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"

OLD = """        import os

        if os.environ.get("LEAPQUANT_ALIGN_FOLDS", "1").strip() == "0":
            return force_flush
"""
NEW = """        import os

        # LEAPQUANT_FOLD_POLICY=auto (14 Nov 2026): with the radix cache off nothing reads a
        # snapshot, so the track-boundary flushes are dropped (for LeapQuant each was a
        # single-request fold that put the request out of phase, and aligned folds then pulled
        # every request into an early fold); with the radix cache on, every request folds on
        # its own step (measured 2-3% faster than aligned folds there).
        if os.environ.get("LEAPQUANT_FOLD_POLICY", "").strip() == "auto":
            if not get_memory().disable_radix_cache:
                return force_flush
            force_flush = torch.zeros_like(force_flush)
        if os.environ.get("LEAPQUANT_ALIGN_FOLDS", "1").strip() == "0":
            return force_flush
"""


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sglang")
    path = os.path.join(root, BACKEND)
    text = open(path, encoding="utf-8").read()
    if "LEAPQUANT_FOLD_POLICY" in text:
        print("already applied:", path)
        return
    if text.count(OLD) != 1:
        sys.exit(f"expected exactly one match in {path}, found {text.count(OLD)}")
    if "get_memory" not in text:
        sys.exit(f"{path} does not import get_memory")
    open(path, "w", encoding="utf-8").write(text.replace(OLD, NEW))
    print("applied:", path)


if __name__ == "__main__":
    main()
