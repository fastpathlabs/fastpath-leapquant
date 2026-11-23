#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
apply_track_flush.py   (after apply_fold_policy.py and apply_fold_policy_stagger.py; backlog B55)

LEAPQUANT_TRACK_FLUSH=0: drop the radix track-boundary flushes (seq_len % mamba_track_interval == 0) whether the radix
cache is on or off, before the fold policy applies. Unset: unchanged.

Why they can go: ReplaySSM, and so LeapQuant, runs only with the no_buffer radix strategy (SGLang refuses
--mamba-radix-cache-strategy extra_buffer with --enable-linear-replayssm), and with no_buffer nothing reads a decode
state at a track boundary. The decode-time track copy (track_mamba_states_all_layers) runs only with extra_buffer. A
running request is published to the tree only after a prefill chunk (checkpoint_kv_cache; its window is empty
then), and a finished one is donated up to its last fold: MambaComponent.prepare_for_caching_req subtracts the
cursor (storage.settle_replay) from the cached length. For LeapQuant each track flush was a fold of a partly filled
window that put the request out of phase with the aligned folds, so with the radix cache on folds ran on almost
every step (results/round9-equalmem, results/radix-equalmem-final).

With the extra_buffer strategy (not reachable today) the flushes are kept.

Edits an SGLang checkout that carries leapquant_phase1.patch and the two fold-policy edits, idempotently, atomically
(the edited file must parse before it replaces the original):
    python3 integrations/sglang/apply_track_flush.py ~/sglang
"""

import ast
import os
import sys

BACKEND = "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"

ANCHOR = """        import os

        # LEAPQUANT_FOLD_POLICY=auto (14 Nov 2026)"""

INSERT = """        import os

        # LEAPQUANT_TRACK_FLUSH=0 (backlog B55): drop the radix track-boundary flushes, radix
        # cache on or off. ReplaySSM (so LeapQuant) runs only with the no_buffer strategy, and
        # there nothing reads a decode state at a track boundary: a running request is published
        # after a prefill chunk (empty window), a finished one up to its last fold (settle_replay
        # takes the cursor off the cached length). Each flush was a fold of a partly filled window
        # that put the request out of phase with the aligned folds. Kept with extra_buffer.
        if os.environ.get("LEAPQUANT_TRACK_FLUSH", "").strip() == "0" and not getattr(
            get_exec().mamba, "enable_mamba_extra_buffer", False
        ):
            force_flush = torch.zeros_like(force_flush)

        # LEAPQUANT_FOLD_POLICY=auto (14 Nov 2026)"""


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sglang")
    path = os.path.join(root, BACKEND)
    text = open(path, encoding="utf-8").read()
    if "LEAPQUANT_TRACK_FLUSH" in text:
        print("already applied:", path)
        return
    if '== "stagger"' not in text:
        sys.exit("apply_fold_policy.py and apply_fold_policy_stagger.py first")
    if text.count(ANCHOR) != 1:
        sys.exit(f"anchor: expected exactly one match in {path}, found {text.count(ANCHOR)}")
    text = text.replace(ANCHOR, INSERT)
    ast.parse(text)                                   # must parse before it replaces the original
    tmp = path + ".tmp"
    open(tmp, "w", encoding="utf-8").write(text)
    os.replace(tmp, path)
    print("applied:", path)


if __name__ == "__main__":
    main()
