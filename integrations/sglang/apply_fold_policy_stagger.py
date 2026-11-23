#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
apply_fold_policy_stagger.py   (after apply_fold_policy.py; backlog B53)

LEAPQUANT_FOLD_POLICY=stagger: each request folds on its own phase of the window, (seq_len + slot) % 16 == 0, so about
1/16 of the requests fold on every step instead of all of them every 16th step (the aligned fold step is the p99
spike between tokens). After a request's first, early fold its forced folds coincide with its full windows. With the
radix cache off the track-boundary flushes are dropped (as with "auto"); with it on they are kept (snapshots need them).
Other values of LEAPQUANT_FOLD_POLICY are unchanged.

Edits an SGLang checkout that carries leapquant_phase1.patch and apply_fold_policy.py, idempotently, atomically
(the edited file must parse before it replaces the original):
    python3 integrations/sglang/apply_fold_policy_stagger.py ~/sglang
"""

import ast
import os
import sys

BACKEND = "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py"

EDITS = [
    ("signature",
     """    def _leapquant_align_flush(
        self, force_flush: torch.Tensor, write_pos: torch.Tensor, cache_len: int
    ) -> torch.Tensor:
""",
     """    def _leapquant_align_flush(
        self,
        force_flush: torch.Tensor,
        write_pos: torch.Tensor,
        cache_len: int,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
"""),
    ("stagger branch",
     """            force_flush = torch.zeros_like(force_flush)
        if os.environ.get("LEAPQUANT_ALIGN_FOLDS", "1").strip() == "0":
""",
     """            force_flush = torch.zeros_like(force_flush)
        # LEAPQUANT_FOLD_POLICY=stagger (backlog B53): each request folds on its own phase,
        # (seq_len + slot) % cache_len == 0, so ~1/cache_len of the requests fold on every
        # step and there is no aligned fold step; no alignment below.
        if (
            os.environ.get("LEAPQUANT_FOLD_POLICY", "").strip() == "stagger"
            and seq_lens_cpu is not None
            and slots is not None
        ):
            rows = force_flush.shape[0]
            lengths = seq_lens_cpu[:rows].to(torch.int64)
            if lengths.shape[0] < rows:
                lengths = torch.cat(
                    [lengths, torch.zeros((rows - lengths.shape[0],), dtype=torch.int64)]
                )
            lengths = lengths.to(device=force_flush.device)
            phase = (lengths + slots[:rows].to(torch.int64).clamp(min=0)) % cache_len == 0
            if get_memory().disable_radix_cache:
                force_flush = torch.zeros_like(force_flush)
            return force_flush | phase.to(force_flush.dtype)
        if os.environ.get("LEAPQUANT_ALIGN_FOLDS", "1").strip() == "0":
"""),
    ("eager call",
     """                    replayssm_force_flush = self._leapquant_align_flush(
                        replayssm_force_flush, replayssm_write_pos, L
                    )
""",
     """                    replayssm_force_flush = self._leapquant_align_flush(
                        replayssm_force_flush,
                        replayssm_write_pos,
                        L,
                        seq_lens_cpu=forward_batch.seq_lens_cpu,
                        slots=slots,
                    )
"""),
    ("graph call",
     """                    force_flush_dev = self._leapquant_align_flush(
                        force_flush_dev,
                        write_pos_buf[safe_slots],
                        mamba_pool.linear_replayssm_cache_len,
                    )
""",
     """                    force_flush_dev = self._leapquant_align_flush(
                        force_flush_dev,
                        write_pos_buf[safe_slots],
                        mamba_pool.linear_replayssm_cache_len,
                        seq_lens_cpu=seq_lens_cpu,
                        slots=safe_slots,
                    )
"""),
]


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sglang")
    path = os.path.join(root, BACKEND)
    text = open(path, encoding="utf-8").read()
    if '== "stagger"' in text:
        print("already applied:", path)
        return
    if "LEAPQUANT_FOLD_POLICY" not in text:
        sys.exit("apply_fold_policy.py first")
    for what, old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"{what}: expected exactly one match in {path}, found {text.count(old)}")
        text = text.replace(old, new)
    ast.parse(text)                                   # must parse before it replaces the original
    tmp = path + ".tmp"
    open(tmp, "w", encoding="utf-8").write(text)
    os.replace(tmp, path)
    print("applied:", path)


if __name__ == "__main__":
    main()
