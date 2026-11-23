#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
apply_layer_switch_ring.py   (after apply_layer_switch.py; backlog B30)

With LeapQuant on, leapquant_phase1.patch sizes SGLang's ReplaySSM decode ring for one slot: LeapQuant keeps its own
window and never reads the ring. With a subset of the layers (LEAPQUANT_LAYERS, mirror state) the other layers decode
with SGLang's window kernel, which keeps its records in that ring for every slot: with one slot they were written out
of bounds (results/layer-switch, L2 and L3: a device-side assert in a later kernel, store_kvcache). With a subset the
ring keeps its full size (about 9.4 MB per slot in fp32, outside SGLang's memory budget, as without LeapQuant).

Edits an SGLang checkout that carries leapquant_phase1.patch, idempotently, atomically:
    python3 integrations/sglang/apply_layer_switch_ring.py ~/sglang
"""

import ast
import os
import sys

STATE = "python/sglang/srt/layers/attention/linear/leapquant_state.py"

OLD = '''def leapquant_ring_slots(slots: int) -> int:
    """The ReplaySSM decode ring is not used by LeapQuant: one slot is enough."""
    return 1 if leapquant_enabled() else slots
'''
NEW = '''def leapquant_ring_slots(slots: int) -> int:
    """The ReplaySSM decode ring is not used by LeapQuant: one slot is enough. Not with a subset of the
    layers (LEAPQUANT_LAYERS, mirror state): the other layers decode with SGLang's window kernel, which
    keeps its records there for every slot (backlog B30: with one slot they were written out of bounds)."""
    if not leapquant_enabled():
        return slots
    if os.environ.get("LEAPQUANT_LAYERS", "all").strip().lower() not in ("", "all"):
        return slots
    return 1
'''


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sglang")
    path = os.path.join(root, STATE)
    text = open(path, encoding="utf-8").read()
    if "LEAPQUANT_LAYERS" in text and "backlog B30" in text:
        print("already applied:", path)
        return
    if text.count(OLD) != 1:
        sys.exit(f"expected exactly one match in {path}, found {text.count(OLD)}")
    text = text.replace(OLD, NEW)
    ast.parse(text)                                   # must parse before it replaces the original
    tmp = path + ".tmp"
    open(tmp, "w", encoding="utf-8").write(text)
    os.replace(tmp, path)
    print("applied:", path)


if __name__ == "__main__":
    main()
