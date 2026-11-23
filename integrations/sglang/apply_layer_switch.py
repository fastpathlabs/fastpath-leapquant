#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
apply_layer_switch.py

State-cache plan v2, task TQ.5 (backlog B30): LEAPQUANT_LAYERS selects the model layers that carry LeapQuant
("all" by default; e.g. "30", "0-13", "0-19,24"). Layers outside the selection decode with SGLang's own window
kernel on the fp32 state, so this needs the mirror state (LEAPQUANT_STATE=mirror, the default); with the
compressed state a subset is refused at start (the other layers would have no state).

Edits an SGLang checkout that already carries integrations/sglang/leapquant_phase1.patch, idempotently:
    python3 integrations/sglang/apply_layer_switch.py ~/sglang
Then regenerate the patch from that checkout (git diff > integrations/sglang/leapquant_phase1.patch).
"""

import os
import sys

BACKEND = "python/sglang/srt/layers/attention/linear/gdn_backend.py"

INIT_OLD = """            self.leapquant = build_leapquant_state(
                self.state_pool,
                model_runner.device,
                get_exec().mamba.linear_replayssm_cache_len,
            )
"""
INIT_NEW = """            self.leapquant = build_leapquant_state(
                self.state_pool,
                model_runner.device,
                get_exec().mamba.linear_replayssm_cache_len,
            )
            # LEAPQUANT_LAYERS (task TQ.5): the layers that carry LeapQuant; the others keep
            # SGLang's window kernel on the fp32 state (mirror state only).
            import os as _os

            from leapquant_engine import parse_layer_selection

            self.leapquant_layers = parse_layer_selection(
                _os.environ.get("LEAPQUANT_LAYERS", "all"), list(self.state_layer_indices)
            )
            if (
                self.leapquant is not None
                and self.leapquant.compressed_only
                and self.leapquant_layers != set(self.state_layer_indices)
            ):
                raise ValueError(
                    "LEAPQUANT_LAYERS selects a subset of the layers: the compressed state keeps no "
                    "fp32 state for the others. Use LEAPQUANT_STATE=mirror."
                )
"""
DECODE_OLD = """        if self.leapquant is not None:
            core_attn_out = self.leapquant.decode("""
DECODE_NEW = """        if self.leapquant is not None and layer.layer_id in self.leapquant_layers:
            core_attn_out = self.leapquant.decode("""


def main():
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sglang")
    path = os.path.join(root, BACKEND)
    text = open(path, encoding="utf-8").read()
    if "leapquant_layers" in text:
        print("already applied:", path)
        return
    for old, new, what in ((INIT_OLD, INIT_NEW, "init"), (DECODE_OLD, DECODE_NEW, "decode")):
        if text.count(old) != 1:
            sys.exit(f"{what}: expected exactly one match in {path}, found {text.count(old)}")
        text = text.replace(old, new)
    open(path, "w", encoding="utf-8").write(text)
    print("applied:", path)


if __name__ == "__main__":
    main()
