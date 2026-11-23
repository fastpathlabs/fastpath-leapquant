#!/usr/bin/env bash
# State-cache plan v2, task TR.2 (backlog B36): the 9 recorded traces at rank 0, 2 and 4, with and without the
# Hadamard rotation of keys and state (reference core option rotate_keys, replay flag --rotate-keys 1).
# Rank 4 and rank 2 without rotation are S20's "base" and "r2"; this adds r0, r0rot, r2rot, r4rot.
# Gate GR (TR.3): the rotation is adopted only if, on all 9 traces including the three layer-30 ones, the error at
# half the rank with rotation is at most today's error at full rank without it.
#   OUT=results/s20_ablations runbook/rotation_experiment.sh        (from the repository root, after make)
set -u
cd "$(dirname "$0")/.."
exec bash runbook/s20_ablations.sh "base:" "r2:--compensators 2" "r0:--compensators 0" \
    "r0rot:--compensators 0 --rotate-keys 1" "r2rot:--compensators 2 --rotate-keys 1" "r4rot:--rotate-keys 1"
