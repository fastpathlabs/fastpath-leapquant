#!/usr/bin/env bash
# harness_setup.sh: puts the pinned InferenceX client (inferencex-e2e at commit fabf7ba, harness/config/profiles.json)
# on the GPU host and checks its source hashes. Downloads the commit's tarball from GitHub (public; nothing else is
# sent), extracts only inferencex-e2e/, and verifies the pinned files. The client runs with the SGLang virtualenv's
# interpreter (deviation D-IX-2: uv is not installed). Idempotent.
#   runbook/harness_setup.sh              (from the tree root; target $HOME/lqharness/upstream/inferencex-e2e)
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=fabf7ba3f6682ecf61b23ebd1aeb35dd98aee0ab
TARGET="${LQH_UPSTREAM:-$HOME/lqharness/upstream}"
mkdir -p "$TARGET"
if [ ! -f "$TARGET/inferencex-e2e/infx/bench/fixed_seq.py" ]; then
    work=$(mktemp -d)
    curl -sSL -o "$work/infx.tar.gz" "https://codeload.github.com/SemiAnalysisAI/InferenceX/tar.gz/$COMMIT"
    sha256sum "$work/infx.tar.gz" | tee "$TARGET/inferencex_tarball.sha256"
    tar -xzf "$work/infx.tar.gz" -C "$work" "InferenceX-$COMMIT/inferencex-e2e"
    rm -rf "$TARGET/inferencex-e2e"
    mv "$work/InferenceX-$COMMIT/inferencex-e2e" "$TARGET/inferencex-e2e"
    rm -rf "$work"
    echo "$COMMIT" > "$TARGET/inferencex-e2e/PINNED_COMMIT"
fi
VENV="${VENV_SGLANG:-$HOME/venv-sglang}"
PYTHONPATH=harness "$VENV/bin/python" -m lqh infx-command --verify "$TARGET/inferencex-e2e"
PYTHONPATH="$TARGET/inferencex-e2e" "$VENV/bin/python" -c "import infx.bench_serving.benchmark_serving as m; print('pinned client imports OK')"
