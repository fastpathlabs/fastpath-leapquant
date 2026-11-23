#!/usr/bin/env bash
# Stage A. Machine and environments. Run once on the GPU machine.
#
# Differences from the plan, both on purpose:
#   - SGLang is installed from source at the pinned commit, not from PyPI, so
#     that the baselines of stage B and the patched server of stage F run the
#     same code.
#   - The CUDA kernel library is built here (needs nvcc from the CUDA toolkit).
#   - PyTorch is installed in the build that matches the GPU driver (--torch-backend=auto).
#     The default wheel can be built for a newer CUDA than the driver supports.
#   - The vLLM environment is created by stage B3, the only step that uses it.
set -euo pipefail
source "$(dirname "$0")/env.sh"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$RESULTS/gpu.csv"

command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

if [ ! -d "$SGLANG_DIR" ]; then
    git clone https://github.com/sgl-project/sglang.git "$SGLANG_DIR"
fi
git -C "$SGLANG_DIR" checkout "$SGLANG_COMMIT"
# SGLang's source build compiles Rust extensions, so it needs cargo, and some of
# those build C libraries with cmake.
if ! command -v cmake > /dev/null; then
    sudo -n apt-get update -qq && sudo -n apt-get install -y -qq cmake nasm pkg-config build-essential libssl-dev protobuf-compiler \
        || { echo "cmake is missing and cannot be installed without sudo: install cmake, nasm, pkg-config, libssl-dev, protobuf-compiler"; exit 1; }
fi
if ! command -v cargo > /dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
[ -d "$VENV_SGLANG" ] || uv venv "$VENV_SGLANG" --python 3.12
if ! "$VENV_SGLANG/bin/python" -c "import sglang" 2> /dev/null; then
    ( source "$VENV_SGLANG/bin/activate" && uv pip install --prerelease=allow -e "$SGLANG_DIR/python" --torch-backend=auto )
fi

# SGLang pins one PyTorch version. If that version has no build for this machine's
# CUDA level, the install above silently falls back to a CPU-only PyTorch. Look for
# a build of the same version that the driver can run (older CUDA 12 builds do).
if ! "$VENV_SGLANG/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2> /dev/null; then
    pinned="$("$VENV_SGLANG/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")"
    for backend in cu128 cu126 cu124; do
        ( source "$VENV_SGLANG/bin/activate" && uv pip install -q --reinstall-package torch --reinstall-package torchaudio \
            --reinstall-package torchvision "torch==$pinned" torchaudio torchvision --torch-backend="$backend" ) || continue
        if "$VENV_SGLANG/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2> /dev/null; then
            echo "SGLang environment: using torch $pinned built for $backend"; break
        fi
    done
    "$VENV_SGLANG/bin/python" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
        || { echo "PyTorch $pinned cannot use this GPU with the installed driver; a newer NVIDIA driver is needed"; exit 1; }
fi

[ -d "$VENV_TRACE" ] || uv venv "$VENV_TRACE" --python 3.12
if ! "$VENV_TRACE/bin/python" -c "import torch, transformers, accelerate, numpy" 2> /dev/null; then
    ( source "$VENV_TRACE/bin/activate" && uv pip install torch transformers accelerate numpy --torch-backend=auto )
fi

# The reference core and the CPU kernel, with every check that needs no GPU.
cd "$LEAPQUANT_HOME"
make
make test 2>&1 | tail -1 | tee "$RESULTS/make_test.txt"
( source "$VENV_TRACE/bin/activate" && make selftest kernel-test kernel-selftest PYTHON=python3 2>&1 | tee "$RESULTS/make_checks.txt" | tail -3 )

# The GPU kernel.
if command -v nvcc > /dev/null; then
    nvcc --version | tail -1 | tee "$RESULTS/nvcc.txt"
    ( source "$VENV_TRACE/bin/activate" && python3 -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda)" | tee -a "$RESULTS/nvcc.txt" )
    make cuda
    echo "built $LEAPQUANT_BUILD_DIR/libleapquant_cuda.so"
else
    echo "nvcc not found. Install the CUDA toolkit whose major version matches torch's CUDA"
    echo "(see torch.version.cuda), put nvcc on PATH, then run:  make cuda"
    exit 1
fi

echo
echo "Pass criteria: make test ends with '0 failed'; the self-check line ends in '(ok)';"
echo "kernel-test ends with '0 failed'; kernel-selftest ends with 'RESULT: PASS'."
