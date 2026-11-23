# Status against the implementation plan

> This file describes the state on 12 November 2026, before any GPU run. It is kept for the record.
> Current state: the README and the run records in `results/`.

12 November 2026. Everything that can be built and checked without a GPU is built and
checked. Nothing has run on a GPU yet. To run it all on the GPU machine: `./run_on_cuda.sh`.

## Where each step stands

| Step | State | What is left |
|---|---|---|
| 1. Reference core | Done. `make test`: 2,127,027 checks, 0 failed. `make selftest`: self-check ok. | Nothing. |
| 2. Baseline on the H100 | Scripts ready (`runbook/stage_b_baselines.sh`). | Run them. Gate. |
| 3. Error on real states | Recorder fixed and run on a small model. Scripts ready (`runbook/stage_c_traces.sh`). | Record Qwen3.5-9B on the H100. Gate. |
| 4. GPU kernel, standalone | Written. The CPU build of the same source is bit-identical to the reference. The CUDA file is not compiled yet. | `make cuda`, then `runbook/stage_d_e_kernel.sh d` and `e`. |
| 5. SGLang integration | Patch written, not run: decode kernel, and prefix-cache checkpoints in the LeapQuant format. The fp32 state of active requests is still kept. | Run it. |
| 6. Accuracy with the model in the loop | `eval/accuracy_bench.py`: GSM8K, AIME 2025, MMLU-Pro, with a paired comparison against the fp32 run. Its scoring is tested against a fake server. | Run it (`run_on_cuda.sh`, step `f-accuracy`). |
| 7. Scale test | Script ready (`runbook/stage_f_end_to_end.sh serving`). | Run it. |
| 8. vLLM port | First port written, not run: prefix caching off, no speculative decoding. `docs/VLLM_PORT.md`. | Run it (`run_on_cuda.sh`, steps `g-*`). |

## Tasks

| Task | State |
|---|---|
| T1 Reference core and tests | Done. |
| T2 Recorder and replay tool | Done, including the first run on a real model (Qwen3.5-0.8B, CPU). Not yet run on the 9B model. |
| T3 Baseline measurements, commands checked in | Commands checked in. No measurements. |
| T4 Traces from Qwen3.5-9B, replay tables | Script ready. No traces. |
| T5 GPU kernel: decode step | Written. Verified as a CPU build. Not compiled for a GPU. |
| T6 GPU kernel: fold | Same. |
| T7 Replay tool writes per-token outputs | Done: `leapquant_replay --write-outputs`. |
| T8 State pool layout for the int8 format | Done in the kernel (`StatePool`). In SGLang it sits beside the fp32 state (phase 1). |
| T9 Flag and decode dispatch | In the patch: `--enable-linear-leapquant`. Not run. |
| T10 Fold at window end, on forced flush, after prefill | In the kernel and in `EngineState.decode`. Tested on a CPU. Not run in SGLang. |
| T11 Checkpoint store accepts the boundary state | Done: `CheckpointStore`, used by SGLang's checkpoint pool when the flag is on. A checkpoint is bit-identical to the boundary state of the request it came from (tested on a CPU). Not run in SGLang. |
| T12 Accuracy runs | Script ready and self-tested. No runs. |
| T13 Scale test and report | Script ready. |
| T14 vLLM port | First port written (`integrations/vllm/`). Its glue is bit-identical to the reference on a CPU. Not run in vLLM. Prefix caching is not supported yet. |

## What was found on the way

### 1. The recorder's self-check failed on a correct trace. Fixed.

`record_trace.py` had never been run on a model. Run on Qwen3.5-0.8B, the self-check reported
a difference of 3.8e-03 and printed its warning. The cause: the transformers kernel computes
in float32 and rounds its output to bfloat16 at the very end. The recorder stored the rounded
output, which is 0.4% off. The recorder now calls the kernel with float32 inputs, stores the
float32 output and does the rounding itself. The model receives the same numbers as before.
After the fix the self-check reports 4e-06 to 4e-05 on three layers and says ok.

### 2. The stage D pass criterion cannot be met by any GPU kernel. Replaced.

The plan asks for kernel outputs within 1e-4 of the reference on every token of every trace.
Rounding makes that impossible. Two builds of the same code that differ only in the last bit
of some floats (here: with and without fused multiply-add) agree to 1e-7 on the first token.
Sooner or later one entry tips to the other int8 level or fp16 value, and from then on the two
runs hold different, equally valid roundings of the same state. On a real trace of 199 tokens
they ended 5e-2 apart. Both were equally close to fp32: 0.0726% and 0.0725% output error.

A GPU differs from a CPU in the last bit of `exp` and `log`, so the same will happen there.
`kernel/verify_kernel.py` therefore checks three things instead:

- Inside a window, with kernel and reference starting each window from the same state, the
  outputs agree: median below 1e-6, 99% below 1e-4, worst below 1e-3.
- Given the same full-precision matrix, the kernel's fold loses no more than the reference's.
- Running freely over the whole trace, the kernel is as close to fp32 as the reference is.

On a CPU all three hold exactly, with every difference equal to zero.

### 3. The reference core gave different numbers depending on compiler flags. Fixed.

With the original Makefile, the compiler was free to fuse a multiply and an add into one
instruction, and whether it did depended on how a loop was written. A specification that
others must match "number for number" should not depend on that. The Makefile now builds
everything with `-ffp-contract=off`. All reference tests still pass.

### 4. First numbers on real states

Qwen3.5-0.8B, one code prompt, 280 generated tokens, 16 heads per layer. Output error
against fp32:

| Layer | bf16, rounded every window | LeapQuant |
|---|---|---|
| 0 | 0.110% | 0.009% |
| 9 | 0.201% | 0.153% |
| 22 | 0.112% | 0.106% |

On these three layers LeapQuant is at least as accurate as bf16 with the window, at 1.19
bytes per entry instead of 2. On the synthetic inputs of the plan it was the other way round.
This is a small model, a short trace and one prompt. It does not replace stage C on
Qwen3.5-9B.

## How the kernel was verified without a GPU

- The arithmetic is one header, `kernel/leapquant_kernel_blocks.h`, written as the work of one
  GPU thread block. The CUDA file and the CPU file both include it.
- `make kernel-test`: the CPU build against the reference core, token by token, for five
  shapes including Qwen3.5-9B's (32 value heads, 16 key heads, 128 x 128), with requests that
  join at different steps, scrambled pool slots, padded rows and forced folds. Every output,
  every fold and every final state is bit-identical.
- `make kernel-race-check`: each block is run by 5, 32 and 128 real threads that meet at a
  barrier where the GPU build calls `__syncthreads()`, under ThreadSanitizer. No race is
  reported, and the results are the same bytes as with one thread. As a check of the check, I
  removed one barrier: ThreadSanitizer reported the race.
- `make kernel-selftest`: the Python bindings, the verification tool and the engine glue,
  against the reference core. Bit-identical.
- On three real traces from Qwen3.5-0.8B the CPU kernel is bit-identical to the reference.
- `make cuda-syntax-check`: the CUDA file parses with clang's CUDA front end against
  stand-in headers.

What this does not cover: the real CUDA headers, the kernel launches, and anything about speed.

## Where the work differs from the plan

| Plan | What was done | Why |
|---|---|---|
| Stage D: 1e-4 on every token | Three criteria in `verify_kernel.py` | Finding 2. |
| Kernel reads the int8 matrix as `[key][value]` | Stored `[value][key]` | SGLang's state is `[V][K]`, and each GPU thread then reads one contiguous run of bytes. The reference core is unchanged. |
| Kernel for Gated DeltaNet and KDA | Gated DeltaNet only | The target is Qwen3.5. KDA stays in the reference core. |
| SGLang state pool holds the int8 format | Phase 1 keeps the fp32 state and adds the LeapQuant state | It changes no cache code, so it can be tried first. It gives the speed measurement, not the memory saving. |
| Stage A installs SGLang from PyPI | Installed from source at the pinned commit | Baselines and the patched server then run the same code. |
| Reference core untouched | One read-only accessor added (`Head::rounded_residual`), `-ffp-contract=off` in the Makefile, `--write-outputs` in the replay tool | Needed to compare a kernel with the reference. `leapquant_reference.zip` is the original. |

## Risks for the first GPU run

- `kernel/leapquant_kernel_cuda.cu` has never been compiled by nvcc. Expect small fixes.
- nvcc must match the CUDA major version of the installed torch, or the library will not load.
- The kernel is written for correctness. `kernel/tune_kernel.py` tries its build options
  (threads per block, 8-byte loads of the int8 matrix, fused multiply-add), verifies every
  build and keeps the fastest correct one. That is as far as tuning goes without a profiler.
  The record dot products still use few threads and the fold keeps its matrix in global
  memory. Stage E may show the kernel slower than SGLang's tuned kernels. That would be a
  statement about this version, not about the method.
- The build disables fused multiply-add on the GPU (`FMAD=false`) so that folds match the
  reference exactly. Try `FMAD=true` for speed once stage D passes.
- The SGLang patch has never been imported. See `integrations/sglang/README.md` for the
  three assumptions to check.
- Every SGLang and vLLM flag in `runbook/` comes from the plan, which took it from source
  reading. None has been run.

## Still open after this round

- **The fp32 state of active requests is not dropped.** SGLang's prefill kernels write it, and
  its cache, copy, host-offload and transfer code all index it by slot. Removing it means
  giving every one of those an indirection to a small fp32 staging pool. That is a change to
  SGLang's memory pool that I will not write without being able to run it. What is done
  instead covers the plan's memory user story (US2, cached prefixes): checkpoints are stored
  in the LeapQuant format. Active requests cost more memory than today, not less.
- **A request restored from a LeapQuant checkpoint.** This is not a defect. Compared with the
  request it was copied from, the restored request differs by 0.6% in the CPU engine test,
  because the two carry different roundings. Compared with fp32, which is what matters, they
  are equally good: 0.53% output error for the original, 0.56% for the restored request. The
  one extra rounding is the checkpoint itself, as with any lossy checkpoint format.
- **SGLang's size estimate for the checkpoint pool** still assumes its own int8 layout and is
  about 15% low for LeapQuant checkpoints (19,456 against 16,896 bytes per head).
- **LiveCodeBench and GPQA Diamond** are not in the accuracy script (code sandbox; gated data).
- **vLLM with prefix caching.** The first port covers prefix caching off. `docs/VLLM_PORT.md`.
