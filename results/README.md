# Results

Each stage script writes its logs to `results/<date>-<host>/`. Check that directory in
after a run, and fill in the tables below. Until then every number is "not measured".

## Measured so far, without a GPU

| What | Result | How to repeat |
|---|---|---|
| Reference core tests | 2,127,027 checks, 0 failed | `make test` |
| Batched kernel against the reference core, CPU build | 2,081,375 checks, 0 failed; bit-identical on every token, fold and final state | `make kernel-test` |
| Kernel as an emulated GPU thread block (5, 32 and 128 threads per block) | Same bytes as one thread; no data race under ThreadSanitizer | `make kernel-race-check` |
| Kernel under AddressSanitizer and UndefinedBehaviorSanitizer | 0 failed | `make kernel-sanitize` |
| Python tools and engine glue against the reference | PASS, bit-identical | `make kernel-selftest` |
| CUDA source | Parses with clang's CUDA front end against stand-in headers | `make cuda-syntax-check` |
| Trace recorder on a real model | Qwen3.5-0.8B on a CPU: self-check 7.5e-06 (ok) | see `docs/STATUS.md` |

## Measured on the GPU (H100 PCIe 80 GB, driver 570, CUDA 12.8, 13 November 2026)

| What | Result |
|---|---|
| `make cuda` | Compiled with nvcc 12.8 at the first attempt, no source change. |
| GPU kernel against the reference, synthetic trace (32 heads, 128 x 128, 400 tokens) | `RESULT: PASS`. Inside a window: median difference 2.1e-07, 99% below 1.1e-05, worst 4.8e-05. Folds: 50 of 50 bit-identical to the reference, 0 of 26,214,400 int8 entries differ. |
| Free run against fp32 | Output error: kernel 0.8537%, reference 0.8537%. The two free runs end 1.9e-02 apart, as predicted in `docs/STATUS.md`, finding 2. |
| Engine glue on the GPU (`engine_test.py`, `block_engine_test.py --device cuda`) | Output error against the fp32 engine 0.365% and 0.121%, the same as on the CPU. |

## Stage B. Baselines on the H100

Output token throughput (tokens per second) and mean inter-token latency (ms).

| Configuration | Concurrency 32, long | Concurrency 128, long | Concurrency 512, short |
|---|---|---|---|
| 1. fp32 state, no window | | | |
| 2. fp32 state, 16-token window | | | |
| 3. bf16 state, 16-token window | | | |
| 4. LeapQuant, phase 1 (stage F) | | | |

Gate: configuration 2 against configuration 1.

Share of GPU time in the Gated DeltaNet decode kernel (stage B3): batch 32: ___, batch 128: ___.

## Stage C. Error on real states

One replay table per trace: `results/<run>/c_replay_*.txt`. Gate: the LeapQuant row against
the "bf16, rounded every window" row.

## Stage D. GPU kernel against the reference

`results/<run>/d_verify_*.txt`. Every file must end with `RESULT: PASS`.

## Stage E. Kernel speed

`results/<run>/e_kernel_speed_*.log`.
