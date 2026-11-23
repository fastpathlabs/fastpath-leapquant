#!/usr/bin/env python3
"""
tune_kernel.py

Builds the GPU kernel in every combination of its tuning options, checks each
build against the reference core, times it, and installs the fastest build
that is still correct as build/libleapquant_cuda.so.

    python3 kernel/tune_kernel.py                      # on the GPU machine, after `make kernel`
    python3 kernel/tune_kernel.py --trace traces/explain/trace_layer_13.bin

Options that are tried (see the Makefile)
    CUDA_THREADS  64, 128, 256   threads per block
    FOLD_SHARED   1, 0           the fold's matrix in the block's shared memory, or in scratch memory
    FMAD          false, true    let the GPU fuse multiply and add
    (WORD_LOADS is fixed at 1: the first sweep found it 17% faster and bit-identical.)

None of them changes what is computed, except FMAD=true, which changes the last
bit of some results. Every build must pass kernel/verify_kernel.py to be
considered. The speed that is compared is the row "leapquant, with gates" of
kernel/bench_kernel.py at the largest batch size: one decode step of one layer
as the SGLang patch runs it.

This tunes build options only. Changes to the kernel source (more threads for
the record dot products, the fold's matrix in shared memory) need a profiler
and are not automated here.
"""

import argparse
import itertools
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(command, env=None, log=None):
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if log:
        with open(log, "w", encoding="utf-8") as handle:
            handle.write(result.stdout)
    return result.returncode, result.stdout


def main():
    parser = argparse.ArgumentParser(description="Try the build options of the GPU kernel and keep the fastest correct one.")
    parser.add_argument("--trace", help="a recorded trace to verify with (default: a synthetic 32-head, 128 x 128 trace)")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--max-tokens", type=int, default=400, help="tokens of the trace used for verification")
    parser.add_argument("--threads", type=int, nargs="+", default=[64, 128, 256])
    args = parser.parse_args()

    tune_dir = os.path.join(ROOT, "build", "tune")
    os.makedirs(tune_dir, exist_ok=True)

    trace = args.trace
    if trace is None:
        trace = os.path.join(tune_dir, "synthetic_trace.bin")
        sys.path.insert(0, ROOT)
        from record_trace import run_self_test
        run_self_test(trace, heads=32, key_channels=128, value_channels=128, tokens=200)

    results = []
    for threads, fold_shared, fmad in itertools.product(args.threads, (1, 0), ("false", "true")):
        word_loads = 1          # settled by the first sweep (17% faster, bit-identical)
        tag = f"t{threads}_s{fold_shared}_fmad{fmad}"
        library = os.path.join(tune_dir, f"libleapquant_cuda_{tag}.so")
        status, output = run(["make", "cuda", f"CUDA_THREADS={threads}", f"WORD_LOADS={word_loads}", f"FMAD={fmad}",
                              f"FOLD_SHARED={fold_shared}", f"CUDA_LIB={library}"],
                             log=os.path.join(tune_dir, f"{tag}.build.log"))
        if status != 0:
            print(f"{tag}: build failed, see build/tune/{tag}.build.log")
            results.append((tag, None, None, (threads, fold_shared, fmad)))
            continue
        status, output = run([sys.executable, "kernel/verify_kernel.py", trace, "--device", "cuda", "--rows", "2",
                              "--max-tokens", str(args.max_tokens), "--kernel-library", library],
                             log=os.path.join(tune_dir, f"{tag}.verify.log"))
        correct = status == 0 and "RESULT: PASS" in output
        environment = dict(os.environ, LEAPQUANT_CUDA_LIBRARY=library)
        status, output = run([sys.executable, "kernel/bench_kernel.py", "--device", "cuda", "--batch-sizes"] +
                             [str(batch) for batch in args.batch_sizes], env=environment,
                             log=os.path.join(tune_dir, f"{tag}.bench.log"))
        times = [float(match) for match in re.findall(r"leapquant, with gates\s+([0-9.]+)", output)] if status == 0 else []
        results.append((tag, correct, times, (threads, fold_shared, fmad)))
        shown = "  ".join(f"{value:8.4f}" for value in times) if times else "benchmark failed"
        print(f"{tag:<24} verify {'PASS' if correct else 'FAIL'}   ms per step at batch {args.batch_sizes}: {shown}", flush=True)

    usable = [entry for entry in results if entry[1] and entry[2]]
    if not usable:
        print("\nNo build passed verification and ran the benchmark. Logs are in build/tune/. Nothing was installed.")
        sys.exit(1)
    best = min(usable, key=lambda entry: entry[2][-1])
    # Fused multiply-add makes folds differ from the reference in the last bit. Only take
    # it for a real gain: on an H100 it measured under 1%, which is noise.
    exact = [entry for entry in usable if entry[3][2] == "false"]
    if exact:
        best_exact = min(exact, key=lambda entry: entry[2][-1])
        if best_exact[2][-1] <= 1.03 * best[2][-1]:
            best = best_exact
    threads, fold_shared, fmad = best[3]
    shutil.copyfile(os.path.join(tune_dir, f"libleapquant_cuda_{best[0]}.so"),
                    os.path.join(ROOT, "build", "libleapquant_cuda.so"))
    with open(os.path.join(ROOT, "build", "tuned_options.txt"), "w", encoding="utf-8") as handle:
        handle.write(f"CUDA_THREADS={threads} WORD_LOADS=1 FOLD_SHARED={fold_shared} FMAD={fmad}\n")
    print(f"\nFastest correct build at batch {args.batch_sizes[-1]}: {best[0]} ({best[2][-1]:.4f} ms per step).")
    print(f"Installed as build/libleapquant_cuda.so. To rebuild it:  make cuda CUDA_THREADS={threads} "
          f"FOLD_SHARED={fold_shared} FMAD={fmad}")


if __name__ == "__main__":
    main()
