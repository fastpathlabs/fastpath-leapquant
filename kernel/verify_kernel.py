#!/usr/bin/env python3
"""
verify_kernel.py

Stage D of the implementation plan: feeds a recorded trace to the batched
kernel (CPU or GPU build) and to the C++ reference core, and compares them.

    python3 kernel/verify_kernel.py traces/trace_layer_00.bin --device cuda
    python3 kernel/verify_kernel.py selftest.bin --device cpu --rows 3

Why this is not a plain "outputs equal to 1e-4 on every token" comparison
    Rounding amplifies. If the kernel and the reference differ in the last bit
    of one float (a GPU's exp() is not the CPU's), sooner or later one entry of
    the state tips to the other int8 level or the other fp16 value. From then
    on the two runs hold different, equally valid roundings of the same state,
    and they drift apart until they differ by about as much as LeapQuant
    differs from fp32. Measured on a CPU with two builds of the same code:
    1e-7 at the first token, 1e-2 after about 80 tokens. No GPU kernel can pass
    a 1e-4 test over a whole trace, and failing it says nothing about a bug.

What is compared instead. Each trace is run through the kernel twice.

    A. Inside a window. Kernel and reference start every window from the same
       boundary state. Their outputs must agree token by token.
           pass: median difference < 1e-6, 99% of all (token, head) pairs
                 < 1e-4, worst < 1e-3
       The difference of one (token, head) pair is |kernel - reference| divided by
       |reference|, but by no less than 1% of the median |reference| of the run: an
       output that happens to be near zero is judged by the size of the outputs
       around it, not by its own (seen on one trace, where |reference| was 4e-5 of
       the median and the plain relative difference 3e-3).
       With --fast (the Triton kernels) the limits are 1e-5, 1e-3 and 1e-2: they
       add in another order, so their differences start larger.
    B. Folds. Kernel and reference are given the same full-precision matrix
       and each turns it into a boundary state. Counted: folds that are
       bit-identical, int8 entries and fp16 values that differ.
           pass: the kernel's fold loses at most 5% more than the reference's
                 (one-sided: a more accurate kernel passes)
    C. Free run. Kernel and reference each run the whole trace alone. Both are
       compared with the fp32 state.
           pass: the kernel's output error is at most 10% above the reference's
                 (one-sided). Not-a-number, infinity, or a nonzero result where the
                 reference is zero always fail.
           With --fast the per-trace limit is 50% and the set of traces is judged by
           kernel/summarize_verify.py: a free run that is not bit-identical drifts
           along its own trajectory, and on some traces that moves the ratio by 10-15
           points between equally valid builds (see summarize_verify.py).

    The CPU build must also be bit-identical to the reference in A, B and C.
    That is checked when --device cpu is used.

--fast
    The decode step and the fold of kernel/leapquant_triton.py (tensor cores)
    take the place of the exact kernel's; everything else is the same.

Rows
    --rows N replays the trace as N requests that join the batch at different
    steps (so their windows end at different steps), in scrambled pool slots,
    next to one padded row.

Needs: torch, numpy, and the libraries built by `make kernel` (and `make cuda`
for --device cuda).
"""

import argparse
import ctypes
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from leapquant_kernel import (  # noqa: E402
    DEFAULT_SWEEPS, Kernel, StatePool, library_path, low_bit_scale_group, next_tokens_in_window,
    paper_compensator_count, scale_group_count,
)

try:
    import leapquant_triton as fast_kernels  # noqa: E402
except Exception:  # noqa: BLE001
    fast_kernels = None

MAGIC = b"LQTRACE1"
OUTPUT_MAGIC = b"LQOUTPUT"


class Trace:
    """A trace file written by record_trace.py (format: see leapquant_replay.cpp)."""

    def __init__(self, path, max_tokens=None):
        with open(path, "rb") as handle:
            if handle.read(8) != MAGIC:
                sys.exit(f"error: {path} is not a LeapQuant trace file")
            (self.key_channels, self.value_channels, self.heads, decay_values,
             has_initial_state, self.has_expected_output) = struct.unpack("<6i", handle.read(24))
        if decay_values != 1:
            sys.exit("error: this trace has one decay per key channel (KDA). The kernel handles Gated DeltaNet only.")
        offset = 32
        entry_count = self.key_channels * self.value_channels
        self.initial_state = None
        if has_initial_state:
            self.initial_state = np.fromfile(path, dtype="<f4", count=self.heads * entry_count, offset=offset)
            self.initial_state = self.initial_state.reshape(self.heads, self.key_channels, self.value_channels)
            offset += 4 * self.heads * entry_count
        K, V = self.key_channels, self.value_channels
        floats_per_head = 2 + K + V + K + (V if self.has_expected_output else 0)
        self.tokens = (os.path.getsize(path) - offset) // (4 * self.heads * floats_per_head)
        if max_tokens is not None:
            self.tokens = min(self.tokens, max_tokens)
        if self.tokens < 1:
            sys.exit("error: the trace holds no complete token")
        self._data = np.memmap(path, dtype="<f4", mode="r", offset=offset, shape=(self.tokens, self.heads, floats_per_head))

    def token(self, index):
        """Returns (alpha [H], beta [H], key [H, K], value [H, V], query [H, K]) of one token, as contiguous arrays."""
        K, V = self.key_channels, self.value_channels
        block = np.ascontiguousarray(self._data[index])
        return (np.ascontiguousarray(block[:, 0]), np.ascontiguousarray(block[:, 1]),
                np.ascontiguousarray(block[:, 2:2 + K]), np.ascontiguousarray(block[:, 2 + K:2 + K + V]),
                np.ascontiguousarray(block[:, 2 + K + V:2 + 2 * K + V]))


def _float_pointer(array):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


class ReferenceLayer:
    """One layer of reference heads plus fp32 truth heads (kernel/reference_shim.cpp)."""

    def __init__(self, library, trace, window_length, compensator_count, bits_per_entry, sweeps, use_smoothing,
                 scale_group=0):
        self._library = library
        self.heads = trace.heads
        self.key_channels = trace.key_channels
        self.value_channels = trace.value_channels
        self.compensator_count = compensator_count
        self.scale_group = scale_group
        self.scale_group_count = scale_group_count(trace.key_channels, scale_group)
        self._handle = ctypes.c_void_p(library.lqref_create(
            trace.heads, trace.key_channels, trace.value_channels, window_length, compensator_count,
            bits_per_entry, sweeps, 1 if use_smoothing else 0, scale_group))
        if not self._handle:
            sys.exit("error: the reference core rejected the settings")
        self.leapquant_output = np.zeros((self.heads, self.value_channels), dtype=np.float32)
        self.truth_output = np.zeros((self.heads, self.value_channels), dtype=np.float32)

    def load_state(self, states, which=3):
        """states: [H, K, V]. which: 1 = LeapQuant heads, 2 = truth heads, 3 = both."""
        states = np.ascontiguousarray(states, dtype=np.float32)
        self._library.lqref_load_state(self._handle, _float_pointer(states), which)

    def step(self, alpha, beta, key, value, query):
        self._library.lqref_step(self._handle, _float_pointer(alpha), _float_pointer(beta), _float_pointer(key),
                                 _float_pointer(value), _float_pointer(query),
                                 _float_pointer(self.leapquant_output), _float_pointer(self.truth_output))
        return self.leapquant_output, self.truth_output

    def tokens_in_window(self):
        return self._library.lqref_tokens_in_window(self._handle)

    def state(self, which=1):
        """Full-precision state [H, K, V]. which: 1 = LeapQuant heads, 2 = truth heads."""
        states = np.zeros((self.heads, self.key_channels, self.value_channels), dtype=np.float32)
        self._library.lqref_copy_state(self._handle, _float_pointer(states), which)
        return states

    def boundary_state(self):
        """The boundary state in pool layout, as a dict of torch tensors."""
        H, K, V, R = self.heads, self.key_channels, self.value_channels, self.compensator_count
        residual = np.zeros((H, V, K), dtype=np.int8)
        if self.scale_group_count == 1:
            step = np.zeros((H, V), dtype=np.float32)
        else:
            step = np.zeros((H, V, self.scale_group_count), dtype=np.float32)
        scale = np.zeros((H, K), dtype=np.float32)
        keys = np.zeros((H, R, K), dtype=np.uint16)
        corrections = np.zeros((H, R, V), dtype=np.uint16)
        self._library.lqref_copy_boundary_state(
            self._handle, residual.ctypes.data_as(ctypes.c_void_p), _float_pointer(step), _float_pointer(scale),
            keys.ctypes.data_as(ctypes.c_void_p), corrections.ctypes.data_as(ctypes.c_void_p))
        return {
            "rounded_residual": torch.from_numpy(residual),
            "value_channel_step": torch.from_numpy(step),
            "key_row_scale": torch.from_numpy(scale),
            "compensator_keys": torch.from_numpy(keys.view(np.float16)),
            "compensator_corrections": torch.from_numpy(corrections.view(np.float16)),
        }


def load_reference_library():
    path = library_path("leapquant_reference")
    if not os.path.exists(path):
        sys.exit(f"error: {path} not found. Build it with:  make kernel")
    library = ctypes.CDLL(path)
    library.lqref_create.restype = ctypes.c_void_p
    library.lqref_create.argtypes = [ctypes.c_int] * 9
    library.lqref_load_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    library.lqref_step.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_float)] * 7
    library.lqref_tokens_in_window.argtypes = [ctypes.c_void_p]
    library.lqref_tokens_in_window.restype = ctypes.c_int
    library.lqref_copy_state.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
    library.lqref_copy_boundary_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                                                  ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    return library


# Below this size a reference vector counts as zero, and the difference is judged absolutely.
ABSOLUTE_FLOOR = 1e-12


def relative_difference(first, second):
    """
    Per head: length of (first - second) / length of second. Inputs [H, n] numpy arrays.
    If the reference is zero, a nonzero candidate is NOT "no difference": the result is infinity.
    Not-a-number or infinite inputs also give infinity, so they can never pass a limit.
    """
    first64, second64 = first.astype(np.float64), second.astype(np.float64)
    difference = np.linalg.norm(first64 - second64, axis=-1)
    reference = np.linalg.norm(second64, axis=-1)
    result = np.where(reference > ABSOLUTE_FLOOR, difference / np.maximum(reference, ABSOLUTE_FLOOR),
                      np.where(difference > ABSOLUTE_FLOOR, np.inf, 0.0))
    bad = ~(np.isfinite(first64).all(axis=-1) & np.isfinite(second64).all(axis=-1))
    return np.where(bad, np.inf, result)


def floored_relative(difference_norms, reference_norms, share=0.01):
    """
    |kernel - reference| / max(|reference|, share * median |reference|), per (token, head).
    Inputs: flat float64 arrays of norms. Not-a-number or infinite norms give infinity.
    """
    finite = np.isfinite(difference_norms) & np.isfinite(reference_norms)
    floor = share * float(np.median(reference_norms[finite])) if finite.any() else 0.0
    floor = max(floor, ABSOLUTE_FLOOR)
    result = difference_norms / np.maximum(reference_norms, floor)
    return np.where(finite, result, np.inf)


def not_worse(candidate, reference, allowed_excess, floor):
    """One-sided limit: the candidate may be better than the reference by any amount, and worse by at
    most `allowed_excess` (a share) plus an absolute `floor`. A candidate that is not finite fails."""
    return bool(np.isfinite(candidate)) and candidate <= reference * (1.0 + allowed_excess) + floor

def main():
    parser = argparse.ArgumentParser(description="Compare the batched LeapQuant kernel with the reference core on a trace.")
    parser.add_argument("trace")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--rows", type=int, default=2, help="requests the trace is replayed as, per comparison")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--compensators", type=int, default=None,
                        help="omitted: rank 4 at 8 bits, rank 6 at 6 and 4 bits")
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--scale-group", type=int, default=None,
                        help="key rows per column step; omitted: 0 at 8 bits, 8 at 6 and 4 bits")
    parser.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS)
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--kernel-library", metavar="PATH",
                        help="load this kernel library instead of the one in build/. With --device cpu the "
                             "bit-identity check is then reported but not required, so that a CPU build with "
                             "other floating-point flags can stand in for a GPU")
    parser.add_argument("--fast", action="store_true",
                        help="the decode step and fold of kernel/leapquant_triton.py (needs --device cuda)")
    parser.add_argument("--kernel-sweeps", type=int, default=None,
                        help="--fast: sweeps of the fast fold's fit (default: --sweeps; 0 = half sweep)")
    parser.add_argument("--fit-every", type=int, default=1,
                        help="--fast: the full fit (--kernel-sweeps) on every n-th fold, half sweeps between")
    parser.add_argument("--fit-bf16", action="store_true", help="the fast fold's fit in bf16 without rescaling")
    parser.add_argument("--fold-one-pass", action="store_true",
                        help="the fast fold's records and compensators as one fp16 product each (no lo part)")
    parser.add_argument("--reference-outputs", metavar="PATH",
                        help="a file written by `leapquant_replay --write-outputs`; the free-running reference "
                             "of this tool must reproduce it bit for bit")
    args = parser.parse_args()
    if fast_kernels is not None:
        fast_kernels.FOLD_FIT_BF16 = args.fit_bf16
        fast_kernels.FOLD_EXACT_M = not args.fold_one_pass

    trace = Trace(args.trace, args.max_tokens)
    H, K, V, L = trace.heads, trace.key_channels, trace.value_channels, args.window
    compensators = paper_compensator_count(args.bits) if args.compensators is None else args.compensators
    scale_group = low_bit_scale_group(args.bits) if args.scale_group is None else args.scale_group
    device = torch.device(args.device)
    if args.fast and (device.type != "cuda" or fast_kernels is None):
        sys.exit("error: --fast needs --device cuda and triton")
    if args.fast and 0 < scale_group < K:
        sys.exit("error: --fast keeps one step per column; pass --scale-group 0 or omit --fast")
    kernel = Kernel(device, library=args.kernel_library)
    reference_library = load_reference_library()

    # Batch layout: row 0 is a padded row. Then `rows` synchronised requests,
    # then `rows` free-running requests. Request r of each kind joins at step 5 * r.
    rows = args.rows
    batch = 1 + 2 * rows
    slots = batch + 2
    start_step = [None] + [5 * r for r in range(rows)] * 2
    is_synchronised = [False] + [True] * rows + [False] * rows
    slot_of_row = [-1] + [slots - 1 - index for index in range(2 * rows)]   # scrambled on purpose
    total_steps = trace.tokens + 5 * (rows - 1)

    pool = StatePool(slots, H, K, V, window_length=L, compensator_count=compensators, bits_per_entry=args.bits,
                     power_iteration_sweeps=args.sweeps, use_smoothing=not args.no_smoothing, device=device,
                     scale_group=scale_group)
    references = [None] + [ReferenceLayer(reference_library, trace, L, compensators, args.bits, args.sweeps,
                                          not args.no_smoothing, scale_group) for _ in range(2 * rows)]

    print(f"trace: {args.trace}")
    print(f"  {trace.tokens} tokens, {H} heads, state {K} x {V}, window {L}, {compensators} compensators, "
          f"{args.bits} bits, scale group {scale_group}, {args.sweeps} sweeps")
    print(f"  kernel: {'kernel/leapquant_triton.py (decode step, fold) + ' if args.fast else ''}{kernel.library_file} on {device}")
    if args.fast:
        print(f"  fast fold: {args.sweeps if args.kernel_sweeps is None else args.kernel_sweeps} sweeps"
              f"{'' if args.fit_every == 1 else f' on every {args.fit_every}th fold, half sweeps between'}")
    print(f"  {rows} synchronised + {rows} free-running requests + 1 padded row; "
          f"stored per head: {pool.boundary_bytes_per_head():,} + {pool.window_bytes_per_head():,} bytes")

    # ---- Starting state (use case UC3) -----------------------------------------
    real_rows = list(range(1, batch))
    if trace.initial_state is not None:
        state_value_major = np.ascontiguousarray(trace.initial_state.transpose(0, 2, 1))        # [H, V, K]
        states = torch.from_numpy(np.broadcast_to(state_value_major, (len(real_rows), H, V, K)).copy()).to(device)
        kernel.store(pool, torch.tensor([slot_of_row[row] for row in real_rows], dtype=torch.int32, device=device), states)
        for row in real_rows:
            references[row].load_state(trace.initial_state, which=3)

    tokens_in_window = torch.zeros(batch, dtype=torch.int32, device=device)
    fast_fold_calls = 0
    in_window_differences = []
    in_window_difference_norms, in_window_reference_norms = [], []
    fold_count = fold_identical = 0
    residual_entries = residual_differ = half_values = half_differ = 0
    kernel_fold_error = reference_fold_error = fold_matrix_length = 0.0
    kernel_error = reference_error = truth_length = 0.0
    worst_free_difference = 0.0
    free_reference_outputs = [] if args.reference_outputs else None
    padded_row_ok = True
    bit_identical = True

    for step_index in range(total_steps):
        token_of_row = [None] * batch
        for row in real_rows:
            token_index = step_index - start_step[row]
            if 0 <= token_index < trace.tokens:
                token_of_row[row] = token_index
        active_rows = [row for row in real_rows if token_of_row[row] is not None]

        decay = np.ones((batch, H), dtype=np.float32)
        beta = np.zeros((batch, H), dtype=np.float32)
        key = np.zeros((batch, H, K), dtype=np.float32)
        query = np.zeros((batch, H, K), dtype=np.float32)
        value = np.zeros((batch, H, V), dtype=np.float32)
        step_slot = np.full(batch, -1, dtype=np.int32)
        inputs_of_row = {}
        for row in active_rows:
            inputs_of_row[row] = trace.token(token_of_row[row])
            decay[row], beta[row], key[row], value[row], query[row] = inputs_of_row[row]
            step_slot[row] = slot_of_row[row]

        slot_tensor = torch.from_numpy(step_slot).to(device)
        out = torch.full((batch, H, V), 123.0, dtype=torch.float32, device=device)
        step_inputs = (torch.from_numpy(decay).to(device), torch.from_numpy(beta).to(device), torch.from_numpy(key).to(device),
                       torch.from_numpy(query).to(device), torch.from_numpy(value).to(device))
        if args.fast:
            fast_kernels.decode_step(pool, slot_tensor, tokens_in_window, *step_inputs, out)
        else:
            kernel.decode_step(pool, slot_tensor, tokens_in_window, *step_inputs, out)
        kernel_output = out.cpu().numpy()
        padded_row_ok = padded_row_ok and not kernel_output[0].any()

        # Which rows fold after this token. The kernel's full-precision state of
        # the synchronised ones is read first: it is the matrix both sides fold.
        cursor = tokens_in_window.cpu().numpy()
        folding_rows = [row for row in active_rows if cursor[row] + 1 >= L]
        folding_synchronised = [row for row in folding_rows if is_synchronised[row]]
        matrix_before_fold = None
        if folding_synchronised:
            fold_slots = torch.tensor([slot_of_row[row] for row in folding_synchronised], dtype=torch.int32, device=device)
            full_window = torch.full((len(folding_synchronised),), L, dtype=torch.int32, device=device)
            matrix_before_fold = torch.empty((len(folding_synchronised), H, V, K), dtype=torch.float32, device=device)
            kernel.export(pool, fold_slots, full_window, matrix_before_fold)
            matrix_before_fold = matrix_before_fold.cpu().numpy()

        if args.fast:
            due = fast_kernels.records_to_fold(tokens_in_window, L, None, tokens_added=1)
            fold_sweeps = args.sweeps if args.kernel_sweeps is None else args.kernel_sweeps
            if bool(((due > 0) & (slot_tensor >= 0)).any()):
                fast_fold_calls += 1
                if (fast_fold_calls - 1) % args.fit_every != 0:
                    fold_sweeps = 0
            fast_kernels.fold(pool, slot_tensor, due, sweeps=fold_sweeps)
        else:
            kernel.fold(pool, slot_tensor, tokens_in_window, tokens_added=1)
        active_mask = torch.from_numpy(step_slot >= 0).to(device)
        new_tokens_in_window = next_tokens_in_window(tokens_in_window, L, active=active_mask)

        for row in active_rows:
            alpha_r, beta_r, key_r, value_r, query_r = inputs_of_row[row]
            reference_output, truth_output = references[row].step(alpha_r, beta_r, key_r, value_r, query_r)
            if is_synchronised[row]:
                in_window_differences.append(relative_difference(kernel_output[row], reference_output))
                in_window_difference_norms.append(np.linalg.norm(kernel_output[row].astype(np.float64) - reference_output, axis=-1))
                in_window_reference_norms.append(np.linalg.norm(reference_output.astype(np.float64), axis=-1))
            else:
                difference = relative_difference(kernel_output[row], reference_output).max()
                worst_free_difference = max(worst_free_difference, float(difference))
                kernel_error += float(((kernel_output[row].astype(np.float64) - truth_output) ** 2).sum())
                reference_error += float(((reference_output.astype(np.float64) - truth_output) ** 2).sum())
                truth_length += float((truth_output.astype(np.float64) ** 2).sum())
                if free_reference_outputs is not None and row == 1 + rows:
                    free_reference_outputs.append(reference_output.copy())
            if not np.array_equal(kernel_output[row], reference_output):
                bit_identical = False

        # ---- Folds of the synchronised rows --------------------------------------
        for position, row in enumerate(folding_synchronised):
            matrix = matrix_before_fold[position]                                    # [H, V, K]
            reference = references[row]
            reference.load_state(np.ascontiguousarray(matrix.transpose(0, 2, 1)), which=1 | 4)   # continues the request
            reference_boundary = reference.boundary_state()
            slot = slot_of_row[row]

            identical = True
            for name in StatePool.BOUNDARY_FIELDS:
                ours = getattr(pool, name)[slot].cpu()
                theirs = reference_boundary[name]
                same = torch.equal(ours.view(torch.int16) if ours.dtype == torch.float16 else ours,
                                   theirs.view(torch.int16) if theirs.dtype == torch.float16 else theirs)
                identical = identical and same
                if name == "rounded_residual":
                    residual_entries += ours.numel()
                    residual_differ += int((ours != theirs).sum())
                elif ours.dtype == torch.float16:
                    half_values += ours.numel()
                    half_differ += int((ours.view(torch.int16) != theirs.view(torch.int16)).sum())
            fold_count += 1
            fold_identical += 1 if identical else 0
            bit_identical = bit_identical and identical

            # How much each fold lost: its boundary state against the matrix it was given.
            one_slot = torch.tensor([slot], dtype=torch.int32, device=device)
            empty_window = torch.zeros(1, dtype=torch.int32, device=device)
            kernel_after = torch.empty((1, H, V, K), dtype=torch.float32, device=device)
            kernel.export(pool, one_slot, empty_window, kernel_after)
            kernel_after = kernel_after.cpu().numpy()[0].astype(np.float64)
            reference_after = reference.state(which=1).transpose(0, 2, 1).astype(np.float64)
            kernel_fold_error += float(((kernel_after - matrix) ** 2).sum())
            reference_fold_error += float(((reference_after - matrix) ** 2).sum())
            fold_matrix_length += float((matrix.astype(np.float64) ** 2).sum())

            # Synchronise: the kernel continues from the reference's boundary state.
            pool.write_boundary_state(slot, reference_boundary)

        tokens_in_window = new_tokens_in_window
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # ---- State at the end, free-running rows ---------------------------------------
    free_rows = [row for row in real_rows if not is_synchronised[row]]
    final = torch.empty((len(free_rows), H, V, K), dtype=torch.float32, device=device)
    kernel.export(pool, torch.tensor([slot_of_row[row] for row in free_rows], dtype=torch.int32, device=device),
                  tokens_in_window[free_rows].contiguous(), final)
    final = final.cpu().numpy().transpose(0, 1, 3, 2)                                 # [rows, H, K, V]
    kernel_state_error = reference_state_error = 0.0
    for position, row in enumerate(free_rows):
        truth = references[row].state(which=2).astype(np.float64)
        truth_norm = np.linalg.norm(truth)
        kernel_state_error = max(kernel_state_error, float(np.linalg.norm(final[position] - truth) / truth_norm))
        reference_state_error = max(reference_state_error,
                                    float(np.linalg.norm(references[row].state(which=1) - truth) / truth_norm))
        if not np.array_equal(final[position], references[row].state(which=1)):
            bit_identical = False

    # ---- Report ----------------------------------------------------------------------
    plain = np.concatenate([d.ravel() for d in in_window_differences])
    differences = floored_relative(np.concatenate(in_window_difference_norms), np.concatenate(in_window_reference_norms))
    median, p99, worst = np.median(differences), np.quantile(differences, 0.99), differences.max()
    limits = (1e-5, 1e-3, 1e-2) if args.fast else (1e-6, 1e-4, 1e-3)
    pass_a = median < limits[0] and p99 < limits[1] and worst < limits[2]
    print("\n  A. inside a window (kernel and reference start every window from the same state)")
    print(f"     output difference over {differences.size:,} (token, head) pairs: "
          f"median {median:.1e}, 99% below {p99:.1e}, worst {worst:.1e}")
    print(f"     (plain relative difference, not floored: worst {plain.max():.1e})")
    print(f"     limits: median < {limits[0]:.0e}, 99% < {limits[1]:.0e}, worst < {limits[2]:.0e} .......... {'pass' if pass_a else 'FAIL'}")

    pass_b = True
    print("\n  B. folds (the same full-precision matrix in, a boundary state out)")
    if fold_count == 0:
        print("     no fold happened: the trace is shorter than one window")
    else:
        kernel_loss = (kernel_fold_error / fold_matrix_length) ** 0.5
        reference_loss = (reference_fold_error / fold_matrix_length) ** 0.5
        pass_b = not_worse(kernel_loss, reference_loss, 0.05, 1e-7)
        print(f"     {fold_count} folds of {H} heads; bit-identical to the reference: {fold_identical}")
        print(f"     int8 entries that differ: {residual_differ:,} of {residual_entries:,}; "
              f"fp16 compensator values that differ: {half_differ:,} of {half_values:,}")
        print(f"     rounding error of the fold: kernel {100 * kernel_loss:.4f}%, reference {100 * reference_loss:.4f}%")
        print(f"     limit: kernel at most 5% worse than the reference ......... {'pass' if pass_b else 'FAIL'}")

    kernel_output_error = (kernel_error / truth_length) ** 0.5
    reference_output_error = (reference_error / truth_length) ** 0.5
    # --fast: per trace only a gross error fails (50%); the set of traces is judged by
    # kernel/summarize_verify.py (geometric mean at most 10% above the reference's). See there why.
    limit_c = 0.50 if args.fast else 0.10
    pass_c = not_worse(kernel_output_error, reference_output_error, limit_c, 1e-7)
    print("\n  C. free run (kernel and reference each run the whole trace alone), against fp32")
    print(f"     output error: kernel {100 * kernel_output_error:.4f}%, reference {100 * reference_output_error:.4f}%")
    print(f"     state error at the end: kernel {100 * kernel_state_error:.4f}%, reference {100 * reference_state_error:.4f}%")
    print(f"     largest difference between the two free runs: {worst_free_difference:.1e} (not a pass criterion)")
    print(f"     limit: kernel at most {100 * limit_c:.0f}% worse than the reference ........ {'pass' if pass_c else 'FAIL'}"
          + ("   (--fast: the set of traces is judged by kernel/summarize_verify.py)" if args.fast else ""))

    print(f"\n  padded row gives zeros ....................................... {'pass' if padded_row_ok else 'FAIL'}")

    passed = pass_a and pass_b and pass_c and padded_row_ok
    if device.type == "cpu" and args.kernel_library is None:
        print(f"  CPU build is bit-identical to the reference .................. {'pass' if bit_identical else 'FAIL'}")
        passed = passed and bit_identical
    else:
        print(f"  bit-identical to the reference: {'yes' if bit_identical else 'no (expected on a GPU)'}")

    if args.reference_outputs:
        with open(args.reference_outputs, "rb") as handle:
            if handle.read(8) != OUTPUT_MAGIC:
                sys.exit(f"error: {args.reference_outputs} was not written by leapquant_replay --write-outputs")
            file_heads, file_values, file_tokens = struct.unpack("<3i", handle.read(12))
            from_replay = np.frombuffer(handle.read(), dtype="<f4").reshape(file_tokens, file_heads, file_values)
        ours = np.stack(free_reference_outputs)
        count = min(len(ours), file_tokens)
        same = file_heads == H and file_values == V and np.array_equal(ours[:count], from_replay[:count])
        print(f"  reference in this tool equals leapquant_replay, {count} tokens ... {'pass' if same else 'FAIL'}")
        passed = passed and same

    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
