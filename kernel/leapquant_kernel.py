"""
leapquant_kernel.py

Python bindings for the batched LeapQuant kernel (leapquant_kernel.h).

The same four calls run on a CPU or on a GPU, depending on the device of the
state pool:

    pool   = StatePool(slots=256, value_heads=32, device="cuda")
    kernel = Kernel(pool.device)

    # once per request, after prefill (use case UC3)
    kernel.store(pool, slot, prefill_state)              # [B, HV, V, K] fp32, destroyed

    # every decode step, for every layer
    kernel.decode_step(pool, slot, tokens_in_window, decay, beta, key, query, value, out)
    kernel.fold(pool, slot, tokens_in_window, tokens_added=1, force=force)

    # once per decode step, after all layers
    tokens_in_window = next_tokens_in_window(tokens_in_window, pool.window_length, force)

Libraries
    CPU   build/libleapquant_cpu.so   (.dylib on macOS)   make kernel
    GPU   build/libleapquant_cuda.so                      make cuda   (needs nvcc)
    Set LEAPQUANT_BUILD_DIR to look somewhere else.

Rules for every tensor passed in
    - It must already have the right dtype, shape and device, and be contiguous.
      Nothing is converted for you. On a GPU the kernel runs after the call
      returns, so a temporary copy made here could be freed before it is read.
    - Index tensors (slot, tokens_in_window, force) are int32.
"""

import ctypes
import os
import sys

import torch


# ---- C structs, in the field order of leapquant_kernel.h ----------------------

class _LqPool(ctypes.Structure):
    _fields_ = [
        ("rounded_residual", ctypes.c_void_p),
        ("value_channel_step", ctypes.c_void_p),
        ("key_row_scale", ctypes.c_void_p),
        ("compensator_keys", ctypes.c_void_p),
        ("compensator_corrections", ctypes.c_void_p),
        ("window_keys", ctypes.c_void_p),
        ("window_corrections", ctypes.c_void_p),
        ("window_log_decays", ctypes.c_void_p),
        ("slot_count", ctypes.c_int32),
        ("value_heads", ctypes.c_int32),
        ("key_channels", ctypes.c_int32),
        ("value_channels", ctypes.c_int32),
        ("window_length", ctypes.c_int32),
        ("compensator_count", ctypes.c_int32),
        ("largest_integer_level", ctypes.c_int32),
        ("power_iteration_sweeps", ctypes.c_int32),
        ("use_smoothing", ctypes.c_int32),
        ("fit_together", ctypes.c_int32),
        ("profile_stop_phase", ctypes.c_int32),
        ("scale_group", ctypes.c_int32),
    ]


class _LqStep(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int32),
        ("key_heads", ctypes.c_int32),
        ("slot", ctypes.c_void_p),
        ("tokens_in_window", ctypes.c_void_p),
        ("decay", ctypes.c_void_p),
        ("beta", ctypes.c_void_p),
        ("key", ctypes.c_void_p),
        ("query", ctypes.c_void_p),
        ("value", ctypes.c_void_p),
        ("output", ctypes.c_void_p),
        ("raw_qkv", ctypes.c_void_p),
        ("raw_a", ctypes.c_void_p),
        ("raw_b", ctypes.c_void_p),
        ("A_log", ctypes.c_void_p),
        ("dt_bias", ctypes.c_void_p),
        ("scale", ctypes.c_float),
        ("raw_dtype", ctypes.c_int32),
        ("raw_stride", ctypes.c_int64),
    ]


class _LqFold(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int32),
        ("tokens_added", ctypes.c_int32),
        ("slot", ctypes.c_void_p),
        ("tokens_in_window", ctypes.c_void_p),
        ("force", ctypes.c_void_p),
        ("scratch", ctypes.c_void_p),
        ("export_by_slot", ctypes.c_void_p),
    ]


class _LqStore(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int32),
        ("slot", ctypes.c_void_p),
        ("state", ctypes.c_void_p),
        ("source_by_slot", ctypes.c_void_p),
        ("skip_slot", ctypes.c_void_p),
        ("source_index", ctypes.c_void_p),
        ("source_stride", ctypes.c_int64),
    ]


class _LqExport(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_int32),
        ("slot", ctypes.c_void_p),
        ("tokens_in_window", ctypes.c_void_p),
        ("state", ctypes.c_void_p),
    ]


_ERROR_TEXT = {
    -1: "LQ_ERROR_BAD_POOL: a pool size is out of range (see LQ_MAX_* in leapquant_kernel.h) or a pool pointer is null",
    -2: "LQ_ERROR_BAD_ARGUMENT: a call argument is out of range or a pointer is null",
    -3: "LQ_ERROR_CUDA: the CUDA runtime reported an error when the kernel was launched",
}


def build_directory():
    """Where the compiled libraries are looked for."""
    override = os.environ.get("LEAPQUANT_BUILD_DIR")
    if override:
        return override
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")


def library_path(name):
    """Full path of build/lib<name>.so, or .dylib on macOS."""
    extension = "dylib" if sys.platform == "darwin" else "so"
    return os.path.join(build_directory(), f"lib{name}.{extension}")


def _address(tensor):
    """Address of a tensor's first element, or null for an empty tensor."""
    if tensor is None or tensor.numel() == 0:
        return None
    return ctypes.c_void_p(tensor.data_ptr())


# Power-iteration sweeps per compensator used by the engine-side code and tools.
# The reference core's own default is 8. 2 sweeps cost a quarter of the fold's
# compensator time; kernel/verify_kernel.py and leapquant_replay --sweeps 2 show
# what they cost in accuracy.
# 15 Nov 2026: ONE sweep is the numerical policy (Asif's decision on backlog B52: the fold 0.88 against 1.09 ms at
# 512 rows, fold loss 1.0126 against 1.0060 of the exact kernel's, nine traces PASS; results/fold_sweeps1).
# LEAPQUANT_SWEEPS=2 restores the earlier policy (runs before 15 Nov used 2).
DEFAULT_SWEEPS = int(os.environ.get("LEAPQUANT_SWEEPS", "1"))


def paper_compensator_count(bits_per_entry):
    """Paper v2, section 4.1. 8-bit keeps 4 compensators. 6-bit and 4-bit use 6."""
    if bits_per_entry in (4, 6):
        return 6
    return 4


def low_bit_scale_group(bits_per_entry):
    """Key rows that share one column step. 0 is the whole column. 6-bit and 4-bit use 8."""
    if bits_per_entry in (4, 6):
        return 8
    return 0


def scale_group_count(key_channels, scale_group):
    """Steps stored for one value column. 1 when `scale_group` is 0 or covers the column."""
    if scale_group <= 0 or scale_group >= key_channels:
        return 1
    return (key_channels + scale_group - 1) // scale_group


class StatePool:
    """
    The LeapQuant state of one layer for `slots` requests.

    Tensors (K = key_channels, V = value_channels, R = compensator_count, L = window_length):
        rounded_residual         int8     [slots, HV, V, K]   } boundary state:
        value_channel_step       float32  [slots, HV, V]      } what a prefix-cache
                                 or [slots, HV, V, G] when G key-row groups share the column
        key_row_scale            float32  [slots, HV, K]      } checkpoint keeps
        compensator_keys         float16  [slots, HV, R, K]   }
        compensator_corrections  float16  [slots, HV, R, V]   }
        window_keys              float16  [slots, HV, L, K]   window buffer:
        window_corrections       float16  [slots, HV, L, V]   only requests that
        window_log_decays        float16  [slots, HV, L]      are generating need it

    The number of records in a request's window is not kept here. The caller
    owns it (see the module text).
    """

    BOUNDARY_FIELDS = ("rounded_residual", "value_channel_step", "key_row_scale",
                       "compensator_keys", "compensator_corrections")
    WINDOW_FIELDS = ("window_keys", "window_corrections", "window_log_decays")

    def __init__(self, slots, value_heads, key_channels=128, value_channels=128, window_length=16,
                 compensator_count=None, bits_per_entry=8, power_iteration_sweeps=DEFAULT_SWEEPS,
                 use_smoothing=True, device="cpu", with_window=True, fit_together=True, scale_group=None):
        if not 2 <= bits_per_entry <= 8:
            raise ValueError("bits_per_entry must be between 2 and 8")
        # None follows the paper: rank 4 at 8 bits, rank 6 and groups of 8 at 6 and 4 bits.
        # An explicit count or group is left alone, so an 8-bit run and a rank-4 ablation stay put.
        if compensator_count is None:
            compensator_count = paper_compensator_count(bits_per_entry)
        if scale_group is None:
            scale_group = low_bit_scale_group(bits_per_entry)
        self.slots = slots
        self.value_heads = value_heads
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.window_length = window_length
        self.compensator_count = compensator_count
        self.bits_per_entry = bits_per_entry
        self.power_iteration_sweeps = power_iteration_sweeps
        self.use_smoothing = bool(use_smoothing)
        self.scale_group = int(scale_group)
        self.scale_group_count = scale_group_count(key_channels, self.scale_group)
        self.device = torch.device(device)
        self.profile_stop_phase = 0     # profiling only; see kernel/profile_kernel.py
        # All compensators in one joint fit (the reference's Settings::fit_compensators_together).
        self.fit_together = fit_together

        heads = (slots, value_heads)
        self.rounded_residual = torch.zeros(heads + (value_channels, key_channels), dtype=torch.int8, device=self.device)
        # G = 1 keeps the old [slots, HV, V] shape. A real group is [slots, HV, V, G].
        if self.scale_group_count == 1:
            step_shape = heads + (value_channels,)
        else:
            step_shape = heads + (value_channels, self.scale_group_count)
        self.value_channel_step = torch.zeros(step_shape, dtype=torch.float32, device=self.device)
        self.key_row_scale = torch.ones(heads + (key_channels,), dtype=torch.float32, device=self.device)
        self.compensator_keys = torch.zeros(heads + (compensator_count, key_channels), dtype=torch.float16, device=self.device)
        self.compensator_corrections = torch.zeros(heads + (compensator_count, value_channels), dtype=torch.float16, device=self.device)
        # with_window=False: a pool that only stores boundary states (a checkpoint
        # store). It can be stored to and exported from, but not decoded or folded.
        self.with_window = with_window
        window_slots = heads if with_window else (0, value_heads)
        self.window_keys = torch.zeros(window_slots + (window_length, key_channels), dtype=torch.float16, device=self.device)
        self.window_corrections = torch.zeros(window_slots + (window_length, value_channels), dtype=torch.float16, device=self.device)
        self.window_log_decays = torch.zeros(window_slots + (window_length,), dtype=torch.float16, device=self.device)

    # ---- sizes ------------------------------------------------------------------
    def boundary_bytes_per_head(self):
        return sum(getattr(self, name)[0, 0].numel() * getattr(self, name).element_size() for name in self.BOUNDARY_FIELDS)

    def window_bytes_per_head(self):
        if not self.with_window:
            return 0
        return sum(getattr(self, name)[0, 0].numel() * getattr(self, name).element_size() for name in self.WINDOW_FIELDS)

    # ---- whole-slot operations -----------------------------------------------------
    def clear(self, slots=None):
        """Sets slots to the zero state (all slots if `slots` is None). The caller resets its cursor."""
        index = slice(None) if slots is None else slots
        self.rounded_residual[index] = 0
        self.value_channel_step[index] = 0.0
        self.key_row_scale[index] = 1.0
        self.compensator_keys[index] = 0.0
        self.compensator_corrections[index] = 0.0

    def copy_boundary_state(self, source_slots, destination_slots):
        """Copies boundary states between slots of this pool. The caller sets the cursor of the destination to 0."""
        for name in self.BOUNDARY_FIELDS:
            tensor = getattr(self, name)
            tensor[destination_slots] = tensor[source_slots]

    def read_boundary_state(self, slots):
        """
        Returns a copy of the boundary state of `slots` as a dict of tensors.
        This is the prefix-cache checkpoint (use cases UC4 and UC5): the int8
        state, stored as is. The window of the slot must be empty; force a fold
        first if it is not.
        """
        return {name: getattr(self, name)[slots].clone() for name in self.BOUNDARY_FIELDS}

    def write_boundary_state(self, slots, checkpoint):
        """Writes a checkpoint made by read_boundary_state() into `slots`. The caller sets their cursor to 0."""
        for name in self.BOUNDARY_FIELDS:
            getattr(self, name)[slots] = checkpoint[name].to(self.device)

    # ---- for the kernel -----------------------------------------------------------
    def c_struct(self):
        return _LqPool(
            rounded_residual=_address(self.rounded_residual),
            value_channel_step=_address(self.value_channel_step),
            key_row_scale=_address(self.key_row_scale),
            compensator_keys=_address(self.compensator_keys),
            compensator_corrections=_address(self.compensator_corrections),
            window_keys=_address(self.window_keys),
            window_corrections=_address(self.window_corrections),
            window_log_decays=_address(self.window_log_decays),
            slot_count=self.slots,
            value_heads=self.value_heads,
            key_channels=self.key_channels,
            value_channels=self.value_channels,
            window_length=self.window_length,
            compensator_count=self.compensator_count,
            largest_integer_level=(1 << (self.bits_per_entry - 1)) - 1,
            power_iteration_sweeps=self.power_iteration_sweeps,
            use_smoothing=1 if self.use_smoothing else 0,
            fit_together=1 if self.fit_together else 0,
            profile_stop_phase=self.profile_stop_phase,
            scale_group=self.scale_group,
        )


def next_tokens_in_window(tokens_in_window, window_length, force=None, active=None):
    """
    The cursor update after one decode step (after ALL layers ran).

        folded      -> 0
        otherwise   -> tokens_in_window + 1
    Rows where `active` is False (padded rows) keep their cursor.
    Uses tensor operations only, so it can be captured in a CUDA graph.
    """
    records = tokens_in_window + 1
    folded = records >= window_length
    if force is not None:
        folded = folded | (force != 0)
    result = torch.where(folded, torch.zeros_like(records), records)
    if active is not None:
        result = torch.where(active, result, tokens_in_window)
    return result


class Kernel:
    """The four kernel entry points, on the CPU or on the GPU."""

    def __init__(self, device="cpu", library=None):
        self.device = torch.device(device)
        self.is_cuda = self.device.type == "cuda"
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError(f"the LeapQuant kernel runs on cpu or cuda, not on {self.device.type}")
        # LEAPQUANT_CUDA_LIBRARY selects another build of the GPU kernel (kernel/tune_kernel.py uses it).
        if library is None and self.is_cuda:
            library = os.environ.get("LEAPQUANT_CUDA_LIBRARY") or None
        path = library or library_path("leapquant_cuda" if self.is_cuda else "leapquant_cpu")
        if not os.path.exists(path):
            target = "cuda" if self.is_cuda else "kernel"
            raise FileNotFoundError(f"{path} not found. Build it with:  make {target}")
        self.library_file = path
        self._library = ctypes.CDLL(path)
        prefix = "lq_cuda_" if self.is_cuda else "lq_cpu_"
        self._decode_step = getattr(self._library, prefix + "decode_step")
        self._fold = getattr(self._library, prefix + "fold")
        self._store = getattr(self._library, prefix + "store")
        self._export = getattr(self._library, prefix + "export")
        for function in (self._decode_step, self._fold, self._store, self._export):
            function.restype = ctypes.c_int
        self._scratch = None

    # ---- helpers ---------------------------------------------------------------
    def _call(self, function, pool, arguments):
        pool_struct = pool.c_struct()
        if self.is_cuda:
            stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            status = function(ctypes.byref(pool_struct), ctypes.byref(arguments), stream)
        else:
            status = function(ctypes.byref(pool_struct), ctypes.byref(arguments))
        if status != 0:
            raise RuntimeError(_ERROR_TEXT.get(status, f"LeapQuant kernel error {status}"))

    def _check(self, name, tensor, dtype, shape):
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, not {tensor.dtype}")
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}, not {tuple(tensor.shape)}")
        if tensor.device.type != self.device.type:
            raise ValueError(f"{name} is on {tensor.device}, the kernel runs on {self.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def _check_pool(self, pool):
        if pool.device.type != self.device.type:
            raise ValueError(f"the pool is on {pool.device}, the kernel runs on {self.device}")

    def scratch_for(self, pool, rows):
        """Working memory for fold(): [rows, HV, V, K] fp32. Kept and reused; grown when needed."""
        shape = (rows, pool.value_heads, pool.value_channels, pool.key_channels)
        if self._scratch is None or self._scratch.shape[0] < rows or tuple(self._scratch.shape[1:]) != shape[1:]:
            self._scratch = torch.empty(shape, dtype=torch.float32, device=self.device)
        return self._scratch

    # ---- the four entry points ----------------------------------------------------
    def decode_step(self, pool, slot, tokens_in_window, decay, beta, key, query, value, out):
        """
        One token for each of B requests (use case UC1).

            slot              int32   [B]         pool slot per row; negative = padded row
            tokens_in_window  int32   [B]         records already in the window, 0 .. L-1
            decay             float32 [B, HV]     alpha, in (0, 1]
            beta              float32 [B, HV]
            key, query        float32 [B, H, K]   as the recurrence uses them (normalised; query scaled)
            value             float32 [B, HV, V]
            out               float32 [B, HV, V]  filled
        Appends one record per row. Does not fold: call fold() next.
        """
        self._check_pool(pool)
        rows = slot.shape[0]
        key_heads = key.shape[1]
        self._check("slot", slot, torch.int32, (rows,))
        self._check("tokens_in_window", tokens_in_window, torch.int32, (rows,))
        self._check("decay", decay, torch.float32, (rows, pool.value_heads))
        self._check("beta", beta, torch.float32, (rows, pool.value_heads))
        self._check("key", key, torch.float32, (rows, key_heads, pool.key_channels))
        self._check("query", query, torch.float32, (rows, key_heads, pool.key_channels))
        self._check("value", value, torch.float32, (rows, pool.value_heads, pool.value_channels))
        self._check("out", out, torch.float32, (rows, pool.value_heads, pool.value_channels))
        step = _LqStep(rows=rows, key_heads=key_heads, slot=_address(slot),
                       tokens_in_window=_address(tokens_in_window), decay=_address(decay), beta=_address(beta),
                       key=_address(key), query=_address(query), value=_address(value), output=_address(out))
        self._call(self._decode_step, pool, step)
        return out

    RAW_DTYPES = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}

    def decode_step_raw(self, pool, slot, tokens_in_window, mixed_qkv, a, b, A_log, dt_bias, scale, key_heads, out):
        """
        decode_step() with the gates and the normalisation done inside the kernel.

            mixed_qkv   [B, >= 2*H*K + HV*V]  packed (q | k | v), float32, bfloat16 or float16
            a, b        [B, HV]   same dtype as mixed_qkv
            A_log, dt_bias   float32 [HV]
            scale       float, applied to the normalised query
        """
        self._check_pool(pool)
        rows = slot.shape[0]
        dtype = mixed_qkv.dtype
        if dtype not in self.RAW_DTYPES:
            raise TypeError(f"mixed_qkv must be float32, bfloat16 or float16, not {dtype}")
        width = 2 * key_heads * pool.key_channels + pool.value_heads * pool.value_channels
        if mixed_qkv.dim() != 2 or mixed_qkv.shape[0] != rows or mixed_qkv.shape[1] < width or mixed_qkv.stride(1) != 1:
            raise ValueError(f"mixed_qkv must be [{rows}, >= {width}] with unit stride in its last dimension")
        self._check("slot", slot, torch.int32, (rows,))
        self._check("tokens_in_window", tokens_in_window, torch.int32, (rows,))
        self._check("a", a, dtype, (rows, pool.value_heads))
        self._check("b", b, dtype, (rows, pool.value_heads))
        self._check("A_log", A_log, torch.float32, (pool.value_heads,))
        self._check("dt_bias", dt_bias, torch.float32, (pool.value_heads,))
        self._check("out", out, torch.float32, (rows, pool.value_heads, pool.value_channels))
        step = _LqStep(rows=rows, key_heads=key_heads, slot=_address(slot), tokens_in_window=_address(tokens_in_window),
                       output=_address(out), raw_qkv=_address(mixed_qkv), raw_a=_address(a), raw_b=_address(b),
                       A_log=_address(A_log), dt_bias=_address(dt_bias), scale=float(scale),
                       raw_dtype=self.RAW_DTYPES[dtype], raw_stride=mixed_qkv.stride(0))
        self._call(self._decode_step, pool, step)
        return out

    def fold(self, pool, slot, tokens_in_window, tokens_added=1, force=None, scratch=None, export_by_slot=None):
        """
        Closes the window of every row where tokens_in_window + tokens_added >= L,
        or where force != 0 and the window is not empty (use cases UC2 and UC5).
        Other rows are untouched. tokens_added is 1 right after decode_step(), else 0.

            scratch         float32 [>= B, HV, V, K]  working memory; made and kept by this object if None
            export_by_slot  float32 [slots, HV, V, K] or None. Every row that folds also writes its
                            full-precision state (before rounding) to the entry of its slot.
        """
        self._check_pool(pool)
        rows = slot.shape[0]
        state_shape = (pool.value_heads, pool.value_channels, pool.key_channels)
        self._check("slot", slot, torch.int32, (rows,))
        self._check("tokens_in_window", tokens_in_window, torch.int32, (rows,))
        if force is not None:
            self._check("force", force, torch.int32, (rows,))
        if scratch is None:
            scratch = self.scratch_for(pool, rows)
        elif scratch.shape[0] < rows:
            raise ValueError(f"scratch holds {scratch.shape[0]} rows, {rows} are needed")
        else:
            self._check("scratch", scratch, torch.float32, (scratch.shape[0],) + state_shape)
        if export_by_slot is not None:
            self._check("export_by_slot", export_by_slot, torch.float32, (pool.slots,) + state_shape)
        fold = _LqFold(rows=rows, tokens_added=tokens_added, slot=_address(slot),
                       tokens_in_window=_address(tokens_in_window), force=_address(force), scratch=_address(scratch),
                       export_by_slot=_address(export_by_slot))
        self._call(self._fold, pool, fold)

    def store(self, pool, slot, state, source_by_slot=None, skip_slot=None, source_index=None):
        """
        Turns full-precision states into boundary states (use case UC3).

            state           float32 [>= B, HV, V, K]  used as working memory and DESTROYED.
                            It is the input, unless source_by_slot is given.
            source_by_slot  float32 [slots, HV, V, K] or None. The input of a row is then the entry
                            of its slot; it is copied into `state` and not changed.
            skip_slot       int32 [slots] or None. Rows whose slot has a nonzero entry are skipped.
            source_index    int32 [B] or None. If given, a row reads entry source_index[row] of
                            source_by_slot instead of the entry of its pool slot. source_by_slot may
                            then have any number of entries, and padding between entries (its
                            stride along the first dimension is passed to the kernel).
        The caller sets the cursor of the stored slots to 0.
        """
        self._check_pool(pool)
        rows = slot.shape[0]
        state_shape = (pool.value_heads, pool.value_channels, pool.key_channels)
        self._check("slot", slot, torch.int32, (rows,))
        if state.shape[0] < rows:
            raise ValueError(f"state holds {state.shape[0]} rows, {rows} are needed")
        self._check("state", state, torch.float32, (state.shape[0],) + state_shape)
        source_stride = 0
        if source_by_slot is not None and source_index is None:
            self._check("source_by_slot", source_by_slot, torch.float32, (pool.slots,) + state_shape)
        elif source_by_slot is not None:
            # Indexed by the caller: any number of entries, possibly with padding between them.
            self._check("source_index", source_index, torch.int32, (rows,))
            if source_by_slot.dtype != torch.float32 or tuple(source_by_slot.shape[1:]) != state_shape:
                raise ValueError(f"source_by_slot must be float32 [entries, {state_shape}], "
                                 f"not {source_by_slot.dtype} {tuple(source_by_slot.shape)}")
            if source_by_slot.device.type != self.device.type:
                raise ValueError(f"source_by_slot is on {source_by_slot.device}, the kernel runs on {self.device}")
            dense_inner = (state_shape[1] * state_shape[2], state_shape[2], 1)
            if tuple(source_by_slot.stride()[1:]) != dense_inner:
                raise ValueError("each entry of source_by_slot must be contiguous")
            source_stride = source_by_slot.stride(0)
        if skip_slot is not None:
            self._check("skip_slot", skip_slot, torch.int32, (pool.slots,))
        store = _LqStore(rows=rows, slot=_address(slot), state=_address(state),
                         source_by_slot=_address(source_by_slot), skip_slot=_address(skip_slot),
                         source_index=_address(source_index), source_stride=source_stride)
        self._call(self._store, pool, store)

    def export(self, pool, slot, tokens_in_window, out):
        """
        Writes the current state (boundary state plus window) in full precision.
            out     float32 [B, HV, V, K]   filled
        """
        self._check_pool(pool)
        rows = slot.shape[0]
        self._check("slot", slot, torch.int32, (rows,))
        if tokens_in_window is not None:     # None = every window is empty
            self._check("tokens_in_window", tokens_in_window, torch.int32, (rows,))
        self._check("out", out, torch.float32, (rows, pool.value_heads, pool.value_channels, pool.key_channels))
        request = _LqExport(rows=rows, slot=_address(slot), tokens_in_window=_address(tokens_in_window),
                            state=_address(out))
        self._call(self._export, pool, request)
        return out
