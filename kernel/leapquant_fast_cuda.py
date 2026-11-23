# SPDX-License-Identifier: Apache-2.0
"""
leapquant_fast_cuda.py

The hand-written CUDA fold of the serving path (kernel/leapquant_fast_fold.cu), loaded with ctypes.
leapquant_triton.fold() calls it when LEAPQUANT_FOLD_BACKEND=cuda (or FOLD_BACKEND is set) and the pool
is within its limits; otherwise the Triton fold runs.

Build on the GPU machine:  make fast-fold   ->  build/libleapquant_fastfold.so
(LEAPQUANT_BUILD_DIR overrides the folder, as for the exact kernels.)
"""

import ctypes
import os

import torch

LIBRARY_NAME = "libleapquant_fastfold.so"


class _Args(ctypes.Structure):
    _fields_ = [
        ("residual", ctypes.c_void_p),
        ("channel_step", ctypes.c_void_p),
        ("row_scale", ctypes.c_void_p),
        ("comp_c", ctypes.c_void_p),
        ("comp_k", ctypes.c_void_p),
        ("win_c", ctypes.c_void_p),
        ("win_k", ctypes.c_void_p),
        ("win_g", ctypes.c_void_p),
        ("slot", ctypes.c_void_p),
        ("records", ctypes.c_void_p),
        ("order", ctypes.c_void_p),
        ("count", ctypes.c_void_p),
        ("value_heads", ctypes.c_int32),
        ("key_channels", ctypes.c_int32),
        ("value_channels", ctypes.c_int32),
        ("window_length", ctypes.c_int32),
        ("compensators", ctypes.c_int32),
        ("sweeps", ctypes.c_int32),
        ("use_smoothing", ctypes.c_int32),
        ("levels", ctypes.c_float),
        ("source", ctypes.c_void_p),
        ("source_stride", ctypes.c_longlong),
        ("source_index", ctypes.c_void_p),
        ("has_index", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


class _DecodeArgs(ctypes.Structure):
    _fields_ = [
        ("mixed_qkv", ctypes.c_void_p), ("a", ctypes.c_void_p), ("b", ctypes.c_void_p),
        ("A_log", ctypes.c_void_p), ("dt_bias", ctypes.c_void_p), ("out", ctypes.c_void_p),
        ("residual", ctypes.c_void_p), ("channel_step", ctypes.c_void_p), ("row_scale", ctypes.c_void_p),
        ("comp_c", ctypes.c_void_p), ("comp_k", ctypes.c_void_p),
        ("win_c", ctypes.c_void_p), ("win_k", ctypes.c_void_p), ("win_g", ctypes.c_void_p),
        ("slot", ctypes.c_void_p), ("write_pos", ctypes.c_void_p),
        ("stride_qkv", ctypes.c_longlong), ("stride_a", ctypes.c_longlong), ("stride_b", ctypes.c_longlong),
        ("scale", ctypes.c_float),
        ("rows", ctypes.c_int32), ("key_heads", ctypes.c_int32), ("value_heads", ctypes.c_int32),
        ("key_channels", ctypes.c_int32), ("value_channels", ctypes.c_int32), ("window_length", ctypes.c_int32),
        ("compensators", ctypes.c_int32), ("in_dtype", ctypes.c_int32), ("out_dtype", ctypes.c_int32),
        ("pre_gated", ctypes.c_int32), ("l2norm", ctypes.c_int32), ("variant", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
    ]


# Which CUDA decode runs (LEAPQUANT_DECODE_BACKEND=cuda): 0 = v5 (one warp per head, plain loads); 2, 3 or 4 = v7
# (heads pipelined through shared memory with that many stages, int8 tensor cores; window of 16 records only);
# 8 = v8 (one warp per head, no block barriers, the levels loaded straight into tensor-core fragments; window 16);
# 9 = v8 with 4 consecutive value rows per lane in the last part (one 8-byte load per record);
# 10 = 9 + the next item's first chunks loaded early (measured 2% slower than 9: not used).
DECODE_VARIANT = int(os.environ.get("LEAPQUANT_CUDA_DECODE_VARIANT", "9"))   # v9: gate G3 met with it, 14 Nov


_DTYPE_CODE = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_LIB = None
_LOAD_ERROR = None
_GRID = {}
_FP32_PARAMS = {}


def library_path():
    build = os.environ.get("LEAPQUANT_BUILD_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
    return os.path.join(build, LIBRARY_NAME)


def _library():
    global _LIB, _LOAD_ERROR
    if _LIB is not None:
        return _LIB
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR
    try:
        lib = ctypes.CDLL(library_path())
        lib.lq_fast_fold.argtypes = [ctypes.POINTER(_Args), ctypes.c_int, ctypes.c_void_p]
        lib.lq_fast_fold.restype = ctypes.c_int
        lib.lq_fast_fold_init.restype = ctypes.c_int
        lib.lq_fast_fold_blocks_per_sm.argtypes = [ctypes.c_int]
        lib.lq_fast_fold_blocks_per_sm.restype = ctypes.c_int
        lib.lq_fast_store.argtypes = [ctypes.POINTER(_Args), ctypes.c_int, ctypes.c_void_p]
        lib.lq_fast_store.restype = ctypes.c_int
        lib.lq_fast_store_blocks_per_sm.argtypes = [ctypes.c_int]
        lib.lq_fast_store_blocks_per_sm.restype = ctypes.c_int
        if hasattr(lib, "lq_fast_decode"):
            lib.lq_fast_decode.argtypes = [ctypes.POINTER(_DecodeArgs), ctypes.c_void_p]
            lib.lq_fast_decode.restype = ctypes.c_int
            lib.lq_fast_decode_args_size.restype = ctypes.c_int
            if lib.lq_fast_decode_args_size() != ctypes.sizeof(_DecodeArgs):
                raise OSError(f"{LIBRARY_NAME}: decode argument layout differs ({lib.lq_fast_decode_args_size()} "
                              f"against {ctypes.sizeof(_DecodeArgs)} bytes); rebuild it")
        lib.lq_fast_fold_args_size.restype = ctypes.c_int
        if lib.lq_fast_fold_args_size() != ctypes.sizeof(_Args):
            raise OSError(f"{LIBRARY_NAME}: argument layout differs ({lib.lq_fast_fold_args_size()} against "
                          f"{ctypes.sizeof(_Args)} bytes); rebuild it")
        error = lib.lq_fast_fold_init()
        if error:
            raise OSError(f"{LIBRARY_NAME}: setting the kernels' shared memory failed (CUDA error {error})")
    except OSError as exc:
        _LOAD_ERROR = exc
        raise
    _LIB = lib
    return lib


def available():
    """True when the library loads (and a GPU is present)."""
    if not torch.cuda.is_available():
        return False
    try:
        _library()
        return True
    except OSError:
        return False


def supports(pool, export_by_slot=None):
    """The kernel's limits: 128 x 128 heads, one step per value column, at most 16 records and 8 compensators."""
    return (export_by_slot is None and pool.device.type == "cuda" and pool.key_channels == 128
            and pool.value_channels == 128 and 1 <= pool.compensator_count <= 8 and 1 <= pool.window_length <= 16
            and getattr(pool, "with_window", True) and getattr(pool, "scale_group_count", 1) == 1
            and pool.bits_per_entry <= 8)


def _grid(device, compensators, work_items, store=False):
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, compensators <= 4, store)
    if key not in _GRID:
        lib = _library()
        per_sm = (lib.lq_fast_store_blocks_per_sm if store else lib.lq_fast_fold_blocks_per_sm)(compensators)
        _GRID[key] = torch.cuda.get_device_properties(index).multi_processor_count * max(1, per_sm)
    return max(1, min(work_items, _GRID[key]))


def _args(pool, slot, records, order, count, sweeps, source=None, source_index=None):
    with_window = getattr(pool, "with_window", True)
    return _Args(
        pool.rounded_residual.data_ptr(), pool.value_channel_step.data_ptr(), pool.key_row_scale.data_ptr(),
        pool.compensator_corrections.data_ptr(), pool.compensator_keys.data_ptr(),
        pool.window_corrections.data_ptr() if with_window else None,
        pool.window_keys.data_ptr() if with_window else None,
        pool.window_log_decays.data_ptr() if with_window else None,
        slot.data_ptr(), records.data_ptr() if records is not None else None, order.data_ptr(), count.data_ptr(),
        pool.value_heads, pool.key_channels, pool.value_channels, pool.window_length, pool.compensator_count,
        int(sweeps), int(bool(pool.use_smoothing)), float((1 << (pool.bits_per_entry - 1)) - 1),
        source.data_ptr() if source is not None else None, source.stride(0) if source is not None else 0,
        source_index.data_ptr() if source_index is not None else None, int(source_index is not None), 0,
    )


def _launch(entry, pool, slot, args, store):
    device = slot.device
    with torch.cuda.device(device):
        grid = _grid(device, pool.compensator_count, slot.shape[0] * pool.value_heads, store)
        error = entry(ctypes.byref(args), grid, ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if error:
        raise RuntimeError(f"the CUDA {'store' if store else 'fold'} failed to launch "
                           f"(error {error}; -1 = pool outside the kernel's limits)")


def fold(pool, slot, records, order, count, sweeps):
    """Folds the rows of the work list (order, count), as leapquant_triton.fold. No checks of the
    tensors beyond what fold() did; the caller has built the work list on the current stream."""
    _launch(_library().lq_fast_fold, pool, slot, _args(pool, slot, records, order, count, sweeps), store=False)


def store_supports(pool):
    """The store needs no window; otherwise the fold's limits."""
    return (pool.device.type == "cuda" and pool.key_channels == 128 and pool.value_channels == 128
            and 1 <= pool.compensator_count <= 8 and getattr(pool, "scale_group_count", 1) == 1
            and pool.bits_per_entry <= 8)


def decode_supports(pool, mixed_qkv, a, b, out):
    """The CUDA decode's limits: 128 x 128 heads, at most 16 records and 8 compensators, one step per column,
    mixed_qkv / a / b of one dtype (fp32, fp16 or bf16), the library built with the decode."""
    lib = _library()
    basic = (hasattr(lib, "lq_fast_decode") and pool.key_channels == 128 and pool.value_channels == 128
             and 1 <= pool.window_length <= 16 and 1 <= pool.compensator_count <= 8
             and getattr(pool, "scale_group_count", 1) == 1 and mixed_qkv.dtype in _DTYPE_CODE
             and a.dtype == mixed_qkv.dtype and b.dtype == mixed_qkv.dtype and out.dtype in _DTYPE_CODE
             and a.stride(-1) == 1 and b.stride(-1) == 1)
    if not basic or DECODE_VARIANT == 0:
        return basic
    # v7 copies q, k and v in 16-byte pieces: 16-byte aligned rows; a window of exactly 16 records.
    size = mixed_qkv.element_size()
    return (pool.window_length == 16 and mixed_qkv.data_ptr() % 16 == 0
            and (mixed_qkv.stride(0) * size) % 16 == 0 and mixed_qkv.stride(-1) == 1)


def _fp32(tensor):
    """An fp32 copy of a small parameter, made once (stable across CUDA-graph captures)."""
    if tensor.dtype == torch.float32 and tensor.is_contiguous():
        return tensor
    key = (tensor.data_ptr(), tensor.dtype, tensor.numel())
    if key not in _FP32_PARAMS:
        _FP32_PARAMS[key] = tensor.detach().float().contiguous()
    return _FP32_PARAMS[key]


def decode(pool, mixed_qkv, a, b, A_log, dt_bias, scale, slot, write_pos, out, key_heads, pre_gated, l2norm):
    """One decode token for every row (the arguments of leapquant_triton._launch)."""
    lib = _library()
    if pre_gated:
        A_log_ptr = dt_bias_ptr = None            # not read
    else:
        A_log_ptr, dt_bias_ptr = _fp32(A_log).data_ptr(), _fp32(dt_bias).data_ptr()
    args = _DecodeArgs(
        mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr(), A_log_ptr, dt_bias_ptr, out.data_ptr(),
        pool.rounded_residual.data_ptr(), pool.value_channel_step.data_ptr(), pool.key_row_scale.data_ptr(),
        pool.compensator_corrections.data_ptr(), pool.compensator_keys.data_ptr(),
        pool.window_corrections.data_ptr(), pool.window_keys.data_ptr(), pool.window_log_decays.data_ptr(),
        slot.data_ptr(), write_pos.data_ptr(),
        mixed_qkv.stride(0), a.stride(0), b.stride(0), float(scale),
        mixed_qkv.shape[0], key_heads, pool.value_heads, pool.key_channels, pool.value_channels, pool.window_length,
        pool.compensator_count, _DTYPE_CODE[mixed_qkv.dtype], _DTYPE_CODE[out.dtype], int(bool(pre_gated)),
        int(bool(l2norm)), DECODE_VARIANT, 0,
    )
    device = mixed_qkv.device
    with torch.cuda.device(device):
        error = lib.lq_fast_decode(ctypes.byref(args), ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream))
    if error:
        raise RuntimeError(f"the CUDA decode failed to launch (error {error}; -1 = pool outside the kernel's limits)")
    return out


def store(pool, slot, source, source_index, order, count, sweeps):
    """Stores (adopts) source states into the pool for the rows of the work list, as leapquant_triton.store.
    `sweeps` is the number of sweeps of the fit (the Triton store's SWEEPS + 2)."""
    _launch(_library().lq_fast_store, pool, slot,
            _args(pool, slot, None, order, count, sweeps, source=source, source_index=source_index), store=True)
