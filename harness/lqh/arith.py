"""Dimensionally consistent comparisons (prompt rev. 3, section 14): time ratios are not speed ratios, byte
reductions are not capacity increases, and configured slots are not measured bytes."""
from __future__ import annotations


def time_and_speed(old_time: float, new_time: float) -> dict:
    """old 0.3821 ms -> new 0.331775 ms: 13.17% less time = 1.1517x speed (15.17% more throughput)."""
    return {"time_ratio": new_time / old_time, "time_reduction": 1 - new_time / old_time,
            "speedup_x": old_time / new_time, "speed_increase": old_time / new_time - 1}


def bytes_and_capacity(old_bytes: float, new_bytes: float) -> dict:
    """2 -> 1.19 bytes per entry: 40.5% less storage; ideal capacity with no other cost 2/1.19 = 1.681x (+68.1%)."""
    return {"byte_ratio": new_bytes / old_bytes, "byte_reduction": 1 - new_bytes / old_bytes,
            "ideal_capacity_x": old_bytes / new_bytes, "ideal_capacity_increase": old_bytes / new_bytes - 1,
            "caveat": "ideal: no other memory category; not equal total GPU memory, not SLO-qualified capacity"}


def conditional_cycle(decode_ms: float, fold_ms: float, window: int = 16) -> float:
    """A CONDITIONAL model (decode + fold / window), never a timed result: 0.2524 + 1.27/16 = 0.331775 ms."""
    return decode_ms + fold_ms / window


def fold_budget(target_ms: float, decode_ms: float, window: int = 16) -> float:
    """Incremental fold time allowed for a per-token target if every other cost is unchanged."""
    return (target_ms - decode_ms) * window
