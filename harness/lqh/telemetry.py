"""Device and server telemetry: GPU memory / power / clock samples, device identity, SGLang server-log events,
memory allocations, and memory snapshots at the points the specification names.

memory_and_power.jsonl records (all with t_unix; t_run when the run clock is known):
  {"kind": "gpu_sample", gpu_index, gpu_uuid, memory_used_mib, memory_total_mib, power_w, sm_clock_mhz,
   mem_clock_mhz, temperature_c, utilization_pct}
  {"kind": "allocation", category, gib, source}       parsed from the server log (weights, KV cache, recurrent
                                                       state, LeapQuant state, CUDA graphs, ...)
  {"kind": "snapshot", label, ...}                     before_model_load, after_pools, steady, peak, after_drain
server_events.jsonl records: {"kind": "decode"|"prefill"|"retract"|"capacity", t_unix, running, queue, ...}
"""
from __future__ import annotations

import calendar
import os
import re
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QUERY = ("timestamp,index,uuid,name,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu,"
         "utilization.gpu,pci.bus_id,driver_version")


def _float(text: str) -> float | None:
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return None


def query_devices() -> list[dict]:
    """GPU identity from nvidia-smi; [] when there is no NVIDIA driver (then every device field is unknown)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version,pci.bus_id",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    devices = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            devices.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2],
                            "memory_total_mib": _float(parts[3]), "driver": parts[4], "pci_bus_id": parts[5]})
    return devices


def gpu_memory_used_mib() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        values = [_float(v) for v in out.split()]
        return sum(v for v in values if v is not None)
    except (OSError, subprocess.SubprocessError):
        return None


class GpuSampler:
    """nvidia-smi sampling in a subprocess (default every 500 ms) into a CSV file; parse() turns it into records."""

    def __init__(self, path: Path, interval_ms: int = 500) -> None:
        self.path = Path(path)
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self.available:
            return
        self.handle = open(self.path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits",
                                      f"-lms={self.interval_ms}"], stdout=self.handle, stderr=subprocess.DEVNULL)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.handle.close()

    def records(self, origin_unix: float | None) -> list[dict]:
        if not self.available or not self.path.exists():
            return [{"kind": "sampler_unavailable", "reason": "no GPU sampling for this run (nvidia-smi absent or mock)"}]
        return parse_samples(self.path.read_text(encoding="utf-8"), origin_unix)


def parse_samples(text: str, origin_unix: float | None) -> list[dict]:
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue
        try:
            stamp = datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f")
        except ValueError:
            continue
        # nvidia-smi prints local time; convert with the host's offset at that moment.
        t_unix = time.mktime(stamp.timetuple()) + stamp.microsecond / 1e6
        record = {"kind": "gpu_sample", "t_unix": t_unix, "gpu_index": int(parts[1]), "gpu_uuid": parts[2],
                  "gpu_name": parts[3], "memory_used_mib": _float(parts[4]), "memory_total_mib": _float(parts[5]),
                  "power_w": _float(parts[6]), "sm_clock_mhz": _float(parts[7]), "mem_clock_mhz": _float(parts[8]),
                  "temperature_c": _float(parts[9]), "utilization_pct": _float(parts[10])}
        if origin_unix is not None:
            record["t_run"] = t_unix - origin_unix
        out.append(record)
    return out


# ---- SGLang server log ------------------------------------------------------------------------------------------
_STAMP = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:[.,](\d+))?[^\]]*\]\s*(.*)$")
_FIELD = re.compile(r"([#A-Za-z][A-Za-z _()/#-]*?):\s*(-?[0-9.]+|True|False)")
_ALLOC = [
    ("weights", re.compile(r"Load weight end.*?mem usage=([0-9.]+) GB")),
    ("recurrent_state_ssm", re.compile(r"Mamba Cache is allocated.*?ssm_state size: ([0-9.]+)GB")),
    ("recurrent_state_conv", re.compile(r"Mamba Cache is allocated.*?conv_state size: ([0-9.]+)GB")),
    ("kv_cache_k", re.compile(r"KV Cache is allocated.*?K size: ([0-9.]+) GB")),
    ("kv_cache_v", re.compile(r"KV Cache is allocated.*?V size: ([0-9.]+) GB")),
    ("leapquant_state", re.compile(r"LeapQuant decode state allocated:.*?([0-9.]+) GB")),
    ("cuda_graph_decode", re.compile(r"Capture cuda graph end.*?mem usage=([0-9.]+) GB")),
    ("cuda_graph_prefill", re.compile(r"Capture target prefill CUDA graph end.*?mem usage=([0-9.]+) GB")),
]
_CAPACITY = [
    ("max_mamba_cache_size", re.compile(r"max_mamba_cache_size: (\d+)")),
    ("kv_tokens", re.compile(r"KV Cache is allocated.*?#tokens: (\d+)")),
    ("max_total_num_tokens", re.compile(r"max_total_num_tokens=(\d+)")),
    ("max_running_requests", re.compile(r"max_running_requests=(\d+)")),
    ("leapquant_slots", re.compile(r"LeapQuant decode state allocated: \d+ layers x (\d+) slots")),
]


def _line_time(date: str, fraction: str | None, utc: bool = False) -> float:
    value = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    # SGLang logs the server host's local time. The harness runs on that host (local time = the host's); logs read
    # elsewhere (carried history) pass utc=True: the GPU host's clock is UTC.
    t = calendar.timegm(value.timetuple()) if utc else time.mktime(value.timetuple())
    if fraction:
        t += float("0." + fraction)
    return t


def parse_server_log(text: str, start_unix: float | None = None, end_unix: float | None = None,
                     origin_unix: float | None = None, utc: bool = False) -> tuple[list[dict], list[dict], dict]:
    """(server events in [start, end], allocation records, capacity facts). Log timestamps have one-second
    resolution; events are kept when their second overlaps the interval."""
    events: list[dict] = []
    allocations: list[dict] = []
    capacity: dict[str, Any] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = _STAMP.match(line)
        if not match:
            continue
        date, fraction, message = match.groups()
        t_unix = _line_time(date, fraction, utc)
        for category, pattern in _ALLOC:
            found = pattern.search(message)
            if found:
                allocations.append({"kind": "allocation", "category": category, "gib": float(found.group(1)),
                                    "t_unix": t_unix, "source": f"server log line {number}",
                                    "note": "mem usage after this step (cumulative)" if "mem usage" in pattern.pattern
                                    else "size of this allocation"})
        for name, pattern in _CAPACITY:
            found = pattern.search(message)
            if found and name not in capacity:
                capacity[name] = int(found.group(1))
        if "server_args=" in message:
            continue
        kind = "decode" if message.startswith("Decode batch") else "prefill" if message.startswith("Prefill batch") \
            else None
        retracted = re.search(r"#retracted_reqs: (\d+)", message)
        if retracted:
            kind = kind or "retract"
        if kind is None:
            continue
        if start_unix is not None and t_unix < start_unix - 1.0:
            continue
        if end_unix is not None and t_unix > end_unix + 1.0:
            continue
        fields = {}
        for key, value in _FIELD.findall(message):
            key = key.strip().lower().replace("#", "n_").replace(" ", "_").replace("(", "").replace(")", "") \
                .replace("/", "_per_")
            fields[key] = value == "True" if value in ("True", "False") else float(value)
        record = {"kind": kind, "t_unix": t_unix, "line": number}
        if origin_unix is not None:
            record["t_run"] = t_unix - origin_unix
        for src, dst in (("n_running-req", "running"), ("n_queue-req", "queue"), ("full_token_usage", "token_usage"),
                         ("token_usage", "token_usage"), ("mamba_usage", "mamba_usage"), ("cuda_graph", "cuda_graph"),
                         ("gen_throughput_token_per_s", "gen_tps"), ("input_throughput_token_per_s", "input_tps"),
                         ("n_new-token", "new_tokens"), ("n_cached-token", "cached_tokens"),
                         ("n_new-seq", "new_seqs"), ("n_full_token", "tokens")):
            if src in fields:
                record[dst] = fields[src]
        if retracted:
            record["retracted"] = int(retracted.group(1))
        events.append(record)
    return events, allocations, capacity


def summarise_server(events: list[dict]) -> dict[str, dict]:
    from .metrics import distribution, scalar
    decode = [e for e in events if e["kind"] == "decode"]
    out = {
        "server_running_requests": distribution([e["running"] for e in decode if "running" in e], "requests",
                                                "decode log line"),
        "server_queue_depth": distribution([e["queue"] for e in events if "queue" in e], "requests", "log line"),
        "server_token_usage": distribution([e["token_usage"] for e in events if "token_usage" in e], "fraction",
                                           "log line"),
        "server_retractions": scalar(sum(e.get("retracted", 0) for e in events), "events"),
        "server_cuda_graph_fraction": scalar(
            (sum(1 for e in decode if e.get("cuda_graph")) / len(decode)) if decode else None, "fraction",
            "no_decode_lines"),
        "server_reported_gen_tps": distribution([e["gen_tps"] for e in decode if "gen_tps" in e], "tok/s",
                                                "decode log line"),
    }
    if not events:
        for key in out:
            out[key]["null_reason"] = out[key].get("null_reason") or "no_server_log_lines_in_interval"
    return out


def summarise_gpu(samples: list[dict], start_unix: float, end_unix: float) -> dict[str, dict]:
    from .metrics import distribution, scalar
    inside = [s for s in samples if s.get("kind") == "gpu_sample" and start_unix <= s["t_unix"] <= end_unix]
    out = {
        "gpu_memory_used_mib": distribution([s["memory_used_mib"] for s in inside if s.get("memory_used_mib") is not None], "MiB", "sample"),
        "gpu_power_w": distribution([s["power_w"] for s in inside if s.get("power_w") is not None], "W", "sample"),
        "gpu_sm_clock_mhz": distribution([s["sm_clock_mhz"] for s in inside if s.get("sm_clock_mhz") is not None], "MHz", "sample"),
        "gpu_temperature_c": distribution([s["temperature_c"] for s in inside if s.get("temperature_c") is not None], "C", "sample"),
    }
    totals = [s["memory_total_mib"] for s in samples if s.get("kind") == "gpu_sample" and s.get("memory_total_mib")]
    out["gpu_memory_total_mib"] = scalar(totals[0] if totals else None, "MiB", "no_samples")
    if not inside:
        for key in ("gpu_memory_used_mib", "gpu_power_w", "gpu_sm_clock_mhz", "gpu_temperature_c"):
            out[key]["null_reason"] = "no_gpu_samples_in_interval"
    return out


def snapshots(samples: list[dict], marks: dict[str, float]) -> list[dict]:
    """Memory snapshots: the sample nearest each marked time, plus the peak. marks: label -> t_unix."""
    gpu = [s for s in samples if s.get("kind") == "gpu_sample" and s.get("memory_used_mib") is not None]
    out = []
    for label, t in marks.items():
        if not gpu or t is None:
            out.append({"kind": "snapshot", "label": label, "value": None, "null_reason": "no_samples"})
            continue
        nearest = min(gpu, key=lambda s: abs(s["t_unix"] - t))
        out.append({"kind": "snapshot", "label": label, "t_unix": nearest["t_unix"],
                    "memory_used_mib": nearest["memory_used_mib"], "offset_s": nearest["t_unix"] - t,
                    "scope": "whole device (nvidia-smi), allocation and reservation not separated"})
    if gpu:
        peak = max(gpu, key=lambda s: s["memory_used_mib"])
        out.append({"kind": "snapshot", "label": "peak", "t_unix": peak["t_unix"],
                    "memory_used_mib": peak["memory_used_mib"], "scope": "whole device (nvidia-smi)"})
    return out


def wait_gpu_free(threshold_mib: float = 2048, timeout_s: float = 300) -> bool:
    """Waits until the device holds less than threshold memory (the previous server is really gone)."""
    if not shutil.which("nvidia-smi"):
        return True
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        used = gpu_memory_used_mib()
        if used is not None and used < threshold_mib:
            return True
        time.sleep(3)
    return False


def host_name() -> str:
    try:
        return os.uname().nodename
    except AttributeError:  # pragma: no cover
        return "unknown"


def utc_from_unix(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
