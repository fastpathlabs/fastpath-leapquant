"""Raw evidence records of one run and their normalised view.

On disk (one directory per run):
  requests.jsonl        one line per request attempt (schema lqh.request.v1): ids, phase, token counts, the
                        scheduled / send / first-token / last-token / terminal / deadline times, status.
  stream_events.jsonl.gz one line per request attempt (kind "chunk_series"): the arrival time of every streamed
                        content chunk as integer microseconds after the send, and the exact token count of each
                        chunk when the server reported it (continuous usage statistics), else null.
All times are seconds on the run's client monotonic clock (origin: run_manifest.clock), except records whose
`clock_domain` is "request_relative" (converted from a client that keeps only per-request offsets): their t_send
is 0 and they cannot be placed on the run timeline (steady-window metrics are then null with a reason).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .util import iter_jsonl

STATUSES = ("success", "failed", "timeout", "cancelled", "incomplete")
PHASES = ("warmup", "measured", "sustained", "probe")
REQUESTS_FILE = "requests.jsonl"
STREAM_FILE = "stream_events.jsonl.gz"


@dataclass
class RequestView:
    """One request attempt with its chunk arrivals as absolute times on the run clock."""

    request_id: str
    attempt: int
    phase: str
    status: str
    input_tokens: int | None
    output_tokens: int | None
    output_tokens_source: str
    requested_output_tokens: int | None
    t_scheduled: float | None
    t_send: float
    t_first: float | None
    t_last: float | None
    t_terminal: float | None
    t_deadline: float | None
    clock_domain: str = "run"
    chunk_times: list[float] = field(default_factory=list)
    chunk_tokens: list[int] | None = None
    error: str | None = None
    retry_of: str | None = None
    session_id: str | None = None
    worker: int | None = None

    @property
    def key(self) -> tuple[str, int]:
        return (self.request_id, self.attempt)

    @property
    def success(self) -> bool:
        return self.status == "success"

    @property
    def tokens_known_per_chunk(self) -> bool:
        return self.chunk_tokens is not None and len(self.chunk_tokens) == len(self.chunk_times)

    @property
    def token_exact_gaps(self) -> bool:
        """True when every chunk carried exactly one token: chunk gaps are then token gaps."""
        return self.tokens_known_per_chunk and all(n == 1 for n in self.chunk_tokens)  # type: ignore[union-attr]

    def tokens_between(self, start: float, end: float) -> tuple[int | None, bool]:
        """Output tokens whose chunk arrived in [start, end). (count, exact). Without per-chunk counts, chunks are
        counted as one token each and exact is False."""
        if self.clock_domain != "run":
            return None, False
        if self.tokens_known_per_chunk:
            total = sum(n for t, n in zip(self.chunk_times, self.chunk_tokens) if start <= t < end)  # type: ignore[arg-type]
            return total, True
        return sum(1 for t in self.chunk_times if start <= t < end), False


def series_to_times(t_send: float, dt_us: Iterable[int]) -> list[float]:
    return [t_send + d / 1e6 for d in dt_us]


def load_requests(run_dir: str | Path) -> list[dict]:
    path = Path(run_dir) / REQUESTS_FILE
    return [record for _, record in iter_jsonl(path)]


def load_series(run_dir: str | Path) -> dict[tuple[str, int], dict]:
    path = Path(run_dir) / STREAM_FILE
    out: dict[tuple[str, int], dict] = {}
    if not path.exists():
        return out
    for _, record in iter_jsonl(path):
        out[(record["request_id"], int(record.get("attempt", 1)))] = record
    return out


def build_views(requests: list[dict], series: dict[tuple[str, int], dict]) -> list[RequestView]:
    views = []
    for record in requests:
        key = (record["request_id"], int(record.get("attempt", 1)))
        chunk = series.get(key)
        t_send = float(record["t_send"])
        times: list[float] = []
        tokens: list[int] | None = None
        if chunk is not None:
            times = series_to_times(t_send, chunk.get("dt_us") or [])
            if chunk.get("tokens") is not None:
                tokens = [int(n) for n in chunk["tokens"]]
        views.append(RequestView(
            request_id=record["request_id"],
            attempt=int(record.get("attempt", 1)),
            phase=record.get("phase", "measured"),
            status=record["status"],
            input_tokens=record.get("input_tokens"),
            output_tokens=record.get("output_tokens"),
            output_tokens_source=record.get("output_tokens_source", "unknown"),
            requested_output_tokens=record.get("requested_output_tokens"),
            t_scheduled=record.get("t_scheduled"),
            t_send=t_send,
            t_first=record.get("t_first_token"),
            t_last=record.get("t_last_token"),
            t_terminal=record.get("t_terminal"),
            t_deadline=record.get("t_deadline"),
            clock_domain=record.get("clock_domain", "run"),
            chunk_times=times,
            chunk_tokens=tokens,
            error=record.get("error"),
            retry_of=record.get("retry_of"),
            session_id=record.get("session_id"),
            worker=record.get("worker"),
        ))
    return views


def load_views(run_dir: str | Path) -> list[RequestView]:
    return build_views(load_requests(run_dir), load_series(run_dir))
