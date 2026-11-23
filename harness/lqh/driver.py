"""Load generator: one streaming HTTP client, three workload contracts.

  finite     InferenceX-equivalent finite cohort: 2 x concurrency warmup requests (the first prompt, like the pinned
             client), then num_prompts (default 10 x concurrency) requests at an infinite offered rate capped by
             max concurrency. Duration: main scheduling start to the last completion (drain included).
  sustained  AA-SLT method: `concurrency` workers, each replacing a finished query immediately; ramp, a steady
             phase of fixed length (180 s by default), then drain (no new queries; in-flight queries finish or are
             marked incomplete at the drain deadline). Unique prompts (no prefix-cache reuse between queries).
  open_loop  Poisson arrivals at a fixed offered rate for a fixed time (seeded schedule, the same for every arm),
             then drain. Attempted and achieved rates are reported separately.

Requests go to /v1/completions with exact-length token-ID prompts, ignore_eos, temperature 0 and
stream_options {include_usage, continuous_usage_stats}, so every streamed chunk carries the server's cumulative
completion-token count: per-chunk token counts are exact, not inferred. The client is standard library only
(asyncio streams, HTTP/1.1, chunked transfer, server-sent events), so its timestamps are taken where bytes arrive.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import re
import sys
import time
import traceback
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .util import utc_now, write_json

_EVENT_END = re.compile(rb"\r?\n\r?\n")


class Clock:
    """Run clock: monotonic seconds since the origin, with the origin's UTC and Unix time recorded."""

    def __init__(self) -> None:
        self.origin_ns = time.perf_counter_ns()
        self.origin_unix = time.time()
        self.origin_utc = utc_now()

    def now(self) -> float:
        return (time.perf_counter_ns() - self.origin_ns) / 1e9

    def to_unix(self, t: float) -> float:
        return self.origin_unix + t

    def describe(self) -> dict:
        return {"origin_utc": self.origin_utc, "origin_unix": self.origin_unix, "domain": "client_monotonic",
                "sync": "time.perf_counter_ns offsets from one origin; Unix time sampled once at the origin"}


@dataclass
class Attempt:
    request_id: str
    attempt: int
    phase: str
    input_tokens: int
    requested_output_tokens: int
    t_scheduled: float | None = None
    t_send: float = 0.0
    t_first: float | None = None
    t_last: float | None = None
    t_terminal: float | None = None
    t_deadline: float | None = None
    status: str = "incomplete"
    error: str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    output_tokens: int | None = None
    output_tokens_source: str = "unknown"
    input_tokens_reported: int | None = None
    done_seen: bool = False
    empty_chunks: int = 0
    worker: int | None = None
    dt_us: array = field(default_factory=lambda: array("q"))
    tokens: array = field(default_factory=lambda: array("i"))
    tokens_known: bool = True
    _cum: int = 0

    def record(self) -> dict:
        return {
            "request_id": self.request_id, "attempt": self.attempt, "retry_of": None, "session_id": None,
            "parent_id": None, "worker": self.worker, "phase": self.phase, "clock_domain": "run",
            "input_tokens": self.input_tokens, "input_tokens_reported": self.input_tokens_reported,
            "requested_output_tokens": self.requested_output_tokens, "output_tokens": self.output_tokens,
            "output_tokens_source": self.output_tokens_source, "t_scheduled": self.t_scheduled,
            "t_send": self.t_send, "t_first_token": self.t_first, "t_last_token": self.t_last,
            "t_terminal": self.t_terminal, "t_deadline": self.t_deadline, "status": self.status,
            "error": self.error, "finish_reason": self.finish_reason, "http_status": self.http_status,
            "content_chunks": len(self.dt_us), "empty_chunks": self.empty_chunks,
            "exact_chunk_tokens": self.tokens_known, "done_seen": self.done_seen,
        }

    def series(self) -> dict:
        return {"request_id": self.request_id, "attempt": self.attempt, "kind": "chunk_series", "clock_domain": "run",
                "dt_us": list(self.dt_us), "tokens": list(self.tokens) if self.tokens_known else None,
                "chunk_semantics": "exact_tokens_per_chunk" if self.tokens_known else "one_chunk_unknown_tokens",
                "terminal_dt_us": None if self.t_terminal is None else int(round((self.t_terminal - self.t_send) * 1e6))}


class Target:
    def __init__(self, base_url: str, model: str, endpoint: str = "/v1/completions") -> None:
        parsed = urlparse(base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.model = model
        self.endpoint = endpoint
        self.base_url = base_url


async def _read_head(reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
    line = await reader.readline()
    if not line:
        raise ConnectionError("connection closed before the status line")
    parts = line.decode("latin-1").split(" ", 2)
    status = int(parts[1])
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if line in (b"\r\n", b"\n", b""):
            break
        name, _, value = line.decode("latin-1").partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers


async def _body(reader: asyncio.StreamReader, headers: dict[str, str]):
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await reader.readline()
            if not size_line:
                raise ConnectionError("stream ended inside a chunked body")
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while True:
                    trailer = await reader.readline()
                    if trailer in (b"\r\n", b"\n", b""):
                        return
            data = await reader.readexactly(size)
            await reader.readexactly(2)
            yield data
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining > 0:
            data = await reader.read(min(1 << 16, remaining))
            if not data:
                raise ConnectionError("stream ended before content-length")
            remaining -= len(data)
            yield data
    else:
        while True:
            data = await reader.read(1 << 16)
            if not data:
                return
            yield data


def _handle_event(attempt: Attempt, payload: bytes, t: float) -> None:
    if payload == b"[DONE]":
        attempt.done_seen = True
        return
    obj = json.loads(payload)
    if obj.get("error") or obj.get("object") == "error":
        error = obj.get("error") or obj.get("message")
        attempt.error = json.dumps(error)[:2000]
        return
    choices = obj.get("choices")
    usage = obj.get("usage")
    if choices:
        choice = choices[0]
        text = choice.get("text") or ""
        if choice.get("finish_reason"):
            attempt.finish_reason = choice["finish_reason"]
        delta: int | None = None
        if usage and usage.get("completion_tokens") is not None:
            cumulative = int(usage["completion_tokens"])
            delta = cumulative - attempt._cum
            attempt._cum = cumulative
            if delta < 0:
                delta = None
        if delta == 0 and not text:
            attempt.empty_chunks += 1
            return
        if delta is None:
            attempt.tokens_known = False
        if attempt.t_first is None:
            attempt.t_first = t
        attempt.t_last = t
        attempt.dt_us.append(int(round((t - attempt.t_send) * 1e6)))
        attempt.tokens.append(delta if delta is not None else 0)
    elif usage:
        if usage.get("completion_tokens") is not None:
            attempt.output_tokens = int(usage["completion_tokens"])
            attempt.output_tokens_source = "usage_final"
        if usage.get("prompt_tokens") is not None:
            attempt.input_tokens_reported = int(usage["prompt_tokens"])


async def stream_completion(target: Target, attempt: Attempt, payload: dict, clock: Clock) -> None:
    """Sends one streaming request and fills `attempt` in place (so a timeout keeps the partial output)."""
    body = json.dumps(payload, separators=(",", ":")).encode()
    head = (f"POST {target.endpoint} HTTP/1.1\r\nHost: {target.host}:{target.port}\r\n"
            f"Content-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n").encode()
    writer = None
    # The send time is taken before the connection opens, as the pinned client does (its timer starts before
    # aiohttp's session.post), so connection setup is part of TTFT in both.
    attempt.t_send = clock.now()
    try:
        reader, writer = await asyncio.open_connection(target.host, target.port, limit=1 << 22)
        writer.write(head + body)
        await writer.drain()
        status, headers = await _read_head(reader)
        attempt.http_status = status
        if status != 200:
            text = b""
            async for piece in _body(reader, headers):
                text += piece
                if len(text) > 4096:
                    break
            attempt.error = f"HTTP {status}: {text[:500].decode('utf-8', 'replace')}"
            attempt.status = "failed"
            attempt.t_terminal = clock.now()
            return
        buffer = b""
        async for piece in _body(reader, headers):
            t = clock.now()
            buffer += piece
            while True:
                match = _EVENT_END.search(buffer)
                if not match:
                    break
                event, buffer = buffer[:match.start()], buffer[match.end():]
                data = [line[5:].strip() for line in event.split(b"\n")
                        if line.rstrip(b"\r").startswith(b"data:")]
                if data:
                    _handle_event(attempt, b"\n".join(d.rstrip(b"\r") for d in data), t)
                if attempt.error:
                    break
            if attempt.error:
                break
        attempt.t_terminal = clock.now()
        if attempt.output_tokens is None and attempt._cum:
            attempt.output_tokens, attempt.output_tokens_source = attempt._cum, "usage_continuous"
        if attempt.output_tokens is None and len(attempt.dt_us):
            attempt.output_tokens, attempt.output_tokens_source = len(attempt.dt_us), "chunk_count"
        if attempt.error:
            attempt.status = "failed"
        elif attempt.t_first is None:
            attempt.status, attempt.error = "failed", "no content chunk received"
        elif not attempt.done_seen:
            attempt.status, attempt.error = "failed", "stream ended without [DONE]"
        else:
            attempt.status = "success"
    except asyncio.CancelledError:
        attempt.t_terminal = None
        raise
    except Exception as exc:  # connection reset, malformed stream: a failed attempt, with what had arrived
        attempt.status = "failed"
        attempt.error = f"{type(exc).__name__}: {exc}"[:2000]
        attempt.t_terminal = clock.now()
    finally:
        if writer is not None:
            writer.close()


# ---- prompts --------------------------------------------------------------------------------------------------
class PromptMaker:
    """Exact-length token-ID prompts. The first four tokens encode the request index, so no two prompts share a
    prefix (no accidental prefix-cache hits); the rest comes from a seeded pool. Token ids are drawn from
    [low, high) to stay clear of special tokens."""

    def __init__(self, seed: int, low: int = 1000, high: int = 150000, pool: int = 1 << 20) -> None:
        self.seed, self.low, self.high = seed, low, high
        rng = random.Random(seed)
        span = high - low
        self.pool = array("i", (low + rng.randrange(span) for _ in range(pool)))
        self.rng = random.Random(seed + 1)

    def make(self, index: int, length: int) -> list[int]:
        span = self.high - self.low
        head = []
        value = index
        for _ in range(4):
            head.append(self.low + value % span)
            value //= span
        start = self.rng.randrange(len(self.pool) - length - 1) if length < len(self.pool) - 1 else 0
        body = list(self.pool[start:start + max(0, length - 4)])
        return (head + body)[:length]


def lengths(rng: random.Random, target: int, ratio: float, count: int) -> list[int]:
    """Uniform lengths in [int(target x ratio), target], as the pinned client draws them; ratio 1.0 = exact."""
    low = int(target * ratio)
    return [target if low >= target else rng.randint(low, target) for _ in range(count)]


# ---- writers --------------------------------------------------------------------------------------------------
class EvidenceWriter:
    """Writes requests.jsonl and stream_events.jsonl.gz as attempts finish (memory stays bounded)."""

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.requests = open(out_dir / "requests.jsonl", "w", encoding="utf-8")
        self.stream = gzip.open(out_dir / "stream_events.jsonl.gz", "wt", encoding="utf-8", compresslevel=3)
        self.counts = {"written": 0}

    def write(self, attempt: Attempt) -> None:
        self.requests.write(json.dumps(attempt.record(), allow_nan=False, separators=(",", ":")) + "\n")
        self.stream.write(json.dumps(attempt.series(), allow_nan=False, separators=(",", ":")) + "\n")
        self.counts["written"] += 1

    def close(self) -> None:
        self.requests.close()
        self.stream.close()


def _payload(target: Target, prompt: list[int], max_tokens: int, ignore_eos: bool) -> dict:
    return {"model": target.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": ignore_eos, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True}}


async def _one(target: Target, clock: Clock, writer: EvidenceWriter, attempt: Attempt, payload: dict,
               timeout: float | None) -> Attempt:
    if timeout is not None:
        attempt.t_deadline = clock.now() + timeout
    task = asyncio.ensure_future(stream_completion(target, attempt, payload, clock))
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        attempt.status, attempt.error = "timeout", f"deadline of {timeout} s"
        attempt.t_terminal = None
    except asyncio.CancelledError:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        attempt.status = "incomplete"
        attempt.t_terminal = None
        writer.write(attempt)
        raise
    writer.write(attempt)
    return attempt


# ---- workloads ------------------------------------------------------------------------------------------------
async def run_finite(target: Target, cfg: dict, clock: Clock, writer: EvidenceWriter) -> dict:
    conc = int(cfg["concurrency"])
    num_prompts = int(cfg.get("num_prompts") or 10 * conc)
    warmups = int(cfg.get("num_warmups", 2 * conc))
    rng = random.Random(cfg.get("seed", 0))
    maker = PromptMaker(cfg.get("seed", 0), cfg.get("token_low", 1000), cfg.get("token_high", 150000))
    in_lens = lengths(rng, cfg["isl"], cfg.get("random_range_ratio", 1.0), num_prompts)
    out_lens = lengths(rng, cfg["osl"], cfg.get("random_range_ratio", 1.0), num_prompts)
    prompts = [maker.make(i, n) for i, n in enumerate(in_lens)]
    timeout = cfg.get("request_timeout_s")
    phases: dict[str, dict] = {}
    if warmups:
        sem = asyncio.Semaphore(conc)
        start = clock.now()

        async def warm(i: int) -> None:
            async with sem:
                attempt = Attempt(f"w{i:06d}", 1, "warmup", in_lens[0], out_lens[0], t_scheduled=start)
                await _one(target, clock, writer, attempt, _payload(target, prompts[0], out_lens[0], True), timeout)

        await asyncio.gather(*(warm(i) for i in range(warmups)))
        phases["warmup"] = {"start": start, "end": clock.now(), "requests": warmups}
    sem = asyncio.Semaphore(conc)
    main_start = clock.now()

    async def measured(i: int) -> Attempt:
        attempt = Attempt(f"m{i:06d}", 1, "measured", in_lens[i], out_lens[i], t_scheduled=clock.now())
        async with sem:
            return await _one(target, clock, writer, attempt,
                              _payload(target, prompts[i], out_lens[i], cfg.get("ignore_eos", True)), timeout)

    results = await asyncio.gather(*(measured(i) for i in range(num_prompts)))
    main_end = clock.now()
    phases["main"] = {"start": main_start, "end": main_end, "requests": num_prompts}
    return {"phases": phases, "requests": {"measured": num_prompts, "warmup": warmups,
                                           "succeeded": sum(1 for a in results if a.status == "success")}}


async def run_sustained(target: Target, cfg: dict, clock: Clock, writer: EvidenceWriter) -> dict:
    conc = int(cfg["concurrency"])
    ramp_min = float(cfg.get("ramp_min_s", 30.0))
    steady_s = float(cfg.get("steady_s", 180.0))
    drain_timeout = float(cfg.get("drain_timeout_s", 900.0))
    stagger = bool(cfg.get("stagger_start", True))
    rng = random.Random(cfg.get("seed", 0))
    maker = PromptMaker(cfg.get("seed", 0), cfg.get("token_low", 1000), cfg.get("token_high", 150000))
    timeout = cfg.get("request_timeout_s")
    t0 = clock.now()
    counter = {"next": 0}
    first_done = [False] * conc
    ramp_end = {"t": None}
    steady_end = {"t": None}
    ramp_ready = asyncio.Event()

    def next_index() -> int:
        counter["next"] += 1
        return counter["next"] - 1

    async def ramp_watch() -> None:
        await asyncio.sleep(ramp_min)
        while not all(first_done):
            await asyncio.sleep(0.05)
        ramp_end["t"] = clock.now()
        steady_end["t"] = ramp_end["t"] + steady_s
        ramp_ready.set()

    async def worker(w: int) -> None:
        if stagger and conc > 1:
            await asyncio.sleep(ramp_min * w / conc)
        while True:
            if steady_end["t"] is not None and clock.now() >= steady_end["t"]:
                return
            i = next_index()
            isl = lengths(rng, cfg["isl"], cfg.get("random_range_ratio", 1.0), 1)[0]
            osl = lengths(rng, cfg["osl"], cfg.get("random_range_ratio", 1.0), 1)[0]
            attempt = Attempt(f"s{i:07d}", 1, "sustained", isl, osl, t_scheduled=clock.now(), worker=w)
            await _one(target, clock, writer, attempt, _payload(target, maker.make(i, isl), osl,
                                                                cfg.get("ignore_eos", True)), timeout)
            first_done[w] = True

    watcher = asyncio.ensure_future(ramp_watch())
    tasks = [asyncio.ensure_future(worker(w)) for w in range(conc)]
    await ramp_ready.wait()
    await asyncio.sleep(max(0.0, steady_end["t"] - clock.now()))
    drain_start = clock.now()
    done, pending = await asyncio.wait(tasks, timeout=drain_timeout)
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    watcher.cancel()
    drain_end = clock.now()
    return {"phases": {"ramp": {"start": t0, "end": ramp_end["t"], "rule": f"min {ramp_min} s and every worker "
                                "completed one query", "stagger_start": stagger},
                       "steady": {"start": ramp_end["t"], "end": steady_end["t"]},
                       "drain": {"start": drain_start, "end": drain_end, "timeout_s": drain_timeout,
                                 "unfinished_at_deadline": len(pending)}},
            "requests": {"issued": counter["next"]}}


async def run_open_loop(target: Target, cfg: dict, clock: Clock, writer: EvidenceWriter) -> dict:
    rate = float(cfg["rate_rps"])
    duration = float(cfg.get("arrival_s", 120.0))
    drain_timeout = float(cfg.get("drain_timeout_s", 900.0))
    rng = random.Random(cfg.get("arrival_seed", 1))
    maker = PromptMaker(cfg.get("seed", 0), cfg.get("token_low", 1000), cfg.get("token_high", 150000))
    timeout = cfg.get("request_timeout_s")
    schedule = []
    t = 0.0
    while True:
        t += rng.expovariate(rate)
        if t > duration:
            break
        schedule.append(t)
    t0 = clock.now()

    async def arrival(i: int, offset: float) -> Attempt:
        await asyncio.sleep(max(0.0, t0 + offset - clock.now()))
        attempt = Attempt(f"o{i:06d}", 1, "measured", cfg["isl"], cfg["osl"], t_scheduled=t0 + offset)
        return await _one(target, clock, writer, attempt,
                          _payload(target, maker.make(i, cfg["isl"]), cfg["osl"], cfg.get("ignore_eos", True)),
                          timeout)

    tasks = [asyncio.ensure_future(arrival(i, o)) for i, o in enumerate(schedule)]
    await asyncio.sleep(max(0.0, t0 + duration - clock.now()))
    arrivals_end = clock.now()
    done, pending = await asyncio.wait(tasks, timeout=drain_timeout) if tasks else (set(), set())
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    end = clock.now()
    return {"phases": {"arrivals": {"start": t0, "end": arrivals_end, "offered_rate_rps": rate,
                                    "scheduled": len(schedule)},
                       "main": {"start": t0, "end": end},
                       "drain": {"start": arrivals_end, "end": end, "unfinished_at_deadline": len(pending)}},
            "requests": {"scheduled": len(schedule)}}


LANES = {"finite": run_finite, "sustained": run_sustained, "open_loop": run_open_loop}


async def _run(target: Target, lane: str, cfg: dict, out_dir: Path, clock: Clock) -> dict:
    writer = EvidenceWriter(out_dir)
    summary: dict[str, Any] = {"schema": "lqh.driver_summary.v1", "lane": lane, "config": cfg,
                               "target": {"base_url": target.base_url, "model": target.model,
                                          "endpoint": target.endpoint},
                               "clock": clock.describe(), "started_utc": utc_now(), "exit": "ok"}
    try:
        summary.update(await LANES[lane](target, cfg, clock, writer))
    except Exception as exc:
        summary["exit"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        summary["traceback"] = traceback.format_exc()[-4000:]
    finally:
        writer.close()
    summary["ended_utc"] = utc_now()
    summary["records_written"] = writer.counts["written"]
    write_json(out_dir / "driver_summary.json", summary)
    return summary


def run(base_url: str, model: str, lane: str, cfg: dict, out_dir: str | Path, endpoint: str = "/v1/completions",
        clock: Clock | None = None) -> dict:
    clock = clock or Clock()
    return asyncio.run(_run(Target(base_url, model, endpoint), lane, cfg, Path(out_dir), clock))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.driver", description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lane", choices=sorted(LANES), required=True)
    parser.add_argument("--config", required=True, help="JSON object: isl, osl, concurrency, ...")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = run(args.base_url, args.model, args.lane, json.loads(args.config), args.out)
    print(json.dumps({k: summary[k] for k in ("exit", "records_written", "phases") if k in summary}, indent=1))
    return 0 if summary["exit"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
