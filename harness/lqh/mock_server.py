"""Deterministic mock of an OpenAI-compatible streaming server, for harness smoke tests ONLY.

A mock establishes harness behaviour (accounting, phases, validation, reports); its speed is never a performance
result and runs against it are tagged engine.mock = true, status smoke_only, and excluded from every chart.

Model of the server: a request waits `prefill_ms_per_1k_input` per 1,000 prompt tokens (plus `prefill_base_ms`),
then emits `tokens_per_chunk` tokens per streamed chunk; the time per token grows with the number of active
requests: decode_ms x (1 + active / knee). Failure injection: every `fail_every`-th request (by arrival order)
fails with an error event after `fail_after_tokens` tokens; every `http_error_every`-th gets HTTP 500.
It writes SGLang-style "Decode batch" log lines once per second when --log is given.

  python -m lqh.mock_server --port 31000 [--config '{"decode_ms": 5}'] [--log server.log]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime

DEFAULTS = {
    "model": "mock-model",
    "decode_ms": 4.0,
    "knee": 64.0,
    "prefill_base_ms": 5.0,
    "prefill_ms_per_1k_input": 4.0,
    "tokens_per_chunk": 1,
    "fail_every": 0,
    "fail_after_tokens": 3,
    "http_error_every": 0,
    "max_running": 1 << 30,
}


class MockServer:
    def __init__(self, config: dict, log_path: str | None) -> None:
        self.cfg = {**DEFAULTS, **config}
        self.active = 0
        self.queued = 0
        self.arrivals = 0
        self.generated = 0
        self.log_path = log_path
        self.slots = asyncio.Semaphore(int(self.cfg["max_running"]))

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            method, path, _ = line.decode("latin-1").split(" ", 2)
            headers = {}
            while True:
                h = await reader.readline()
                if h in (b"\r\n", b"\n", b""):
                    break
                k, _, v = h.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            body = await reader.readexactly(int(headers.get("content-length", "0") or 0))
            if method == "GET" and path == "/health":
                await self._plain(writer, 200, b"ok")
            elif method == "GET" and path == "/v1/models":
                await self._plain(writer, 200, json.dumps({"object": "list", "data": [
                    {"id": self.cfg["model"], "object": "model"}]}).encode(), "application/json")
            elif method == "GET" and path == "/get_server_info":
                await self._plain(writer, 200, json.dumps({"mock": True, "config": self.cfg}).encode(),
                                  "application/json")
            elif method == "POST" and path == "/v1/completions":
                await self._complete(writer, json.loads(body))
            else:
                await self._plain(writer, 404, b"not found")
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
            except Exception:
                pass

    async def _plain(self, writer, status: int, body: bytes, ctype: str = "text/plain") -> None:
        writer.write(f"HTTP/1.1 {status} X\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\n"
                     f"Connection: close\r\n\r\n".encode() + body)
        await writer.drain()

    async def _chunk(self, writer, payload: str) -> None:
        data = f"data: {payload}\n\n".encode()
        writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        await writer.drain()

    async def _complete(self, writer, request: dict) -> None:
        self.arrivals += 1
        number = self.arrivals
        cfg = self.cfg
        if cfg["http_error_every"] and number % cfg["http_error_every"] == 0:
            await self._plain(writer, 500, b'{"error": "injected HTTP 500"}', "application/json")
            return
        prompt = request.get("prompt")
        n_in = len(prompt) if isinstance(prompt, list) else max(1, len(str(prompt)) // 4)
        n_out = int(request.get("max_tokens") or 16)
        continuous = bool((request.get("stream_options") or {}).get("continuous_usage_stats"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
                     b"Connection: close\r\n\r\n")
        self.queued += 1
        async with self.slots:
            self.queued -= 1
            self.active += 1
            try:
                await asyncio.sleep((cfg["prefill_base_ms"] + cfg["prefill_ms_per_1k_input"] * n_in / 1000.0) / 1000.0)
                sent = 0
                fail_here = cfg["fail_every"] and number % cfg["fail_every"] == 0
                while sent < n_out:
                    k = min(int(cfg["tokens_per_chunk"]), n_out - sent)
                    if sent > 0:
                        per_token = cfg["decode_ms"] * (1.0 + self.active / cfg["knee"]) / 1000.0
                        await asyncio.sleep(per_token * k)
                    sent += k
                    self.generated += k
                    if fail_here and sent >= cfg["fail_after_tokens"]:
                        await self._chunk(writer, json.dumps({"error": {"message": "injected failure",
                                                                        "code": 500}}))
                        writer.write(b"0\r\n\r\n")
                        await writer.drain()
                        return
                    chunk = {"id": f"cmpl-{number}", "object": "text_completion", "model": cfg["model"],
                             "choices": [{"index": 0, "text": "x" * k,
                                          "finish_reason": "length" if sent >= n_out else None}]}
                    if continuous:
                        chunk["usage"] = {"prompt_tokens": n_in, "completion_tokens": sent,
                                          "total_tokens": n_in + sent}
                    await self._chunk(writer, json.dumps(chunk))
                await self._chunk(writer, json.dumps({"id": f"cmpl-{number}", "object": "text_completion",
                                                      "choices": [], "usage": {"prompt_tokens": n_in,
                                                                               "completion_tokens": sent,
                                                                               "total_tokens": n_in + sent}}))
                await self._chunk(writer, "[DONE]")
                writer.write(b"0\r\n\r\n")
                await writer.drain()
            finally:
                self.active -= 1

    async def log_loop(self) -> None:
        if not self.log_path:
            return
        last = 0
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(f"[{_stamp()}] server_args={{'mock': True, 'max_running_requests': {self.cfg['max_running']}}}\n")
            log.write(f"[{_stamp()}] max_total_num_tokens=100000, chunked_prefill_size=8192, "
                      f"max_running_requests={min(self.cfg['max_running'], 4096)}, context_len=4096\n")
            log.flush()
            while True:
                await asyncio.sleep(1.0)
                rate = self.generated - last
                last = self.generated
                log.write(f"[{_stamp()}] Decode batch, #running-req: {self.active}, #full token: 0, "
                          f"full token usage: 0.00, cuda graph: True, gen throughput (token/s): {rate:.2f}, "
                          f"#queue-req: {self.queued}\n")
                log.flush()


def _stamp() -> str:
    # Local time, as SGLang's log lines are (the parser reads both the same way).
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


async def serve(port: int, config: dict, log_path: str | None, ready: asyncio.Event | None = None) -> None:
    server = MockServer(config, log_path)
    listener = await asyncio.start_server(server.handle, "127.0.0.1", port, backlog=4096)
    logger = asyncio.ensure_future(server.log_loop())
    if ready is not None:
        ready.set()
    try:
        async with listener:
            await listener.serve_forever()
    finally:
        logger.cancel()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.mock_server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--config", default="{}")
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)
    try:
        asyncio.run(serve(args.port, json.loads(args.config), args.log))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
