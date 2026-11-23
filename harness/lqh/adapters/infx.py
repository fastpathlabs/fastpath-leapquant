"""Thin adapter for the pinned InferenceX fixed-sequence client (inferencex-e2e @ fabf7ba).

It runs the pinned client's benchmark module with exactly the argv of infx.bench.fixed_seq.client_argv (verified in
the pinned source; harness/config/profiles.json) plus --save-detailed (deviation D-IX-1), keeps the client's raw
output under native/, then converts the per-request arrays into the harness's records:
  - times are request-relative (the client keeps TTFT and chunk gaps, not send times): t_send = 0,
    t_first = ttft, chunk arrivals = ttft + cumulative ITL, t_last = the client's latency (last content chunk);
    the terminal response time is unknown (null), so terminal-E2E metrics are null for this lane;
  - per-chunk token counts are unknown; when a response has exactly as many content chunks as output tokens, the
    chunks are labelled "inferred_single_token_chunks" (each chunk one token), otherwise counts stay null;
  - success = the client recorded no error and an output length > 0 (its own rule for completed requests).
The client's own aggregates are kept as `native` next to the harness's recomputation (parity check in analyze).

Host setup: LQH_INFX_DIR = a checkout of inferencex-e2e at the pinned commit (runbook/harness_setup.sh); the
interpreter is the SGLang virtualenv's (deviation D-IX-2).
"""
from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..util import harness_root, read_json, sha256_file, utc_now, write_json

PINNED_FILES = ("infx/bench/fixed_seq.py", "infx/bench_serving/benchmark_serving.py",
                "infx/bench_serving/backend_request_func.py", "infx/bench_serving/benchmark_outcome.py")


def pinned_hashes() -> dict[str, str]:
    profiles = read_json(harness_root() / "config" / "profiles.json")["profiles"]
    infx = next(p for p in profiles if p["profile_id"] == "infx_fixed_seq")
    return {k.replace("inferencex-e2e/", ""): v for k, v in infx["source"]["files_sha256"].items()}


def verify_checkout(infx_dir: Path) -> list[str]:
    """Problems with the checkout (missing or changed pinned files); empty when it matches the pins."""
    problems = []
    expected = pinned_hashes()
    for name in PINNED_FILES:
        path = infx_dir / name
        if not path.exists():
            problems.append(f"{name} missing")
        elif sha256_file(path) != expected.get(name):
            problems.append(f"{name} differs from the pinned revision")
    return problems


def client_argv(python: str, base_url: str, model: str, tokenizer: str, isl: int, osl: int, ratio: float, conc: int,
                num_prompts: int, result: Path, seed: int = 0) -> list[str]:
    """client_argv(Point) of infx/bench/fixed_seq.py @ fabf7ba, item for item, plus --save-detailed and --seed."""
    return [
        python, "-m", "infx.bench_serving.benchmark_serving",
        "--model", model,
        "--backend", "vllm",
        "--base-url", base_url,
        "--dataset-name", "random",
        "--random-input-len", str(isl),
        "--random-output-len", str(osl),
        "--random-range-ratio", str(ratio),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(conc),
        "--request-rate", "inf",
        "--ignore-eos",
        "--save-result",
        "--num-warmups", str(2 * conc),
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--result-dir", str(result.parent),
        "--result-filename", result.name,
        "--tokenizer", tokenizer,
        "--save-detailed",            # D-IX-1: keep the per-request arrays
        "--seed", str(seed),          # the client's default (0), stated explicitly
    ]


NATIVE_KEYS = ("duration", "completed", "total_input_tokens", "total_output_tokens", "request_throughput",
               "request_goodput", "output_throughput", "total_token_throughput", "max_concurrency", "num_prompts",
               "benchmark_start_time_unix", "benchmark_end_time_unix", "benchmark_outcome", "request_rate")


def convert(result: dict, out_dir: Path, osl: int, isl: int) -> dict:
    """Pinned-client result (with --save-detailed) -> requests.jsonl + stream_events.jsonl.gz. Returns counts."""
    inputs, outputs = result.get("input_lens") or [], result.get("output_lens") or []
    ttfts, itls, errors = result.get("ttfts") or [], result.get("itls") or [], result.get("errors") or []
    n = len(outputs)
    if not (len(inputs) == len(ttfts) == len(itls) == len(errors) == n):
        raise ValueError(f"per-request arrays differ in length: inputs {len(inputs)}, outputs {n}, "
                         f"ttfts {len(ttfts)}, itls {len(itls)}, errors {len(errors)}")
    single = 0
    with open(out_dir / "requests.jsonl", "w", encoding="utf-8") as req, \
            gzip.open(out_dir / "stream_events.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as stream:
        for i in range(n):
            error = errors[i] or None
            out_len = int(outputs[i] or 0)
            success = error is None and out_len > 0
            gaps = itls[i] or []
            dt = []
            if success:
                t = float(ttfts[i])
                dt.append(int(round(t * 1e6)))
                for gap in gaps:
                    t += float(gap)
                    dt.append(int(round(t * 1e6)))
            chunks = len(dt)
            tokens = None
            semantics = "one_chunk_unknown_tokens"
            if success and chunks == out_len:
                tokens, semantics = [1] * chunks, "inferred_single_token_chunks"
                single += 1
            record = {
                "request_id": f"m{i:06d}", "attempt": 1, "retry_of": None, "session_id": None, "parent_id": None,
                "phase": "measured", "clock_domain": "request_relative", "input_tokens": int(inputs[i]),
                "input_tokens_reported": None, "requested_output_tokens": osl,
                "output_tokens": out_len if success else None,
                "output_tokens_source": "upstream_usage", "t_scheduled": None, "t_send": 0.0,
                "t_first_token": dt[0] / 1e6 if dt else None, "t_last_token": dt[-1] / 1e6 if dt else None,
                "t_terminal": None, "t_deadline": None, "status": "success" if success else "failed",
                "error": error[:2000] if error else None, "finish_reason": None, "http_status": None,
                "content_chunks": chunks, "empty_chunks": None, "exact_chunk_tokens": tokens is not None,
                "done_seen": None,
            }
            req.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            stream.write(json.dumps({"request_id": record["request_id"], "attempt": 1, "kind": "chunk_series",
                                     "clock_domain": "request_relative", "dt_us": dt, "tokens": tokens,
                                     "chunk_semantics": semantics, "terminal_dt_us": None},
                                    separators=(",", ":")) + "\n")
    return {"requests": n, "inferred_single_token_chunks": single}


def run_point(base_url: str, manifest: dict, point: dict, out_dir: Path, clock) -> dict:
    """Runs one fixed-sequence point with the pinned client; writes native/ and the converted records."""
    out_dir = Path(out_dir).resolve()
    native_dir = out_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"schema": "lqh.driver_summary.v1", "lane": "finite", "client": "pinned InferenceX",
                     "started_utc": utc_now(), "exit": "ok", "phases": {}}
    infx_dir = Path(os.path.expandvars(os.environ.get("LQH_INFX_DIR", "$HOME/lqharness/upstream/inferencex-e2e")))
    problems = verify_checkout(infx_dir)
    if problems:
        summary.update(exit="error", error="pinned client unavailable: " + "; ".join(problems))
        write_json(out_dir / "driver_summary.json", summary)
        return summary
    python = os.path.expandvars(os.environ.get("LQH_INFX_PYTHON", sys.executable))
    model = (manifest.get("server") or {}).get("served_model") or point.get("model", "default")
    tokenizer = point.get("tokenizer") or os.environ.get("LQH_TOKENIZER", "Qwen/Qwen3.5-9B")
    result = native_dir / "result.json"
    argv = client_argv(python, base_url, model, tokenizer, point["isl"], point["osl"],
                       point.get("random_range_ratio", 1.0), point["concurrency"], point["num_prompts"], result,
                       point.get("seed", 0))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(infx_dir) + os.pathsep + env.get("PYTHONPATH", "")
    env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
    summary["argv"] = argv
    with open(native_dir / "client.log", "w", encoding="utf-8") as log:
        started = time.time()
        code = subprocess.run(argv, cwd=native_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                              timeout=point.get("client_timeout_s", 14400)).returncode
    summary["client_exit_code"] = code
    summary["client_wall_s"] = time.time() - started
    if not result.exists():
        summary.update(exit="error", error=f"the pinned client wrote no result (exit {code}); see native/client.log")
        write_json(out_dir / "driver_summary.json", summary)
        return summary
    data = json.loads(result.read_text(encoding="utf-8"))
    summary["native"] = {k: data.get(k) for k in NATIVE_KEYS if k in data}
    summary["native"]["aggregates"] = {k: v for k, v in data.items()
                                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
    summary["native"]["result_sha256"] = sha256_file(result)
    summary["native"]["client_exit_code"] = code
    summary["native"]["note"] = ("a non-zero exit with a result file is the client's failure gate (more than 5% "
                                 "failed requests) or its count check; the result is kept and validated")
    summary["conversion"] = convert(data, out_dir, point["osl"], point["isl"])
    summary["ended_utc"] = utc_now()
    write_json(out_dir / "driver_summary.json", summary)
    return summary


def main(argv=None) -> int:
    """python -m lqh infx-command --verify DIR | --base-url U --model M --tokenizer T --isl I --osl O --conc C"""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m lqh infx-command")
    parser.add_argument("--verify", help="check a checkout of inferencex-e2e against the pinned hashes")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--isl", type=int, default=1024)
    parser.add_argument("--osl", type=int, default=1024)
    parser.add_argument("--conc", type=int, default=16)
    parser.add_argument("--result", default="result.json")
    args = parser.parse_args(argv)
    if args.verify:
        problems = verify_checkout(Path(args.verify))
        print("pinned checkout OK" if not problems else "\n".join(problems))
        return 1 if problems else 0
    print(" ".join(client_argv("python3", args.base_url, args.model, args.tokenizer, args.isl, args.osl, 1.0,
                               args.conc, 10 * args.conc, Path(args.result))))
    return 0
