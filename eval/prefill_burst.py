#!/usr/bin/env python3
"""
prefill_burst.py

Prefill-dominated bursts against a running SGLang server: N requests of L random token ids each, sent at once, M new
tokens each, so the burst's wall time is mostly prefill. Repeated R times with fresh token ids (and a cache flush
before each burst), so no request hits the radix cache. Optionally the server's torch profiler records the steps of
one burst (--profile-dir: POST /start_profile before the burst at index --profile-burst).

    python3 prefill_burst.py --port 30000 --out burst.json [--requests 256 --length 128 --new-tokens 4 --bursts 5]

Prints per burst the wall time, prefill tokens per second over the wall time and the median / max request latency.
Uses only the Python standard library.
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request


def post(port, path, body, timeout=600):
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except ValueError:
        return raw.decode(errors="replace")


def burst(port, prompts, new_tokens):
    latencies = [None] * len(prompts)

    def one(index):
        start = time.perf_counter()
        post(port, "/generate", {"input_ids": prompts[index],
                                 "sampling_params": {"temperature": 0, "max_new_tokens": new_tokens,
                                                     "ignore_eos": True}})
        latencies[index] = time.perf_counter() - start

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return time.perf_counter() - start, latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--bursts", type=int, default=5)
    parser.add_argument("--vocab", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--profile-burst", type=int, default=2)
    parser.add_argument("--profile-steps", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    results = []
    for index in range(-1, args.bursts):                  # burst -1: warm-up, not reported
        prompts = [[rng.randrange(1000, args.vocab) for _ in range(args.length)] for _ in range(args.requests)]
        try:
            post(args.port, "/flush_cache", None, timeout=60)
        except Exception as error:                        # noqa: BLE001 - a busy cache refuses; report and go on
            print("flush_cache:", error)
        time.sleep(1.0)
        if args.profile_dir and index == args.profile_burst:
            print("start_profile:", post(args.port, "/start_profile",
                                         {"output_dir": args.profile_dir, "num_steps": args.profile_steps,
                                          "activities": ["CPU", "GPU"]}))
        wall, latencies = burst(args.port, prompts, args.new_tokens)
        if index < 0:
            continue
        item = {"burst": index, "wall_s": wall, "prefill_tok_s": args.requests * args.length / wall,
                "latency_median_s": statistics.median(latencies), "latency_max_s": max(latencies)}
        results.append(item)
        print(f"burst {index}: wall {wall:.3f} s  {item['prefill_tok_s']:,.0f} prompt tok/s  "
              f"latency median {item['latency_median_s']:.3f} s, max {item['latency_max_s']:.3f} s")
    walls = [item["wall_s"] for item in results]
    summary = {"requests": args.requests, "length": args.length, "new_tokens": args.new_tokens,
               "wall_median_s": statistics.median(walls), "wall_mean_s": statistics.mean(walls),
               "bursts": results}
    print(f"median wall {summary['wall_median_s']:.3f} s over {len(walls)} bursts")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)


if __name__ == "__main__":
    main()
