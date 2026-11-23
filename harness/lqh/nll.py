"""Teacher-forced negative log-likelihood by position, read through the SERVING ENGINE's own API (prompt rev. 3,
section 15): the configuration measured is the configuration served, never a separate copy of the model.

  python -m lqh.nll measure --base-url http://127.0.0.1:30000 --docs docs.jsonl --lengths 1024,4096,8192 --out DIR
  python -m lqh.nll compare REFERENCE_DIR CANDIDATE_DIR [...]          paired by document, bootstrap over documents
  python -m lqh.nll make-docs --tokenizer Qwen/Qwen3.5-9B --dataset emozilla/pg19-test --count 8 --min-tokens 33000 --out docs.jsonl

The engine's /generate endpoint returns input-token log-probabilities (return_logprob, logprob_start_len 0) for an
identical token sequence; NLL at position i = -log p(token_i | tokens_<i). Positions are bucketed (1-1k, 1k-2k,
2k-4k, 4k-8k, 8k-16k, 16k-32k); a document's bucket value is its mean over the bucket's tokens (the token
denominator); the uncertainty unit is the DOCUMENT (percentile bootstrap over documents), never the token.
Teacher forcing is a diagnostic: it does not replace free-running task quality. With SGLang the input tokens are
processed by the prefill path, so a compressed state is formed at prefill-chunk boundaries (the server's
--chunked-prefill-size, recorded); a chunk size of 16 makes that every window (closest to decode-time folding).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

from .util import read_jsonl, utc_now, write_json, write_jsonl

BUCKETS = [(1, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 1 << 20)]


def generate_logprobs(base_url: str, ids: list[int], timeout: float = 3600) -> list[float | None]:
    body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                       "return_logprob": True, "logprob_start_len": 0}).encode()
    request = urllib.request.Request(f"{base_url}/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    entries = reply["meta_info"]["input_token_logprobs"]
    return [None if (e is None or e[0] is None) else float(e[0]) for e in entries]


def measure(base_url: str, docs: list[dict], lengths: list[int], out: Path, server_info: dict | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc in docs:
        for length in lengths:
            ids = doc["input_ids"][:length]
            if len(ids) < length:
                rows.append({"doc_id": doc["doc_id"], "length": length, "null_reason": "document shorter than length"})
                continue
            logprobs = generate_logprobs(base_url, ids)
            rows.append({"doc_id": doc["doc_id"], "length": length, "nll": [None if lp is None else -lp for lp in logprobs]})
    write_jsonl(out / "nll_by_position.jsonl", rows)
    summary = {"schema": "lqh.nll.v1", "measured_utc": utc_now(), "base_url": base_url, "lengths": lengths,
               "documents": [d["doc_id"] for d in docs], "server_info": server_info,
               "chunked_prefill_size": (server_info or {}).get("chunked_prefill_size"),
               "buckets": bucket_means(rows)}
    write_json(out / "nll_summary.json", summary)
    return summary


def bucket_means(rows: list[dict]) -> dict:
    """Per (length, bucket): each document's mean NLL over the bucket's positions, then the mean over documents."""
    out: dict = {}
    for row in rows:
        if "nll" not in row:
            continue
        for lo, hi in BUCKETS:
            values = [v for i, v in enumerate(row["nll"], start=1) if lo <= i < hi and v is not None]
            if not values:
                continue
            key = f"L{row['length']}:{lo}-{hi}"
            out.setdefault(key, {})[row["doc_id"]] = sum(values) / len(values)
    return {key: {"per_document": docs, "mean": sum(docs.values()) / len(docs), "documents": len(docs),
                  "tokens_denominator": "positions inside the bucket, per document; documents weighted equally"}
            for key, docs in out.items()}


def compare(reference: Path, candidates: list[Path], resamples: int = 5000, seed: int = 7) -> dict:
    ref = json.loads((reference / "nll_summary.json").read_text())["buckets"]
    result = {"reference": str(reference), "computed_utc": utc_now(), "unit_of_uncertainty": "document",
              "candidates": {}}
    rng = random.Random(seed)
    for cand in candidates:
        mine = json.loads((cand / "nll_summary.json").read_text())["buckets"]
        rows = {}
        for key, entry in mine.items():
            if key not in ref:
                continue
            docs = sorted(set(entry["per_document"]) & set(ref[key]["per_document"]))
            deltas = [entry["per_document"][d] - ref[key]["per_document"][d] for d in docs]
            if not deltas:
                continue
            boot = sorted(sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(resamples))
            rows[key] = {"delta_nll_mean": sum(deltas) / len(deltas), "documents": len(deltas),
                         "low": boot[int(0.025 * resamples)], "high": boot[int(0.975 * resamples) - 1],
                         "relative_perplexity_change": math.exp(sum(deltas) / len(deltas)) - 1}
        result["candidates"][str(cand)] = rows
    return result


def make_docs(tokenizer: str, dataset: str, split: str, field: str, count: int, min_tokens: int, out: Path) -> int:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    rows = []
    for index, row in enumerate(load_dataset(dataset, split=split, streaming=True)):
        ids = tok.encode(row[field], add_special_tokens=False)
        if len(ids) >= min_tokens:
            rows.append({"doc_id": f"{dataset}:{split}:{index}", "input_ids": ids[:min_tokens],
                         "source": dataset, "license_note": "Project Gutenberg public-domain texts (PG-19)"})
        if len(rows) >= count:
            break
    write_jsonl(out, rows)
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.nll")
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--base-url", required=True)
    m.add_argument("--docs", required=True)
    m.add_argument("--lengths", default="1024,4096,8192,16384,32768")
    m.add_argument("--max-docs", type=int, default=8)
    m.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("reference")
    c.add_argument("candidates", nargs="+")
    c.add_argument("--out", required=True)
    d = sub.add_parser("make-docs")
    d.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    d.add_argument("--dataset", default="emozilla/pg19-test")
    d.add_argument("--split", default="test")
    d.add_argument("--field", default="text")
    d.add_argument("--count", type=int, default=8)
    d.add_argument("--min-tokens", type=int, default=33000)
    d.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.cmd == "make-docs":
        print(make_docs(args.tokenizer, args.dataset, args.split, args.field, args.count, args.min_tokens, Path(args.out)))
        return 0
    if args.cmd == "compare":
        write_json(Path(args.out), compare(Path(args.reference), [Path(c) for c in args.candidates]))
        return 0
    info = {}
    try:
        with urllib.request.urlopen(f"{args.base_url}/get_server_info", timeout=30) as response:
            raw = json.load(response)
        info = {k: raw.get(k) for k in ("chunked_prefill_size", "mamba_ssm_dtype", "disable_radix_cache",
                                        "max_running_requests", "model_path")}
    except Exception as exc:
        info = {"error": str(exc)}
    docs = read_jsonl(args.docs)[: args.max_docs]
    summary = measure(args.base_url, docs, [int(x) for x in args.lengths.split(",")], Path(args.out), info)
    print(json.dumps({k: v["mean"] for k, v in summary["buckets"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
