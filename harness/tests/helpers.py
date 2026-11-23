"""Builds synthetic run directories for the fixtures: a manifest, request records, chunk series and a driver summary,
with artifact hashes, exactly as the runner writes them. Tests then break one thing at a time."""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS))

from lqh.util import config_hash, sha256_file  # noqa: E402

GPU = {"index": 0, "uuid": "GPU-test-0", "name": "NVIDIA H100 PCIe", "memory_total_mib": 81559.0, "driver": "580",
       "pci_bus_id": "00000000:01:00.0"}


def request(rid, t_send, chunks, tokens=None, status="success", phase="measured", requested=None, t_terminal=None,
            t_scheduled=None, attempt=1, error=None):
    """chunks: absolute chunk arrival times; tokens: per-chunk counts (None = unknown)."""
    n = sum(tokens) if tokens is not None else len(chunks)
    record = {"request_id": rid, "attempt": attempt, "phase": phase, "clock_domain": "run", "input_tokens": 100,
              "requested_output_tokens": requested if requested is not None else n,
              "output_tokens": n if status == "success" else None, "output_tokens_source": "usage_final",
              "t_scheduled": t_scheduled if t_scheduled is not None else t_send, "t_send": t_send,
              "t_first_token": chunks[0] if chunks else None, "t_last_token": chunks[-1] if chunks else None,
              "t_terminal": t_terminal if t_terminal is not None else ((chunks[-1] + 0.001) if chunks else None),
              "t_deadline": None, "status": status, "error": error, "content_chunks": len(chunks),
              "exact_chunk_tokens": tokens is not None}
    if status != "success" and t_terminal is None:
        # As the driver records them: a timeout or an unresolved (incomplete) request has no terminal response.
        record["t_terminal"] = None if status in ("timeout", "incomplete") else \
            ((chunks[-1] + 0.001) if chunks else t_send + 0.01)
    series = {"request_id": rid, "attempt": attempt, "kind": "chunk_series", "clock_domain": "run",
              "dt_us": [int(round((t - t_send) * 1e6)) for t in chunks], "tokens": tokens}
    return record, series


def make_run(requests, phases, lane="finite", num_prompts=None, gpus=None, extra_manifest=None, slo_id=None,
             root=None, ignore_eos=True, mock=False):
    root = Path(root or tempfile.mkdtemp(prefix="lqh_fixture_"))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "requests.jsonl", "w") as handle:
        for record, _ in requests:
            handle.write(json.dumps(record) + "\n")
    with gzip.open(root / "stream_events.jsonl.gz", "wt") as handle:
        for _, series in requests:
            handle.write(json.dumps(series) + "\n")
    (root / "driver_summary.json").write_text(json.dumps({"exit": "ok", "phases": phases, "lane": lane}))
    (root / "server_events.jsonl").write_text(json.dumps({"kind": "capacity"}) + "\n")
    (root / "memory_and_power.jsonl").write_text("")
    config = {"launch": "mock" if mock else "sglang", "flags": ["x"]}
    measured = sum(1 for r, _ in requests if r["phase"] == "measured")
    manifest = {
        "schema": "lqh.run_manifest.v1", "run_id": f"fixture.{root.name}", "campaign_id": "test", "experiment_id": "test.exp",
        "plan_id": "plan", "profile_id": "local_diagnostics", "profile_version": "v1", "cell_id": "local_diagnostics:cell",
        "pair_id": "p", "block_id": 1, "comparison_class": "diagnostic", "is_carried": False, "evidence_kind": "measured_raw",
        "candidate": {"candidate_id": "cand", "config_hash": config_hash(config), "config": config},
        "execution": {"status": "completed"}, "clock": {"origin_utc": "2026-11-15T00:00:00Z", "origin_unix": 1_000_000.0,
                                                         "domain": "client_monotonic"},
        "device": {"gpus": gpus if gpus is not None else [GPU], "allocated_physical_gpus": len(gpus) if gpus is not None else 1},
        "workload": {"lane": lane, "num_prompts": num_prompts if num_prompts is not None else measured,
                     "ignore_eos": ignore_eos, "concurrency": 4},
        "engine": {"mock": mock}, "slo_id": slo_id, "pricing_id": None, "artifacts": {},
        "started_utc": "2026-11-15T00:00:00Z",
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    for name in ("requests.jsonl", "stream_events.jsonl.gz", "driver_summary.json"):
        manifest["artifacts"][name] = {"sha256": sha256_file(root / name), "bytes": (root / name).stat().st_size}
    (root / "run_manifest.json").write_text(json.dumps(manifest))
    return root


def simple_finite(n=4, tokens_per_request=8, gap=0.01, start=1.0):
    """n requests, each with `tokens_per_request` single-token chunks, sent at start + i*0.1."""
    reqs = []
    for i in range(n):
        send = start + i * 0.1
        chunks = [send + 0.05 + k * gap for k in range(tokens_per_request)]
        reqs.append(request(f"m{i}", send, chunks, [1] * tokens_per_request))
    end = max(r["t_terminal"] for r, _ in reqs)
    return reqs, {"main": {"start": start, "end": end}}
