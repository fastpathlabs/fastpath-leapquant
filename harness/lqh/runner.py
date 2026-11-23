"""Paired runner: pinned contract -> paired blocks -> server + workload driver -> immutable raw evidence ->
validation -> metrics. One interface for every profile.

A plan (campaign.json "plans") names candidates, points and a number of blocks. Each block starts every candidate's
server in a balanced order (forward, reversed, rotated: no arm always runs first), runs all the plan's points on it,
stops it and waits for the GPU to empty. Every point becomes one run directory:
  <out>/<plan>/b<block>/<candidate>/<point id>/
    run_manifest.json  requests.jsonl  stream_events.jsonl.gz  driver_summary.json  server_events.jsonl
    memory_and_power.jsonl  validation.json  metrics.json  [native/ for the pinned InferenceX client]
Servers are left alone between points of one block (the profile's reset rule: a restart per block and arm).
A point that cannot run is still written, with execution status not_run / unsupported / crashed and its reason.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import analyze, driver, telemetry
from .util import config_hash, harness_root, read_json, repo_root, sha256_file, utc_now, write_json, write_jsonl

CONFIG = harness_root() / "config"


def load(name: str) -> dict:
    return read_json(CONFIG / f"{name}.json")


def candidate_by_id(candidates: dict, candidate_id: str) -> dict:
    for entry in candidates["candidates"]:
        if entry["candidate_id"] == candidate_id:
            return entry
    raise KeyError(f"unknown candidate {candidate_id}")


def block_order(ids: list[str], block: int) -> list[str]:
    """Balanced orders: block 1 forward, 2 reversed, 3 rotated by one, 4 rotated and reversed, ..."""
    order = list(ids)
    shift = (block - 1) // 2
    order = order[shift % len(order):] + order[:shift % len(order)]
    return order if block % 2 == 1 else list(reversed(order))


def expand_env(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


# ---- server lifecycle -----------------------------------------------------------------------------------------
class Server:
    def __init__(self, candidate: dict, port: int, log_path: Path, model: str, python: str) -> None:
        self.candidate = candidate
        self.port = port
        self.log_path = Path(log_path).resolve()      # the server runs in another working directory
        self.model = model
        self.python = python
        self.proc: subprocess.Popen | None = None
        self.marks: dict[str, float] = {}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def command(self) -> list[str]:
        cfg = self.candidate["config"]
        if cfg.get("launch") == "mock":
            return [sys.executable, "-m", "lqh.mock_server", "--port", str(self.port),
                    "--config", json.dumps(cfg.get("mock", {})), "--log", str(self.log_path)]
        return [expand_env(self.python), "-m", "sglang.launch_server", "--model-path", self.model,
                "--port", str(self.port), *[expand_env(f) for f in cfg.get("server_flags", [])]]

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        for key in self.candidate["config"].get("env_unset", []):
            env.pop(key, None)
        for key, value in (self.candidate["config"].get("env") or {}).items():
            env[key] = expand_env(str(value))
        if self.candidate["config"].get("launch") == "mock":
            env["PYTHONPATH"] = str(harness_root()) + os.pathsep + env.get("PYTHONPATH", "")
        return env

    def start(self, timeout_s: float = 1800) -> bool:
        self.marks["before_model_load"] = time.time()
        self.marks["memory_before_model_load_mib"] = telemetry.gpu_memory_used_mib()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.log_path, "a", encoding="utf-8")
        self.proc = subprocess.Popen(self.command(), env=self.environment(), stdout=handle,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     cwd=expand_env(self.candidate["config"].get("cwd", str(repo_root()))))
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                    if response.status == 200:
                        self.marks["after_pools"] = time.time()
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def served_model(self) -> str:
        try:
            with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=10) as response:
                return json.load(response)["data"][0]["id"]
        except Exception:
            return self.model

    def server_info(self) -> dict:
        try:
            with urllib.request.urlopen(f"{self.base_url}/get_server_info", timeout=10) as response:
                return json.load(response)
        except Exception:
            return {}

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=30)
        if self.candidate["config"].get("launch") != "mock":
            telemetry.wait_gpu_free()
        self.marks["after_stop"] = time.time()


# ---- one point ------------------------------------------------------------------------------------------------
def point_id(point: dict) -> str:
    if point.get("lane") == "open_loop":
        return f"{point['variant']}_rate{point['rate_rps']}"
    return f"{point['variant']}_c{point['concurrency']}"


def _build_facts(campaign: dict) -> dict:
    """What the run was produced with: runbook/manifest.sh output on the GPU host (sources, binaries, versions)."""
    script = repo_root() / "runbook" / "manifest.sh"
    facts: dict[str, Any] = {}
    if script.exists() and os.environ.get("LQH_SKIP_BUILD_MANIFEST") != "1":
        out = Path(os.environ.get("TMPDIR", "/tmp")) / f"lqh_manifest_{os.getpid()}.txt"
        try:
            subprocess.run(["bash", str(script), str(out)], timeout=120, check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for line in out.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(":")
                facts[key.strip()] = value.strip()
        except (OSError, subprocess.SubprocessError):
            facts["manifest_error"] = "runbook/manifest.sh failed"
    return facts


def run_point(campaign: dict, plan_id: str, plan: dict, profile: dict, candidate: dict, point: dict, block: int,
              order_position: int, server: Server | None, out_dir: Path, build: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = candidate["config"]
    lane = point["lane"]
    cell_id = f"{profile['profile_id']}:{point_id(point)}"
    run_id = f"{campaign['campaign_id']}.{plan_id}.b{block}.{candidate['candidate_id']}.{point_id(point)}"
    workload = {k: v for k, v in point.items() if k not in ("variant",)}
    workload.update(variant=point["variant"], lane=lane, ignore_eos=point.get("ignore_eos", True))
    if lane == "finite":
        workload.setdefault("num_prompts", 10 * point["concurrency"])
        workload.setdefault("num_warmups", 2 * point["concurrency"])
    devices = telemetry.query_devices()
    if cfg.get("launch") == "mock":
        devices = [{"index": 0, "uuid": "MOCK-0", "name": "mock device (no GPU)", "memory_total_mib": 0.0,
                    "driver": "none", "pci_bus_id": "none"}]
    clock = driver.Clock()
    manifest: dict[str, Any] = {
        "schema": "lqh.run_manifest.v1", "run_id": run_id, "campaign_id": campaign["campaign_id"],
        "experiment_id": f"{campaign['campaign_id']}.{plan_id}", "plan_id": plan_id,
        "profile_id": profile["profile_id"], "profile_version": profile["version"], "cell_id": cell_id,
        "pair_id": f"{plan_id}.b{block}.{point_id(point)}", "block_id": block, "block_order_position": order_position,
        "comparison_class": plan.get("comparison_class", "diagnostic") if cfg.get("launch") != "mock" else "smoke",
        "is_carried": False, "origin_experiment_id": None, "source_group_id": None, "source_artifact_sha256": None,
        "source_row_id": None, "evidence_kind": "measured_raw",
        "candidate": {"candidate_id": candidate["candidate_id"], "config_hash": config_hash(cfg), "config": cfg,
                      "role": candidate.get("role"), "quality_status": candidate.get("quality_status")},
        "execution": {"status": "completed", "reason": None},
        "clock": clock.describe(),
        "device": {"host": telemetry.host_name(), "allocated_physical_gpus": len(devices) or None, "gpus": devices,
                   "instances": 1},
        "workload": workload,
        "engine": {"name": cfg.get("engine", "sglang"), "commit": cfg.get("engine_commit"),
                   "mock": cfg.get("launch") == "mock"},
        "server": None, "build": build, "slo_id": plan.get("slo_id", campaign.get("default_slo_id")),
        "pricing_id": campaign.get("pricing_id"), "native": None, "artifacts": {},
        "started_utc": utc_now(), "ended_utc": None,
    }
    if server is not None:
        info = server.server_info()
        manifest["server"] = {"command": server.command(), "log": str(server.log_path),
                              "nondefault_flags": cfg.get("server_flags", []), "env": cfg.get("env", {}),
                              "configured_slots": cfg.get("configured_slots"),
                              "max_running_requests": info.get("max_running_requests") or cfg.get("max_running_requests"),
                              "served_model": server.served_model(),
                              "server_info_keys": sorted(info.keys())[:200]}
    sampler = telemetry.GpuSampler(out_dir / "gpu_samples.csv")
    if cfg.get("launch") == "mock":
        sampler.available = False      # a mock run has no device: GPU samples would belong to an unlisted GPU
    sampler.start()
    t_start = time.time()
    if lane == "infx_upstream":
        from .adapters import infx
        summary = infx.run_point(server.base_url if server else point["base_url"], manifest, point, out_dir, clock)
    else:
        model = server.served_model() if server else point.get("model", "default")
        summary = driver.run(server.base_url if server else point["base_url"], model, lane, workload, out_dir,
                             clock=clock)
    t_after = time.time()
    time.sleep(2.0)  # after-drain memory sample
    sampler.stop()
    if summary.get("native"):
        manifest["native"] = summary["native"]
        manifest["workload"]["lane"] = "finite"
        manifest["clock"]["domain"] = "request_relative"
    if summary.get("exit") != "ok":
        manifest["execution"] = {"status": "crashed", "reason": summary.get("error")}
    # Telemetry: GPU samples, server-log events, allocations, memory snapshots.
    samples = sampler.records(clock.origin_unix)
    events: list[dict] = []
    allocations: list[dict] = []
    capacity: dict = {}
    if server is not None and server.log_path.exists():
        text = server.log_path.read_text(encoding="utf-8", errors="replace")
        events, allocations, capacity = telemetry.parse_server_log(text, t_start - 5, t_after + 5, clock.origin_unix)
    marks = {}
    if server is not None:
        marks = {"before_model_load": server.marks.get("before_model_load"),
                 "after_pools": server.marks.get("after_pools")}
    phases = summary.get("phases") or {}
    steady = phases.get("steady") or phases.get("main") or {}
    if steady.get("start") is not None:
        marks["steady"] = clock.origin_unix + (steady["start"] + steady["end"]) / 2.0
    marks["after_drain"] = t_after + 1.5
    snaps = telemetry.snapshots(samples, marks)
    if server is not None and server.marks.get("memory_before_model_load_mib") is not None:
        snaps.append({"kind": "snapshot", "label": "before_model_load_direct", "t_unix": server.marks["before_model_load"],
                      "memory_used_mib": server.marks["memory_before_model_load_mib"]})
    write_jsonl(out_dir / "server_events.jsonl", [{"kind": "capacity", **capacity}] + events)
    write_jsonl(out_dir / "memory_and_power.jsonl", samples + allocations + snaps)
    manifest["ended_utc"] = utc_now()
    for name in ("requests.jsonl", "stream_events.jsonl.gz", "driver_summary.json", "server_events.jsonl",
                 "memory_and_power.jsonl"):
        path = out_dir / name
        if path.exists():
            manifest["artifacts"][name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    write_json(out_dir / "run_manifest.json", manifest)
    return analyze.compute(out_dir)


def write_not_run(campaign: dict, plan_id: str, plan: dict, profile: dict, candidate: dict, point: dict, block: int,
                  out_dir: Path, status: str, reason: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = candidate["config"]
    manifest = {
        "schema": "lqh.run_manifest.v1",
        "run_id": f"{campaign['campaign_id']}.{plan_id}.b{block}.{candidate['candidate_id']}.{point_id(point)}",
        "campaign_id": campaign["campaign_id"], "experiment_id": f"{campaign['campaign_id']}.{plan_id}",
        "plan_id": plan_id, "profile_id": profile["profile_id"], "profile_version": profile["version"],
        "cell_id": f"{profile['profile_id']}:{point_id(point)}", "pair_id": None, "block_id": block,
        "comparison_class": plan.get("comparison_class", "diagnostic"), "is_carried": False,
        "evidence_kind": "planned",
        "candidate": {"candidate_id": candidate["candidate_id"], "config_hash": config_hash(cfg), "config": cfg},
        "execution": {"status": status, "reason": reason},
        "clock": {"origin_utc": utc_now(), "domain": "client_monotonic"},
        "device": {"gpus": [], "allocated_physical_gpus": None},
        "workload": {"lane": point["lane"] if point["lane"] != "infx_upstream" else "finite", **point},
        "engine": {"mock": cfg.get("launch") == "mock"}, "started_utc": utc_now(),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    analyze.compute(out_dir)


# ---- a plan ---------------------------------------------------------------------------------------------------
def run_plan(plan_id: str, out_root: Path, port: int, blocks: list[int] | None = None,
             campaign_path: Path | None = None, budget_s: float | None = None) -> dict:
    campaign = read_json(campaign_path or CONFIG / "campaign.json")
    out_root = Path(out_root).resolve()                # subprocesses (servers, pinned client) use other cwds
    profiles = {p["profile_id"]: p for p in load("profiles")["profiles"]}
    candidates = load("candidates")
    plan = campaign["plans"][plan_id]
    budget_s = budget_s or plan.get("budget_s")
    started = time.time()
    build = _build_facts(campaign)
    model = campaign["model"]["served_path"]
    python = campaign["engine"].get("python", "python3")
    done: list[dict] = []
    problems: list[str] = []          # crashed, invalid or unrun points: the command exits non-zero
    status_path = out_root / plan_id / "STATUS.jsonl"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    finished = set()
    if status_path.exists():
        for line in status_path.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("state") == "done":
                finished.add((entry["block"], entry["candidate"]))
    for block in blocks or range(1, int(plan.get("blocks", 1)) + 1):
        for position, candidate_id in enumerate(block_order(plan["candidates"], block), start=1):
            if (block, candidate_id) in finished:
                continue
            candidate = candidate_by_id(candidates, candidate_id)
            points = plan["points"]
            if plan.get("seed_per_block"):
                # Paired workload seeds (16 Nov 2026): every arm of a block gets the same fresh seed (point seed +
                # block number); blocks differ. The run directory and run id do not change; the manifest records it.
                points = [dict(p, seed=int(p.get("seed", 0)) + block) for p in points]
            run_dirs = [(point, out_root / plan_id / f"b{block}" / candidate_id / point_id(point))
                        for point in points]
            if budget_s and time.time() - started > budget_s:
                for point, run_dir in run_dirs:
                    write_not_run(campaign, plan_id, plan, profiles[point["profile"]], candidate, point, block,
                                  run_dir, "not_run", f"plan budget of {budget_s:.0f} s used up")
                    problems.append(f"b{block} {candidate_id} {point_id(point)}: not_run (plan budget)")
                continue
            log = out_root / plan_id / f"b{block}" / candidate_id / "server.log"
            server = Server(candidate, port, log, model, python)
            if not server.start():
                server.stop()
                for point, run_dir in run_dirs:
                    write_not_run(campaign, plan_id, plan, profiles[point["profile"]], candidate, point, block,
                                  run_dir, "crashed", f"server did not start; see {log}")
                    problems.append(f"b{block} {candidate_id} {point_id(point)}: crashed (server did not start)")
                _status(status_path, block, candidate_id, "server_failed")
                continue
            try:
                for point, run_dir in run_dirs:
                    unsupported = point_unsupported(candidate, point)
                    if unsupported:
                        write_not_run(campaign, plan_id, plan, profiles[point["profile"]], candidate, point, block,
                                      run_dir, "unsupported", unsupported)
                        continue
                    doc = run_point(campaign, plan_id, plan, profiles[point["profile"]], candidate, point, block,
                                    position, server, run_dir, build)
                    done.append({"run_dir": str(run_dir), "status": doc["status"]})
                    if doc["status"] != "measured":
                        problems.append(f"b{block} {candidate_id} {point_id(point)}: {doc['status']}")
                    print(f"{utc_now()} {doc['run_id']}: {doc['status']} "
                          f"{(doc.get('headline') or {}).get('deployment_output_tps')}", flush=True)
                if plan.get("plateau_extension"):
                    plateau_sweep(campaign, plan_id, plan, profiles, candidate, block, position, server, out_root,
                                  build, started, budget_s)
            finally:
                server.stop()
            _status(status_path, block, candidate_id, "done")
    return {"plan": plan_id, "runs": done, "problems": problems}


def plateau_sweep(campaign: dict, plan_id: str, plan: dict, profiles: dict, candidate: dict, block: int,
                  position: int, server: Server, out_root: Path, build: dict, started: float,
                  budget_s: float | None) -> dict:
    """AA-SLT method: after the listed points, add `step` concurrent queries per phase until the DECLARED local plateau
    rule stops the sweep (campaign.json plateau_rule_aa_slt). Every point stays visible, including those past the
    plateau or failing their SLO; the stop reason is written to plateau_<candidate>.json."""
    rule = plan["plateau_extension"]
    base = [p for p in plan["points"] if p["lane"] == "sustained"][-1]
    history = []
    for point in plan["points"]:
        if point["lane"] != "sustained":
            continue
        path = out_root / plan_id / f"b{block}" / candidate["candidate_id"] / point_id(point) / "metrics.json"
        if path.exists():
            doc = read_json(path)
            history.append((point["concurrency"], (doc.get("headline") or {}).get("deployment_output_tps"),
                            ((doc.get("metrics") or {}).get("error_rate") or {}).get("value")))
    conc = base["concurrency"]
    small = 0
    reason = "max_concurrency"
    previous = history[-1][1] if history else None
    while conc + rule["step"] <= rule["max"]:
        if budget_s and time.time() - started > budget_s:
            reason = "plan_budget_used"
            break
        conc += rule["step"]
        point = dict(base, concurrency=conc)
        run_dir = out_root / plan_id / f"b{block}" / candidate["candidate_id"] / point_id(point)
        doc = run_point(campaign, plan_id, plan, profiles[point["profile"]], candidate, point, block, position,
                        server, run_dir, build)
        tps = (doc.get("headline") or {}).get("deployment_output_tps")
        errors = ((doc.get("metrics") or {}).get("error_rate") or {}).get("value")
        history.append((conc, tps, errors))
        print(f"{utc_now()} {doc['run_id']}: {doc['status']} {tps}", flush=True)
        if errors is not None and errors > rule["max_error_rate"]:
            reason = f"error_rate {errors:.3f} above {rule['max_error_rate']}"
            break
        if tps is None:
            reason = "point not measured"
            break
        gain = (tps / previous - 1.0) if previous else None
        small = small + 1 if gain is not None and gain < rule["min_gain"] else 0
        previous = tps
        if small >= rule["patience"]:
            reason = f"plateau: {rule['patience']} successive +{rule['step']} steps each under {rule['min_gain']:.0%}"
            break
    result = {"candidate": candidate["candidate_id"], "block": block, "rule": rule, "stop_reason": reason,
              "points": [{"concurrency": c, "steady_output_tps": t, "error_rate": e} for c, t, e in history],
              "label": "local plateau rule (declared in campaign.json before the run), not prescribed by AA"}
    write_json(out_root / plan_id / f"b{block}" / f"plateau_{candidate['candidate_id']}.json", result)
    return result


def point_unsupported(candidate: dict, point: dict) -> str | None:
    """A point beyond a candidate's declared limits is recorded as unsupported (with the reason), not dropped."""
    limit = candidate["config"].get("max_client_concurrency")
    if limit is not None and point.get("concurrency", 0) > limit:
        return f"concurrency {point['concurrency']} above this candidate's declared limit {limit}"
    return None


def _status(path: Path, block: int, candidate: str, state: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"t": utc_now(), "block": block, "candidate": candidate, "state": state}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.runner")
    parser.add_argument("plan")
    parser.add_argument("--out", required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--blocks", default=None, help="comma-separated block numbers (default: all)")
    parser.add_argument("--campaign", default=None)
    args = parser.parse_args(argv)
    blocks = [int(b) for b in args.blocks.split(",")] if args.blocks else None
    result = run_plan(args.plan, Path(args.out), args.port, blocks, Path(args.campaign) if args.campaign else None)
    print(json.dumps({"plan": result["plan"], "runs": len(result["runs"]), "problems": result["problems"]}))
    # A crashed, invalid or unrun point must not look like success to the job that called this (16 Nov 2026).
    return 3 if result["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
