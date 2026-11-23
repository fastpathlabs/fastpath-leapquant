"""python -m lqh.campaign_manifest --out FILE [--build-manifest MANIFEST_TXT] [--job-log LOG]

campaign_manifest.json (multi-GPU benchmark plan, 15 Nov 2026, sections 2 and 12): the pins, configurations,
hardware, budgets and planned/actual work of the campaign in one file. Run on the GPU host (it reads nvidia-smi,
lscpu, /proc/meminfo and the build manifest written by runbook/manifest.sh). What cannot be pinned is recorded with a
null and the reason, never guessed (no git repository on the host, no container).
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from .util import harness_root

CONFIG = harness_root() / "config"


def run(argv: list[str]) -> str | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def gpus() -> list[dict]:
    fields = ["uuid", "name", "pci.bus_id", "memory.total", "power.limit", "power.default_limit", "clocks.max.sm",
              "clocks.max.mem", "driver_version", "pcie.link.gen.max", "pcie.link.width.max", "persistence_mode",
              "compute_mode"]
    out = run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"]) or ""
    rows = []
    for line in out.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(fields, values)))
    topo = run(["nvidia-smi", "topo", "-m"])
    for row in rows:
        row["form_factor"] = "PCIe" if "PCIe" in row.get("name", "") else ("SXM" if "SXM" in row.get("name", "") else None)
    return rows + ([{"topology_matrix": topo}] if topo else [])


def host() -> dict:
    lscpu = run(["lscpu"]) or ""
    cpu = re.search(r"Model name:\s*(.+)", lscpu)
    cores = re.search(r"^CPU\(s\):\s*(\d+)", lscpu, re.M)
    numa = re.search(r"NUMA node\(s\):\s*(\d+)", lscpu)
    mem = None
    try:
        m = re.search(r"MemTotal:\s*(\d+) kB", Path("/proc/meminfo").read_text())
        mem = round(int(m.group(1)) / 2**20, 1) if m else None
    except OSError:
        pass
    return {"hostname": os.uname().nodename, "cpu_model": cpu.group(1).strip() if cpu else None,
            "logical_cpus": int(cores.group(1)) if cores else None, "numa_nodes": int(numa.group(1)) if numa else None,
            "ram_gib": mem, "kernel": os.uname().release,
            "open_file_limit_note": "soft 1,024 by default; runbook/harness_plan.sh raises it to the hard limit (15 Nov)",
            "network_interconnect": "single host, single GPU: no NVLink/NVSwitch or inter-node network in use"}


def build_facts(path: Path | None) -> dict:
    if not path or not path.exists():
        return {"null_reason": "no build manifest given"}
    facts = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            facts[key.strip()] = value.strip()
    return facts


def jobs(log: Path | None) -> list[dict]:
    if not log or not log.exists():
        return []
    rows: dict[str, dict] = {}
    for line in log.read_text().splitlines():
        m = re.match(r"(\d{3})[a-z]?_(\S+) (start|end) (\d\d:\d\d)(?: exit (\d+))?", line)
        if m:
            row = rows.setdefault(m.group(1), {"job": m.group(1), "name": m.group(2)})
            row[m.group(3)] = m.group(4)
            if m.group(5) is not None:
                row["exit"] = int(m.group(5))
            continue
        m = re.match(r"\S+Z (\d{3})_(\S+) (queued|STOPPED|edited)(.*)", line)
        if m:
            row = rows.setdefault(m.group(1), {"job": m.group(1), "name": m.group(2)})
            row.setdefault("notes", []).append((m.group(3) + m.group(4))[:300])
    return [rows[k] for k in sorted(rows)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--build-manifest", default=None)
    parser.add_argument("--job-log", default=None)
    args = parser.parse_args(argv)
    configs = {name: json.loads((CONFIG / name).read_text()) for name in
               ("campaign.json", "campaign_q38_27b.json", "candidates.json", "slo.json", "pricing.json", "profiles.json")
               if (CONFIG / name).exists()}
    doc = {
        "schema": "lqh.campaign_manifest.v1",
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "specification": ["the internal test-harness specification of 14 Nov 2026 (revision 3)",
                          "LeapQuant multi-GPU benchmark plan (15 Nov 2026), adopted 15 Nov ~23:00 UTC "
                          "with the constraint that tests finish within a few hours"],
        "source": {"git_commit": None, "git_null_reason": "the source tree on the host and the local copy are not git "
                   "repositories (a git tag/commit is open item X2, on Asif's request only)",
                   "dirty_patch_hash": None, "container_digest": None,
                   "container_null_reason": "bare-metal virtualenvs, no container",
                   "config_sha256": {name: sha256(CONFIG / name) for name in configs},
                   "build": build_facts(Path(args.build_manifest) if args.build_manifest else None)},
        "models": {
            "qwen3.5-9b": configs.get("campaign.json", {}).get("model"),
            "qwen3.8-27b": configs.get("campaign_q38_27b.json", {}).get("model"),
            "qwen3.6-35b-a3b": {"status": "not started (prompt stage 5: after the dense model passes)"}},
        "numerical_policy": configs.get("campaign.json", {}).get("numerical_policy"),
        "candidates": {c["candidate_id"]: {"config_hash": None, "server_flags": c["config"].get("server_flags"),
                                           "env": c["config"].get("env"), "slots": c["config"].get("configured_slots"),
                                           "label": c.get("label")}
                       for c in configs.get("candidates.json", {}).get("candidates", [])},
        "slo": configs.get("slo.json"),
        "pricing": configs.get("pricing.json"),
        "hardware": {"gpus": gpus(), "allocated_gpus": 1, "host": host(),
                     "curves": "one hardware type (H100 PCIe 80 GB); no other hardware in this campaign"},
        "jobs": jobs(Path(args.job_log) if args.job_log else None),
        "blocked_or_not_run": [
            {"item": "multi-GPU stages 2-5 (DP 1/2/4/8 for the 9B, TP 1/2/4/8 for the 27B, equal-budget TP/DP/EP)",
             "reason": "one GPU on this host; no authorization to provision more (more GPUs need the owners' "
                       "approval); TP support in the LeapQuant integration is also untested"},
            {"item": "Qwen3.6-35B-A3B (MoE)", "reason": "plan stage 5, after the 27B passes; not in the time box"},
            {"item": "profile C (product open-loop), profile D (AgentX), profile E (mixed-length wave study)",
             "reason": "not in the time box; AgentX harness not installed (backlog B70)"},
            {"item": "10-block confirmations; full quality suites (GPQA-Diamond 198, MMLU-Pro, LiveCodeBench)",
             "reason": "not in the time box; quality stays pending, so performance claims stay experimental"},
            {"item": "racecheck of the decode kernel at batch 512", "reason": "interrupted at 600 s on 15 Nov (job 112); "
             "needs an hour or more; not in the time box"},
            {"item": "compounding probe repair", "reason": "the mirror state is never read back between prefill chunks; "
             "needs a code change and a sensitivity test"}],
    }
    for cid, entry in doc["candidates"].items():
        cand = next(c for c in configs["candidates.json"]["candidates"] if c["candidate_id"] == cid)
        from .util import config_hash
        entry["config_hash"] = config_hash(cand["config"])
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}: {len(doc['candidates'])} candidates, {len(doc['jobs'])} jobs, "
          f"{len([g for g in doc['hardware']['gpus'] if 'uuid' in g])} GPU(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
