"""Carried historical evidence (prompt rev. 3, sections 7, 11, 16): earlier runs in the harness's report, labelled as
carried, with their original identifiers, never counted as repetitions of new runs and never mixed into new
paired comparisons.

  python -m lqh.carried serving RESULTS_ROOT --out DIR
      One row per historical client log (sglang.benchmark.serving closed-loop points, open-loop rates): every metric
      the client reported (evidence_kind reported_aggregate: the per-request events were not kept), the client's
      settings, the source file's SHA-256, and from the same session: configured slots, admitted running and waiting
      requests over the point's interval (from the server log), peak GPU memory (.mem), build identity (manifest.txt).
      Flags: copied (the same file content in two experiments), mixed_build (one arm label, several binaries),
      build_unknown (no manifest), single_run. Writes carried_serving.json / .csv.
  python -m lqh.carried accuracy --reference NAME=RUN.json --arm NAME=RUN.json ... --out DIR [--label TEXT]
      accuracy_bench runs (paper lane) -> quality_items_<arm>.jsonl + quality_decision.json for the lane's own tasks.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

from .telemetry import parse_server_log
from .util import read_json, sha256_file, utc_now, write_json, write_jsonl

METRIC_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 /()%._-]{1,80}?):\s+(-?[0-9][0-9,]*\.?[0-9]*(?:[eE][-+]?\d+)?)\s*$")
SERVING_NAME = re.compile(r"^serving_(?P<label>.+)_(?P<shape>[a-z]+[0-9a-z]*)_c(?P<conc>\d+)\.log$")
OPENLOOP_NAME = re.compile(r"^openloop_(?P<label>.+)_rate(?P<rate>[0-9.]+)\.log$")
ARGS = re.compile(r"benchmark_args=Namespace\((.*)\)")
ARG_KEYS = ("dataset_name", "random_input_len", "random_output_len", "random_range_ratio", "num_prompts",
            "max_concurrency", "request_rate", "seed", "warmup_requests", "flush_cache", "goodput", "disable_ignore_eos",
            "backend", "output_file")
BUILD_KEYS = ("fast_cuda_source_hash", "libleapquant_fastfold.so", "libleapquant_cuda.so", "kernel_source_hash",
              "engine_glue_hash", "fast_kernels_hash", "sglang_diff_hash", "environment", "build_dir", "time_utc")


def result_block(text: str) -> dict[str, float]:
    """Every 'Name: number' line of the last 'Serving Benchmark Result' block."""
    start = text.rfind("Serving Benchmark Result")
    if start < 0:
        return {}
    out: dict[str, float] = {}
    for line in text[start:].splitlines()[1:]:
        line = line.strip()
        if line.startswith("=") and out:
            if len(line.strip("=")) == 0:
                break
        match = METRIC_LINE.match(line)
        if match:
            out[match.group(1).strip()] = float(match.group(2).replace(",", ""))
    return out


def client_args(text: str) -> dict:
    match = ARGS.search(text)
    if not match:
        return {}
    out = {}
    for key in ARG_KEYS:
        found = re.search(rf"\b{key}=([^,]+?)(?:, [a-z_]+=|\)$)", match.group(1) + ")")
        if found:
            out[key] = found.group(1).strip().strip("'")
    return out


def arm_of(label: str) -> tuple[str, str | None, str | None]:
    """'off_bf16_window_r2' -> ('bf16_window', 'off', 'r2')."""
    match = re.match(r"^(?:(on|off)_)?(.+?)(?:_(r\d+))?$", label)
    return (match.group(2), match.group(1), match.group(3)) if match else (label, None, None)


def build_facts(folder: Path) -> dict | None:
    manifest = folder / "manifest.txt"
    if not manifest.exists():
        return None
    facts = {}
    for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
        key, _, value = line.partition(":")
        if key.strip() in BUILD_KEYS:
            facts[key.strip()] = value.strip()
    return facts


def build_identity(facts: dict | None) -> str | None:
    if not facts:
        return None
    parts = [facts.get(k, "") for k in ("fast_cuda_source_hash", "libleapquant_fastfold.so", "libleapquant_cuda.so",
                                        "fast_kernels_hash", "engine_glue_hash")]
    return "|".join(p.split(" built")[0] for p in parts) if any(parts) else None


def serving_rows(root: Path) -> list[dict]:
    rows = []
    logs = sorted(root.glob("*/serving_*.log")) + sorted(root.glob("*/openloop_*.log"))
    server_cache: dict[Path, str] = {}
    for log in logs:
        folder, name = log.parent, log.name
        match = SERVING_NAME.match(name) or OPENLOOP_NAME.match(name)
        if not match:
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        metrics = result_block(text)
        if not metrics:
            continue
        label = match.group("label")
        arm, radix, rep = arm_of(label)
        row = {"row_id": f"{folder.name}/{name}", "origin_experiment_id": folder.name, "source_file": str(log.resolve()),
               "source_artifact_sha256": sha256_file(log), "is_carried": True, "evidence_kind": "reported_aggregate",
               "lane": "open_loop" if name.startswith("openloop_") else "finite", "label": label, "arm": arm,
               "radix": radix, "repetition_label": rep, "metrics": metrics, "client": client_args(text)}
        if "conc" in match.groupdict() and match.group("conc"):
            row.update(shape=match.group("shape"), client_concurrency=int(match.group("conc")))
        if "rate" in match.groupdict():
            row.update(offered_rate_rps=float(match.group("rate")))
        mem = log.with_suffix(".mem")
        if mem.exists():
            found = re.search(r"peak_memory_used_mib (\d+)", mem.read_text())
            row["peak_gpu_memory_mib"] = int(found.group(1)) if found else None
        server_log = folder / f"server_{label}.log"
        if not server_log.exists():
            candidates = sorted(folder.glob(f"server_*{arm}*.log"))
            server_log = candidates[0] if len(candidates) == 1 else server_log
        if server_log.exists():
            if server_log not in server_cache:
                server_cache[server_log] = server_log.read_text(encoding="utf-8", errors="replace")
            duration = metrics.get("Benchmark duration (s)") or 0.0
            end = os.path.getmtime(log)
            events, allocations, capacity = parse_server_log(server_cache[server_log], end - duration - 120, end + 5,
                                                             utc=True)
            decode = [e for e in events if e["kind"] == "decode"]
            row["server"] = {"log": str(server_log.resolve()), "log_end_unix": end, "configured_slots": capacity.get("leapquant_slots") or capacity.get("max_mamba_cache_size"),
                             "max_running_requests": capacity.get("max_running_requests"), "kv_tokens": capacity.get("kv_tokens"),
                             "admitted_running_max": max((e.get("running", 0) for e in decode), default=None),
                             "waiting_max": max((e.get("queue", 0) for e in events if "queue" in e), default=None),
                             "retractions": sum(e.get("retracted", 0) for e in events),
                             "interval": "approximate: [client log mtime - duration - 120 s, mtime + 5 s]",
                             "allocations_gib": {a["category"]: a["gib"] for a in allocations}}
        facts = build_facts(folder)
        row["build"] = facts
        row["build_identity"] = build_identity(facts)
        rows.append(row)
    # Flags across rows.
    by_hash = defaultdict(set)
    for row in rows:
        by_hash[row["source_artifact_sha256"]].add(row["origin_experiment_id"])
    builds_per_arm = defaultdict(set)
    for row in rows:
        if row["build_identity"]:
            builds_per_arm[row["arm"]].add(row["build_identity"])
    for row in rows:
        flags = ["single_run"]
        if len(by_hash[row["source_artifact_sha256"]]) > 1:
            flags.append("copied: same file content in " + ", ".join(sorted(by_hash[row["source_artifact_sha256"]])))
        if len(builds_per_arm[row["arm"]]) > 1 and "leapquant" in row["arm"]:
            flags.append(f"mixed_build: the label '{row['arm']}' covers {len(builds_per_arm[row['arm']])} builds")
        if not row["build"]:
            flags.append("build_unknown: no manifest.txt in the experiment folder")
        server = row.get("server") or {}
        if server.get("admitted_running_max") and row.get("client_concurrency") and \
                server["admitted_running_max"] < row["client_concurrency"]:
            flags.append(f"server ran at most {server['admitted_running_max']} of {row['client_concurrency']} client requests")
        row["flags"] = flags
    return rows


def write_serving(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "carried_serving.json", {"schema": "lqh.carried_serving.v1", "computed_utc": utc_now(),
                                              "rows": rows, "note": "carried history: client-reported aggregates; never "
                                              "repetitions of new runs; never in new paired comparisons"})
    keys: list[str] = []
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat["flags"] = "; ".join(row["flags"])
        for k, v in row["metrics"].items():
            flat[f"metric: {k}"] = v
        for k, v in (row.get("server") or {}).items():
            if not isinstance(v, dict):
                flat[f"server: {k}"] = v
        for k, v in (row.get("client") or {}).items():
            flat[f"client: {k}"] = v
        flat_rows.append(flat)
        keys += [k for k in flat if k not in keys]
    with open(out / "carried_serving.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for flat in flat_rows:
            writer.writerow(flat)


def accuracy(reference: tuple[str, Path], arms: list[tuple[str, Path]], out: Path, label: str, margin: float) -> dict:
    """accuracy_bench runs -> quality_items_<arm>.jsonl and a decision for the lane's own tasks."""
    from .quality import compare_task, load_arm
    out.mkdir(parents=True, exist_ok=True)
    tasks: set[str] = set()
    for name, path in [reference] + arms:
        run = read_json(path)
        rows = []
        for task, items in run["tasks"].items():
            tasks.add(task)
            for item in items:
                rows.append({"schema": "lqh.quality_item.v1", "dataset": task, "sample_id": f"{task}:{item['id']}",
                             "question": item["question"], "sample": item["sample"],
                             "seed": None if run.get("seed") is None else run["seed"] + item["sample"], "arm": name,
                             "prompt_sha256": item.get("prompt_sha256"), "answer": item["answer"], "gold": item["gold"],
                             "correct": item["correct"], "capped": bool(item.get("truncated")),
                             "finish_reason": item.get("finish_reason"), "completion_tokens": item.get("completion_tokens"),
                             "error": item.get("error"), "max_tokens": run.get("max_tokens"),
                             "temperature": run.get("temperature"), "top_p": run.get("top_p"),
                             "source_run": str(path), "source_sha256": sha256_file(path)})
        write_jsonl(out / f"quality_items_{name}.jsonl", rows)
    ref = load_arm(out / f"quality_items_{reference[0]}.jsonl")
    doc = {"schema": "lqh.quality_decision.v1", "computed_utc": utc_now(), "label": label, "reference_arm": reference[0],
           "margin_points": margin, "tasks_declared": sorted(tasks), "is_carried": True, "arms": {},
           "decision_rule": "lane-specific: noninferior on every task of this lane at the margin"}
    for name, _ in arms:
        mine = load_arm(out / f"quality_items_{name}.jsonl")
        doc["arms"][name] = {task: compare_task(ref.get(task, []), items, margin) for task, items in mine.items()}
    doc["decision"] = {f"{name}_noninferior_on_every_task": all(r["verdict"] == "PASS" for r in tasks_.values())
                       for name, tasks_ in doc["arms"].items()}
    doc["decision"]["note"] = ("prompt hashes were not recorded by these historical runs (null in both arms); the cohorts "
                               "are joined by question id, sample and seed")
    doc["reference_cap_rate"] = {t: sum(i["capped"] for i in items) / len(items) for t, items in ref.items()}
    write_json(out / "quality_decision.json", doc)
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.carried")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serving")
    s.add_argument("root")
    s.add_argument("--out", required=True)
    a = sub.add_parser("accuracy")
    a.add_argument("--reference", required=True, help="NAME=RUN.json")
    a.add_argument("--arm", action="append", required=True, help="NAME=RUN.json (repeatable)")
    a.add_argument("--out", required=True)
    a.add_argument("--label", default="carried paper-lane accuracy")
    a.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.cmd == "serving":
        rows = serving_rows(Path(args.root))
        write_serving(rows, Path(args.out))
        flagged = sum(1 for r in rows if len(r["flags"]) > 1)
        print({"rows": len(rows), "experiments": len({r["origin_experiment_id"] for r in rows}), "flagged": flagged})
        return 0
    ref_name, ref_path = args.reference.split("=", 1)
    arms = [(a.split("=", 1)[0], Path(a.split("=", 1)[1])) for a in args.arm]
    doc = accuracy((ref_name, Path(ref_path)), arms, Path(args.out), args.label, args.margin)
    for arm, tasks in doc["arms"].items():
        for task, r in tasks.items():
            print(arm, task, r["verdict"], round(r.get("difference", 0), 4), round(r.get("low") or 0, 4),
                  round(r.get("high") or 0, 4), "cap", round(r.get("cap_rate_arm", 0), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
