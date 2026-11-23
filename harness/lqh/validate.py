"""Validation between raw evidence and every public number.

validate_run() checks one run directory and gives it exactly one status:
  not_run | unsupported | skipped | crashed | invalid | measured | smoke_only
("qualified" is decided later, per campaign cell, from the measured runs, the SLO and the candidate's quality and
lifecycle status.) A run is "measured" only when every error-level check passes; anything unknown stays unknown.
validate_campaign() maps runs onto the expected cells of a plan and keeps expected / attempted / completed /
parsed / accepted counts distinct. Neither ever turns an incomplete input into a pass.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import schema_lite
from .records import REQUESTS_FILE, STREAM_FILE
from .util import StrictJsonError, config_hash, iter_jsonl, read_json, sha256_file, write_json

EPS = 1e-6
RUN_STATUSES = ("not_run", "unsupported", "skipped", "crashed", "invalid", "measured", "smoke_only")
UNKNOWN = (None, "", "unknown", "Unknown", "N/A", "[N/A]")


class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []

    def add(self, check_id: str, ok: bool, detail: str = "", severity: str = "error") -> bool:
        self.checks.append({"id": check_id, "ok": bool(ok), "severity": severity, "detail": detail})
        return ok

    @property
    def errors(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"] and c["severity"] == "error"]

    @property
    def warnings(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"] and c["severity"] == "warning"]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_request_records(records: list[dict], report: Report, exact_output: bool,
                          series: dict[tuple[str, int], dict]) -> dict[str, int]:
    """Record-level checks. Returns counts (parsed, measured, duplicates, ...)."""
    seen: Counter = Counter()
    problems: dict[str, list[str]] = defaultdict(list)
    for record in records:
        errors = schema_lite.check(record, "request")
        if errors:
            problems["schema"].append(f"{record.get('request_id')}: {errors[0]}")
            continue
        key = (record["request_id"], record.get("attempt", 1))
        seen[key] += 1
        times = [record.get(name) for name in ("t_scheduled", "t_send", "t_first_token", "t_last_token", "t_terminal")]
        present = [t for t in times if t is not None]
        if any(not _finite(t) for t in present):
            problems["non_finite_time"].append(record["request_id"])
            continue
        if any(b < a - EPS for a, b in zip(present, present[1:])):
            problems["timestamp_reversal"].append(record["request_id"])
        status = record["status"]
        n_out = record.get("output_tokens")
        requested = record.get("requested_output_tokens")
        if n_out is not None and n_out < 0:
            problems["negative_tokens"].append(record["request_id"])
        if n_out is not None and requested is not None and n_out > requested:
            problems["impossible_token_count"].append(f"{record['request_id']}: {n_out} > requested {requested}")
        if status == "success":
            relative = record.get("clock_domain") == "request_relative"
            if record.get("t_first_token") is None or (record.get("t_terminal") is None and not relative):
                problems["success_without_timestamps"].append(record["request_id"])
            if n_out is None:
                problems["success_without_token_count"].append(record["request_id"])
            if exact_output and requested is not None and n_out is not None and n_out != requested \
                    and record.get("phase") != "warmup":
                problems["output_length_mismatch"].append(f"{record['request_id']}: {n_out} != {requested}")
        chunk = series.get(key)
        if chunk is not None and record.get("clock_domain", "run") == "run":
            dt = chunk.get("dt_us") or []
            if any(b < a for a, b in zip(dt, dt[1:])):
                problems["chunk_time_reversal"].append(record["request_id"])
            if dt and record.get("t_first_token") is not None:
                first = record["t_send"] + dt[0] / 1e6
                if abs(first - record["t_first_token"]) > 2e-6:
                    problems["first_chunk_mismatch"].append(record["request_id"])
            tokens = chunk.get("tokens")
            if tokens is not None:
                if len(tokens) != len(dt):
                    problems["chunk_token_length_mismatch"].append(record["request_id"])
                elif status == "success" and n_out is not None and sum(tokens) != n_out \
                        and record.get("output_tokens_source") != "upstream_usage":
                    problems["chunk_token_sum_mismatch"].append(f"{record['request_id']}: {sum(tokens)} != {n_out}")
    duplicates = [f"{k[0]}#{k[1]}" for k, n in seen.items() if n > 1]
    report.add("V010_request_schema", not problems["schema"], "; ".join(problems["schema"][:5]))
    report.add("V011_duplicate_request_ids", not duplicates, ", ".join(duplicates[:10]))
    report.add("V012_non_finite_times", not problems["non_finite_time"], ", ".join(problems["non_finite_time"][:10]))
    report.add("V013_timestamp_order", not problems["timestamp_reversal"],
               "send <= first <= last <= terminal violated: " + ", ".join(problems["timestamp_reversal"][:10]))
    report.add("V014_token_counts", not (problems["negative_tokens"] or problems["impossible_token_count"]),
               "; ".join((problems["negative_tokens"] + problems["impossible_token_count"])[:10]))
    report.add("V015_success_fields", not (problems["success_without_timestamps"]
                                          or problems["success_without_token_count"]),
               "; ".join((problems["success_without_timestamps"] + problems["success_without_token_count"])[:10]))
    report.add("V016_exact_output_length", not problems["output_length_mismatch"],
               "; ".join(problems["output_length_mismatch"][:10]), severity="warning")
    report.add("V017_chunk_series", not (problems["chunk_time_reversal"] or problems["first_chunk_mismatch"]
                                        or problems["chunk_token_length_mismatch"]),
               "; ".join((problems["chunk_time_reversal"] + problems["first_chunk_mismatch"]
                          + problems["chunk_token_length_mismatch"])[:10]))
    report.add("V018_chunk_token_sum", not problems["chunk_token_sum_mismatch"],
               "; ".join(problems["chunk_token_sum_mismatch"][:10]))
    return {"parsed": len(records), "duplicates": len(duplicates),
            "output_length_mismatches": len(problems["output_length_mismatch"])}


def check_devices(manifest: dict, samples: list[dict], report: Report) -> None:
    device = manifest.get("device") or {}
    gpus = device.get("gpus") or []
    allocated = device.get("allocated_physical_gpus")
    unknown = [g.get("index") for g in gpus
               if any(g.get(f) in UNKNOWN for f in ("name", "uuid", "memory_total_mib"))]
    report.add("V020_device_listed", bool(gpus), "no GPU recorded in the manifest")
    report.add("V021_device_fields_known", not unknown, f"GPU fields unknown for index {unknown}")
    uuids = {g.get("uuid") for g in gpus}
    report.add("V022_device_count_consistent", allocated == len(uuids) and allocated is not None,
               f"allocated_physical_gpus={allocated}, distinct listed GPUs={len(uuids)}")
    sampled = {s.get("gpu_uuid") for s in samples if s.get("kind") == "gpu_sample"}
    foreign = sampled - uuids
    report.add("V023_samples_from_allocated_devices", not foreign, f"samples from unlisted GPUs {sorted(map(str, foreign))}")


def check_phases(summary: dict, records: list[dict], lane: str, report: Report, native: dict | None = None) -> None:
    phases = summary.get("phases") or {}
    bad = [name for name, p in phases.items()
           if not (isinstance(p, dict) and _finite(p.get("start")) and _finite(p.get("end")) and p["end"] >= p["start"])]
    report.add("V030_phases_well_formed", not bad, f"bad phases: {bad}")
    if lane == "finite" and native:
        # Pinned-client lane: the client's own duration is the cohort duration (drain included).
        duration = native.get("duration")
        report.add("V031_native_duration", _finite(duration) and duration > 0, f"client duration {duration!r}")
        return
    if lane == "finite":
        main = phases.get("main")
        report.add("V031_main_phase_present", main is not None, "finite cohort without a main phase")
        if main:
            early = [r["request_id"] for r in records if r.get("phase") == "measured"
                     and r.get("clock_domain", "run") == "run" and r["t_send"] < main["start"] - EPS]
            late = [r["request_id"] for r in records if r.get("phase") == "measured"
                    and r.get("clock_domain", "run") == "run" and r.get("t_terminal") is not None
                    and r["t_terminal"] > main["end"] + 1e-3]
            report.add("V032_measured_inside_main", not early and not late,
                       f"sent before main start: {early[:5]}, ended after main end: {late[:5]}")
            warm = phases.get("warmup")
            if warm:
                report.add("V033_warmup_before_main", warm["end"] <= main["start"] + EPS,
                           "warmup overlaps the measured phase")
    if lane == "sustained":
        names = ("ramp", "steady", "drain")
        report.add("V034_sustained_phases", all(n in phases for n in names), f"needs {names}")
        if all(n in phases for n in names):
            ordered = phases["ramp"]["end"] <= phases["steady"]["start"] + EPS and \
                phases["steady"]["end"] <= phases["drain"]["start"] + EPS
            report.add("V035_sustained_phase_order", ordered, "ramp, steady and drain overlap or are out of order")


def check_status_lines(lines: Iterable[str], report: Report) -> str | None:
    """Runbook STATUS.txt semantics: the LAST verdict of a step wins; an earlier PASS never survives a later FAIL;
    a SKIPPED line is not a pass. Returns the overall verdict (PASS / FAIL / SKIPPED / None when no step)."""
    verdicts: dict[str, str] = {}
    for line in lines:
        match = re.match(r"^(\S+)\s+(PASS|FAIL|INCONCLUSIVE)\b(.*)$", line.strip())
        if not match:
            continue
        name, verdict, rest = match.groups()
        if "SKIPPED" in rest:
            verdict = "SKIPPED"
        verdicts[name] = verdict
    if not verdicts:
        report.add("V040_status_steps", False, "no step verdicts in STATUS", severity="warning")
        return None
    failed = sorted(n for n, v in verdicts.items() if v in ("FAIL", "INCONCLUSIVE"))
    skipped = sorted(n for n, v in verdicts.items() if v == "SKIPPED")
    report.add("V041_status_final_verdicts", not failed, f"steps whose last verdict is not PASS: {failed}")
    report.add("V042_status_skipped", not skipped, f"skipped steps (not passes): {skipped}", severity="warning")
    if failed:
        return "FAIL"
    if skipped:
        return "SKIPPED"
    return "PASS"


def validate_run(run_dir: str | Path, write: bool = True) -> dict:
    run_dir = Path(run_dir)
    report = Report()
    result: dict[str, Any] = {"run_dir": str(run_dir), "schema": "lqh.validation.v1"}
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        report.add("V001_manifest_present", False, "run_manifest.json missing")
        return _finish(result, report, "invalid", run_dir, write)
    try:
        manifest = read_json(manifest_path)
    except (StrictJsonError, OSError) as exc:
        report.add("V001_manifest_present", False, f"unreadable: {exc}")
        return _finish(result, report, "invalid", run_dir, write)
    result["run_id"] = manifest.get("run_id")
    schema_errors = schema_lite.check(manifest, "run_manifest")
    report.add("V002_manifest_schema", not schema_errors, "; ".join(schema_errors[:5]))
    declared = manifest.get("execution", {}).get("status")
    if declared in ("not_run", "unsupported", "skipped"):
        report.add("V003_declared_not_measured", True, f"declared {declared}: {manifest['execution'].get('reason')}")
        return _finish(result, report, declared, run_dir, write)
    candidate = manifest.get("candidate") or {}
    if candidate.get("config") is not None:
        recomputed = config_hash(candidate["config"])
        report.add("V004_candidate_hash", recomputed == candidate.get("config_hash"),
                   f"manifest {candidate.get('config_hash')} != recomputed {recomputed}")
    # Artifact hashes: what the manifest names must exist and be byte-identical.
    for name, meta in (manifest.get("artifacts") or {}).items():
        path = run_dir / name
        if not path.exists():
            report.add(f"V005_artifact:{name}", False, "listed in the manifest but missing")
            continue
        actual = sha256_file(path)
        report.add(f"V005_artifact:{name}", actual == meta.get("sha256"),
                   f"sha256 {actual[:16]} != manifest {str(meta.get('sha256'))[:16]}")
    crashed = manifest.get("execution", {}).get("status") == "crashed"
    summary_path = run_dir / "driver_summary.json"
    if not summary_path.exists():
        report.add("V006_driver_summary", False, "driver_summary.json missing (the client did not finish)")
        return _finish(result, report, "crashed", run_dir, write)
    summary = read_json(summary_path)
    if summary.get("exit") not in (None, "ok"):
        report.add("V007_driver_exit", False, f"driver exit: {summary.get('exit')}: {summary.get('error')}")
        return _finish(result, report, "crashed", run_dir, write)
    # Raw request records.
    try:
        records = [record for _, record in iter_jsonl(run_dir / REQUESTS_FILE)]
    except (StrictJsonError, OSError) as exc:
        report.add("V008_requests_readable", False, str(exc))
        return _finish(result, report, "invalid", run_dir, write)
    series: dict[tuple[str, int], dict] = {}
    if (run_dir / STREAM_FILE).exists():
        try:
            for _, chunk in iter_jsonl(run_dir / STREAM_FILE):
                series[(chunk["request_id"], int(chunk.get("attempt", 1)))] = chunk
        except (StrictJsonError, OSError, KeyError) as exc:
            report.add("V009_stream_readable", False, str(exc))
    lane = manifest.get("workload", {}).get("lane", "finite")
    exact_output = bool(manifest.get("workload", {}).get("ignore_eos"))
    counts = check_request_records(records, report, exact_output, series)
    measured = [r for r in records if r.get("phase") in ("measured", "sustained")]
    # A measured request whose connection was refused found no server: it had died or never came up, and the run then
    # measures the outage, not the configuration (job 114, 15 Nov 2026: sglang_best ran out of GPU memory during its
    # warmup and was published as 0.0 tok/s "measured").
    refused = [r for r in measured if r.get("status") != "success"
               and re.search(r"ConnectionRefused|Connect call failed", str(r.get("error") or ""))]
    report.add("V052_server_reachable", not refused,
               f"{len(refused)} of {len(measured)} measured requests found no server: {str(refused[0].get('error'))[:160]}"
               if refused else "")
    # A request the CLIENT could not send (its own resources ran out) measures the load generator, not the server
    # (AA-SLT at 1,024 users, 15 Nov 2026: OSError 24 "Too many open files" on 2.3 million attempts per arm).
    client_side = [r for r in measured if r.get("status") != "success"
                   and re.search(r"Errno 24|Too many open files|Errno 99|Cannot assign requested address|"
                                 r"Errno 105|No buffer space", str(r.get("error") or ""))]
    report.add("V054_client_resources", not client_side,
               f"{len(client_side)} of {len(measured)} measured attempts failed inside the client: "
               f"{str(client_side[0].get('error'))[:160]}" if client_side else "")
    # Failure storm: a sustained lane replaces every finished attempt, failed ones included, so an instant failure
    # becomes millions of attempts. Half the attempts failing (and at least 50) is not served load.
    failed = [r for r in measured if r.get("status") != "success"]
    storm = len(failed) >= 50 and len(failed) >= 0.5 * max(1, len(measured))
    report.add("V053_failure_storm", not storm,
               f"{len(failed)} of {len(measured)} measured attempts failed" if storm else "")
    expected = manifest.get("workload", {}).get("num_prompts") if lane == "finite" else None
    logical = {r["request_id"] for r in measured}
    if expected is not None:
        report.add("V019_expected_requests", len(logical) == expected,
                   f"expected {expected} measured requests, records for {len(logical)}")
    check_phases(summary, records, lane, report, manifest.get("native"))
    # The pinned InferenceX client runs its warmups itself and exports only the measured cohort, so the records hold
    # no warmup (the exported warmup count is 0). Its own log states the count and the order; that is the evidence
    # (multi-GPU benchmark plan, 15 Nov 2026: "audit the exported warmup count of zero against the 2C claim").
    client_log = run_dir / "native" / "client.log"
    if client_log.exists():
        text = client_log.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"num_warmups=(\d+)", text)
        requested = int(m.group(1)) if m else None
        m = re.search(r"Warming up with (\d+) requests", text)
        started = int(m.group(1)) if m else None
        done_at, main_at = text.find("Warmup completed"), text.find("Starting main benchmark run")
        conc = (manifest.get("workload") or {}).get("concurrency")
        expected = 2 * int(conc) if isinstance(conc, (int, float)) else None
        before_main = 0 <= done_at < main_at
        result["native_warmups"] = {"requested": requested, "started": started, "completed": done_at >= 0,
                                    "before_measured_cohort": before_main, "expected_2c": expected,
                                    "evidence": "native/client.log"}
        conforming = (requested is not None and requested == started and before_main
                      and (expected is None or requested == expected))
        report.add("V035_native_warmups", conforming,
                   f"client log: num_warmups={requested}, started {started}, completed {done_at >= 0}, before the "
                   f"measured cohort {before_main}, expected 2C={expected}", severity="warning")
    samples: list[dict] = []
    if (run_dir / "memory_and_power.jsonl").exists():
        try:
            samples = [s for _, s in iter_jsonl(run_dir / "memory_and_power.jsonl")]
        except StrictJsonError as exc:
            report.add("V024_memory_power_readable", False, str(exc))
    check_devices(manifest, samples, report)
    native = manifest.get("native") or {}
    outcome = native.get("benchmark_outcome")
    if outcome:
        # The upstream gate passes with up to 5% failures; the project's own requirement is separate.
        report.add("V050_upstream_gate", outcome.get("status") == "passed",
                   f"upstream outcome {outcome}", severity="warning")
        report.add("V051_upstream_failures_disclosed", outcome.get("failed", 0) == 0,
                   f"upstream gate status {outcome.get('status')} with {outcome.get('failed')} failed requests",
                   severity="warning")
    result["counts"] = {"expected": expected, "attempted": len(measured), "parsed": counts["parsed"],
                        "completed": sum(1 for r in measured if r["status"] == "success"),
                        "duplicates": counts["duplicates"]}
    status = "measured"
    if manifest.get("engine", {}).get("mock"):
        status = "smoke_only"
    if report.errors:
        status = "crashed" if crashed or refused else "invalid"
    return _finish(result, report, status, run_dir, write)


def _finish(result: dict, report: Report, status: str, run_dir: Path, write: bool) -> dict:
    result["status"] = status
    result["execution_validity"] = "valid" if status in ("measured", "smoke_only") else (
        "not_applicable" if status in ("not_run", "unsupported", "skipped") else "invalid")
    result["checks"] = report.checks
    result["errors"] = [c["id"] for c in report.errors]
    result["warnings"] = [c["id"] for c in report.warnings]
    if write and run_dir.exists():
        write_json(run_dir / "validation.json", result)
    return result


# ---- campaign level -------------------------------------------------------------------------------------------
def cell_key(manifest: dict) -> tuple:
    return (manifest.get("cell_id"), manifest.get("candidate", {}).get("candidate_id"), manifest.get("block_id"))


def validate_campaign(expected_cells: list[dict], runs: list[tuple[dict, dict]]) -> dict:
    """expected_cells: plan cells {cell_id, candidate_id, block_id, ...}. runs: (manifest, validation) pairs.
    Every planned cell gets a status; duplicate repetitions are rejected, not averaged; runs that are not in the
    plan are listed as unexpected (never silently added)."""
    by_key: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for manifest, validation in runs:
        by_key[cell_key(manifest)].append((manifest, validation))
    ledger = []
    counts = Counter()
    planned = set()
    for cell in expected_cells:
        key = (cell["cell_id"], cell["candidate_id"], cell["block_id"])
        planned.add(key)
        counts["expected"] += 1
        found = by_key.get(key, [])
        entry = dict(cell)
        if not found:
            entry["status"] = "not_run"
        elif len(found) > 1:
            entry["status"] = "invalid"
            entry["reason"] = f"duplicate_repetition: {[m.get('run_id') for m, _ in found]}"
            counts["attempted"] += 1
        else:
            manifest, validation = found[0]
            entry["run_id"] = manifest.get("run_id")
            entry["status"] = validation.get("status", "invalid")
            if entry["status"] not in ("not_run", "unsupported", "skipped"):
                counts["attempted"] += 1
            if entry["status"] in ("measured", "invalid", "smoke_only"):
                counts["completed"] += 1
                counts["parsed"] += 1
            if entry["status"] == "measured":
                counts["accepted"] += 1
            entry["errors"] = validation.get("errors", [])
        ledger.append(entry)
    unexpected = [m.get("run_id") for key, items in by_key.items() if key not in planned for m, _ in items]
    return {"counts": {k: counts[k] for k in ("expected", "attempted", "completed", "parsed", "accepted")},
            "cells": ledger, "unexpected_runs": unexpected,
            "status_totals": dict(Counter(e["status"] for e in ledger))}


def detect_copied_rows(rows: list[dict]) -> list[dict]:
    """Historical rows whose raw source content hash appears in more than one experiment: the same measurement
    carried into another round. They are one observation, never independent repetitions."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        digest = row.get("source_artifact_sha256")
        if digest:
            by_hash[digest].append(row)
    out = []
    for digest, items in by_hash.items():
        experiments = {r.get("origin_experiment_id") for r in items}
        if len(experiments) > 1:
            out.append({"source_artifact_sha256": digest, "experiments": sorted(map(str, experiments)),
                        "rows": [r.get("source_row_id") for r in items]})
    return out
