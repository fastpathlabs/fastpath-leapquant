"""compute-sanitizer evidence: sanitizer_results.json records built from complete logs.

An exit code alone does not identify a memory violation, so every record carries the tool, the exact command and
options, the device/build identity, the exit status, the error summary, a classification of the reported errors, the
complete log's hash, and the expected coverage. Classification:
  clean              the tool's summary line reports 0 errors / 0 hazards
  api_errors_only    every reported error is a CUDA API error (e.g. Triton's launcher calling cuFuncSetAttribute),
                     no memory-access or race report
  violations         memory-access errors, uninitialised reads or race hazards reported
  interrupted        no summary line (the run did not finish)
  missing_log        no log file
  python -m lqh.sanitizer historical RESULTS_ROOT --out sanitizer_results.json
  python -m lqh.sanitizer record LOG --tool memcheck --command "..." --exit 0 [--manifest manifest.txt]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .util import sha256_file, utc_now, write_json

SUMMARY_MEM = re.compile(r"ERROR SUMMARY: (\d+) errors?")
SUMMARY_RACE = re.compile(r"RACECHECK SUMMARY: (\d+) hazards displayed \((\d+) errors, (\d+) warnings\)")
API_ERROR = re.compile(r"Program hit CUDA_ERROR_\w+ \(error \d+\) due to .* on CUDA API call to (\w+)")
ACCESS = re.compile(r"Invalid (__global__|__shared__|__local__) (read|write)|Uninitialized __global__ memory read|"
                    r"Address .* is out of bounds|misaligned")
HAZARD = re.compile(r"Potential (RAW|WAR|WAW) hazard|Race reported")


def classify(text: str | None, tool: str) -> dict:
    if text is None:
        return {"classification": "missing_log", "errors": None}
    out: dict = {}
    race = SUMMARY_RACE.search(text)
    mem = SUMMARY_MEM.search(text)
    api_calls = API_ERROR.findall(text)
    access = len(ACCESS.findall(text))
    hazards = len(HAZARD.findall(text))
    out.update(api_errors=len(api_calls), api_calls=sorted(set(api_calls)), access_reports=access,
               hazard_reports=hazards)
    if tool == "racecheck":
        if not race:
            return {**out, "classification": "interrupted", "errors": None, "summary": None}
        errors, warnings = int(race.group(2)), int(race.group(3))
        out.update(summary=race.group(0), errors=errors, warnings=warnings, hazards_displayed=int(race.group(1)))
        out["classification"] = "clean" if errors == 0 and warnings == 0 and int(race.group(1)) == 0 else "violations"
        return out
    if not mem:
        return {**out, "classification": "interrupted", "errors": None, "summary": None}
    errors = int(mem.group(1))
    out.update(summary=mem.group(0), errors=errors)
    if errors == 0:
        out["classification"] = "clean"
    elif access == 0 and len(api_calls) == errors:
        out["classification"] = "api_errors_only"
    else:
        out["classification"] = "violations"
    return out


def record(log: Path, tool: str, command: str | None, exit_status: int | None, manifest: Path | None,
           coverage: str, evidence_kind: str) -> dict:
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else None
    build = {}
    if manifest is not None and manifest.exists():
        for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            if key.strip() in ("gpu", "fast_cuda_source_hash", "libleapquant_fastfold.so", "libleapquant_cuda.so",
                               "build_dir", "time_utc", "environment", "nvcc"):
                build[key.strip()] = value.strip()
    version = None
    if text:
        found = re.search(r"COMPUTE-SANITIZER.*?(\d{4}\.\d+\.\d+)?", text[:400])
        version = found.group(1) if found and found.group(1) else None
    return {"schema": "lqh.sanitizer_result.v1", "log": str(log), "log_sha256": sha256_file(log) if log.exists() else None,
            "tool": tool, "tool_version": version, "command": command, "exit_status": exit_status,
            "build": build or None, "expected_coverage": coverage, "evidence_kind": evidence_kind,
            **classify(text, tool), "recorded_utc": utc_now(),
            "note": "complete stdout/stderr is the log file itself (hash above)"}


HISTORICAL = [
    ("fast_fold_v1/F1a.log", "memcheck", "compute-sanitizer --tool memcheck --error-exitcode 1 python3 kernel/fast_fold_test.py --quick", "CUDA fold v1 + store, fast_fold_test quick cases"),
    ("fast_fold_v1/F1a_rerun.log", "memcheck", "compute-sanitizer --tool memcheck --error-exitcode 1 python3 kernel/fast_fold_test.py --quick", "CUDA fold v1 + store (the run reported as 'exit code 1')"),
    ("fast_fold_v1/F1a_memcheck2.log", "memcheck", "compute-sanitizer --tool memcheck (see runbook/fast_fold_check.sh)", "CUDA fold v1"),
    ("fast_fold_v1/F1b.log", "racecheck", "compute-sanitizer --tool racecheck --racecheck-report all --kernel-name kns=fast_fold_kernel --error-exitcode 1 python3 kernel/fast_fold_test.py --quick", "CUDA fold v1 shared memory"),
    ("fast_fold_v1/F1b_rerun.log", "racecheck", "as F1b", "CUDA fold v1 shared memory"),
    ("fast_fold_v3/memcheck_v3_x0.txt", "memcheck", "compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 1 python3 kernel/fast_fold_test.py --quick", "CUDA fold v3"),
    ("fast_fold_v3/racecheck_v3_x0.txt", "racecheck", "compute-sanitizer --tool racecheck --racecheck-report all --kernel-name kns=fast_fold_kernel", "CUDA fold v3"),
    ("fast_fold_v4/memcheck.txt", "memcheck", "compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 1 python3 kernel/fast_fold_test.py --quick", "CUDA fold v4"),
    ("fast_fold_v4/racecheck.txt", "racecheck", "compute-sanitizer --tool racecheck --racecheck-report all --kernel-name kns=fast_fold_kernel", "CUDA fold v4"),
    ("fast_fold_v6/memcheck.txt", "memcheck", "compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 1 python3 kernel/fast_fold_test.py --quick", "CUDA fold v6 + CUDA store"),
    ("fast_fold_v6/racecheck.txt", "racecheck", "compute-sanitizer --tool racecheck --racecheck-report all", "CUDA fold v6 + CUDA store"),
    ("fast_decode_v5/memcheck.txt", "memcheck", "compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 1 python3 kernel/bench_triton.py (small)", "CUDA decode v5"),
    ("fast_decode_v7/memcheck.txt", "memcheck", "as v5", "CUDA decode v7"),
    ("fast_decode_v7/racecheck.txt", "racecheck", "compute-sanitizer --tool racecheck --racecheck-report all --kernel-name kns=decode_kernel", "CUDA decode v7"),
    ("fast_decode_v7/racecheck_gs.txt", "racecheck", "compute-sanitizer --tool racecheck (Gram-Schmidt fold)", "CUDA fold, batched Gram-Schmidt"),
    ("fast_decode_v8/memcheck.txt", "memcheck", "as v5", "CUDA decode v8"),
    ("fast_decode_v8/racecheck.txt", "racecheck", "compute-sanitizer --tool racecheck --racecheck-report all --kernel-name kns=decode8_kernel", "CUDA decode v8"),
    ("fast_decode_v9/memcheck.txt", "memcheck", "LEAPQUANT_CUDA_DECODE_VARIANT=9 compute-sanitizer --tool memcheck --report-api-errors no --error-exitcode 1 python3 kernel/bench_triton.py --batch-sizes 4 --windows 1 --no-exact-times --repeats 2 --fold-backend cuda --decode-backend cuda", "CUDA decode v9 (batch 4 only)"),
    ("fast_decode_v9/racecheck.txt", "racecheck", "LEAPQUANT_CUDA_DECODE_VARIANT=9 compute-sanitizer --tool racecheck --racecheck-report all --kernel-name kns=decode8_kernel --error-exitcode 1 python3 kernel/bench_triton.py --batch-sizes 4 ...", "CUDA decode v9 (batch 4 only)"),
]


def historical(root: Path) -> dict:
    records = []
    for rel, tool, command, coverage in HISTORICAL:
        log = root / rel
        manifest = log.parent / "manifest.txt"
        rec = record(log, tool, command, None, manifest if manifest.exists() else None, coverage, "measured_raw")
        rec["exit_status_note"] = "not recorded in the log; the STATUS line of the run is the only exit evidence"
        records.append(rec)
    resolved = next((r for r in records if r["log"].endswith("F1a_rerun.log")), None)
    return {"schema": "lqh.sanitizer_results.v1", "computed_utc": utc_now(), "records": records,
            "claims": [{"claim": "a memory-check run exited with code 1 (second review)",
                        "resolution": (f"log found: {resolved['log']}: {resolved.get('summary')}; classification "
                                       f"{resolved['classification']} ({resolved.get('api_errors')} CUDA API errors on "
                                       f"{', '.join(resolved.get('api_calls') or [])}; {resolved.get('access_reports')} "
                                       f"memory-access reports). The exit code came from API errors raised by the Triton "
                                       f"launcher, not from a reported memory violation; the run is 'api_errors_only', "
                                       f"not 'clean'.") if resolved else "log not found: unsupported history"},
                       {"claim": "sanitizers clean on the final kernels (fold, store, decode, export) at batch 1, 32, 512",
                        "resolution": "historical clean logs cover fold v6 and decode v9 at small batches only; the current "
                                      "checks of the final binary are queued (runbook/harness_sanitizers.sh)"}]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.sanitizer")
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("historical")
    h.add_argument("root")
    h.add_argument("--out", required=True)
    r = sub.add_parser("record")
    r.add_argument("log")
    r.add_argument("--tool", required=True)
    r.add_argument("--command", default=None)
    r.add_argument("--exit", type=int, default=None)
    r.add_argument("--manifest", default=None)
    r.add_argument("--coverage", default="")
    r.add_argument("--append", required=True, help="sanitizer_results.json to append to")
    args = parser.parse_args(argv)
    if args.cmd == "historical":
        write_json(Path(args.out), historical(Path(args.root)))
        return 0
    from .util import read_json
    path = Path(args.append)
    doc = read_json(path) if path.exists() else {"schema": "lqh.sanitizer_results.v1", "records": []}
    doc["records"].append(record(Path(args.log), args.tool, args.command, args.exit,
                                 Path(args.manifest) if args.manifest else None, args.coverage, "measured_raw"))
    write_json(path, doc)
    return 0 if doc["records"][-1]["classification"] in ("clean",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
