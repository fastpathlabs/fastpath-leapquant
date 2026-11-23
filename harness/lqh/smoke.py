"""End-to-end smoke on the deterministic mock server: plan -> evidence -> validation -> metrics -> report -> checker.

  python -m lqh smoke [--out DIR] [--port 31950]

Harness behaviour only. The runs are tagged mock / smoke_only and can never enter a performance claim.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from .checker import check_bundle
from .report import build
from .runner import run_plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh smoke")
    parser.add_argument("--out", default=None)
    parser.add_argument("--port", type=int, default=31950)
    args = parser.parse_args(argv)
    root = Path(args.out or tempfile.mkdtemp(prefix="lqh_smoke_")).resolve()
    os.environ.setdefault("LQH_SKIP_BUILD_MANIFEST", "1")
    run_plan("smoke_mock", root / "results", args.port)
    bundle = root / "bundle"
    result = build([root / "results"], bundle, title="Harness smoke (mock server)")
    check = check_bundle(bundle)
    (bundle / "check_report.json").write_text(json.dumps(check, indent=1))
    print(json.dumps({"bundle": str(bundle), "runs": result["runs"], "statuses": result["statuses"],
                      "checked": check["checked"], "mismatches": len(check["failed"]),
                      "uncovered": len(check["uncovered"]), "hash_failures": len(check["hash_failures"]),
                      "ok": check["ok"]}))
    for failure in check["failed"][:10]:
        print("MISMATCH", failure)
    for missing in check["uncovered"][:10]:
        print("UNCOVERED", missing)
    all_valid = set(result["statuses"]) == {"smoke_only"}
    if not all_valid:
        print("SMOKE FAIL: not every mock run is valid:", result["statuses"])
    return 0 if check["ok"] and all_valid else 1
