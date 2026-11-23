"""python -m lqh.eligibility RESULTS_DIR... [--out FILE]

Separates what one word, "measured", used to cover (multi-GPU benchmark plan, 15 Nov 2026, section 4):
  execution_status       the validator's status (measured / invalid / crashed / not_run ...)
  protocol_status        conforming, divergent (the run left the profile's recipe) or unverified (evidence missing)
  service_eligible       the operating point meets the service policy: every run of the SLO's request predicate
                         and global constraints, plus attainment >= 95% of the attempted measured requests and
                         completion >= 99% (the prompt's suggested internal gates; not an industry standard)
  quality_eligible       the exact configuration passed the frozen quality rule (none has: qualification pending)
  publication_eligible   all of the above, and the comparison has at least 5 paired blocks (checked per claim)
One row per run directory, from its validation.json and metrics.json (no re-measurement). The independent checker
does not yet reproduce these rows; they are a derived diagnostic until it does.
"""
import argparse
import json
import sys
from pathlib import Path

ATTAINMENT_MIN = 0.95
COMPLETION_MIN = 0.99
QUALITY_STATUS = "pending: the 87-question screen (job 121) is inconclusive at the 1-point margin"


def _value(metrics: dict, name: str, field: str = "value"):
    entry = metrics.get(name)
    return entry.get(field) if isinstance(entry, dict) else None


def offered_logical(metrics: dict) -> tuple[int | None, int]:
    """(offered logical requests, retries): the declared cohort size, or the logical requests attempted when larger or
    when no size is declared (sustained lane); a retry is the same logical request."""
    attempted = _value(metrics, "requests_attempted") or 0
    retries = int(_value(metrics, "retries") or 0)
    expected = _value(metrics, "requests_expected")
    return (max(int(expected or 0), int(attempted) - retries) or None), retries


def service_rule(metrics: dict, slo: dict | None) -> tuple[bool | None, list[str]]:
    """The service gate of one run (None without an SLO). Quality is a separate gate. The independent checker
    re-derives the same rule in its own code (lqh.checker).
    slo-interactive-v2 (global_constraints.denominator = offered_logical_requests): joint attainment >= attainment_min
    and complete success >= complete_success_min, both over offered logical requests, and at least one qualifying
    request; the p99 token gap is recorded separately, not gated.
    v1: the SLO's global error and p99 token-gap limits, at least one qualifying request, attainment >= ATTAINMENT_MIN
    and completion >= COMPLETION_MIN over attempted measured requests."""
    if not slo:
        return None, ["no SLO"]
    reasons = []
    limits = slo.get("global_constraints") or {}
    if limits.get("denominator") == "offered_logical_requests":
        offered, retries = offered_logical(metrics)
        qualifying = _value(metrics, "slo_qualifying_requests") or 0
        succeeded = _value(metrics, "requests_succeeded") or 0
        if retries:
            reasons.append(f"{retries} retries: logical-request counts needed (attempt counts would overstate)")
        for name, count, label in (("attainment_min", qualifying, "joint attainment"),
                                   ("complete_success_min", succeeded, "complete success")):
            limit = limits.get(name)
            share = count / offered if offered else None
            if limit is not None and (share is None or share < limit):
                reasons.append(f"{label} {share if share is None else round(share, 4)} < {limit} "
                               f"({count} of {offered} offered)")
        if limits.get("error_rate_max") is not None:
            error_rate = _value(metrics, "error_rate")
            if error_rate is None or error_rate > limits["error_rate_max"]:
                reasons.append(f"error rate {error_rate} > {limits['error_rate_max']}")
        if not qualifying > 0:
            reasons.append("no qualifying request")
        return not reasons, reasons
    error_rate = _value(metrics, "error_rate")
    if limits.get("error_rate_max") is not None and (error_rate is None or error_rate > limits["error_rate_max"]):
        reasons.append(f"error rate {error_rate} > {limits['error_rate_max']}")
    if limits.get("p99_token_gap_ms_max") is not None:
        gap = _value(metrics, limits.get("p99_token_gap_metric", "itl_token_ms"), "p99")
        if gap is None or gap > limits["p99_token_gap_ms_max"]:
            reasons.append(f"p99 token gap {gap if gap is None else round(gap, 1)} ms > {limits['p99_token_gap_ms_max']}")
    if not (_value(metrics, "slo_qualifying_requests") or 0) > 0:
        reasons.append("no qualifying request")
    attainment = _value(metrics, "slo_qualifying_fraction")
    if attainment is None or attainment < ATTAINMENT_MIN:
        reasons.append(f"attainment {attainment if attainment is None else round(attainment, 4)} < {ATTAINMENT_MIN}")
    completion = _value(metrics, "completion_rate")
    if completion is None or completion < COMPLETION_MIN:
        reasons.append(f"completion {completion if completion is None else round(completion, 4)} < {COMPLETION_MIN}")
    return not reasons, reasons


def classify(run_dir: Path) -> dict:
    row = {"run_dir": str(run_dir)}
    validation = json.loads((run_dir / "validation.json").read_text()) if (run_dir / "validation.json").exists() else {}
    doc = json.loads((run_dir / "metrics.json").read_text()) if (run_dir / "metrics.json").exists() else {}
    manifest = json.loads((run_dir / "run_manifest.json").read_text()) if (run_dir / "run_manifest.json").exists() else {}
    metrics = doc.get("metrics") or {}
    row.update(run_id=manifest.get("run_id"), plan_id=manifest.get("plan_id"), cell_id=manifest.get("cell_id"),
               candidate=(manifest.get("candidate") or {}).get("candidate_id"), block=manifest.get("block_id"))
    status = validation.get("status") or doc.get("status") or "missing"
    row["execution_status"] = status
    # Protocol: divergent when the validator found a recipe break, unverified when evidence is missing.
    divergent, unverified = [], []
    for check in validation.get("checks", []):
        if check.get("ok"):
            continue
        if check["id"] in ("V053_failure_storm", "V054_client_resources", "V052_server_reachable"):
            divergent.append(check["id"])
        elif check["id"].startswith("V035") or check["id"].startswith("V05"):
            unverified.append(check["id"])
    warm = validation.get("native_warmups")
    if (manifest.get("workload") or {}).get("lane") == "finite" and (run_dir / "native").exists() and not warm:
        unverified.append("native warmups not checked (no client log)")
    row["protocol_status"] = ("divergent: " + ", ".join(divergent) if divergent else
                              "unverified: " + ", ".join(unverified) if unverified else "conforming")
    # Service: the shared rule (the report shows it; the checker re-derives it).
    reasons = [] if status == "measured" else [f"execution {status}"]
    _, rule_reasons = service_rule(metrics, (doc.get("slo") or {}).get("definition"))
    reasons += rule_reasons
    attainment = _value(metrics, "slo_qualifying_fraction")
    completion = _value(metrics, "completion_rate")
    row["service_eligible"] = not reasons
    row["service_reasons"] = "; ".join(reasons)
    row["quality_eligible"] = False
    row["quality_reason"] = QUALITY_STATUS
    row["publication_eligible"] = False
    row["publication_reason"] = "quality not qualified" + ("" if row["service_eligible"] else "; not service-eligible")
    headline = doc.get("headline") or {}
    row["output_tps"] = headline.get("deployment_output_tps")
    row["goodput_tps"] = _value(metrics, "output_goodput_tps")
    row["attainment"] = attainment
    row["completion"] = completion
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--out", default=None, help="write the rows as JSON lines here")
    args = parser.parse_args(argv)
    rows = []
    for root in args.roots:
        for manifest in sorted(Path(root).rglob("run_manifest.json")):
            rows.append(classify(manifest.parent))
    if args.out:
        Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        tps = f"{r['output_tps']:.0f}" if isinstance(r["output_tps"], (int, float)) else "-"
        print(f"{r['plan_id']}\t{r['cell_id']}\t{r['candidate']}\tb{r['block']}\t{r['execution_status']}\t"
              f"{r['protocol_status']}\tservice={'yes' if r['service_eligible'] else 'no'}\t{tps}\t{r['service_reasons']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
