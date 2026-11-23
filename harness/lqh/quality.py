"""Quality qualification of the 16-bit control and LeapQuant (test-harness prompt rev. 3, section 13).

  python -m lqh.quality RESULTS_DIR [--campaign harness/config/campaign.json] [--reference fp32]

Reads quality_items_<arm>.jsonl (written by eval/accuracy_bench.py --items-out), joins every arm with the reference
arm by sample id (task, question, sample, seed) and prompt hash, and decides per task:
  - several samples per question: question-clustered paired difference, t interval over questions;
    when every question has the same share in both arms (zero variance) the interval falls back to Newcombe's
    hybrid interval on the pooled paired answers, so it is never the degenerate 0..0;
  - one sample per question: Newcombe's hybrid interval (method 10);
  - verdict PASS / INCONCLUSIVE / FAIL at the declared margin (noninferiority), INVALID for disjoint or duplicated
    cohorts or a changed prompt.
Capped answers (finish_reason length) count as wrong; their rate is reported per arm and task. The decision rule is
read from campaign.json (frozen before the run) and applied as written; a task that was not run makes the rule's
"every task" condition unmet.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .stats import newcombe_paired, noninferiority, t_quantile
from .util import harness_root, read_json, read_jsonl, utc_now, write_json


def load_arm(path: Path) -> dict[str, list[dict]]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in read_jsonl(path):
        by_task[item["dataset"]].append(item)
    return by_task


def scored(item: dict) -> dict:
    """The fixed-budget primary outcome: a capped answer (finish_reason length) is wrong whatever it contains (a cut
    answer can still hold a number the parser picks up, or code that happens to run)."""
    return {**item, "correct": bool(item["correct"]) and not item.get("capped")}


def compare_task(reference: list[dict], arm: list[dict], margin_points: float) -> dict:
    ref = {i["sample_id"]: scored(i) for i in reference}
    ours = {i["sample_id"]: scored(i) for i in arm}
    out: dict = {"n_reference": len(reference), "n_arm": len(arm)}
    if len(ref) != len(reference) or len(ours) != len(arm):
        return {**out, "verdict": "INVALID", "reason": "duplicated sample ids"}
    shared = sorted(set(ref) & set(ours))
    if len(shared) != len(ref) or len(shared) != len(ours):
        return {**out, "verdict": "INVALID", "reason": f"cohorts differ: {len(shared)} shared of {len(ref)} / {len(ours)}"}
    changed = [s for s in shared if ref[s].get("prompt_sha256") != ours[s].get("prompt_sha256")]
    if changed:
        return {**out, "verdict": "INVALID", "reason": f"{len(changed)} prompts differ between the arms"}
    failed = sum(1 for s in shared if ours[s].get("error"))
    acc_arm = sum(1 for s in shared if ours[s]["correct"]) / len(shared)
    acc_ref = sum(1 for s in shared if ref[s]["correct"]) / len(shared)
    out.update(samples=len(shared), accuracy_arm=acc_arm, accuracy_reference=acc_ref, difference=acc_arm - acc_ref,
               cap_rate_arm=sum(1 for s in shared if ours[s].get("capped")) / len(shared),
               cap_rate_reference=sum(1 for s in shared if ref[s].get("capped")) / len(shared),
               failed_requests_arm=failed)
    per_question: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for s in shared:
        per_question[ours[s]["question"]].append((ours[s]["correct"], ref[s]["correct"]))
    both = sum(1 for s in shared if ours[s]["correct"] and ref[s]["correct"])
    only_arm = sum(1 for s in shared if ours[s]["correct"] and not ref[s]["correct"])
    only_ref = sum(1 for s in shared if ref[s]["correct"] and not ours[s]["correct"])
    neither = len(shared) - both - only_arm - only_ref
    k = max(len(v) for v in per_question.values())
    if k > 1:
        # The repository's paper-lane definition (eval/accuracy_bench.py clustered_difference_interval): per question
        # the difference of the shares answered correctly; a t interval over questions; the variance floored by the
        # binomial noise of the k samples (pooled share with one success and one failure added), so questions that
        # happen to agree never give a zero-width interval.
        diffs, floors = [], []
        for v in per_question.values():
            mine_ok, ref_ok = [a for a, _ in v], [b for _, b in v]
            diffs.append(sum(mine_ok) / len(mine_ok) - sum(ref_ok) / len(ref_ok))
            pooled = (sum(mine_ok) + sum(ref_ok) + 1) / (len(mine_ok) + len(ref_ok) + 2)
            floors.append(pooled * (1 - pooled) * (1 / len(mine_ok) + 1 / len(ref_ok)))
        n = len(diffs)
        mean = sum(diffs) / n
        variance = sum((d - mean) ** 2 for d in diffs) / max(n - 1, 1)
        variance = max(variance, sum(floors) / n)
        half = t_quantile(0.975, max(n - 1, 1)) * math.sqrt(variance / n)
        low, high = mean - half, mean + half
        method = f"question-clustered t interval with a binomial variance floor, {n} questions x {k} samples"
    else:
        result = newcombe_paired(both, only_arm, only_ref, neither)
        low, high, method = result["low"], result["high"], result["method"]
    verdict = noninferiority(low, high, margin_points / 100.0)
    if failed:
        verdict = "INVALID"
        out["reason"] = f"{failed} failed requests (an execution failure is never scored silently)"
    out.update(low=low, high=high, method=method, discordant=only_arm + only_ref, verdict=verdict,
               margin_points=margin_points)
    return out


def decide(results: Path, campaign: dict, reference: str = "fp32") -> dict:
    rule = campaign["quality_qualification"]
    margin = float(rule.get("margin_points", 1.0))
    tasks_declared = list(rule["tasks"].keys())
    arms = {p.stem.replace("quality_items_", ""): load_arm(p) for p in sorted(results.glob("quality_items_*.jsonl"))}
    doc: dict = {"schema": "lqh.quality_decision.v1", "computed_utc": utc_now(), "results": str(results),
                 "reference_arm": reference, "margin_points": margin, "decision_rule": rule["decision_rule_frozen"],
                 "tasks_declared": tasks_declared, "arms": {}}
    if reference not in arms:
        doc["decision"] = {"status": "INCOMPLETE", "reason": f"reference arm {reference} has no items"}
        return doc
    for name, by_task in arms.items():
        if name == reference:
            continue
        doc["arms"][name] = {task: compare_task(arms[reference].get(task, []), items, margin)
                             for task, items in by_task.items()}
    ref_caps = {task: sum(1 for i in items if i.get("capped")) / len(items) for task, items in arms[reference].items()}
    doc["reference_cap_rate"] = ref_caps

    def every_task_pass(arm: str) -> tuple[bool, list[str]]:
        problems = []
        for task in tasks_declared:
            result = doc["arms"].get(arm, {}).get(task)
            if result is None:
                problems.append(f"{task}: not run")
            elif result["verdict"] != "PASS":
                problems.append(f"{task}: {result['verdict']}")
        return not problems, problems

    window_ok, window_problems = every_task_pass("bf16_window")
    lq_ok, lq_problems = every_task_pass("leapquant")
    doc["decision"] = {
        "bf16_window_primary_product_baseline": window_ok,
        "bf16_window_reasons": window_problems,
        "comparator_for_cost_claims": "bf16_window" if window_ok else "fp32 (bf16_window ineligible as the primary "
                                                                      "baseline; its speed stays visible)",
        "leapquant_quality_qualified": lq_ok,
        "leapquant_reasons": lq_problems,
        "leapquant_claim_status": "quality-qualified" if lq_ok else "experimental: raw performance only, no "
                                                                   "quality-qualified cost claim",
        "cap_rate_target": "under 5% of answers cut",
        "cap_rate_exceeded": sorted({f"{arm}:{task}" for arm, tasks in doc["arms"].items() for task, r in tasks.items()
                                     if (r.get("cap_rate_arm") or 0) >= 0.05}
                                    | {f"{reference}:{t}" for t, c in ref_caps.items() if c >= 0.05}),
    }
    return doc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lqh.quality")
    parser.add_argument("results")
    parser.add_argument("--campaign", default=str(harness_root() / "config" / "campaign.json"))
    parser.add_argument("--reference", default="fp32")
    args = parser.parse_args(argv)
    doc = decide(Path(args.results), read_json(args.campaign), args.reference)
    write_json(Path(args.results) / "quality_decision.json", doc)
    for arm, tasks in doc.get("arms", {}).items():
        for task, r in tasks.items():
            print(f"{arm:14s} {task:14s} {r.get('verdict'):13s} acc {r.get('accuracy_arm', float('nan')):.3f} vs "
                  f"{r.get('accuracy_reference', float('nan')):.3f}  diff {r.get('difference', float('nan')):+.3f} "
                  f"[{r.get('low', float('nan')):+.3f}, {r.get('high', float('nan')):+.3f}]  cap {r.get('cap_rate_arm', 0):.1%}"
                  f"  {r.get('reason', '')}")
    print(json.dumps(doc["decision"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
