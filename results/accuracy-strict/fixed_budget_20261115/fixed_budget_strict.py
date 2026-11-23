"""Re-score the strict paper lane under the frozen fixed-budget rule (a capped answer is wrong, whatever it holds)
and redo the bench's own paired comparisons. Reads the saved runs; writes nothing but stdout.
usage: python fixed_budget_strict.py <tree with eval/accuracy_bench.py> <leapquant.json> <bf16_window.json> <fp32_window.json>"""
import contextlib
import io
import json
import sys

sys.path.insert(0, sys.argv[1] + "/eval")
import accuracy_bench as bench  # noqa: E402

names = {"leapquant": sys.argv[2], "bf16_window": sys.argv[3], "fp32_window": sys.argv[4]}
runs = {}
for name, path in names.items():
    with open(path, encoding="utf-8") as handle:
        run = json.load(handle)
    for task, items in run["tasks"].items():
        changed = 0
        for item in items:
            capped = bool(item.get("truncated")) or item.get("finish_reason") == "length"
            if capped and item["correct"]:
                item["correct"] = False
                changed += 1
            item["capped"] = capped
        capped_count = sum(item["capped"] for item in items)
        print(f"{name:<12} {task:<9} capped {capped_count:>3}/{len(items)}  capped-but-scored-correct (now wrong): {changed}")
    runs[name] = run

for name in names:
    print(f"\n=== {name} ({names[name]}), capped = wrong")
    bench.summarise(runs[name])

for ours, theirs in (("leapquant", "fp32_window"), ("bf16_window", "fp32_window"), ("leapquant", "bf16_window")):
    print(f"\n=== {ours} against {theirs}, capped = wrong")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        verdict = bench.compare(runs[ours], runs[theirs], theirs, 1.0)
    print(buffer.getvalue().rstrip())
