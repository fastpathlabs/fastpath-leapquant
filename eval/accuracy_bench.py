#!/usr/bin/env python3
"""
accuracy_bench.py

Task T12 of the implementation plan: answer accuracy with the model in the loop.
Sends benchmark questions to a running server (any OpenAI-compatible endpoint),
scores the answers and, with --compare, puts two runs side by side.

    # one run per server configuration
    python3 eval/accuracy_bench.py --port 30000 --out results/acc_fp32.json
    python3 eval/accuracy_bench.py --port 30000 --out results/acc_leapquant.json --compare results/acc_fp32.json

    # check this script itself, without a model (it starts a small fake server)
    python3 eval/accuracy_bench.py --self-test

Tasks (three of the five the paper reports; these three can be scored by rule)
    gsm8k      openai/gsm8k, test split, 1,319 word problems, numeric answer
    aime2025   yentinglin/aime_2025, 30 competition problems, integer answer
    mmlu_pro   TIGER-Lab/MMLU-Pro, test split, multiple choice with up to 10
               options. 12,032 questions; --limit picks an evenly spaced sample.
    Not included: LiveCodeBench (needs a code sandbox) and GPQA Diamond (the
    dataset is gated).

The questions are fetched with the `datasets` package (pip install datasets)
and cached in eval/cache/. Without that package the script falls back to the
Hugging Face dataset viewer, page by page, which is slow and rate-limited. The model is asked to reason and to end with "ANSWER: <answer>".
Temperature is 0.

How to read a comparison
    The plan's criterion is "within run-to-run noise of the fp32 run". Two runs
    answer the same questions, so the comparison is paired: only the questions
    on which the two runs disagree carry information. The script prints the
    accuracy of both runs, the number of questions only one of them got right,
    and the difference with its standard error. A difference within two
    standard errors is noise. To see the noise of the server itself, run the
    fp32 server twice and compare those two runs the same way.

Uses only the Python standard library, plus `datasets` for the first download.
"""

import argparse
import concurrent.futures
import http.server
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.parse
import urllib.request

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
VIEWER = "https://datasets-server.huggingface.co/rows"
INSTRUCTION = ("Think step by step. When you are done, write the final answer on its own last line "
               "in the form\nANSWER: <answer>\n")
LETTERS = "ABCDEFGHIJ"


# ---- datasets ------------------------------------------------------------------------

def fetch_rows(dataset, config, split):
    """All rows of one split, through the dataset viewer (100 rows per request), cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{dataset.replace('/', '__')}__{config}__{split}.json")
    if os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as handle:
            return json.load(handle)
    try:
        # Preferred: the `datasets` library fetches one file per split.
        from datasets import load_dataset
        rows = [dict(row) for row in load_dataset(dataset, config, split=split)]
        with open(cache_file, "w", encoding="utf-8") as handle:
            json.dump(rows, handle)
        return rows
    except ImportError:
        print(f"  (the `datasets` package is not installed; fetching {dataset} page by page, which is slow)", file=sys.stderr)
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": offset, "length": 100})
        page = None
        for attempt in range(8):     # the viewer answers 5xx or 429 now and then: wait and try again
            try:
                with urllib.request.urlopen(f"{VIEWER}?{query}", timeout=120) as response:
                    page = json.load(response)
                break
            except Exception as problem:
                if attempt == 7:
                    sys.exit(f"\nerror: cannot fetch {dataset} {config}/{split} at row {offset}: {problem}")
                time.sleep(15.0 * (attempt + 1))
        total = page["num_rows_total"]
        rows.extend(item["row"] for item in page["rows"])
        offset += 100
        time.sleep(1.0)              # stay under the viewer's rate limit
        print(f"\r  fetching {dataset} {config}/{split}: {min(offset, total)}/{total}", end="", file=sys.stderr)
    print(file=sys.stderr)
    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    return rows


def evenly_spaced(items, limit):
    if limit is None or limit >= len(items):
        return items
    return [items[(index * len(items)) // limit] for index in range(limit)]


def load_task(name, limit):
    """Returns a list of {id, prompt, gold, kind} for one task."""
    questions = []
    if name == "gsm8k":
        for index, row in enumerate(fetch_rows("openai/gsm8k", "main", "test")):
            gold = row["answer"].split("####")[-1].strip()
            questions.append({"id": f"gsm8k-{index}", "kind": "number", "gold": gold,
                              "prompt": f"{row['question']}\n\n{INSTRUCTION}The answer is a number."})
    elif name == "aime2025":
        for index, row in enumerate(fetch_rows("yentinglin/aime_2025", "default", "train")):
            questions.append({"id": f"aime2025-{index}", "kind": "number", "gold": str(row["answer"]).strip(),
                              "prompt": f"{row['problem']}\n\n{INSTRUCTION}The answer is an integer."})
    elif name == "aime2024":
        for index, row in enumerate(fetch_rows("HuggingFaceH4/aime_2024", "default", "train")):
            questions.append({"id": f"aime2024-{index}", "kind": "number", "gold": str(row["answer"]).strip(),
                              "prompt": f"{row['problem']}\n\n{INSTRUCTION}The answer is an integer."})
    elif name == "aime2026":
        # MathArena/aime_2026: AIME 2026 I and II, 30 problems (the paper's AIME task, Table 1).
        for row in fetch_rows("MathArena/aime_2026", "default", "train"):
            questions.append({"id": f"aime2026-{row['problem_idx']}", "kind": "number", "gold": str(row["answer"]).strip(),
                              "prompt": f"{row['problem']}\n\n{INSTRUCTION}The answer is an integer."})
    elif name == "gpqa_diamond":
        # Idavidrein/gpqa, config gpqa_diamond (198 questions). Gated: the HF account must accept its terms.
        # The four options are shuffled by a fixed hash of the record id, the same for every run and arm.
        import hashlib
        for row in fetch_rows("Idavidrein/gpqa", "gpqa_diamond", "train"):
            options = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
            options = [str(o).strip() for o in options]
            record = str(row.get("Record ID") or row["Question"][:64])
            order = sorted(range(4), key=lambda i: hashlib.sha256(f"{record}:{i}".encode()).hexdigest())
            shuffled = [options[i] for i in order]
            gold = LETTERS[order.index(0)]
            listing = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(shuffled))
            questions.append({"id": f"gpqa_diamond-{record}", "kind": "letter", "gold": gold,
                              "prompt": f"{row['Question'].strip()}\n\n{listing}\n\n{INSTRUCTION}"
                                        f"The answer is one letter from A to D."})
    elif name == "beyond_aime":
        for index, row in enumerate(fetch_rows("ByteDance-Seed/BeyondAIME", "default", "test")):
            questions.append({"id": f"beyond_aime-{index}", "kind": "number", "gold": str(row["answer"]).strip(),
                              "prompt": f"{row['problem']}\n\n{INSTRUCTION}The answer is an integer."})
    elif name == "livecodebench_v6":
        # livecodebench/code_generation_lite release_v6 in the paper's window (eval/lcb_sandbox.py: data, prompt and
        # the sandboxed scorer). Each question carries what the scorer needs; the private tests stay encoded on the
        # host and are decoded only inside the sandbox.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import lcb_sandbox
        questions = lcb_sandbox.load_questions()
    elif name == "mmlu_pro":
        for row in fetch_rows("TIGER-Lab/MMLU-Pro", "default", "test"):
            options = "\n".join(f"{LETTERS[i]}. {text}" for i, text in enumerate(row["options"]))
            questions.append({"id": f"mmlu_pro-{row['question_id']}", "kind": "letter", "gold": row["answer"].strip(),
                              "prompt": f"{row['question']}\n\n{options}\n\n{INSTRUCTION}"
                                        f"The answer is one letter from A to {LETTERS[len(row['options']) - 1]}."})
    else:
        sys.exit(f"error: unknown task {name}")
    return evenly_spaced(questions, limit)


# ---- scoring -------------------------------------------------------------------------

def extract_answer(text, kind):
    """The model's final answer in a comparable form, or None if none is found."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    marked = re.findall(r"ANSWER\s*:\s*(.+)", text, flags=re.IGNORECASE)
    candidate = marked[-1] if marked else text[-200:]
    candidate = candidate.replace("\\boxed", " ").replace("$", " ").replace("*", " ")
    if kind == "letter":
        match = re.search(r"\b([A-J])\b", candidate.strip().upper())
        return match.group(1) if match else None
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", candidate)
    if not numbers:
        return None
    return normalise_number(numbers[0] if marked else numbers[-1])


def normalise_number(text):
    text = text.replace(",", "").strip().rstrip(".")
    try:
        value = float(text)
    except ValueError:
        return None
    return str(int(value)) if value == int(value) else repr(value)


def is_correct(answer, gold, kind):
    if answer is None:
        return False
    if kind == "letter":
        return answer == gold.upper()
    return answer == normalise_number(gold)


# ---- talking to the server --------------------------------------------------------------

def ask(base_url, model, prompt, max_tokens, timeout, sampling=None, system=None):
    """One chat completion. `sampling` adds temperature, top_p, top_k, seed and chat_template_kwargs
    (thinking mode); without it the request is greedy (temperature 0), as the product lane uses.
    `system`: an optional system message (LiveCodeBench's prompt has one)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    request_body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    request_body.update(sampling or {})
    body = json.dumps(request_body).encode()
    request = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    choice = reply["choices"][0]
    message = choice["message"]
    tokens = (reply.get("usage") or {}).get("completion_tokens", 0)
    # Some servers return the reasoning separately; the answer is in `content`.
    return message.get("content") or message.get("reasoning_content") or "", tokens, choice.get("finish_reason")


def sampling_of(args, sample):
    """Request fields of the paper lane (None for the greedy product lane)."""
    if args.temperature == 0 and args.samples == 1 and args.thinking == "auto":
        return None
    fields = {"temperature": args.temperature, "top_p": args.top_p}
    if args.top_k > 0:
        fields["top_k"] = args.top_k
    if getattr(args, "presence_penalty", 0.0):
        fields["presence_penalty"] = args.presence_penalty
    if args.seed is not None:
        fields["seed"] = args.seed + sample
    if args.thinking != "auto":
        fields["chat_template_kwargs"] = {"enable_thinking": args.thinking == "on"}
    return fields


def run_task(base_url, model, questions, args):
    """One task alone (kept for callers of the earlier interface)."""
    return run_tasks(base_url, model, {"task": questions}, args)["task"]


def score_code_items(items_by_task, questions_by_task, args):
    """Code answers (kind "code", LiveCodeBench) are scored after generation, in the sandbox of eval/lcb_sandbox.py
    (docker, no network, read-only, resource limits): model-written code never runs on the host itself."""
    jobs = []
    for task, items in items_by_task.items():
        by_id = {q["id"]: q for q in questions_by_task[task]}
        for item in items:
            question = by_id[item["question"]]
            if question["kind"] == "code" and item["error"] is None and "_full_text" in item:
                jobs.append((question, item))
    if not jobs:
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import lcb_sandbox
    print(f"  scoring {len(jobs)} code answers in the sandbox ...", file=sys.stderr)

    def one(job):
        question, item = job
        result = lcb_sandbox.score(question, item.pop("_full_text"))
        item["correct"] = bool(result["correct"])
        item["answer"] = result.get("answer")
        item["sandbox"] = {k: v for k, v in result.items() if k not in ("correct", "answer")}

    with concurrent.futures.ThreadPoolExecutor(max_workers=getattr(args, "sandbox_workers", 8)) as pool:
        list(pool.map(one, jobs))
    for items in items_by_task.values():
        for item in items:
            item.pop("_full_text", None)


def run_tasks(base_url, model, questions_by_task, args):
    """Every task's items through ONE pool of --concurrency requests, in the order the tasks are given (put the tasks
    with long answers first), so the server stays full until the last answers instead of draining after each task.
    One item per (question, sample). With --samples k > 1 the ids are '<question>#s<j>', and a comparison is done per
    question on the share of its k samples answered correctly."""
    jobs = [(task, question, sample) for task, questions in questions_by_task.items()
            for question in questions for sample in range(args.samples)]

    def one(job):
        task, question, sample = job
        try:
            text, tokens, finish = ask(base_url, model, question["prompt"], args.max_tokens, args.timeout,
                                       sampling_of(args, sample), system=question.get("system"))
            error = None
        except Exception as problem:   # a failed request makes the run invalid; it is never scored silently
            text, tokens, finish, error = "", 0, None, f"{type(problem).__name__}: {problem}"
        code = question["kind"] == "code"
        capped = finish == "length"
        answer = None if code else extract_answer(text, question["kind"])
        item_id = question["id"] if args.samples == 1 else f"{question['id']}#s{sample}"
        import hashlib
        item = {"id": item_id, "question": question["id"], "sample": sample, "gold": question["gold"],
                # A capped answer (finish_reason length) is wrong in the fixed-budget outcome, whatever it contains.
                "answer": answer, "correct": False if (code or capped) else is_correct(answer, question["gold"], question["kind"]),
                "completion_tokens": tokens, "error": error, "finish_reason": finish, "truncated": finish == "length",
                "prompt_sha256": hashlib.sha256(question["prompt"].encode("utf-8")).hexdigest(),
                "response_sha256": hashlib.sha256((text or "").encode("utf-8")).hexdigest(),
                "text": text if args.keep_text else None}
        if code and not capped:
            item["_full_text"] = text      # scored in the sandbox after generation (score_code_items)
        elif code:
            item["answer"] = "capped"      # not run: a capped answer is wrong
        return task, item

    results = {task: [] for task in questions_by_task}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for task, item in pool.map(one, jobs):
            results[task].append(item)
    score_code_items(results, questions_by_task, args)
    return results


def write_quality_items(args, run):
    """quality_items.jsonl of the test harness: one line per answered item, joinable across arms by sample id
    (task, question, sample, seed) and prompt hash; caps (finish_reason length) and failures are explicit."""
    os.makedirs(os.path.dirname(os.path.abspath(args.items_out)), exist_ok=True)
    with open(args.items_out, "w", encoding="utf-8") as handle:
        for task, items in run["tasks"].items():
            for item in items:
                record = {"schema": "lqh.quality_item.v1", "dataset": task, "sample_id": f"{task}:{item['id']}",
                          "question": item["question"], "sample": item["sample"],
                          "seed": None if args.seed is None else args.seed + item["sample"], "arm": args.arm,
                          "prompt_sha256": item.get("prompt_sha256"), "response_sha256": item.get("response_sha256"),
                          "answer": item["answer"], "gold": item["gold"], "correct": item["correct"],
                          "capped": bool(item.get("truncated")), "finish_reason": item.get("finish_reason"),
                          "completion_tokens": item["completion_tokens"], "error": item["error"],
                          "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
                          "top_k": args.top_k, "presence_penalty": args.presence_penalty}
                handle.write(json.dumps(record) + "\n")
    print(f"  wrote {args.items_out}")


# ---- reports ---------------------------------------------------------------------------

def summarise(run):
    print(f"\n  {'task':<10} {'questions':>9} {'correct':>8} {'accuracy':>9} {'+/-':>6} {'no answer':>10} "
          f"{'failed':>7} {'truncated':>10} {'tokens/answer':>14}")
    for task, items in run["tasks"].items():
        count = len(items)
        correct = sum(item["correct"] for item in items)
        accuracy = correct / count
        error = math.sqrt(accuracy * (1 - accuracy) / count)
        unanswered = sum(item["answer"] is None and item["error"] is None for item in items)
        failed = sum(item["error"] is not None for item in items)
        tokens = sum(item["completion_tokens"] for item in items) / count
        truncated = sum(bool(item.get("truncated")) for item in items)
        print(f"  {task:<10} {count:>9} {correct:>8} {100 * accuracy:>8.2f}% {100 * error:>5.2f}% {unanswered:>10} "
              f"{failed:>7} {truncated:>10} {tokens:>14.0f}")


def wilson_interval(successes, count, z=1.96):
    """Wilson score interval of a proportion."""
    if count == 0:
        return 0.0, 1.0
    p = successes / count
    centre = (p + z * z / (2 * count)) / (1 + z * z / count)
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_difference_interval(both, only_this, only_other, neither, z=1.96):
    """
    95% interval of (candidate accuracy - baseline accuracy) on paired questions: Newcombe's hybrid
    score method for paired proportions (Newcombe 1998, method 10). Unlike the Wald (McNemar) form it
    does not collapse to a point when no question differs: 30 questions with identical answers give an
    interval about +-10 points wide, not 0 .. 0, so a small sample stays INCONCLUSIVE.
    """
    n = both + only_this + only_other + neither
    p1 = (both + only_this) / n
    p2 = (both + only_other) / n
    l1, u1 = wilson_interval(both + only_this, n, z)
    l2, u2 = wilson_interval(both + only_other, n, z)
    denominator = (both + only_this) * (only_other + neither) * (both + only_other) * (only_this + neither)
    cross = both * neither - only_this * only_other
    if denominator <= 0:
        phi = 0.0
    elif cross > 0:
        phi = max(cross - n / 2, 0.0) / math.sqrt(denominator)
    else:
        phi = cross / math.sqrt(denominator)
    d = p1 - p2
    delta = math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    epsilon = math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return d - delta, d + epsilon


def t_quantile_975(df):
    """97.5% quantile of Student's t (Cornish-Fisher; within 0.3% of the table for df >= 5)."""
    z = 1.959964
    df = max(df, 1)
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df * df)


def clustered_difference_interval(ours, theirs):
    """
    Paper lane (several samples per question): the unit is the question. For each question the share of
    its k samples answered correctly, in each run; the paired difference per question; a t interval over
    the questions. The variance is floored by the binomial noise of k samples (one success and one
    failure added per question, both runs pooled), so a small sample whose questions happen to agree
    exactly does not get a zero-width interval.
    Returns (questions, samples per question, difference, low, high).
    """
    shares = {}
    for item in ours.values():
        shares.setdefault(item.get("question", item["id"]), [[], []])[0].append(bool(item["correct"]))
    for item in theirs.values():
        shares.setdefault(item.get("question", item["id"]), [[], []])[1].append(bool(item["correct"]))
    differences, floors = [], []
    k = 1
    for this, base in shares.values():
        k = max(k, len(this))
        differences.append(sum(this) / len(this) - sum(base) / len(base))
        pooled = (sum(this) + sum(base) + 1) / (len(this) + len(base) + 2)
        floors.append(pooled * (1 - pooled) * (1 / len(this) + 1 / len(base)))
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / max(n - 1, 1)
    variance = max(variance, sum(floors) / n)
    half = t_quantile_975(n - 1) * math.sqrt(variance / n)
    return n, k, mean, mean - half, mean + half


def compare(run, other, other_name, margin_points=1.0):
    """
    Paired noninferiority test of `run` (the candidate) against `other` (the baseline).

    A task PASSES only if the lower end of the 95% confidence interval of
    (candidate accuracy - baseline accuracy) is above -margin_points. "No significant
    difference" is not a pass: a small sample that cannot exclude a loss of the margin
    is INCONCLUSIVE. Both runs must hold exactly the same questions, and neither may
    contain failed requests; otherwise the comparison is INVALID.
    Returns "PASS", "INCONCLUSIVE", "FAIL" or "INVALID" (the worst over the tasks).
    """
    print(f"\n  Paired comparison with {other_name} (baseline). Noninferiority margin: {margin_points:.1f} points.")
    print(f"  {'task':<10} {'paired':>7} {'this run':>9} {'baseline':>9} {'only this':>10} {'only base':>10} "
          f"{'difference':>11} {'95% interval':>18}  verdict")
    order = ["PASS", "INCONCLUSIVE", "FAIL", "INVALID"]
    overall = "PASS"
    tasks = sorted(set(run["tasks"]) | set(other["tasks"]))
    for task in tasks:
        ours = {item["id"]: item for item in run["tasks"].get(task, [])}
        theirs = {item["id"]: item for item in other["tasks"].get(task, [])}
        failed = sum(item.get("error") is not None for item in list(ours.values()) + list(theirs.values()))
        if set(ours) != set(theirs) or not ours:
            verdict = "INVALID"
            print(f"  {task:<10} the two runs do not hold the same questions "
                  f"({len(ours)} here, {len(theirs)} in the baseline, {len(set(ours) & set(theirs))} shared)  INVALID")
        elif failed:
            verdict = "INVALID"
            print(f"  {task:<10} {failed} requests failed (transport or server errors)  INVALID")
        else:
            pairs = [(ours[key]["correct"], theirs[key]["correct"]) for key in ours]
            count = len(pairs)
            only_this = sum(a and not b for a, b in pairs)
            only_other = sum(b and not a for a, b in pairs)
            both = sum(a and b for a, b in pairs)
            difference = (only_this - only_other) / count
            if any(item.get("sample", 0) > 0 for item in ours.values()):
                # Several samples per question: question-clustered interval (the samples are not paired).
                _, _, difference, low, high = clustered_difference_interval(ours, theirs)
            else:
                low, high = paired_difference_interval(both, only_this, only_other,
                                                       count - both - only_this - only_other)
            if 100 * low > -margin_points:
                verdict = "PASS"
            elif 100 * high < -margin_points:
                verdict = "FAIL"
            else:
                verdict = "INCONCLUSIVE"
            print(f"  {task:<10} {count:>7} {100 * sum(a for a, _ in pairs) / count:>8.2f}% "
                  f"{100 * sum(b for _, b in pairs) / count:>8.2f}% {only_this:>10} {only_other:>10} "
                  f"{100 * difference:>+10.2f}% {f'{100 * low:+.2f} .. {100 * high:+.2f}':>18}  {verdict}")
        if order.index(verdict) > order.index(overall):
            overall = verdict
    explain = {"PASS": "every task is noninferior within the margin",
               "INCONCLUSIVE": "at least one task has too few questions to exclude a loss of the margin",
               "FAIL": "at least one task is worse than the margin allows",
               "INVALID": "the two runs cannot be compared"}
    print(f"\n  RESULT: {overall}: {explain[overall]}")
    return overall


# ---- self-test ---------------------------------------------------------------------------

def self_test(args):
    """Starts a fake server that knows the gold answers and gets a fixed share of them right."""
    questions = {name: load_task(name, 60) for name in ("gsm8k", "aime2025", "mmlu_pro")}
    gold_of_prompt = {q["prompt"]: q for items in questions.values() for q in items}
    state = {"share_correct": 1.0}

    class FakeServer(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            question = gold_of_prompt[body["messages"][0]["content"]]
            lucky = random.Random(question["id"]).random() < state["share_correct"]
            if question["kind"] == "letter":
                answer = question["gold"] if lucky else LETTERS[(LETTERS.index(question["gold"]) + 1) % 4]
                text = f"<think>Option {answer} fits.</think>\nSome reasoning about 12 and 7.\nANSWER: ({answer})"
            else:
                answer = question["gold"] if lucky else "123456789"
                text = f"We compute 3 + 4 = 7 first.\n**ANSWER: $\\boxed{{{answer}}}$**"
            reply = json.dumps({"choices": [{"message": {"content": text}}], "usage": {"completion_tokens": 42}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    args.concurrency, args.keep_text = 16, False

    runs = {}
    for share in (1.0, 0.7):
        state["share_correct"] = share
        runs[share] = {"tasks": {name: run_task(base_url, "fake", items, args) for name, items in questions.items()}}
    server.shutdown()

    failures = []
    for name, items in runs[1.0]["tasks"].items():
        if not all(item["correct"] for item in items):
            failures.append(f"{name}: a correct answer was scored as wrong")
    for name, items in runs[0.7]["tasks"].items():
        expected = sum(random.Random(q["id"]).random() < 0.7 for q in questions[name])
        if sum(item["correct"] for item in items) != expected:
            failures.append(f"{name}: scored {sum(item['correct'] for item in items)} correct, expected {expected}")
    summarise(runs[0.7])
    # A run against itself has no discordant pairs. With the non-degenerate (Newcombe) interval, 60 questions cannot
    # exclude a 1-point loss, so the verdict is INCONCLUSIVE, never FAIL, and the interval is not the degenerate 0..0.
    if compare(runs[1.0], runs[1.0], "itself") == "FAIL":
        failures.append("a run compared with itself fails")
    low, high = paired_difference_interval(30, 0, 0, 0)
    if not (low < 0 <= high and high - low > 0.05):
        failures.append(f"zero-discordance interval is degenerate: {low} .. {high}")
    if compare(runs[0.7], runs[1.0], "the all-correct run") != "FAIL":
        failures.append("a run with 70% correct was not failed against a run with 100% correct")
    # The two cases a review found the old comparison accepted:
    small = {"tasks": {"aime2025": [dict(item) for item in runs[1.0]["tasks"]["aime2025"]]}}
    worse = {"tasks": {"aime2025": [dict(item) for item in small["tasks"]["aime2025"]]}}
    worse["tasks"]["aime2025"][0]["correct"] = worse["tasks"]["aime2025"][1]["correct"] = False
    if compare(worse, small, "30 of 30 against 28 of 30") == "PASS":
        failures.append("28 of 30 against 30 of 30 was accepted")
    disjoint = {"tasks": {"aime2025": [dict(item, id="other-" + item["id"]) for item in small["tasks"]["aime2025"]]}}
    if compare(disjoint, small, "a run with different questions") != "INVALID":
        failures.append("runs with different questions were compared")
    print("\nself-test:", "PASS" if not failures else "FAIL: " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


def run_quality_gate(args):
    """TQ.6: the long-context quality gate, next to the task run. The gate itself (loss by position band to 32k
    tokens and beyond, recall and code items, paired against a reference arm) is the harness repository's; this
    only calls it and keeps its verdict with the run. Without --quality-gate the run records that no gate ran."""
    import subprocess

    if not args.quality_gate:
        print("\n  quality gate: not run (no --quality-gate command; the gate lives in the harness repository)")
        return {"ran": False}
    environment = dict(os.environ, LEAPQUANT_GATE_PORT=str(args.port), LEAPQUANT_GATE_RUN=args.out or "")
    try:
        done = subprocess.run(args.quality_gate, shell=True, env=environment, capture_output=True, text=True,
                              timeout=args.timeout * 4)
        status, output = done.returncode, (done.stdout + done.stderr)
    except subprocess.TimeoutExpired:
        status, output = None, "timed out"
    verdict = {0: "PASS", 1: "FAIL"}.get(status, "UNDECIDED")
    tail = output.strip().splitlines()[-40:]
    print(f"\n  quality gate: {verdict} (exit status {status})")
    for line in tail[-5:]:
        print(f"    {line}")
    return {"ran": True, "command": args.quality_gate, "exit_status": status, "verdict": verdict, "output_tail": tail}


def test_verdicts():
    """
    Fixtures for every false-pass class of plan v6, task F05 (no server needed):
    disjoint, empty and failed comparisons are INVALID; an underpowered comparison (30 questions, also
    with no question answered differently) is INCONCLUSIVE, not PASS; a clearly better candidate on a
    large sample PASSES; a clearly worse one FAILS; a candidate that is better than the baseline is not
    rejected for being different.
    """
    def run_of(answers, task="t", error=None):
        return {"tasks": {task: [{"id": f"q{index}", "correct": bool(value), "error": error}
                                 for index, value in enumerate(answers)]}}

    cases = []
    thirty_same = [1] + [0] * 29                               # AIME, 14 Nov: 1 of 30, 29 cut on both sides
    cases.append(("30 questions, identical answers", run_of(thirty_same), run_of(thirty_same), "INCONCLUSIVE"))
    cases.append(("30 of 30 against 30 of 30", run_of([1] * 30), run_of([1] * 30), "INCONCLUSIVE"))
    cases.append(("28 of 30 against 30 of 30", run_of([1] * 28 + [0] * 2), run_of([1] * 30), "INCONCLUSIVE"))
    big_base = [1] * 900 + [0] * 100
    big_same = list(big_base)
    cases.append(("1000 questions, identical", run_of(big_same), run_of(big_base), "PASS"))
    big_better = [1] * 950 + [0] * 50
    cases.append(("1000 questions, candidate 5 points better", run_of(big_better), run_of(big_base), "PASS"))
    big_worse = [1] * 850 + [0] * 150
    cases.append(("1000 questions, candidate 5 points worse", run_of(big_worse), run_of(big_base), "FAIL"))
    disjoint = {"tasks": {"t": [{"id": f"other{index}", "correct": True, "error": None} for index in range(30)]}}
    cases.append(("different questions", disjoint, run_of([1] * 30), "INVALID"))
    cases.append(("no questions", {"tasks": {"t": []}}, {"tasks": {"t": []}}, "INVALID"))
    cases.append(("failed requests", run_of([1] * 30, error="timeout"), run_of([1] * 30), "INVALID"))
    cases.append(("a task missing on one side", run_of([1] * 30, task="a"), run_of([1] * 30, task="b"), "INVALID"))
    def sampled(per_question, k=4, task="t"):
        """per_question: list of correct counts (0..k) per question; k samples each."""
        return {"tasks": {task: [{"id": f"q{q}#s{s}", "question": f"q{q}", "sample": s, "correct": s < c, "error": None}
                                 for q, c in enumerate(per_question) for s in range(k)]}}

    cases.append(("30 questions x 4 samples, identical", sampled([1] + [0] * 29), sampled([1] + [0] * 29),
                  "INCONCLUSIVE"))
    cases.append(("400 questions x 4 samples, candidate loses 20 points", sampled([4] * 320 + [0] * 80),
                  sampled([4] * 400), "FAIL"))
    failures = []
    for name, candidate, baseline, expected in cases:
        got = compare(candidate, baseline, name)
        if got != expected:
            failures.append(f"{name}: {got}, expected {expected}")
    print("\nverdict fixtures:", "PASS" if not failures else "FAIL: " + "; ".join(failures))
    sys.exit(0 if not failures else 1)


def main():
    parser = argparse.ArgumentParser(description="Answer accuracy against an OpenAI-compatible server.")
    parser.add_argument("--self-test", action="store_true", help="check this script against a fake server and exit")
    parser.add_argument("--test-verdicts", action="store_true", help="check the verdict logic on fixtures and exit")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default="default")
    parser.add_argument("--tasks", default="gsm8k,aime2025,mmlu_pro")
    parser.add_argument("--limit", type=int, default=1000, help="at most this many questions per task (evenly spaced)")
    parser.add_argument("--max-tokens", type=int, default=8192, help="room for long reasoning; the method is about long outputs")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (product lane)")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0, help="0 = off")
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="OpenAI presence penalty (the paper's Qwen3.5-9B setting is 1.5, Table 5)")
    parser.add_argument("--items-out", help="also write one JSON line per answered item (test-harness quality_items.jsonl)")
    parser.add_argument("--arm", default=None, help="label of the server configuration, recorded in --items-out")
    parser.add_argument("--samples", type=int, default=1, help="samples per question (paper lane: several)")
    parser.add_argument("--seed", type=int, default=None, help="seed of sample 0; sample j uses seed + j")
    parser.add_argument("--thinking", choices=["auto", "on", "off"], default="auto",
                        help="chat_template_kwargs enable_thinking (auto: the template's default)")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--sandbox-workers", type=int, default=8, help="parallel sandboxes scoring code answers")
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--keep-text", action="store_true", help="store the full answers in the output file")
    parser.add_argument("--out", help="write the run to this JSON file")
    parser.add_argument("--compare", help="a JSON file written by an earlier run: the baseline")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="noninferiority margin in percentage points (decide it before the run)")
    parser.add_argument("--report", metavar="RUN.json",
                        help="no server: summarise this saved run (and compare it with --compare)")
    parser.add_argument("--quality-gate", metavar="COMMAND",
                        help="state-cache plan task TQ.6: after the task run, run the long-context quality gate "
                             "(the harness repository's bench/quality runner) against the same server. COMMAND is "
                             "run by the shell with LEAPQUANT_GATE_PORT and LEAPQUANT_GATE_RUN set; its exit status "
                             "(0 pass, 1 fail, other: could not decide) and its last 40 lines go into the run file")
    args = parser.parse_args()

    if args.self_test:
        self_test(args)
    if args.test_verdicts:
        test_verdicts()

    if args.report:
        with open(args.report, encoding="utf-8") as handle:
            run = json.load(handle)
        summarise(run)
        status = 0
        if args.compare:
            with open(args.compare, encoding="utf-8") as handle:
                verdict = compare(run, json.load(handle), args.compare, args.margin)
            status = {"PASS": 0, "INCONCLUSIVE": 3, "FAIL": 4, "INVALID": 5}[verdict]
        sys.exit(status)

    base_url = f"http://{args.host}:{args.port}"
    run = {"base_url": base_url, "max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
           "top_k": args.top_k, "presence_penalty": args.presence_penalty, "samples": args.samples, "seed": args.seed,
           "thinking": args.thinking, "arm": args.arm, "tasks": {}}
    # --tasks a,b:N,c: a task with ":N" keeps N of its questions (evenly spaced, the same for every arm); without, --limit.
    questions_by_task = {}
    for entry in args.tasks.split(","):
        name, _, own_limit = entry.strip().partition(":")
        questions_by_task[name] = load_task(name, int(own_limit) if own_limit else args.limit)
        print(f"{name}: {len(questions_by_task[name])} questions", file=sys.stderr)
    run["question_counts"] = {name: len(q) for name, q in questions_by_task.items()}
    print(f"all tasks through one pool, concurrency {args.concurrency} ...", file=sys.stderr)
    run["tasks"] = run_tasks(base_url, args.model, questions_by_task, args)
    summarise(run)
    run["quality_gate"] = run_quality_gate(args)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(run, handle)
        print(f"\n  wrote {args.out}")
    if args.items_out:
        write_quality_items(args, run)
    failed = sum(item["error"] is not None for items in run["tasks"].values() for item in items)
    if failed:
        print(f"\n  EXECUTION INVALID: {failed} requests failed. Accuracy numbers above count them as wrong answers.")
    status = 2 if failed else 0
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            verdict = compare(run, json.load(handle), args.compare, args.margin)
        status = max(status, {"PASS": 0, "INCONCLUSIVE": 3, "FAIL": 4, "INVALID": 5}[verdict])
    sys.exit(status)


if __name__ == "__main__":
    main()
