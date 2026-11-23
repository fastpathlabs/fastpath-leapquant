#!/usr/bin/env python3
"""
radix_turns_client.py

Two-turn prompts against a running SGLang server with the radix cache on, at temperature 0, so that the second turn
starts from a state the radix cache kept when the first turn finished (a decode state, after a long answer that
crosses several radix track boundaries). Backlog B55: with LEAPQUANT_TRACK_FLUSH=0 those boundaries no longer fold.

    turn 1   "Question: <prompt>\\nAnswer:"                          up to --max-tokens (default 600) new tokens
    turn 2   turn 1's text + its answer + "\\n\\nQuestion: In one sentence, what was the main point above?\\nAnswer:"

Prints per prompt the answer lengths, the tokens turn 2 took from the cache (meta_info cached_tokens) and, with
--compare, how many words each turn shares with the same turn of another run. Exit status 1 when no second turn
took anything from the cache (the check then says nothing), when an answer is empty, or when more than
--max-early-differences second turns disagree with the other run within their first 5 words.

    python3 radix_turns_client.py --port 30000 --out lq.json [--compare reference.json]

Uses only the Python standard library.
"""

import argparse
import json
import sys
import urllib.request

PROMPTS = [
    "Explain step by step how to compute 37 * 43 in your head, then check the result another way.",
    "Summarise in eight sentences how a hash table handles collisions, with the trade-offs of each method.",
    "Write a Python function that merges two sorted lists, with a docstring and three test cases, and explain it.",
    "Describe how a refrigerator moves heat from inside to outside, naming each part of the cycle.",
    "List the planets of the solar system with one fact about each, then say which is hardest to observe and why.",
    "Explain what a binary search does, prove that it ends, and give its running time with a short argument.",
]
FOLLOW_UP = "\n\nQuestion: In one sentence, what was the main point above?\nAnswer:"


def generate(port, text, max_tokens):
    body = json.dumps({
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
    }).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    meta = result.get("meta_info", {})
    return result.get("text", "") or "", int(meta.get("cached_tokens", 0) or 0), int(meta.get("completion_tokens", 0) or 0)


def words_in_common_prefix(first, second):
    count = 0
    for a, b in zip(first.split(), second.split()):
        if a != b:
            break
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", help="a file written by an earlier run of this script")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--max-early-differences", type=int, default=2)
    args = parser.parse_args()

    results = []
    for prompt in PROMPTS:
        first = f"Question: {prompt}\nAnswer:"
        answer1, cached1, tokens1 = generate(args.port, first, args.max_tokens)
        answer2, cached2, tokens2 = generate(args.port, first + answer1 + FOLLOW_UP, 80)
        results.append({"prompt": prompt, "answer1": answer1, "answer2": answer2, "tokens1": tokens1,
                        "tokens2": tokens2, "cached1": cached1, "cached2": cached2})
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=1)

    reference = None
    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            reference = {item["prompt"]: item for item in json.load(handle)}
    failures, early = [], 0
    for item in results:
        line = (f"turn 1: {item['tokens1']:4d} tokens  turn 2: {item['tokens2']:3d} tokens, "
                f"{item['cached2']:4d} from the cache")
        if reference is not None and item["prompt"] in reference:
            other = reference[item["prompt"]]
            same1 = words_in_common_prefix(item["answer1"], other["answer1"])
            same2 = words_in_common_prefix(item["answer2"], other["answer2"])
            line += f"  same words as the other run: turn 1 {same1}/{len(other['answer1'].split())}, " \
                    f"turn 2 {same2}/{len(other['answer2'].split())}"
            if same2 < min(5, len(other["answer2"].split())):
                early += 1
        print(line)
        print("   turn 2:", " ".join(item["answer2"].split())[:160])
        if not item["answer1"].strip() or not item["answer2"].strip():
            failures.append(f"empty answer: {item['prompt'][:40]}")
    if not any(item["cached2"] > 0 for item in results):
        failures.append("no second turn took tokens from the cache")
    if reference is not None and early > args.max_early_differences:
        failures.append(f"{early} second turns differ within their first 5 words (allowed {args.max_early_differences})")
    for failure in failures:
        print("FAIL:", failure)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
