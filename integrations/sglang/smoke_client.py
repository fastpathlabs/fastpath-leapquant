#!/usr/bin/env python3
"""
smoke_client.py

Sends a few fixed prompts to a running SGLang server at temperature 0 and saves
the answers. With --compare it also shows how far each answer agrees with the
answer another server gave to the same prompt.

    python3 smoke_client.py --port 30000 --out window.json
    python3 smoke_client.py --port 30000 --out leapquant.json --compare window.json

What to expect when comparing the LeapQuant server with the window server
    The two servers round the recurrent state differently, so their answers
    are not required to be identical to the last token. But they must start
    the same and stay sensible. An answer that turns into noise or repetition
    after a few words means the kernel or its integration is wrong.

Uses only the Python standard library.
"""

import argparse
import json
import urllib.request

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "List the first ten prime numbers and explain in two sentences why 1 is not prime.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "Explain step by step how to compute 37 * 43 in your head.",
    "Summarise in five sentences how a hash table handles collisions.",
    "Translate into German: 'The train to Munich leaves from platform nine at half past seven.'",
]


def ask(port, prompt, max_tokens):
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)["choices"][0]["message"]["content"] or ""


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
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--max-early-differences", type=int, default=None,
                        help="with --compare: exit with status 1 if more answers than this differ within "
                             "their first 15 words, or if an answer repeats itself")
    args = parser.parse_args()

    answers = [ask(args.port, prompt, args.max_tokens) for prompt in PROMPTS]
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"prompts": PROMPTS, "answers": answers}, handle, indent=1, ensure_ascii=False)
    print(f"wrote {args.out}: {len(answers)} answers")

    if args.compare:
        with open(args.compare, encoding="utf-8") as handle:
            other = json.load(handle)["answers"]
        print(f"\n{'words':>6} {'same first':>11}  prompt")
        short = 0
        for prompt, ours, theirs in zip(PROMPTS, answers, other):
            common = words_in_common_prefix(ours, theirs)
            total = len(ours.split())
            short += common < min(15, total)
            print(f"{total:>6} {common:>11}  {prompt[:60]}")
        print(f"\nanswers that differ within their first 15 words: {short} of {len(PROMPTS)}")
        print("Read the two files side by side. Differences late in an answer are expected;")
        print("an answer that degrades into noise or repetition is a bug.")
        repeating = [index for index, answer in enumerate(answers) if repeats_itself(answer)]
        if repeating:
            print(f"answers that repeat one 8-word phrase 4 times or more: {repeating}")
        if args.max_early_differences is not None:
            if short > args.max_early_differences or repeating:
                print("SMOKE: FAIL")
                raise SystemExit(1)
            print("SMOKE: PASS")


def repeats_itself(answer, length=8, times=4):
    """True if some run of `length` words occurs `times` times or more (a degenerate answer)."""
    words = answer.split()
    counts = {}
    for start in range(len(words) - length + 1):
        phrase = tuple(words[start:start + length])
        counts[phrase] = counts.get(phrase, 0) + 1
        if counts[phrase] >= times:
            return True
    return False


if __name__ == "__main__":
    main()
