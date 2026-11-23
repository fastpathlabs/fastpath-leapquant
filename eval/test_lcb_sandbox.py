#!/usr/bin/env python3
"""
test_lcb_sandbox.py: self-tests of eval/lcb_sandbox.py on the host that runs the quality jobs (docker needed; the
sandbox image built and the window prepared: `python3 eval/lcb_sandbox.py build-image && ... prepare`).

    python3 eval/test_lcb_sandbox.py          # prints one line per case and PASS/FAIL; exit 0 only if all pass

Cases: a correct stdin program passes; a wrong one fails; an infinite loop hits the per-test limit; a program that
ignores SIGALRM hits the global limit; network and DNS are unreachable; the root file system is read-only and the host
is invisible (only /tmp is writable); a process bomb and a thread bomb are contained by the pid limit; a memory bomb
fails without breaking the sandbox; 10 MB on stderr does not disturb grading; call-based Solution classes pass and
fail correctly; extraction follows LiveCodeBench (last fenced block, reasoning removed, unclosed <think> = no code);
the host module never imports pickle; two real problems of the window (abc399_a stdin, LeetCode 3811 call-based,
private tests pickled, decoded only in the sandbox) pass with a correct solution and fail with a wrong one; no
container is left behind.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcb_sandbox  # noqa: E402


def wrap(code, decoy=True):
    """A model answer: reasoning with a decoy code block, then the real answer after </think>."""
    thinking = "Let me think.\n```python\nprint('decoy from the reasoning')\n```\n" if decoy else "Let me think.\n"
    return f"{thinking}</think>\n\nHere is my solution.\n```python\n{code}\n```\n"


def stdin_question(tests, private=None, timeout=None):
    lcb = {"question_id": "synthetic", "fn_name": None, "public_tests": tests,
           "private_tests": json.dumps(private) if private is not None else None, "tests_total": None}
    if timeout:
        lcb["timeout"] = timeout
    return {"id": "synthetic", "kind": "code", "lcb": lcb}


def functional_question(fn_name, tests):
    return {"id": "synthetic_fn", "kind": "code",
            "lcb": {"question_id": "synthetic_fn", "fn_name": fn_name, "public_tests": tests, "private_tests": None,
                    "tests_total": None}}


def t(i, o, kind="stdin"):
    return {"input": i, "output": o, "testtype": kind}


SUM_TESTS = [t("1 2", "3")]
SUM_PRIVATE = [t("10 20", "30"), t("-5 5", "0"), t("1000000000000 1", "1000000000001")]

CASES = []


def case(name, question, text, expect, timeout=None):
    CASES.append((name, question, text, expect, timeout))


case("stdin correct", stdin_question(SUM_TESTS, SUM_PRIVATE), wrap("a, b = map(int, input().split())\nprint(a + b)"),
     {"answer": "pass", "correct": True})
case("stdin wrong answer", stdin_question(SUM_TESTS, SUM_PRIVATE), wrap("a, b = map(int, input().split())\nprint(a - b)"),
     {"answer": "fail:wrong_answer", "correct": False})
case("stdin __main__ block and sys.stdin.read", stdin_question(SUM_TESTS, SUM_PRIVATE),
     wrap("import sys\ndef main():\n    a, b = map(int, sys.stdin.read().split())\n    print(a + b)\n"
          "if __name__ == '__main__':\n    main()"),
     {"answer": "pass", "correct": True})
case("infinite loop: per-test limit", stdin_question(SUM_TESTS, timeout=2), wrap("while True:\n    pass"),
     {"answer": "fail:timeout", "correct": False})
case("SIGALRM ignored: global limit", stdin_question(SUM_TESTS, timeout=1),
     wrap("import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\nwhile True:\n    pass"),
     {"answer": "fail:global_timeout", "correct": False})
case("network and DNS unreachable", stdin_question([t("x", "NET_BLOCKED\nDNS_BLOCKED")]),
     wrap("import socket\n"
          "try:\n    socket.create_connection(('1.1.1.1', 53), timeout=3)\n    print('NET_OPEN')\n"
          "except OSError:\n    print('NET_BLOCKED')\n"
          "try:\n    socket.getaddrinfo('huggingface.co', 443)\n    print('DNS_OPEN')\n"
          "except OSError:\n    print('DNS_BLOCKED')"),
     {"answer": "pass", "correct": True})
case("root file system read-only, host invisible, /tmp writable",
     stdin_question([t("x", "ROOT_RO\nSANDBOX_RO\nNO_HOST_HOME\nTMP_OK")]),
     wrap("import os\n"
          "def tryopen(path):\n"
          "    try:\n        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)\n        os.close(fd)\n        return True\n"
          "    except OSError:\n        return False\n"
          "print('ROOT_WRITABLE' if tryopen('/lcb_escape_test') else 'ROOT_RO')\n"
          "print('SANDBOX_WRITABLE' if tryopen('/sandbox/escape_test') else 'SANDBOX_RO')\n"
          "print('HOST_HOME_VISIBLE' if os.path.exists('/home/ubuntu') else 'NO_HOST_HOME')\n"
          "print('TMP_OK' if tryopen('/tmp/ok') else 'TMP_RO')"),
     {"answer": "pass", "correct": True})
case("process bomb contained (pid limit)", stdin_question([t("x", "CONTAINED")]),
     wrap("import os, sys\nstarted = 0\n"
          "try:\n    for _ in range(2000):\n"
          "        os.posix_spawn(sys.executable, [sys.executable, '-c', 'import time; time.sleep(30)'], {})\n"
          "        started += 1\nexcept OSError:\n    pass\n"
          "print('CONTAINED' if started < 2000 else 'NOT_CONTAINED')"),
     {"answer": "pass", "correct": True})
case("thread bomb contained (pid limit)", stdin_question([t("x", "CONTAINED")]),
     wrap("import threading, time\nstarted = 0\n"
          "try:\n    for _ in range(5000):\n"
          "        threading.Thread(target=time.sleep, args=(30,), daemon=True).start()\n        started += 1\n"
          "except RuntimeError:\n    pass\n"
          "print('CONTAINED' if started < 5000 else 'NOT_CONTAINED')"),
     {"answer": "pass", "correct": True})
case("memory bomb fails, sandbox survives", stdin_question(SUM_TESTS),
     wrap("blob = bytearray(3 * 1024 ** 3)\nprint(len(blob))"),
     {"answer_in": ["fail:runtime_error", "fail:global_timeout"], "correct": False})
case("10 MB on stderr does not disturb grading", stdin_question(SUM_TESTS, SUM_PRIVATE),
     wrap("import sys\nsys.stderr.write('x' * 10_000_000)\na, b = map(int, input().split())\nprint(a + b)"),
     {"answer": "pass", "correct": True})
case("float output: Decimal fallback", stdin_question([t("x", "0.50 2")]), wrap("print('0.5 2.0')"),
     {"answer": "pass", "correct": True})
case("functional correct", functional_question("add", [t("1\n2", "3", "functional"), t("[1, 2]\n[3]", "[1, 2, 3]", "functional")]),
     wrap("class Solution:\n    def add(self, a, b):\n        return a + b"),
     {"answer": "pass", "correct": True})
case("functional tuple compared as list", functional_question("pair", [t("1", "[1, 2]", "functional")]),
     wrap("class Solution:\n    def pair(self, a):\n        return (a, a + 1)"),
     {"answer": "pass", "correct": True})
case("functional wrong answer", functional_question("add", [t("1\n2", "3", "functional")]),
     wrap("class Solution:\n    def add(self, a, b):\n        return a * b"),
     {"answer": "fail:wrong_answer", "correct": False})
case("functional runtime error", functional_question("add", [t("1\n2", "3", "functional")]),
     wrap("class Solution:\n    def add(self, a, b):\n        return a / 0"),
     {"answer": "fail:runtime_error", "correct": False})
case("compile error", stdin_question(SUM_TESTS), wrap("def broken(:\n    pass"),
     {"answer": "fail:runtime_error", "correct": False})
case("no code: unclosed <think>", stdin_question(SUM_TESTS),
     "<think>reasoning that never ends\n```python\nprint(3)\n```\n", {"answer": "no_code", "correct": False})
case("no code: no fenced block", stdin_question(SUM_TESTS), "</think>\nThe answer is to add them.",
     {"answer": "no_code", "correct": False})
case("no think tags: last fenced block", stdin_question(SUM_TESTS, SUM_PRIVATE),
     "```python\nprint(0)\n```\nBetter:\n```python\na, b = map(int, input().split())\nprint(a + b)\n```",
     {"answer": "pass", "correct": True})


def real_cases():
    questions = {q["lcb"]["question_id"]: q for q in lcb_sandbox.load_questions()}
    hamming = questions["abc399_a"]
    reverse = questions["3811"]
    case("real abc399_a (stdin, pickled private tests) correct", hamming,
         wrap("n = int(input())\ns = input().strip()\nt = input().strip()\nprint(sum(1 for a, b in zip(s, t) if a != b))"),
         {"answer": "pass", "correct": True})
    case("real abc399_a wrong", hamming, wrap("n = int(input())\ns = input()\nt = input()\nprint(0)"),
         {"answer": "fail:wrong_answer", "correct": False})
    case("real 3811 (call-based, pickled private tests) correct", reverse,
         wrap("class Solution:\n    def reverseDegree(self, s: str) -> int:\n"
              "        return sum((26 - (ord(c) - 97)) * (i + 1) for i, c in enumerate(s))"),
         {"answer": "pass", "correct": True})
    case("real 3811 wrong (alphabet not reversed)", reverse,
         wrap("class Solution:\n    def reverseDegree(self, s: str) -> int:\n"
              "        return sum((ord(c) - 96) * (i + 1) for i, c in enumerate(s))"),
         {"answer": "fail:wrong_answer", "correct": False})
    return hamming, reverse


def main():
    import re
    failures = []
    source = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lcb_sandbox.py"), encoding="utf-8").read()
    uses_pickle = re.search(r"\bimport\s+[\w\s,]*\bpickle\b|\bfrom\s+pickle\b|\bpickle\s*\.|['\"]pickle['\"]\s*\)",
                            source)
    if uses_pickle:
        failures.append("host module uses pickle")
    print(f"{'static: the host module never imports or calls pickle':58s} {'FAIL' if uses_pickle else 'PASS'}")
    hamming, reverse = real_cases()
    print(f"window: {len(lcb_sandbox.load_questions())} problems; abc399_a tests {hamming['lcb']['tests_total']}, "
          f"3811 tests {reverse['lcb']['tests_total']}; image {lcb_sandbox.IMAGE} {lcb_sandbox.image_id()[:19]}")
    for name, question, text, expect, timeout in CASES:
        started = time.time()
        result = lcb_sandbox.score(question, text, timeout=timeout)
        ok = result["correct"] == expect["correct"]
        if "answer" in expect:
            ok = ok and result["answer"] == expect["answer"]
        if "answer_in" in expect:
            ok = ok and result["answer"] in expect["answer_in"]
        if not ok:
            failures.append(name)
        detail = f"{result['answer']} {result.get('tests_passed')}/{result.get('tests_total')}"
        print(f"{name:58s} {'PASS' if ok else 'FAIL'}  {detail}  {time.time() - started:.1f}s"
              + ("" if ok else f"  first_failure={result.get('first_failure')!r} status={result.get('sandbox_status')}"))
    left = subprocess.run(["docker", "ps", "-a", "--filter", "name=lcb-", "--format", "{{.Names}}"],
                          capture_output=True, text=True).stdout.split()
    print(f"{'no container left behind':58s} {'PASS' if not left else 'FAIL ' + ' '.join(left)}")
    if left:
        failures.append("containers left behind")
    print("\nself-test:", "PASS" if not failures else "FAIL: " + "; ".join(failures))
    return 0 if not failures else 1


HAMMING = "n = int(input())\ns = input().strip()\nt = input().strip()\nprint(sum(1 for a, b in zip(s, t) if a != b))"
REVERSE = ("class Solution:\n    def reverseDegree(self, s: str) -> int:\n"
           "        return sum((26 - (ord(c) - 97)) * (i + 1) for i, c in enumerate(s))")


def e2e(bench_path):
    """The bench end to end against a fake OpenAI server: every window problem is asked once (with LiveCodeBench's
    system message); abc399_a and 3811 get correct code, the others a wrong program; the bench scores them in the
    sandbox. Checks the counts, the two passes, the system message and the sandbox identity on every item."""
    import http.server
    import tempfile
    import threading
    seen = {"requests": 0, "system_ok": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = body["messages"]
            seen["requests"] += 1
            if messages[0]["role"] == "system" and messages[0]["content"] == lcb_sandbox.SYSTEM_MESSAGE_GENERIC:
                seen["system_ok"] += 1
            user = messages[-1]["content"]
            if "Hamming distance" in user:
                code = HAMMING
            elif "reverse degree" in user:
                code = REVERSE
            elif "class Solution" in user:
                code = "class Solution:\n    pass"
            else:
                code = "print(0)"
            reply = {"choices": [{"message": {"content": wrap(code)}, "finish_reason": "stop"}],
                     "usage": {"completion_tokens": 50}}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    with tempfile.TemporaryDirectory() as folder:
        out, items_out = os.path.join(folder, "run.json"), os.path.join(folder, "items.jsonl")
        started = time.time()
        command = [sys.executable, bench_path, "--port", str(port), "--tasks", "livecodebench_v6", "--limit", "100000",
                   "--max-tokens", "1000", "--concurrency", "16", "--temperature", "1.0", "--top-p", "0.95",
                   "--top-k", "20", "--presence-penalty", "1.5", "--thinking", "on", "--seed", "1234",
                   "--arm", "e2e", "--out", out, "--items-out", items_out]
        finished = subprocess.run(command, capture_output=True, text=True, timeout=3600)
        seconds = time.time() - started
        server.shutdown()
        if finished.returncode != 0:
            print(finished.stdout[-2000:], finished.stderr[-2000:])
            print("e2e: FAIL (bench exit", finished.returncode, ")")
            return 1
        items = [json.loads(line) for line in open(items_out, encoding="utf-8")]
        run = json.load(open(out, encoding="utf-8"))
    by_question = {i["question"]: i for i in items}
    problems = []
    if len(items) != 131:
        problems.append(f"{len(items)} items, expected 131")
    if not by_question["livecodebench_v6-abc399_a"]["correct"] or not by_question["livecodebench_v6-3811"]["correct"]:
        problems.append("a correct solution was not scored correct")
    others = [i for q, i in by_question.items() if q not in ("livecodebench_v6-abc399_a", "livecodebench_v6-3811")]
    if any(i["correct"] for i in others):
        problems.append("a wrong solution was scored correct")
    sandboxed = [t for t in run["tasks"]["livecodebench_v6"] if (t.get("sandbox") or {}).get("image_id")]
    if len(sandboxed) != 131:
        problems.append(f"{len(sandboxed)} items carry the sandbox image id, expected 131")
    if seen["system_ok"] != seen["requests"] or seen["requests"] != 131:
        problems.append(f"system message on {seen['system_ok']} of {seen['requests']} requests")
    answers = {}
    for i in items:
        answers[i["answer"]] = answers.get(i["answer"], 0) + 1
    slowest = sorted(((t.get("sandbox") or {}).get("seconds") or 0, t["question"]) for t in run["tasks"]["livecodebench_v6"])[-3:]
    print(f"e2e: 131 problems through the bench and the sandbox in {seconds:.0f} s; answers {answers}; "
          f"slowest scoring {slowest}")
    print("e2e:", "PASS" if not problems else "FAIL: " + "; ".join(problems))
    return 0 if not problems else 1


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--e2e":
        raise SystemExit(e2e(sys.argv[2]))
    raise SystemExit(main())
