#!/usr/bin/env python3
"""
lcb_sandbox.py: LiveCodeBench v6 for eval/accuracy_bench.py (task "livecodebench_v6"): problems, prompt, extraction and
a sandboxed scorer. Standard library only on the host; model-written code and the pickled private tests are only ever
handled inside a locked-down docker container (eval/lcb_sandbox_runner.py).

Problem window: Qwen's "LiveCodeBench v6 (25.02-25.05)". The paper (arXiv 2609.38166, Table 1) uses the model-card
benchmark settings. The Qwen/Qwen3.5-9B card lists "LiveCodeBench v6" (65.6) without dates, next to
Qwen3-Next-80B-A3B-Thinking 68.7 and Qwen3-30B-A3B-Thinking-2507 66.0. Those two values are the rows labelled
"LiveCodeBench v6 (25.02-25.05)" on their own model cards (huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking,
/Qwen3-30B-A3B-Thinking-2507; read 15 Nov 2026), so the window is contest_date from 2025-02-01 to 2025-05-01, with
LiveCodeBench's own inclusive filter (lcb_runner/benchmarks/code_generation.py load_code_generation_dataset:
start <= contest_date <= end).
Data: huggingface.co/datasets/livecodebench/code_generation_lite at commit 0fe84c3912ea0c4d4a78037083943e8f0c4dd505,
file test6.jsonl (the problems release_v6 adds; sha256 bb4c364f...f1f5 checked). Earlier files end before the window:
the dataset README gives release_v5 = May 2023..Jan 2025, and test5.jsonl's latest contest_date is 2025-01-04
(scanned 15 Nov 2026). Result: 131 problems, 2025-02-01..2025-04-06 (release_v6 ends on 6 April):
82 AtCoder (stdin; 18 easy, 19 medium, 45 hard) and 49 LeetCode (functional; 13 easy, 20 medium, 16 hard).
Order: contest_date, then question_id.
Prompt: LiveCodeBench's chat prompt (lcb_runner/prompts/code_generation.py, commit
28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24, LMStyle.OpenAIChat): system PromptConstants.SYSTEM_MESSAGE_GENERIC, user
get_generic_question_template_answer (question; starter code or the stdin format block; "### Answer: ...").
Extraction: LiveCodeBench's extract_code (lcb_runner/utils/extraction_utils.py, same commit): the lines between the
last two lines that contain ``` (fewer than two such lines: no code). Before that, the reasoning is removed: the text
after the last </think>; an opened but unclosed <think> means the answer section was never written (no_code).
Grading: LiveCodeBench's run_test / check_correctness, copied into eval/lcb_sandbox_runner.py: 6 s per test, the
global limit (6 + 1) x tests + 5 s, stdin compared line by line after strip with the Decimal fallback, call-based
problems through Solution().<func_name>(*json args) with tuples compared as lists. A problem passes when every test
result is > 0 (pass_k_utils). Here additionally every test must have a result and there must be at least one test.
Sandbox (one container per answer): docker run --rm -i --network none --read-only --tmpfs /tmp (64 MB)
--memory 2g --memory-swap 2g --pids-limit 128 --cpus 1 --cap-drop ALL --security-opt no-new-privileges
--user 65534:65534, the runner mounted read-only; JSON in on stdin, JSON out on stdout; the host kills the container
(docker kill) at its own limit. Image: python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
plus numpy 1.26.4 (LiveCodeBench's environment has numpy through torch/vllm, so solutions may import it), built once
with `python3 eval/lcb_sandbox.py build-image` (needs the network at build time only); its id goes into every result.
Deviations from LiveCodeBench, deliberate: (1) the container (LiveCodeBench runs the code in a local process with only
its reliability_guard); (2) 2 GB memory and 1 CPU per answer (LiveCodeBench sets no memory limit in run_test);
(3) sortedcontainers and other third-party packages except numpy are absent (as in LiveCodeBench's own environment);
(4) the result reaches the parent through a pipe instead of a multiprocessing.Manager list; (5) output the solution
writes to stdout/stderr outside LiveCodeBench's capture goes to a 64 MB tmpfs file (a solution that writes more there
gets an error there); (6) answers cut at the token budget are not special-cased here: the caller counts capped
answers (finish_reason "length") as wrong.

    python3 eval/lcb_sandbox.py build-image      # once per host (network needed for pip)
    python3 eval/lcb_sandbox.py prepare          # window file from test6.jsonl (huggingface_hub or urllib + token)
    python3 eval/lcb_sandbox.py inventory        # per-problem test counts, decoded inside the sandbox
    python3 eval/lcb_sandbox.py info
"""

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache", "lcb")
RUNNER = os.path.join(HERE, "lcb_sandbox_runner.py")

DATASET = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
SOURCE_FILE = "test6.jsonl"
SOURCE_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
WINDOW_START, WINDOW_END = "2025-02-01", "2025-05-01"
WINDOW_FILE = os.path.join(CACHE, "lcb_v6_20250201_20250501.jsonl")
WINDOW_META = os.path.join(CACHE, "lcb_v6_20250201_20250501.meta.json")
INVENTORY_FILE = os.path.join(CACHE, "lcb_v6_20250201_20250501.inventory.json")

BASE_IMAGE = "python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535"
NUMPY_VERSION = "1.26.4"
IMAGE = os.environ.get("LCB_SANDBOX_IMAGE", "lqh-lcb-sandbox:py311-numpy1264")
TIMEOUT_PER_TEST_S = 6                 # lcb_runner/evaluation/compute_code_generation_metrics.py codegen_metrics
DEFAULT_HOST_LIMIT_S = 1800            # when a problem's test count is unknown

# lcb_runner/prompts/code_generation.py (commit 28fef95e) PromptConstants
SYSTEM_MESSAGE_GENERIC = ("You are an expert Python programmer. You will be given a question (problem specification) "
                          "and will generate a correct Python program that matches the specification and passes all "
                          "tests.")
FORMATTING_MESSAGE_WITH_STARTER_CODE = ("You will use the following starter code to write the solution to the problem "
                                        "and enclose your code within delimiters.")
FORMATTING_WITHOUT_STARTER_CODE = ("Read the inputs from stdin solve the problem and write the answer to stdout (do "
                                   "not directly test on the sample inputs). Enclose your code within delimiters as "
                                   "follows. Ensure that when the python program runs, it reads the inputs, runs the "
                                   "algorithm and writes output to STDOUT.")

DOCKER_FLAGS = ["--network", "none", "--read-only", "--tmpfs", "/tmp:rw,nosuid,nodev,size=64m",
                "--memory", "2g", "--memory-swap", "2g", "--pids-limit", "128", "--cpus", "1",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
                "--ulimit", "nofile=1024:1024", "-e", "PYTHONDONTWRITEBYTECODE=1", "-w", "/tmp"]

_lock = threading.Lock()
_image_id = None
_inventory = None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


# ---- data -----------------------------------------------------------------------------------------------------------

def _hf_token():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token.strip()
    path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    return None


def _source_path():
    """test6.jsonl at the pinned revision: the huggingface_hub cache if that package is there, else a plain download
    into eval/cache/lcb/ (the token, if any, goes only to huggingface.co)."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(DATASET, SOURCE_FILE, repo_type="dataset", revision=DATASET_REVISION)
    except ImportError:
        pass
    import urllib.request
    os.makedirs(CACHE, exist_ok=True)
    target = os.path.join(CACHE, SOURCE_FILE)
    if not os.path.exists(target):
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{DATASET_REVISION}/{SOURCE_FILE}"
        request = urllib.request.Request(url)
        token = _hf_token()
        if token:
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=600) as response, open(target + ".part", "wb") as handle:
            while True:
                chunk = response.read(1 << 22)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(target + ".part", target)
    return target


def prepare():
    """Writes the window file (rows of test6.jsonl in the window, as stored: the private tests stay encoded)."""
    source = _source_path()
    digest = _sha256_file(source)
    if digest != SOURCE_SHA256:
        raise SystemExit(f"error: {source} has sha256 {digest}, expected {SOURCE_SHA256}")
    start, end = datetime.fromisoformat(WINDOW_START), datetime.fromisoformat(WINDOW_END)
    rows = []
    with open(source, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)              # JSON only; private_test_cases stays an opaque string here
            if start <= datetime.fromisoformat(row["contest_date"]) <= end:
                rows.append(row)
    rows.sort(key=lambda r: (datetime.fromisoformat(r["contest_date"]), r["question_id"]))
    os.makedirs(CACHE, exist_ok=True)
    with open(WINDOW_FILE + ".part", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    os.replace(WINDOW_FILE + ".part", WINDOW_FILE)
    counts = {}
    for row in rows:
        kind = "functional" if json.loads(row["metadata"]).get("func_name") else "stdin"
        key = f"{row['platform']}/{kind}/{row['difficulty']}"
        counts[key] = counts.get(key, 0) + 1
    meta = {"dataset": DATASET, "revision": DATASET_REVISION, "source_file": SOURCE_FILE, "source_sha256": digest,
            "window": [WINDOW_START, WINDOW_END], "filter": "start <= contest_date <= end (LiveCodeBench)",
            "problems": len(rows), "counts": dict(sorted(counts.items())),
            "first_date": rows[0]["contest_date"] if rows else None,
            "last_date": rows[-1]["contest_date"] if rows else None,
            "window_file_sha256": _sha256_file(WINDOW_FILE), "prepared_utc": datetime.now(timezone.utc).isoformat()}
    with open(WINDOW_META, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=1)
    return meta


def _window_rows():
    if not os.path.exists(WINDOW_FILE):
        prepare()
    with open(WINDOW_META, encoding="utf-8") as handle:
        meta = json.load(handle)
    if _sha256_file(WINDOW_FILE) != meta["window_file_sha256"]:
        raise SystemExit(f"error: {WINDOW_FILE} does not match {WINDOW_META}; run `python3 eval/lcb_sandbox.py prepare`")
    with open(WINDOW_FILE, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle], meta


def generic_prompt(row):
    """lcb_runner/prompts/code_generation.py get_generic_question_template_answer (commit 28fef95e)."""
    prompt = f"### Question:\n{row['question_content']}\n\n"
    if row["starter_code"]:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{row['starter_code']}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def _load_inventory():
    global _inventory
    with _lock:
        if _inventory is None:
            _inventory = {}
            if os.path.exists(INVENTORY_FILE):
                with open(INVENTORY_FILE, encoding="utf-8") as handle:
                    _inventory = json.load(handle).get("problems", {})
        return _inventory


def load_questions():
    """The window's problems as accuracy_bench questions (kind "code"), in contest_date / question_id order."""
    rows, meta = _window_rows()
    inventory = _load_inventory()
    questions = []
    for row in rows:
        qid = row["question_id"]
        metadata = json.loads(row["metadata"])
        questions.append({
            "id": f"livecodebench_v6-{qid}", "kind": "code", "gold": qid,
            "system": SYSTEM_MESSAGE_GENERIC, "prompt": generic_prompt(row),
            "lcb": {"question_id": qid, "question_title": row["question_title"], "platform": row["platform"],
                    "contest_id": row["contest_id"], "contest_date": row["contest_date"],
                    "difficulty": row["difficulty"], "starter_code": row["starter_code"],
                    "fn_name": metadata.get("func_name") or None,
                    "public_tests": json.loads(row["public_test_cases"]),
                    "private_tests": row["private_test_cases"],          # opaque here; decoded in the sandbox
                    "tests_total": (inventory.get(qid) or {}).get("tests_total"),
                    "window": f"{WINDOW_START}..{WINDOW_END}", "dataset_revision": meta["revision"]},
        })
    return questions


# ---- extraction ------------------------------------------------------------------------------------------------------

def strip_reasoning(text):
    """The answer section: the text after the last </think>. An opened but unclosed <think> (the answer section was
    never written, e.g. cut at the budget) gives None; text without think tags is returned unchanged."""
    text = text or ""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    if "<think>" in text:
        return None
    return text


def extract_code(model_output):
    """lcb_runner/utils/extraction_utils.py extract_code for chat styles (commit 28fef95e)."""
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1: indexlines[-1]])


# ---- sandbox ---------------------------------------------------------------------------------------------------------

def image_id():
    global _image_id
    with _lock:
        if _image_id is None:
            probe = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
                                   capture_output=True, text=True, timeout=60)
            if probe.returncode != 0:
                raise RuntimeError(f"sandbox image {IMAGE} is missing: run `python3 eval/lcb_sandbox.py build-image`")
            _image_id = probe.stdout.strip()
        return _image_id


def build_image():
    dockerfile = (f"FROM {BASE_IMAGE}\n"
                  f"RUN pip install --no-cache-dir numpy=={NUMPY_VERSION} && python -c \"import numpy; "
                  f"print(numpy.__version__)\"\n")
    result = subprocess.run(["docker", "build", "-t", IMAGE, "-"], input=dockerfile, text=True,
                            capture_output=True, timeout=1800)
    if result.returncode != 0:
        raise SystemExit(f"docker build failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return {"image": IMAGE, "image_id": image_id(), "base_image": BASE_IMAGE, "numpy": NUMPY_VERSION}


def run_sandbox(payload, host_limit_s):
    """One container: the payload as JSON on stdin, the runner's JSON result from stdout. Returns (result | None,
    status) where status is "ok", "timeout" (the host killed the container) or "error: ..."."""
    name = f"lcb-{uuid.uuid4().hex[:16]}"
    command = (["docker", "run", "--rm", "-i", "--name", name] + DOCKER_FLAGS +
               ["-v", f"{RUNNER}:/sandbox/runner.py:ro", IMAGE, "python3", "-I", "/sandbox/runner.py"])
    data = json.dumps(payload).encode("utf-8")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = process.communicate(input=data, timeout=host_limit_s)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True, timeout=120)
        try:
            process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
        return None, "timeout"
    if process.returncode != 0:
        return None, f"error: exit {process.returncode}: {err.decode('utf-8', 'replace')[-600:]}"
    try:
        return json.loads(out.decode("utf-8")), "ok"
    except ValueError:
        return None, f"error: unreadable result ({len(out)} bytes): {err.decode('utf-8', 'replace')[-600:]}"


def _host_limit(lcb, timeout):
    total = lcb.get("tests_total")
    if not total:
        return DEFAULT_HOST_LIMIT_S
    return (timeout + 1) * total + 5 + 120          # LiveCodeBench's global limit + container start and decoding


CODES = {-1: "fail:global_timeout", -2: "fail:wrong_answer", -3: "fail:timeout", -4: "fail:runtime_error"}


def score(question, full_text, timeout=None):
    """Grades one answer in the sandbox. Returns {"correct", "answer", tests_total, tests_run, tests_passed,
    first_failure, code_sha256, seconds, image, image_id, timeout_per_test_s, sandbox_status}. Never raises: any
    failure of the scorer itself (docker missing, a bad record) is a "fail:sandbox_error" result, so a run's generated
    answers are never lost to a scoring exception."""
    started = time.time()
    try:
        return _score(question, full_text, timeout, started)
    except Exception as problem:    # noqa: BLE001 (reported in the result, never silent)
        return {"correct": False, "answer": "fail:sandbox_error", "sandbox_status": f"error: {type(problem).__name__}",
                "first_failure": f"{type(problem).__name__}: {problem}"[:600], "image": IMAGE, "image_id": None,
                "tests_total": (question.get("lcb") or {}).get("tests_total"), "tests_run": 0, "tests_passed": 0,
                "seconds": round(time.time() - started, 3)}


def _score(question, full_text, timeout, started):
    lcb = question["lcb"]
    timeout = int(timeout or lcb.get("timeout") or TIMEOUT_PER_TEST_S)
    answer_section = strip_reasoning(full_text)
    code = extract_code(answer_section) if answer_section is not None else ""
    base = {"code_sha256": _sha256_text(code), "image": IMAGE, "timeout_per_test_s": timeout,
            "tests_total": lcb.get("tests_total"), "tests_run": 0, "tests_passed": 0, "first_failure": None}
    if not code.strip():
        reason = "no answer section (unclosed <think>)" if answer_section is None else "no fenced code block"
        return {**base, "correct": False, "answer": "no_code", "first_failure": reason, "sandbox_status": "not_run",
                "image_id": None, "seconds": round(time.time() - started, 3)}
    payload = {"code": code, "fn_name": lcb.get("fn_name"), "public_tests": lcb["public_tests"],
               "private_tests": lcb.get("private_tests"), "timeout": timeout}
    try:
        identity = image_id()
    except RuntimeError as problem:
        return {**base, "correct": False, "answer": "fail:sandbox_error", "first_failure": str(problem),
                "sandbox_status": "error: no image", "image_id": None, "seconds": round(time.time() - started, 3)}
    result, status = run_sandbox(payload, _host_limit(lcb, timeout))
    base.update(image_id=identity, sandbox_status=status)
    if result is None:
        answer = "fail:sandbox_timeout" if status == "timeout" else "fail:sandbox_error"
        return {**base, "correct": False, "answer": answer, "first_failure": status[:600],
                "seconds": round(time.time() - started, 3)}
    results = result.get("results") or []
    total = result.get("tests_total") or 0
    passed_entries = [r is True or (isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0) for r in results]
    correct = total > 0 and len(results) == total and all(passed_entries)
    answer = "pass"
    if not correct:
        failing = next((r for r, ok in zip(results, passed_entries) if not ok), None)
        if failing is False:
            answer = "fail:wrong_answer"
        elif isinstance(failing, (int, float)) and not isinstance(failing, bool):
            answer = CODES.get(int(failing), f"fail:code_{failing}")
        else:
            answer = "fail:incomplete"
    metadata = result.get("metadata") or {}
    return {**base, "correct": correct, "answer": answer, "tests_total": total, "tests_run": len(results),
            "tests_passed": sum(passed_entries),
            "first_failure": None if correct else str(metadata.get("error_message") or metadata)[:600],
            "decode": result.get("decode"), "grading_seconds": result.get("grading_seconds"),
            "seconds": round(time.time() - started, 3)}


# ---- inventory ---------------------------------------------------------------------------------------------------------

def inventory(workers=8):
    """Per-problem test counts and sizes, decoded inside the sandbox (no code is run: the runner's count mode)."""
    import concurrent.futures
    rows, meta = _window_rows()
    identity = image_id()

    def one(row):
        payload = {"mode": "count", "public_tests": json.loads(row["public_test_cases"]),
                   "private_tests": row["private_test_cases"]}
        result, status = run_sandbox(payload, 600)
        return row["question_id"], (result or {"error": status})

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        problems = dict(pool.map(one, rows))
    doc = {"window_meta": meta, "image": IMAGE, "image_id": identity, "problems": problems,
           "computed_utc": datetime.now(timezone.utc).isoformat()}
    with open(INVENTORY_FILE, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=1)
    global _inventory
    with _lock:
        _inventory = problems
    return doc


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    command = argv[0] if argv else "info"
    if command == "build-image":
        print(json.dumps(build_image(), indent=1))
    elif command == "prepare":
        print(json.dumps(prepare(), indent=1))
    elif command == "inventory":
        doc = inventory()
        totals = sorted((p.get("tests_total") or 0) for p in doc["problems"].values())
        errors = [q for q, p in doc["problems"].items() if "error" in p]
        print(f"{len(totals)} problems; tests per problem: min {totals[0]}, median {totals[len(totals) // 2]}, "
              f"max {totals[-1]}, total {sum(totals)}; errors: {errors}")
    elif command == "info":
        rows, meta = _window_rows()
        print(json.dumps({**meta, "image": IMAGE}, indent=1))
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
