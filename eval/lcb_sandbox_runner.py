#!/usr/bin/env python3
# Parts of this file are copied from LiveCodeBench (github.com/LiveCodeBench/LiveCodeBench,
# commit 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24) under the MIT License:
#
# MIT License
#
# Copyright (c) 2024 LiveCodeBench
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The rest of this file: Copyright 2026 FastPathLabs, Apache License 2.0.
"""
lcb_sandbox_runner.py: runs INSIDE the LiveCodeBench sandbox container (eval/lcb_sandbox.py starts it); never on the host.

Reads one JSON request on stdin:
    {"code": str, "fn_name": str | null, "public_tests": [{"input", "output", "testtype"}...],
     "private_tests": str (the dataset's private_test_cases field, as stored) | list | null, "timeout": 6}
and writes one JSON result on the ORIGINAL stdout:
    {"results": [...], "metadata": {...}, "tests_total": n, "decode": "json" | "pickle" | "none"}

The private tests are decoded here, inside the container: LiveCodeBench stores them as JSON or as
base64 -> zlib -> pickle -> JSON (lcb_runner/benchmarks/code_generation.py); a pickle is only ever loaded in here.

Grading is LiveCodeBench's own, copied from github.com/LiveCodeBench/LiveCodeBench at commit
28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24 (MIT licence):
  - lcb_runner/evaluation/testing_util.py: import_string, Capturing, MockStdinWithBuffer, MockBuffer, clean_if_name,
    make_function, call_method, get_function, compile_code, convert_line_to_decimals, get_stripped_lines,
    grade_call_based, grade_stdio, run_test, reliability_guard (unchanged apart from dropping the unused numpy import
    and the debug prints);
  - lcb_runner/evaluation/compute_code_generation_metrics.py: check_correctness (the child process and the global
    timeout (timeout + 1) * tests + 5 s; on no result every test is -1) and sys.set_int_max_str_digits(50000).
    The child's result comes back through a pipe instead of a multiprocessing.Manager list (same values).
Result codes: True passed, -1 global timeout, -2 wrong answer, -3 time limit (per test), -4 runtime/compile error.
A problem passes when every entry is > 0 (lcb_runner/evaluation/pass_k_utils.py).
Anything the solution writes to file descriptor 1 goes to stderr; the result is written to a saved copy of the original
stdout, which the child closes first.
"""

import ast
import base64
import faulthandler
import json
import multiprocessing
import os
import pickle
import platform
import signal
import sys
import time
import zlib
from io import StringIO
from types import ModuleType
from unittest.mock import mock_open, patch

sys.set_int_max_str_digits(50000)

import_string = "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\nfrom heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\nfrom random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\nfrom operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\nfrom builtins import *\nfrom typing import *\nimport string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\nimport copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\nimport operator\nimport io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"

RESULT_FD = None


def truncatefn(s, length=300):
    if isinstance(s, str):
        pass
    else:
        s = str(s)
    if len(s) <= length:
        return s

    return s[: length // 2] + "...(truncated) ..." + s[-length // 2:]


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


class MockStdinWithBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs
        self._stringio = StringIO(inputs)
        self.buffer = MockBuffer(inputs)

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self._stringio.readline(*args)

    def readlines(self, *args):
        return self.inputs.split("\n")

    def __getattr__(self, name):
        return getattr(self._stringio, name)


class MockBuffer:
    def __init__(self, inputs: str):
        self.inputs = inputs.encode("utf-8")

    def read(self, *args):
        return self.inputs

    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def clean_if_name(code: str) -> str:
    try:
        astree = ast.parse(code)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            condition = last_block.test
            if ast.unparse(condition).strip() == "__name__ == '__main__'":
                code = (
                    ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)  # type: ignore
                )
    except:  # noqa: E722 (as in LiveCodeBench)
        pass

    return code


def make_function(code: str) -> str:
    try:
        import_stmts = []
        all_other_stmts = []
        astree = ast.parse(code)
        for stmt in astree.body:
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                import_stmts.append(stmt)
            else:
                all_other_stmts.append(stmt)

        function_ast = ast.FunctionDef(
            name="wrapped_function",
            args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
            ),
            body=all_other_stmts,
            decorator_list=[],
            lineno=-1,
        )
        main_code = (
            import_string
            + "\n"
            + ast.unparse(import_stmts)  # type: ignore
            + "\n"
            + ast.unparse(function_ast)  # type: ignore
        )
        return main_code
    except Exception:
        return code


def call_method(method, inputs):

    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    mock_stdin = MockStdinWithBuffer(inputs)

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", mock_stdin)
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit:
            pass
        finally:
            pass

    return _inner_call_method(method)


def get_function(compiled_sol, fn_name: str):  # type: ignore
    try:
        assert hasattr(compiled_sol, fn_name)
        return getattr(compiled_sol, fn_name)
    except Exception:
        return


def compile_code(code: str, timeout: int):
    signal.alarm(timeout)
    try:
        tmp_sol = ModuleType("tmp_sol", "")
        exec(code, tmp_sol.__dict__)
        if "class Solution" in code:
            compiled_sol = tmp_sol.Solution()
        else:
            compiled_sol = tmp_sol

        assert compiled_sol is not None
    finally:
        signal.alarm(0)

    return compiled_sol


def convert_line_to_decimals(line: str):
    from decimal import Decimal
    try:
        decimal_line = [Decimal(elem) for elem in line.split()]
    except:  # noqa: E722 (as in LiveCodeBench)
        return False, []
    return True, decimal_line


def get_stripped_lines(val: str):
    val = val.strip()

    return [val_line.strip() for val_line in val.split("\n")]


def grade_call_based(code: str, all_inputs: list, all_outputs: list, fn_name: str, timeout: int):
    code = import_string + "\n\n" + code
    compiled_sol = compile_code(code, timeout)

    if compiled_sol is None:
        return

    method = get_function(compiled_sol, fn_name)

    if method is None:
        return

    all_inputs = [
        [json.loads(line) for line in inputs.split("\n")] for inputs in all_inputs
    ]

    all_outputs = [json.loads(output) for output in all_outputs]

    total_execution = 0
    all_results = []
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()
        try:
            start = time.time()
            prediction = method(*gt_inp)
            total_execution += time.time() - start
            signal.alarm(0)

            # don't penalize model if it produces tuples instead of lists
            if isinstance(prediction, tuple):
                prediction = list(prediction)

            tmp_result = prediction == gt_out

            all_results.append(tmp_result)

            if not tmp_result:
                return all_results, {
                    "output": truncatefn(prediction),
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                    "error_code": -2,
                    "error_message": "Wrong Answer",
                }
        except Exception as e:
            signal.alarm(0)
            if "timeoutexception" in repr(e).lower():
                all_results.append(-3)
                return all_results, {
                    "error": repr(e),
                    "error_code": -3,
                    "error_message": "Time Limit Exceeded",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }
            else:
                all_results.append(-4)
                return all_results, {
                    "error": repr(e),
                    "error_code": -4,
                    "error_message": "Runtime Error",
                    "inputs": truncatefn(gt_inp),
                    "expected": truncatefn(gt_out),
                }

        finally:
            signal.alarm(0)
            faulthandler.disable()

    return all_results, {"execution time": total_execution}


def grade_stdio(code: str, all_inputs: list, all_outputs: list, timeout: int):
    code = clean_if_name(code)

    code = make_function(code)

    compiled_sol = compile_code(code, timeout)
    if compiled_sol is None:
        return

    method = get_function(compiled_sol, "wrapped_function")

    if method is None:
        return

    all_results = []
    total_execution_time = 0
    for idx, (gt_inp, gt_out) in enumerate(zip(all_inputs, all_outputs)):
        signal.alarm(timeout)
        faulthandler.enable()

        signal.alarm(timeout)
        with Capturing() as captured_output:
            try:
                start = time.time()
                call_method(method, gt_inp)
                total_execution_time += time.time() - start
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if "timeoutexception" in repr(e).lower():
                    all_results.append(-3)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -3,
                        "error_message": "Time Limit Exceeded",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }
                else:
                    all_results.append(-4)
                    return all_results, {
                        "error": repr(e),
                        "error_code": -4,
                        "error_message": "Runtime Error",
                        "inputs": truncatefn(gt_inp),
                        "expected": truncatefn(gt_out),
                    }

            finally:
                signal.alarm(0)
                faulthandler.disable()

        prediction = captured_output[0]

        stripped_prediction_lines = get_stripped_lines(prediction)
        stripped_gt_out_lines = get_stripped_lines(gt_out)

        WA_send_args = {
            "output": truncatefn(prediction),
            "inputs": truncatefn(gt_inp),
            "expected": truncatefn(gt_out),
            "error_code": -2,
        }

        if len(stripped_prediction_lines) != len(stripped_gt_out_lines):
            all_results.append(-2)
            WA_send_args["error_message"] = "Wrong answer: mismatched output length"
            return all_results, WA_send_args

        for output_line_idx, (
            stripped_prediction_line,
            stripped_gt_out_line,
        ) in enumerate(zip(stripped_prediction_lines, stripped_gt_out_lines)):
            WA_send_args["error_message"] = (
                f"Wrong answer at {output_line_idx=}: {truncatefn(stripped_prediction_line)} != {truncatefn(stripped_gt_out_line)}"
            )

            if stripped_prediction_line == stripped_gt_out_line:
                continue

            success, decimal_prediction_line = convert_line_to_decimals(stripped_prediction_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args
            success, decimal_gtout_line = convert_line_to_decimals(stripped_gt_out_line)
            if not success:
                all_results.append(-2)
                return all_results, WA_send_args

            if decimal_prediction_line == decimal_gtout_line:
                continue

            all_results.append(-2)
            return all_results, WA_send_args
        all_results.append(True)

    return all_results, {"execution time": total_execution_time}


def run_test(sample, test=None, timeout=6):
    signal.signal(signal.SIGALRM, timeout_handler)

    # Disable functionalities that can make destructive changes to the test.
    reliability_guard()

    in_outs = json.loads(sample["input_output"])

    if in_outs.get("fn_name") is None:
        which_type = "standard_input"
        method_name = None
    else:
        which_type = "call_based"
        method_name = in_outs["fn_name"]

    if which_type == "call_based":
        signal.alarm(timeout)
        try:
            results, metadata = grade_call_based(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                fn_name=method_name,
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            return [-4], {
                "error_code": -4,
                "error_message": f"Error during testing: {e}",
            }
        finally:
            signal.alarm(0)
    else:
        signal.alarm(timeout)
        try:
            results, metadata = grade_stdio(
                code=test,
                all_inputs=in_outs["inputs"],
                all_outputs=in_outs["outputs"],
                timeout=timeout,
            )
            return results, metadata
        except Exception as e:
            return [-4], {
                "error_code": -4,
                "error_message": f"Error during testing: {e}",
            }
        finally:
            signal.alarm(0)


def reliability_guard(maximum_memory_bytes=None):
    """LiveCodeBench's guard (NOT a sandbox; the container is): disables destructive functions in the child."""

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.quit = None

    import os as _os

    _os.environ["OMP_NUM_THREADS"] = "1"

    _os.kill = None
    _os.system = None
    _os.putenv = None
    _os.remove = None
    _os.removedirs = None
    _os.rmdir = None
    _os.fchdir = None
    _os.setuid = None
    _os.fork = None
    _os.forkpty = None
    _os.killpg = None
    _os.rename = None
    _os.renames = None
    _os.truncate = None
    _os.replace = None
    _os.unlink = None
    _os.fchmod = None
    _os.fchown = None
    _os.chmod = None
    _os.chown = None
    _os.chroot = None
    _os.fchdir = None
    _os.lchflags = None
    _os.lchmod = None
    _os.lchown = None
    _os.getcwd = None
    _os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    builtins.help = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


# ---- LiveCodeBench compute_code_generation_metrics.check_correctness, with a pipe -----------------------------------

def _temp_run(sample, generation, timeout, connection):
    try:
        os.close(RESULT_FD)          # the solution must not reach the result channel
    except OSError:
        pass
    res, metadata = run_test(sample, test=generation, timeout=timeout)
    connection.send((res, metadata))
    connection.close()


def check_correctness(sample, generation, timeout):
    receiver, sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(target=_temp_run, args=(sample, generation, timeout, sender))
    process.start()
    sender.close()
    count = len(json.loads(sample["input_output"])["inputs"])
    deadline = (timeout + 1) * count + 5
    result = None
    started = time.time()
    try:
        if receiver.poll(deadline):
            result = receiver.recv()
    except (EOFError, OSError):
        result = None
    process.join(timeout=max(0.0, deadline - (time.time() - started)) if result is None else 5)
    if process.is_alive():
        process.kill()
        process.join()
    if not result:
        # consider that all tests failed (LiveCodeBench: "global timeout")
        return [-1 for _ in range(count)], {"error_code": -1, "error_message": "global timeout or the child died",
                                            "child_exit": process.exitcode}
    return result


def decode_private(raw):
    if raw is None or raw == "":
        return [], "none"
    if isinstance(raw, list):
        return raw, "json"
    try:
        return json.loads(raw), "json"
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))), "pickle"


def _write_result(out):
    data = json.dumps(out, default=str).encode("utf-8")
    view = memoryview(data)
    while view:
        written = os.write(RESULT_FD, view)
        view = view[written:]
    os.close(RESULT_FD)


OUTPUT_LOG = "/tmp/solution_output.log"


def main():
    global RESULT_FD
    RESULT_FD = os.dup(1)
    # Whatever reaches file descriptors 1 and 2 outside LiveCodeBench's capture (the solution's stderr, os.write,
    # tracebacks) goes to a file on the 64 MB tmpfs, never to the result channel; its tail is returned.
    try:
        log_fd = os.open(OUTPUT_LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    except OSError:
        log_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    request = json.load(sys.stdin)
    private, decoded = decode_private(request.get("private_tests"))
    tests = list(request.get("public_tests") or []) + list(private)
    if request.get("mode") == "count":    # inventory: no code is run
        sizes = [len(t["input"]) + len(t["output"]) for t in tests]
        _write_result({"tests_total": len(tests), "public": len(request.get("public_tests") or []),
                       "private": len(private), "decode": decoded, "max_test_bytes": max(sizes) if sizes else 0,
                       "total_test_bytes": sum(sizes),
                       "testtypes": sorted({t.get("testtype", "?") for t in tests})})
        return
    timeout = int(request.get("timeout", 6))
    in_outs = {"inputs": [t["input"] for t in tests], "outputs": [t["output"] for t in tests],
               "fn_name": request.get("fn_name") or None}
    sample = {"input_output": json.dumps(in_outs)}
    started = time.time()
    results, metadata = check_correctness(sample, request["code"], timeout)
    fixed = []
    for entry in results:
        if isinstance(entry, bool) or entry is None:
            fixed.append(bool(entry))
        elif isinstance(entry, (int, float)):
            fixed.append(entry)
        else:                        # e.g. a numpy result of `prediction == gt_out` (LiveCodeBench takes .item(0));
            try:                     # such an entry is always followed by a failure code, so the problem fails anyway
                fixed.append(bool(entry))
            except Exception:
                fixed.append(False)
    tail = ""
    try:
        with open(OUTPUT_LOG, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1500))
            tail = handle.read().decode("utf-8", "replace")
    except OSError:
        pass
    _write_result({"results": fixed, "metadata": metadata, "tests_total": len(tests), "decode": decoded,
                   "grading_seconds": round(time.time() - started, 3), "output_tail": tail})


if __name__ == "__main__":
    main()
