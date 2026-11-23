"""q38_probe.py: helpers for runbook/q38_probe.sh, the first GPU contact with Qwen3.8-27B (backlog B75, 15 Nov 2026).

  smoke --port P --model M --arm NAME --out DIR   greedy completions (agreement with fp32 later) and sampled chat
                                                  answers (the FlashInfer top-k/top-p path) against a running server
  pools LOG                                       memory pools from an SGLang server log, as JSON
  plan DIR                                        per-arm cost per state slot and per KV token, the pool budget, and
                                                  the slot count that balances state and KV for 2,048-token requests;
                                                  greedy agreement of each arm with fp32. Writes plan.json and plan.txt.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "The capital of Australia is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Question: A train travels 120 km in 1.5 hours. What is its average speed in km/h?\nAnswer:",
    "In 1905, Albert Einstein published four papers that",
    "The three primary colors of light are",
    "Translate to French: 'The weather is beautiful today.'\nFrench:",
    "List the first five prime numbers:",
    "Photosynthesis is the process by which",
]
REQUEST_TOKENS = 2048          # 1k/1k: 1,024 prompt + 1,024 answer tokens (InferenceX), ~2,000 for AA-SLT
SLOTS_PER_REQUEST = 3          # SGLang's no_buffer radix strategy: 3 state slots per running request


def post(url, body, timeout=600):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def smoke(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    record = {"arm": args.arm, "model": args.model, "greedy": [], "sampled": [], "errors": []}
    for prompt in PROMPTS:
        try:
            started = time.time()
            reply = post(f"{base}/v1/completions", {"model": args.model, "prompt": prompt, "max_tokens": 96,
                                                     "temperature": 0.0})
            choice = reply["choices"][0]
            record["greedy"].append({"prompt": prompt, "text": choice["text"], "finish_reason": choice.get("finish_reason"),
                                     "completion_tokens": (reply.get("usage") or {}).get("completion_tokens"),
                                     "seconds": round(time.time() - started, 3)})
        except Exception as error:
            record["errors"].append(f"greedy: {type(error).__name__}: {error}"[:300])
    for seed in (1, 2):
        try:
            reply = post(f"{base}/v1/chat/completions", {
                "model": args.model, "messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}],
                "max_tokens": 64, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5,
                "seed": seed})
            choice = reply["choices"][0]
            message = choice.get("message") or {}
            record["sampled"].append({"seed": seed, "text": (message.get("content") or "") + (message.get("reasoning_content") or ""),
                                      "finish_reason": choice.get("finish_reason"),
                                      "completion_tokens": (reply.get("usage") or {}).get("completion_tokens")})
        except Exception as error:
            record["errors"].append(f"sampled: {type(error).__name__}: {error}"[:300])
    (out / f"smoke_{args.arm}.json").write_text(json.dumps(record, indent=1))
    ok = (not record["errors"] and len(record["greedy"]) == len(PROMPTS) and len(record["sampled"]) == 2
          and all(g["completion_tokens"] for g in record["greedy"]) and all(s["completion_tokens"] for s in record["sampled"]))
    print(f"{args.arm}: {len(record['greedy'])} greedy, {len(record['sampled'])} sampled, {len(record['errors'])} errors"
          f" -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


def gib(text):
    return float(text)


def pools(log_path):
    text = Path(log_path).read_text(errors="replace")
    found = {"log": str(log_path)}
    m = re.search(r"Load weight end\..*?avail mem=([\d.]+) GB, mem usage=([\d.]+) GB", text)
    if m:
        found["avail_after_weights_gib"], found["weights_gib"] = gib(m.group(1)), gib(m.group(2))
    m = re.search(r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+), conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB", text)
    if m:
        found["slots"], found["conv_gib"], found["ssm_gib"] = int(m.group(1)), gib(m.group(2)), gib(m.group(3))
    m = re.search(r"KV Cache is allocated\. dtype: \S+, #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", text)
    if m:
        found["kv_tokens"], found["kv_gib"] = int(m.group(1)), gib(m.group(2)) + gib(m.group(3))
    m = re.search(r"LeapQuant decode state allocated: (\d+) layers x (\d+) slots x (\d+) heads.*?([\d.]+) GB", text)
    if m:
        found["lq_layers"], found["lq_slots"], found["lq_heads"], found["lq_pool_gib"] = (
            int(m.group(1)), int(m.group(2)), int(m.group(3)), gib(m.group(4)))
    m = re.search(r"Memory pool end\. avail mem=([\d.]+) GB", text)
    if m:
        found["avail_after_pools_gib"] = gib(m.group(1))
    m = re.findall(r"max_total_num_tokens=(\d+),.*?max_running_requests=(\d+),.*?available_gpu_mem=([\d.]+) GB", text)
    if m:
        found["max_running_requests"], found["available_after_graphs_gib"] = int(m[-1][1]), gib(m[-1][2])
    m = re.search(r"mem_fraction_static': ([\d.]+)", text)
    if m:
        found["mem_fraction_static"] = float(m.group(1))
    m = re.search(r"cuda_graph_max_bs_decode': (\d+)", text)
    if m:
        found["cuda_graph_max_bs_decode"] = int(m.group(1))
    return found


def first_difference(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return min(len(a), len(b)) if len(a) != len(b) else None


def plan(args):
    folder = Path(args.dir)
    arms = {}
    for log in sorted(folder.glob("server_*.log")):
        name = log.stem[len("server_"):]
        arms[name] = pools(log)
    result = {"request_tokens": REQUEST_TOKENS, "slots_per_request": SLOTS_PER_REQUEST, "arms": {}}
    lines = [f"Qwen3.8-27B probe: slots balanced for {REQUEST_TOKENS}-token requests ({SLOTS_PER_REQUEST} slots each)"]
    for name, p in arms.items():
        if "kv_tokens" not in p or "slots" not in p:
            result["arms"][name] = {"pools": p, "error": "pools not found in the log (server did not start?)"}
            lines.append(f"  {name}: pools not found in {p['log']}")
            continue
        kv_per_token = p["kv_gib"] * 2**30 / p["kv_tokens"]
        slots = p["slots"]
        state_gib = p["conv_gib"] + p["ssm_gib"]
        per_slot = (p["conv_gib"] * 2**30) / slots
        fixed = 0.0
        if "lq_pool_gib" in p:                       # LeapQuant: the compressed pool has slots + 1 rows; SGLang's
            per_slot += p["lq_pool_gib"] * 2**30 / p["lq_slots"]     # fp32 ssm tensor is fixed (2 rows)
            fixed = p["ssm_gib"] * 2**30
            state_gib += p["lq_pool_gib"]
        else:
            per_slot += p["ssm_gib"] * 2**30 / slots
        budget = state_gib * 2**30 + p["kv_gib"] * 2**30
        per_request = SLOTS_PER_REQUEST * per_slot + REQUEST_TOKENS * kv_per_token
        balanced = int((budget - fixed) // per_request)
        result["arms"][name] = {"pools": p, "kv_bytes_per_token": kv_per_token, "state_bytes_per_slot": per_slot,
                                "budget_gib": budget / 2**30, "balanced_requests": balanced,
                                "balanced_slots": SLOTS_PER_REQUEST * balanced,
                                "probe_requests_kv_bound": p["kv_tokens"] // REQUEST_TOKENS,
                                "probe_requests_state_bound": slots // SLOTS_PER_REQUEST}
        lines.append(f"  {name}: budget {budget / 2**30:.2f} GiB (weights {p.get('weights_gib')} GiB, mem fraction "
                     f"{p.get('mem_fraction_static')}), per slot {per_slot / 2**20:.1f} MiB, KV {kv_per_token / 1024:.0f} KiB"
                     f"/token -> balanced {balanced} requests ({SLOTS_PER_REQUEST * balanced} slots); probe at {slots} slots:"
                     f" KV {p['kv_tokens']:,} tokens, max running {p.get('max_running_requests')}, free after graphs "
                     f"{p.get('available_after_graphs_gib')} GiB")
    smokes = {path.stem[len("smoke_"):]: json.loads(path.read_text()) for path in folder.glob("smoke_*.json")}
    reference = smokes.get("fp32")
    if reference:
        lines.append("  greedy agreement with fp32 (first differing character of 96-token completions; '-' = identical):")
        for name, record in smokes.items():
            if name == "fp32":
                continue
            diffs = []
            for mine, ref in zip(record["greedy"], reference["greedy"]):
                diffs.append(first_difference(mine["text"], ref["text"]))
            result.setdefault("agreement", {})[name] = diffs
            lines.append(f"    {name}: " + ", ".join("-" if d is None else str(d) for d in diffs))
    for name, record in smokes.items():
        lines.append(f"  {name}: {len(record['greedy'])} greedy, {len(record['sampled'])} sampled, "
                     f"{len(record['errors'])} errors" + (f" ({record['errors'][0]})" if record["errors"] else ""))
    (folder / "plan.json").write_text(json.dumps(result, indent=1))
    (folder / "plan.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--port", type=int, default=30000)
    s.add_argument("--model", required=True)
    s.add_argument("--arm", required=True)
    s.add_argument("--out", required=True)
    p = sub.add_parser("pools")
    p.add_argument("log")
    q = sub.add_parser("plan")
    q.add_argument("dir")
    args = parser.parse_args(argv)
    if args.command == "smoke":
        return smoke(args)
    if args.command == "pools":
        print(json.dumps(pools(args.log), indent=1))
        return 0
    return plan(args)


if __name__ == "__main__":
    sys.exit(main())
