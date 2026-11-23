"""q38_candidates.py PLAN_JSON [--write]: the Qwen3.8-27B arms from the first-contact probe (job 123).

Each 27B arm copies its 9B arm (lq_final, bf16_window, fp32_packed: same flags, env and numerical policy) and changes
only: the model (Qwen/Qwen3.8-27B-FP8 at c3a462c from HF_HUB_CACHE=/data/hf/hub, offline), the state slots (the
probe's balance of state and KV for 1k/1k requests, from the server logs) and, if the probe needed it, the CUDA-graph
batch limit. Prints the arms; --write adds or replaces them in harness/config/candidates.json.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "harness" / "config" / "candidates.json"
ARMS = {"leapquant": ("lq_final", "lq_final_q38"), "bf16_window": ("bf16_window", "bf16_window_q38"),
        "fp32": ("fp32_packed", "fp32_packed_q38")}
MODEL = {"model": "Qwen/Qwen3.8-27B-FP8", "model_revision": "c3a462cc3d38cf351f3b26cbd763469eeb74ef81",
         "weights_dtype": "fp8 e4m3, 128x128 blocks (published FP8 checkpoint)"}


def set_flag(flags: list[str], name: str, value: str) -> None:
    if name in flags:
        flags[flags.index(name) + 1] = value
    else:
        flags.extend([name, value])


def build(plan: dict, doc: dict) -> list[dict]:
    by_id = {c["candidate_id"]: c for c in doc["candidates"]}
    out = []
    for probe_arm, (base_id, new_id) in ARMS.items():
        entry = plan["arms"].get(probe_arm)
        if not entry or "balanced_slots" not in entry:
            raise SystemExit(f"probe has no balanced slot count for {probe_arm}: {entry}")
        slots = int(entry["balanced_slots"])
        gbs = (entry.get("pools") or {}).get("cuda_graph_max_bs_decode") or 512
        c = copy.deepcopy(by_id[base_id])
        c["candidate_id"] = new_id
        cfg = c["config"]
        cfg.update(MODEL)
        cfg["configured_slots"] = slots
        cfg["max_running_requests"] = slots // 3
        set_flag(cfg["server_flags"], "--max-mamba-cache-size", str(slots))
        set_flag(cfg["server_flags"], "--cuda-graph-max-bs-decode", str(gbs))
        env = cfg.setdefault("env", {})
        env.update({"HF_HUB_CACHE": "/data/hf/hub", "HF_HUB_OFFLINE": "1"})
        c["label"] = f"{by_id[base_id]['label']} - Qwen3.8-27B FP8, {slots} state slots (balanced for 1k/1k)"
        c["role"] = f"Qwen3.8-27B arm (backlog B75); same policy as {base_id}"
        c["tuning_note"] = (f"job 123 probe (runbook/q38_probe.py plan): pool budget {entry['budget_gib']:.2f} GiB, "
                            f"{entry['state_bytes_per_slot'] / 2**20:.1f} MiB per slot, "
                            f"{entry['kv_bytes_per_token'] / 1024:.0f} KiB per KV token -> {entry['balanced_requests']} "
                            f"requests of 2,048 tokens ({slots} slots); CUDA graphs to batch {gbs}")
        c.pop("decided", None)
        out.append(c)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("plan")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads(Path(args.plan).read_text())
    text = CANDIDATES.read_text(encoding="utf-8")
    doc = json.loads(text)
    arms = build(plan, doc)
    for c in arms:
        flags = c["config"]["server_flags"]
        print(c["candidate_id"], "slots", c["config"]["configured_slots"], "graphs",
              flags[flags.index("--cuda-graph-max-bs-decode") + 1], "|", c["tuning_note"])
    if args.write:
        ids = {c["candidate_id"] for c in arms}
        doc["candidates"] = [c for c in doc["candidates"] if c["candidate_id"] not in ids] + arms
        CANDIDATES.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + ("\n" if text.endswith("\n") else ""),
                              encoding="utf-8")
        print(f"wrote {len(arms)} arms to {CANDIDATES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
