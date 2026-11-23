# LeapQuant test harness (`lqh`)

Implements the internal test-harness specification of 14 Nov 2026 (revision 3, final). One interface runs the pinned
benchmark profiles and publishes **every metric of every run**. Validation sits between raw evidence and every public
number.

```
pinned contract (config/) -> paired runner (lqh.runner) -> workload driver + server (lqh.driver / pinned InferenceX client)
  -> immutable raw evidence (run directory) -> validation (lqh.validate) -> metrics, SLO goodput, cost (lqh.analyze)
  -> publication bundle (lqh.report) -> independent checker (lqh.checker)
```

Standard library only: driver, validator, metrics, checker, tests. Optional: matplotlib for figures, and `jsonschema`
for the schema cross-check in the tests.

## Commands

From the repository root (`PYTHONPATH=harness`) or inside `harness/`:

| Command | What it does |
|---|---|
| `python -m lqh smoke --out DIR` | Mock server end to end: 12 runs (finite, sustained, open loop × 2 arms × 2 blocks), report, checker. Harness behaviour only. |
| `python -m lqh run PLAN --out RESULTS --port 30000` | Runs a plan from `config/campaign.json`: one server per candidate and block (balanced order), every point, telemetry, validation, metrics. On the GPU host use `runbook/harness_plan.sh`. |
| `python -m lqh metrics RUN_DIR...` | Validates a run and computes `metrics.json`. |
| `python -m lqh report RESULTS... --out reports/harness_<name>_<UTC>` | Builds the publication bundle (never overwrites: the folder must be new). |
| `python -m lqh check BUNDLE` | Independent checker: recomputes every displayed number and chart point from raw files. Exit 0 only if all reproduce. |
| `python -m lqh.quality RESULTS` | 16-bit qualification decision from `quality_items_<arm>.jsonl` (rule frozen in the campaign). |
| `python -m lqh.kernel_cycle RESULTS` | Summary of the full-cycle kernel timing (paired ratios, gate). |
| `python -m lqh.nll measure/compare/make-docs` | Teacher-forced NLL by position through the serving API. |
| `python -m lqh.carried serving RESULTS_ROOT --out DIR` / `accuracy --reference fp32=RUN.json --arm NAME=RUN.json` | Carried history: every earlier client log (all its metrics, source hash, configured slots, admitted running / waiting per point, peak memory, build) and the strict paper-lane accuracy runs, labelled carried and never repetitions. |
| `runbook/harness_report.sh "label"` (GPU host) | Carried imports, the campaign report into a new `~/lqexp/reports/harness-campaign_<UTC>`, and the checker; runs automatically at the end of every harness job. |
| `python -m lqh.historical --inventory evidence/historical_inventory.json --out reports/...` | The seven historical graphics re-plotted from raw files, with sidecars. |
| `python -m lqh.evidence` | Paper crosswalk, claim registry, memory reconciliation (`evidence/`). |
| `python -m lqh.sanitizer historical results --out ...` | compute-sanitizer records from complete logs. |
| `python -m lqh infx-command [--verify DIR]` | Prints the pinned InferenceX command, or checks a checkout against the pins. |
| `cd harness/tests && python -m unittest discover -p "test_*.py"` | 74 fixture tests (accounting, cost, validation, statistics, quality, report regressions, adapter, config). |

GPU host runbooks (in the tree root):

- `runbook/harness_setup.sh`: fetches and checks the pinned InferenceX client.
- `runbook/harness_plan.sh`: runs any plan.
- `harness_kernel_cycle.sh`
- `harness_sanitizers.sh`
- `harness_lifecycle.sh`
- `quality_16bit.sh`
- `harness_longcontext.sh`
- `harness_compounding.sh`

## Run directory (one point)

| File | Content |
|---|---|
| `run_manifest.json` | **Identity:** run, experiment, plan, cell, pair and block ids; block order position; profile and version; comparison class; carried flag and origin ids; evidence kind. **Candidate:** id with the SHA-256 of its full configuration. **Clock:** origin as UTC and Unix time, monotonic offsets. **Device:** GPU UUID, name, memory, driver, PCI bus. **Workload, server and build:** command, non-default flags, environment, and the build facts from `runbook/manifest.sh`. **Also:** SLO and pricing ids, native upstream aggregates, and the hash of every artifact. |
| `requests.jsonl` | One line per attempt: logical id, attempt, retry, session and parent ids; phase; input, requested and output tokens with their source; scheduled, send, first-token, last-token, terminal and deadline times; status, error, finish reason. |
| `stream_events.jsonl.gz` | Per attempt: the arrival of every content chunk (µs after send) and its exact token count. The token count comes from SGLang's continuous usage statistics; if unknown it is null, and the chunk semantics are stated. |
| `server_events.jsonl` | Capacity facts, plus decode and prefill lines. Capacity facts: configured slots, KV tokens, max running. Each line: running requests (the actual batch), queue depth, token usage, CUDA graph, generation rate, retractions. |
| `memory_and_power.jsonl` | **GPU samples (0.5 s):** memory, power, SM clock, temperature, utilisation. **Allocations** from the server log: weights, KV, recurrent state, LeapQuant state, graphs. **Snapshots:** before model load, after pools, steady, peak, after drain. |
| `driver_summary.json` | Phases (warmup/main, ramp/steady/drain, arrivals/drain), configuration, exit status. |
| `native/` | Pinned InferenceX client: its raw result JSON (with per-request arrays) and log. |
| `validation.json` | Every check with its id, severity and detail. Final status: `measured`, `invalid`, `crashed`, `not_run`, `unsupported`, `skipped` or `smoke_only`. |
| `metrics.json` | **Every metric of the glossary** (~95 names; 60+ computed per lane, the rest null with a reason). Plus the headline, capacity (configured slots / admitted running / waiting), SLO and global constraints, interactivity slices, native aggregates and parity, site metrics (AA-SLT four, InferenceX per-GPU fields), AA API v3 equivalents, cost, energy. |

## What a bundle publishes

`index.html` contains:

- **Run status** counts.
- **Main table:** one row per plan, cell and candidate. Columns: output tok/s median/min/max, SLO-qualified tok/s, eligible runs, error rate, TTFT/TPOT/ITL.
- **Paired comparisons:** geometric mean of per-block ratios with a 95% t interval. Mean of ratios and ratio of means are shown next to it.
- **Capacity table:** configured slots, server max running, KV tokens, client concurrency, admitted running and waiting requests, for every run.
- **Latency and per-user speed** p50/p90/p95/p99 for every run.
- **Cost:** null with its reason while `config/pricing.json` has no price.
- **Throughput-frontier figures** with sidecars.
- **Claim ledger** and **unmeasured points**.
- **Every run's full metric set**, validation checks, allocations and parity.
- **Glossary.**

Exports: `runs.csv/json` (every metric of every run, long form), `summary.csv/json`, `claims.json`, `unmeasured.csv`,
`glossary.json`, `sources.json`, `displayed_values.json`, `figures/*.png|json|csv`, `RELEASE_NOTES.md`,
`CAPTIONS.md`, `MANIFEST.sha256`, `check_report.json`. Nothing is published or submitted automatically.

## Profiles (config/profiles.json)

| Profile | Contract | Status |
|---|---|---|
| `infx_fixed_seq` | Pinned InferenceX fabf7ba, run through a thin adapter. `client_argv` is followed item by item; the only addition is `--save-detailed`. Uses 2×C warmup, 10×C prompts, infinite rate, ignore-EOS. Drain is included. | Implemented; installed on the host and verified. Parity with the client's own aggregates was exact against the mock. |
| `aa_slt_local` | AA-SLT method: 1000/1000, immediate replacement, 180 s steady phase. Uses separate ramp, steady and drain timestamps and counts in-window tokens only. Sweep 1..64, then +64 to the declared local plateau. | Implemented |
| `agentx_local` | agentx-harness 1.0.14, current preset (verified). | **Blocked:** lifecycle qualification not complete; the harness is not installed. |
| `aa_api_v3_metrics` | Local metric equivalents of AA API v3. | Computed per run. No 5-day history or score is manufactured. |
| `paper_reproduction` | Crosswalk (`evidence/paper_crosswalk.json`), kernel full cycle, 16-bit qualification. | Pure-decode and offline lanes planned. |
| `local_diagnostics` | 128/512, long decode, open loop. | Implemented |

## Measured / blocked ledger (15 Nov 2026, 01:35 UTC)

**Measured by this harness so far:** mock smoke runs only. These have no performance meaning: 12 runs, 2,059
displayed values reproduced by the independent checker. The pinned InferenceX client was also run against the mock
on the GPU host: parity was exact on all 8 aggregates and the checker passed on 142 values.

**Queued on the H100 (job runner, in this order)** after the strict accuracy run (092b):

| Job | Order of work | What |
|---|---|---|
| 100 | step 1 | Full state-advancing layer cycle (one CUDA graph: 16 steps + fold) vs SGLang fp32/bf16 window kernels. 10 interleaved repetitions; batches 512, 128, 32, 8 and 1. |
| 100b | §14 | compute-sanitizer memcheck, initcheck and racecheck on the final kernels and the full-cycle graph, at batches 1, 32 and 512. |
| 100c | §13 | Engine-level lifecycle: restore on outputs (fresh and moved buffers), fused slot copy, fold/store, engine, compressed state. |
| 101 | §12 | Validation pair: bf16_window vs lq_combined on decode 128/512 c64, pinned InferenceX 1k/1k c16, and AA-SLT c16. |
| 102 / 103 | step 2 | 16-bit qualification: AIME 2026 and BeyondAIME. Arms: fp32, bf16_rounded, bf16_window, LeapQuant. Paper sampling, 3 seeds, retraction-free. |
| 103b | step 2 | GPQA-Diamond: runs only if the dataset's terms are accepted, otherwise records BLOCKED. |
| 104 | step 3 | Memory per arm (13 candidates): configured slots, running and waiting requests. |
| 105 | step 4 | Paired pilot: 128/512 at c512, 5 blocks. |
| 105b / 105c | §15 / §13 | Long-context NLL (1k–32k), and the compounding test (one layer at a time vs all layers). |
| 106 / 107 | full matrix | InferenceX 1k/1k and 8k/1k sweeps at concurrency 1..1024 (6 arms); AA-SLT sweep to the plateau (4 arms). |

**Blocked, with reasons:**

- **GPQA-Diamond:** the dataset is gated. The HF account behind the host's token must accept the terms, and a fine-grained token needs read access to gated repositories.
- **LiveCodeBench v6:** needs a sandboxed scorer. Its private tests are pickled, and executing model-written code needs isolation.
- **AgentX:** see the profile table.
- **Dollar metrics:** null until `config/pricing.json` has a price with its source.
- **Paper pure-decode and offline-serving lanes:** planned.
- **The paper's early stop on repetitive output:** method not published, so not reproduced.

**Historical evidence:** `reports/harness-historical_*` has the seven graphics re-plotted from raw files, with
sidecars, plus the archive's problems, sanitizer records, claim registry, paper crosswalk and gate registry.
