# LeapQuant in SGLang

`leapquant_phase1.patch` applies to SGLang commit `c8ea70797157cafab46b3807f850d1c5c061a8d1`
(the file name is historical: the patch now holds phase 1 and phase 2). It adds one flag,
`--enable-linear-leapquant`, and two environment variables:

| Variable | Values | Meaning |
|---|---|---|
| `LEAPQUANT_STATE` | `mirror` (default) | Phase 1. The fp32 state per slot stays; LeapQuant comes on top. Works with the radix cache. Speed only. |
| | `compressed` | Phase 2. No fp32 state per slot: the slot holds the LeapQuant state only (21.3 MB instead of 50.3 MB for Qwen3.5-9B). Works with the radix cache (a cached prefix is a slot's boundary state). Speed and memory. |
| `LEAPQUANT_FAST` | `1` (default) | The fast kernels: `kernel/leapquant_triton.py` (Triton, tensor cores). |
| | `0` | The exact CUDA kernels (bit-identical to the reference core on a CPU; slower). Mirror only. |
| `LEAPQUANT_COMPENSATORS` | `4` (default) | Compensators per head. A default plus `from_layer:count` items, e.g. `4,20:8`: 8 on the model's layers 20 and above. |
| `LEAPQUANT_HOME` | path | This repository, after `make cuda` (the exact kernels are also used for decompression). |
| `LEAPQUANT_FOLD_POLICY` | unset or `stagger` (default from 15 Nov 2026) | Each request folds on its own phase, (tokens + slot) mod 16, so about 1/16 of the batch folds on every step: no fold-step spike in the p99 between tokens (`apply_fold_policy_stagger.py`; the default since `apply_fold_policy_default_stagger.py`, Asif's decision). With the radix cache off the track-boundary flushes are dropped. |
| | `aligned` | The default before 15 Nov: when one request's window is full, every request of the batch folds on that step. |
| | `auto` | Radix cache off: the track-boundary flushes are dropped, then aligned folds. Radix cache on: each request folds on its own step (`apply_fold_policy.py`). |
| `LEAPQUANT_SWEEPS` | unset or `1` (default from 15 Nov 2026) | Sweeps of the fold's compensator fit (`kernel/leapquant_kernel.py` DEFAULT_SWEEPS; backlog B52, Asif's decision). `2` gives the policy of the runs before 15 Nov. |
| `LEAPQUANT_TRACK_FLUSH` | unset (default) | The radix track-boundary flushes (every 256 tokens) are kept unless the fold policy drops them. |
| | `0` | Dropped with the radix cache on or off (`apply_track_flush.py`, backlog B55). With SGLang's `no_buffer` radix strategy, the only one ReplaySSM accepts, nothing reads a decode state at a track boundary: a finished request is cached up to its last fold (`settle_replay`). |

`apply_fold_policy.py`, `apply_fold_policy_stagger.py` and `apply_track_flush.py` are the source of those three
edits (in that order); `leapquant_phase1.patch` already contains them. `apply_fold_policy_default_stagger.py` (15 Nov
2026) runs after them and is NOT in the patch: it makes `stagger` what an unset `LEAPQUANT_FOLD_POLICY` means. `radix_turns_client.py` checks two-turn prompts
whose second turn starts from a radix-cached decode state.

## Status

- Phase 1 with the exact kernels: served in rounds 2-5 (13 Nov 2026, H100), with CUDA graphs on and
  off; answers agree with the fp32 window server.
- Phase 1 and phase 2 with the fast kernels: smoke-tested 14 Nov 2026 (runbook/serving_fast.sh, steps
  S2 and S3): every answer agrees with the fp32 window server for its first 15 words; four of six
  agree to the last word. Serving round 6 measures both against SGLang's own kernels.
- Engine side: kernel/engine_test.py, block_engine_test.py and compressed_test.py (all with `--fast`
  on a GPU), runbook/fast_checks.sh (37 checks, including every recorded trace).

## What the patch changes

| File | Change |
|---|---|
| `arg_groups/fields/exec_.py`, `field_order.py` | The flag `--enable-linear-leapquant`. |
| `arg_groups/attention_hook.py` | Start-up checks: `--enable-linear-replayssm`, a window of at most 16, a float32 state, `LEAPQUANT_HOME`; no mixed chunk, no speculative decoding; `LEAPQUANT_STATE=compressed` refuses `--enable-int8-mamba-checkpoint` and `--enable-hierarchical-cache`. |
| `layers/attention/linear/leapquant_state.py` (new) | Builds one `EngineState` over the state pool and registers it with `MambaPool.register_slot_state`, so it is told when a slot is cleared or copied. Helpers for the two state modes. |
| `layers/attention/linear/gdn_backend.py` | Decode routes to `EngineState.decode`. Compressed: prefill runs on a staging tensor from `EngineState.extend_begin` and is stored by `extend_end`. |
| `layers/attention/hybrid_linear_attn_backend.py` | When one request's window is full, every request of the batch folds on that step (aligned folds); the fold policies and `LEAPQUANT_TRACK_FLUSH` above. |
| `mem_cache/memory_pool.py` | Compressed: `temporal` has two rows, and slot copies (radix cache) are left to the side state. Either mode: the ReplaySSM decode ring, which LeapQuant does not use, has one slot (it was about 9.4 MB per slot in fp32 and was never charged to the memory budget). |
| `configs/mamba_utils.py` | Compressed: the memory budget charges the LeapQuant bytes per slot instead of the fp32 state. |
| `mem_cache/mamba_checkpoint_pool.py` | Mirror with `--enable-int8-mamba-checkpoint`: cached prefixes in the LeapQuant boundary format. |

With the flag off, `self.leapquant` is `None` and every path is the code that is there today.

## How it works

Both modes ride on SGLang's replay-window option: a cursor per slot that counts the records in
the window, and a forced flush on prefix-cache boundaries.

Mirror (phase 1):
1. Prefill is unchanged. It writes the fp32 state `temporal[layer][slot]`.
2. The first decode step of a slot adopts it (stores it as a LeapQuant boundary state).
3. Every decode step reads the int8 state and the window and appends one record.
4. When the window is full, or the flush is forced, the window folds, and the full-precision state
   is written back to `temporal[slot]`, so the prefix cache reads what it reads today.
5. When SGLang clears a slot or copies a cached state into it, the slot is marked for adoption again.

Compressed (phase 2):
1. Prefill of a layer runs on a staging tensor with one row per request of the batch: zero for a
   new request, the request's own state (decompressed) for a later chunk of a chunked prefill.
2. At the end of the layer's prefill the staging rows are stored in the LeapQuant pool.
3. Decode reads and writes only the pool; nothing is adopted or written back.
4. The staging tensor exists only while one layer's prefill runs.
5. The radix cache keeps a slot's state as of its last fold; in this mode that is the slot's
   boundary state. `MambaPool.copy_from` (a prefix hit, a prefix checkpoint) reaches
   `EngineState.copy_slots`, which copies the boundary states; `clear_slots` resets a slot,
   so a new request starts from zero.

## Not supported

- Compressed: `--enable-int8-mamba-checkpoint`, `--enable-hierarchical-cache` (refused at start),
  host offload of slot states (`get_cpu_copy`) and PD disaggregation (they read the fp32 state).
- Either mode: speculative decoding, `--enable-mixed-chunk`, KDA models, unified memory, the GDN
  prefill CUDA graph (`SGLANG_ENABLE_GDN_PREFILL_GRAPH_EXTEND`, compressed only).

## Run it

```
export LEAPQUANT_HOME=~/leapquant          # this repository, after `make cuda`
runbook/stage_f_end_to_end.sh patch
RESULTS=results/run-name runbook/serving_fast.sh   # smoke both modes, then serving against SGLang's kernels
RESULTS=results/run-name runbook/serving_radix.sh  # radix cache on: cache hits on the compressed state, 512 running
RESULTS=results/run-name runbook/accuracy_fast.sh  # GSM8K, MMLU-Pro, AIME: fp32 twice, LeapQuant, bf16 window
```
