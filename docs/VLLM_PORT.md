# vLLM port (task T14): first port

`integrations/vllm/leapquant_first_port.patch` applies to vLLM commit
`145cc6428b65db1c4b26ce8d28c7dec444563561`. It is switched on with `VLLM_GDN_LEAPQUANT=1`.

**Status: untested in vLLM.** The patch byte-compiles and has never been imported or run.
The logic it calls (`BlockEngineState` in `kernel/leapquant_engine.py`) is tested on a CPU by
`kernel/block_engine_test.py`, where it is bit-identical to the reference core.

## Scope

| Supported | Not supported (fails at start-up with a reason) |
|---|---|
| Prefix caching off (mamba cache mode "none") | Cache mode "align" |
| Decode-only batches and the decode rows of mixed batches | Speculative decoding |
| `--mamba-ssm-cache-dtype float32` | Other state dtypes |

## Why vLLM needs more than SGLang did

- vLLM has no window, cursor or flush for these layers. `BlockEngineState` keeps its own
  cursor per layer.
- The state lives in KV-cache blocks. The "slot" is a block id out of the whole block pool, so
  a LeapQuant pool cannot be sized by it. `BlockEngineState` keeps a table from block id to a
  small number of LeapQuant slots.
- Nothing is called when a request ends. When the pool is full, the slots that have not
  decoded for the longest time are reused. If that would take the slot of a request that is
  still decoding, `assign()` raises instead.
- The blocks are padded, so the kernel takes the distance between two blocks as an argument.

## What the patch changes (two files, 204 lines)

| Place in `qwen_gdn_linear_attn.py` | Change |
|---|---|
| `__init__` | Reads the switch and checks the configuration. |
| `bind_kv_cache` (new override) | Builds the state object, before any CUDA graph is captured. |
| `_forward_core_decode_non_spec` | Decode-only batches go to `BlockEngineState.decode`. |
| `_forward_core`, step 2.2 | So do the decode rows of a mixed prefill-and-decode batch. |
| `_forward_core`, after `scatter_states` | `assign()`: the blocks prefill just wrote get a slot and are adopted at their first decode step. |

`leapquant_glue.py` (new) finds this repository through `LEAPQUANT_HOME` and builds the state.

## Assumptions to check on the first run

- With cache mode "none", a request keeps the same state block from the end of prefill to its
  last token.
- `bind_kv_cache` runs once, on the GPU, before the decode CUDA graph is captured.
- `prefill_state_indices` holds one block id per prefill request.
- Kernels launched through `ctypes` on `torch.cuda.current_stream()` are captured by vLLM's
  CUDA graphs like any other kernel.
- The default pool (`max_num_seqs` plus a quarter) is large enough. If the server raises
  "the pool is too small", set `VLLM_GDN_LEAPQUANT_SLOTS` higher. Each slot costs about 21 MB
  over all layers of Qwen3.5-9B.

## What it does not give

Memory. The fp32 block of each request stays allocated. Prefix caching ("align" mode) needs a
flush before vLLM copies a state to a new block, inside `MambaHybridModelState.preprocess_state`;
that is the next step and should be written with a GPU at hand.

## Run it

```
runbook/stage_g_vllm.sh patch
runbook/stage_g_vllm.sh smoke
runbook/stage_g_vllm.sh forward
```
