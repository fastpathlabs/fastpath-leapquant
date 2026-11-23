"""
leapquant_engine.py

The engine side of LeapQuant: everything a serving engine needs around the
kernel, in one object. The SGLang patch (integrations/sglang) is a thin layer
over this file, and kernel/engine_test.py tests this file on a CPU.

This is integration PHASE 1. It replaces the decode kernel and nothing else:

    - The engine keeps its fp32 state tensor (SGLang: `temporal`, one
      [slots, HV, V, K] tensor per layer). Prefill still writes it. The prefix
      cache still copies it. None of that code changes.
    - A request that starts to decode is ADOPTED: its fp32 state is folded into
      a LeapQuant boundary state (use cases UC3 and UC4). From then on the
      decode kernel reads and writes only the LeapQuant pool.
    - Every time a window is folded, the full-precision state is also written
      back to the engine's fp32 tensor. So that tensor is current at every fold
      boundary, which is exactly what SGLang's replay-window option guarantees
      and what its prefix cache already relies on.

What phase 1 does not give: the memory saving. The fp32 tensor stays allocated
and the LeapQuant pool comes on top (about 21 MB per request for Qwen3.5-9B).

Phase 2, compressed only (compressed_only=True; SGLang: LEAPQUANT_STATE=compressed).
The engine has no fp32 state per slot:
    - Prefill (extend) of a layer asks extend_begin() for a small fp32 STAGING
      tensor, one row per request of the batch: zero for a new request, the
      request's own state (decompressed) for a later chunk of a chunked prefill.
      The engine's prefill kernel runs on it, and extend_end() compresses the
      final states into the LeapQuant pool.
    - decode() is called with ssm_states=None: nothing to adopt, nothing to write back.
    - A prefix cache keeps a slot's state as it was at its last fold: that is the
      boundary state. copy_slots() copies boundary states from slot to slot.
  So the state of a request costs its boundary state plus its window, about 21 MB
  for Qwen3.5-9B instead of 50 MB in fp32, and the staging rows exist only while a
  layer's prefill runs.

Two sets of kernels
    exact   kernel/leapquant_kernel_*.{cpp,cu}: bit-identical to the reference core on a CPU.
            What the tests use, and the oracle for the fast kernels.
    fast    kernel/leapquant_triton.py: tensor cores, the arithmetic in another order (see there).
            The default on a GPU. kernel/verify_kernel.py --fast and kernel/bench_triton.py
            check it against the exact kernels and the reference core.

Who is in charge of a slot's state
    valid[layer][slot] == 0   the engine's fp32 tensor. The LeapQuant pool
                              holds garbage for this slot.
    valid[layer][slot] == 1   the LeapQuant pool. The fp32 tensor holds the
                              state as of the last fold.
    The engine sets a slot to 0 whenever it clears it or copies a state into
    it (reset_slots, copy_slots). decode() sets it to 1 when it adopts.

The window cursor (records in the window, one int32 per slot) belongs to the
engine. SGLang's replay-window option already keeps one, advances it once per
decode step and forces a flush on prefix-cache boundaries. decode() takes the
cursor of this step and the force flags as arguments and changes neither.
"""

import os

import torch
import torch.nn.functional as F

from leapquant_kernel import Kernel, StatePool, low_bit_scale_group, paper_compensator_count

try:
    # The fast kernels (Triton, tensor cores). Not on machines without triton or a GPU.
    import leapquant_triton as fast_kernels
except Exception:  # noqa: BLE001 - any import failure means: no fast kernels here
    fast_kernels = None


def fast_kernels_available(device):
    return fast_kernels is not None and torch.device(device).type == "cuda"


def _resolve_fast(fast, device):
    """fast=None: the fast kernels on a GPU where triton is there, else the exact ones."""
    if fast is None:
        return fast_kernels_available(device)
    if fast and not fast_kernels_available(device):
        raise ValueError("the fast LeapQuant kernels need a CUDA device and triton")
    return bool(fast)


def choose_kernel(fast, device, scale_group, key_channels):
    """The fast kernels keep one step per column. A real scale group uses the exact kernel.

    fast=True with a real group is an error. fast=None with a real group falls back
    to the exact kernel. 8-bit (scale_group 0) is unchanged.
    """
    grouped = 0 < scale_group < key_channels
    if grouped and fast is True:
        raise ValueError("grouped column steps use the exact kernel; the fast kernels keep one step per column")
    if grouped:
        return False
    return _resolve_fast(fast, device)


def parse_layer_selection(spec, layer_ids):
    """
    Which model layers carry LeapQuant (state-cache plan v2, task TQ.5): the layer switch behind the
    compounding test (one layer changed against all layers changed) and behind a per-layer policy.

        spec       "all" (or empty), or comma-separated model layer ids and ranges: "30", "0-13", "0-19,24"
        layer_ids  the model's linear-attention layer ids
    Returns the selected ids as a set. Every id named must be a linear-attention layer.
    """
    layer_ids = list(layer_ids)
    spec = (spec or "all").replace(" ", "").lower()
    if spec == "all":
        return set(layer_ids)
    chosen = set()
    for part in spec.split(","):
        if not part:
            continue
        try:
            if "-" in part:
                first, last = (int(value) for value in part.split("-", 1))
                chosen.update(range(first, last + 1))
            else:
                chosen.add(int(part))
        except ValueError as error:
            raise ValueError(f"LEAPQUANT_LAYERS={spec!r}: expected 'all' or e.g. '0-13,30'") from error
    unknown = sorted(chosen - set(layer_ids))
    if unknown and not set(range(min(unknown), max(unknown) + 1)) & set(layer_ids):
        raise ValueError(f"LEAPQUANT_LAYERS={spec!r}: no linear-attention layer among {unknown}")
    # A range may span full-attention layers (e.g. "0-13"); only the linear ones are kept.
    return chosen & set(layer_ids)


def resolve_compensator_counts(compensator_count, layers, bits_per_entry):
    """None becomes the paper rank for `bits_per_entry`. A list or an explicit int is kept."""
    if compensator_count is None:
        return [paper_compensator_count(bits_per_entry)] * layers
    if isinstance(compensator_count, (list, tuple)):
        counts = list(compensator_count)
    else:
        counts = [compensator_count] * layers
    if len(counts) != layers:
        raise ValueError(f"compensator_count has {len(counts)} entries for {layers} layers")
    return counts


class EngineState:
    """LeapQuant state of every linear-attention layer of one engine."""

    def __init__(self, layers, slots, value_heads, key_heads, key_channels, value_channels, max_rows,
                 window_length=16, compensator_count=None, bits_per_entry=8, device="cuda", cursor=None,
                 padding_slot=0, scratch=None, fused_gates=True, fast=None, compressed_only=False,
                 scale_group=None):
        """
        layers        number of linear-attention layers
        slots         size of the engine's state pool (including its padding slot)
        key_heads     number of key heads H, or None to read it from the first decode call
        max_rows      largest decode batch
        cursor        the engine's per-slot cursor tensor (int32 [slots]) or None. If given,
                      reset_slots() and copy_slots() also set the cursor of those slots to 0.
        padding_slot  the slot padded batch rows point at (0 in SGLang), or None
        fast          True: the fast kernels; False: the exact ones; None: fast on a GPU with triton.
                      A real scale group (6-bit and 4-bit, unless scale_group is passed) uses the
                      exact kernel. fast=True with a real group raises.
                      compensator_count may also be a list with one entry per layer.
                      compensator_count=None and scale_group=None follow the paper for bits_per_entry.
        compressed_only  phase 2 (see the module text). Needs the fast kernels.
        """
        self.layers = layers
        self.slots = slots
        self.value_heads = value_heads
        self.key_heads = key_heads
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.window_length = window_length
        self.padding_slot = padding_slot
        # True: the kernel computes the gates itself (one launch). False: gates() does, with about
        # ten small tensor operations per layer; that path is bit-identical to the reference core
        # given the same gates, and is what the exact CPU tests use.
        self.fused_gates = fused_gates
        self.cursor = cursor
        self.device = torch.device(device)
        if scale_group is None:
            scale_group = low_bit_scale_group(bits_per_entry)
        self.scale_group = int(scale_group)
        self.fast = choose_kernel(fast, self.device, self.scale_group, key_channels)
        self.compressed_only = bool(compressed_only)
        if self.compressed_only and not self.fast:
            raise ValueError("compressed_only needs the fast kernels (a CUDA device and triton)")
        self.kernel = Kernel(self.device)
        counts = resolve_compensator_counts(compensator_count, layers, bits_per_entry)
        self.pools = [
            StatePool(slots, value_heads, key_channels, value_channels, window_length=window_length,
                      compensator_count=count, bits_per_entry=bits_per_entry, device=self.device,
                      scale_group=self.scale_group)
            for count in counts
        ]
        self.valid = torch.zeros((layers, slots), dtype=torch.int32, device=self.device)
        # Working memory for adopt and fold of the exact kernels, shared by all layers. The fast
        # kernels need none. (`scratch` lets several EngineState objects share it.)
        if scratch is None and not self.fast:
            scratch = torch.empty((max_rows, value_heads, value_channels, key_channels), dtype=torch.float32,
                                  device=self.device)
        self.scratch = scratch
        # The decode inputs that are the same for every layer of a step (the rows' slots, the records each row
        # folds, the fold's work list) are built at the first layer of the step and reused by the others:
        # about ten small kernels fewer per layer (also inside a CUDA graph). LEAPQUANT_STEP_REUSE=0: per layer.
        self.step_reuse = os.environ.get("LEAPQUANT_STEP_REUSE", "1").strip() != "0"
        self._step_key = None
        self._step_layer = None
        self._step_inputs = None
        # The radix cache's slot copy (compressed state, one per request with the cache on): every boundary tensor of
        # every layer and the slot flags in one launch (leapquant_triton.SlotRows) instead of about 245 small kernels.
        # LEAPQUANT_FUSED_SLOT_COPY=0: field by field. Compiled here, on the padding slot, not at the first copy.
        self._slot_rows = None
        if (self.compressed_only and self.fast and self.device.type == "cuda"
                and os.environ.get("LEAPQUANT_FUSED_SLOT_COPY", "1").strip() != "0"):
            tensors = [getattr(pool, name) for pool in self.pools for name in StatePool.BOUNDARY_FIELDS]
            tensors += [self.valid[layer] for layer in range(layers)]
            self._slot_rows = fast_kernels.SlotRows(tensors)
            warm = torch.full((1,), padding_slot if padding_slot is not None else 0, dtype=torch.int64,
                              device=self.device)
            self._slot_rows.copy(warm, warm)

    def bytes_allocated(self):
        total = (self.scratch.numel() * 4 if self.scratch is not None else 0) + self.valid.numel() * 4
        for pool in self.pools:
            for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS:
                tensor = getattr(pool, name)
                total += tensor.numel() * tensor.element_size()
        return total

    # ---- what the engine calls when a slot changes owner -------------------------
    # (the SlotIndexedState protocol of SGLang's MambaPool.register_slot_state)

    def reset_slots(self, indices):
        """The engine cleared these slots: its fp32 tensor is in charge again."""
        if indices.numel() == 0:
            return
        index = indices.to(device=self.device, dtype=torch.long)
        self.valid[:, index] = 0
        if self.cursor is not None:
            self.cursor[index] = 0

    def copy_slots(self, src_index, dst_index):
        """
        The engine copied states from src to dst (a prefix-cache hit, UC4; a prefix checkpoint).
          mirror:      it copied the fp32 states; dst is adopted again at its first decode step.
          compressed:  there is no fp32 state to copy, so the boundary states are copied here. The
                       engine copies only flushed states (SGLang: MambaPool.copy_from), so the window
                       of dst starts empty; the engine resets its cursor.
        """
        if not self.compressed_only:
            self.reset_slots(dst_index)
            return
        if dst_index.numel() == 0:
            return
        src = src_index.to(device=self.device, dtype=torch.long).reshape(-1)
        dst = dst_index.to(device=self.device, dtype=torch.long).reshape(-1)
        if self._slot_rows is not None and dst.shape[0] == 1 and src.shape[0] == 1:
            # One pair, the radix cache's copy per request: one launch (a slot copied onto itself is harmless).
            self._slot_rows.copy(src, dst)
        else:
            # Several pairs: gather, then scatter (sources and destinations may overlap).
            for pool in self.pools:
                pool.copy_boundary_state(src, dst)
            self.valid[:, dst] = self.valid[:, src]
        if self.cursor is not None:
            self.cursor[dst] = 0

    def get_cpu_slots(self, indices):
        """A host copy of the LeapQuant state of these slots (all layers)."""
        index = indices.to(device=self.device, dtype=torch.long)
        saved = {"valid": self.valid[:, index].to("cpu")}
        for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS:
            saved[name] = [getattr(pool, name)[index].to("cpu") for pool in self.pools]
        return saved

    def load_cpu_slots(self, data, indices):
        index = indices.to(device=self.device, dtype=torch.long)
        if data is None:
            self.valid[:, index] = 0
            return
        self.valid[:, index] = data["valid"].to(self.device)
        for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS:
            for pool, tensor in zip(self.pools, data[name]):
                getattr(pool, name)[index] = tensor.to(self.device)

    def iter_transfer_state_entries(self):
        """Nothing here is offered for transfer between servers (not supported in phase 1)."""
        return iter(())

    # ---- one decode step of one layer -------------------------------------------------
    def gates(self, mixed_qkv, a, b, A_log, dt_bias, scale):
        """
        Turns the layer's projections into the inputs of the recurrence, with the
        formulas of SGLang's packed decode kernel
        (fused_recurrent_gated_delta_rule_packed_decode, use_qk_l2norm_in_kernel=True):

            g     = -exp(A_log) * softplus(a + dt_bias)        decay = exp(g)
            beta  = sigmoid(b), rounded to the dtype of b
            q, k  = L2-normalised with eps 1e-6;  q also times `scale`

        mixed_qkv [B, 2*H*K + HV*V] is packed (q | k | v). Returns float32 tensors
        decay [B, HV], beta [B, HV], key [B, H, K], query [B, H, K], value [B, HV, V].
        """
        rows = mixed_qkv.shape[0]
        K, HV, V = self.key_channels, self.value_heads, self.value_channels
        if self.key_heads is None:
            # Not given at construction: read it from the width of the packed tensor.
            self.key_heads = (mixed_qkv.shape[1] - HV * V) // (2 * K)
        H = self.key_heads
        if mixed_qkv.shape[1] != 2 * H * K + HV * V or H < 1 or HV % H != 0:
            raise ValueError(f"mixed_qkv has width {mixed_qkv.shape[1]}, which does not fit "
                             f"{H} key heads x {K} and {HV} value heads x {V}")
        packed = mixed_qkv.float()
        query = packed[:, :H * K].reshape(rows, H, K)
        key = packed[:, H * K:2 * H * K].reshape(rows, H, K)
        value = packed[:, 2 * H * K:].reshape(rows, HV, V).contiguous()
        query = (query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6) * scale).contiguous()
        key = (key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)).contiguous()
        g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float(), beta=1.0, threshold=20.0)
        decay = torch.exp(g).contiguous()
        beta = torch.sigmoid(b.float()).to(b.dtype).float().contiguous()
        return decay, beta, key, query, value

    def _adopt(self, pool, slot, source, skip_slot, source_index=None):
        """Stores the full-precision state of the rows whose slot is not marked in skip_slot."""
        if self.fast:
            fast_kernels.store(pool, slot, source, skip_slot=skip_slot, source_index=source_index)
        else:
            self.kernel.store(pool, slot, self.scratch, source_by_slot=source, skip_slot=skip_slot,
                              source_index=source_index)

    def _fold(self, pool, slot, write_pos, force, export_by_slot, records=None, work=None):
        """Folds the windows that are full after this token, or forced. `records` and `work`: this step's records
        to fold and work list when already built (see _inputs_for_step)."""
        if self.fast:
            if records is None:
                records = fast_kernels.records_to_fold(write_pos, self.window_length, force, tokens_added=1)
            fast_kernels.fold(pool, slot, records, export_by_slot=export_by_slot, work=work)
        else:
            self.kernel.fold(pool, slot, write_pos, tokens_added=1, force=force, scratch=self.scratch,
                             export_by_slot=export_by_slot)

    def _token(self, pool, slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale, out_dtype=torch.float32):
        """The decode kernel for one batch, with the gates fused or not. Returns [B, HV, V] in out_dtype."""
        rows = mixed_qkv.shape[0]
        if self.fast:
            if self.key_heads is None:
                self.key_heads = (mixed_qkv.shape[1] - self.value_heads * self.value_channels) // (2 * self.key_channels)
            out = torch.empty((rows, self.value_heads, self.value_channels), dtype=out_dtype, device=self.device)
            a = a if a.stride(-1) == 1 else a.contiguous()
            b = b if b.stride(-1) == 1 else b.contiguous()
            fast_kernels.decode_step_raw(pool, slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale, self.key_heads, out)
            return out
        out = torch.empty((rows, self.value_heads, self.value_channels), dtype=torch.float32, device=self.device)
        if self.fused_gates:
            if self.key_heads is None:
                self.key_heads = (mixed_qkv.shape[1] - self.value_heads * self.value_channels) // (2 * self.key_channels)
            self.kernel.decode_step_raw(pool, slot, write_pos, mixed_qkv, a.contiguous(), b.contiguous(),
                                        A_log.float(), dt_bias.float(), scale, self.key_heads, out)
        else:
            decay, beta, key, query, value = self.gates(mixed_qkv, a, b, A_log, dt_bias, scale)
            self.kernel.decode_step(pool, slot, write_pos, decay, beta, key, query, value, out)
        return out if out_dtype == torch.float32 else out.to(out_dtype)

    def decode(self, layer_index, mixed_qkv, a, b, A_log, dt_bias, scale, ssm_states, cache_indices,
               write_pos, force_flush=None, out_dtype=torch.float32):
        """
        One token for each row of the batch, for one layer (use cases UC1, UC2, UC3, UC5).

            mixed_qkv, a, b, A_log, dt_bias, scale   as SGLang's packed decode kernel takes them
            ssm_states      float32 [slots, HV, V, K]  the engine's fp32 state of this layer
            cache_indices   int [B]        slot of each row; rows at the padding slot are skipped
            write_pos       int32 [B]      records in each row's window BEFORE this token (0 .. L-1)
            force_flush     int32 [B] or None   nonzero = fold after this token even if the window is not full

        Returns the output, [B, HV, V] in out_dtype (float32 unless asked otherwise).

        Uses tensor operations and kernel launches only, with no branch on tensor
        values, so the whole call can be captured in a CUDA graph.
        """
        pool = self.pools[layer_index]
        valid = self.valid[layer_index]
        rows = mixed_qkv.shape[0]

        slot, records, work = self._inputs_for_step(layer_index, cache_indices, write_pos, force_flush)

        # 1. Adopt the rows the engine's fp32 tensor is still in charge of.
        #    (Compressed only: ssm_states is None; every request was stored at the end of its prefill.)
        if ssm_states is not None:
            self._adopt(pool, slot, ssm_states, valid)
            # (Padded rows may carry slot -1; the clamp sends them to the padding slot.)
            valid.index_fill_(0, cache_indices.to(torch.long).clamp(min=0), 1)

        # 2. The token.
        out = self._token(pool, slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale, out_dtype)

        # 3. Fold the windows that are full or forced, and keep the fp32 tensor current.
        self._fold(pool, slot, write_pos, force_flush, ssm_states, records=records, work=work)
        return out

    def _inputs_for_step(self, layer_index, cache_indices, write_pos, force_flush):
        """
        (slot, records, work) of a decode step: int32 slots with -1 for padded rows; with the fast kernels also
        the records each row folds after this token and the fold's work list (else None, None). These depend
        only on the step's batch tensors, not on the layer, so they are built at the first layer of a step and
        reused by the next ones. A step begins when the layer index does not increase, or when any of the
        batch tensors is another one (identity, address, length).
        """
        key = (id(cache_indices), cache_indices.data_ptr(), cache_indices.shape[0], cache_indices.dtype,
               id(write_pos), write_pos.data_ptr(), write_pos.shape[0],
               None if force_flush is None else (id(force_flush), force_flush.data_ptr(), force_flush.shape[0]))
        if (self.step_reuse and self._step_inputs is not None and key == self._step_key
                and self._step_layer is not None and layer_index > self._step_layer):
            self._step_layer = layer_index
            return self._step_inputs
        slot = cache_indices.to(torch.int32)
        if self.padding_slot is not None:
            slot = torch.where(slot == self.padding_slot, torch.full_like(slot, -1), slot)
        slot = slot.contiguous()
        records = work = None
        if self.fast:
            records = fast_kernels.records_to_fold(write_pos, self.window_length, force_flush, tokens_added=1)
            work = fast_kernels.fold_work_list(slot, records) if slot.shape[0] > 0 else None
        self._step_key, self._step_layer, self._step_inputs = key, layer_index, (slot, records, work)
        return self._step_inputs

    # ---- prefill, compressed only (phase 2) ------------------------------------------------------
    def _rows(self, cache_indices):
        """int32 slots with -1 for padded rows, and the mask of real rows."""
        slot = cache_indices.to(torch.int32)
        real = slot >= 0
        if self.padding_slot is not None:
            real = real & (slot != self.padding_slot)
        return torch.where(real, slot, torch.full_like(slot, -1)).contiguous(), real

    def extend_begin(self, layer_index, cache_indices):
        """
        The staging tensor for one layer's prefill.

            cache_indices   int [B]   slot of each request of the prefill batch (padding slot = padded row)
        Returns (staging, staging_indices):
            staging          float32 [B + 1, HV, V, K]; row 0 is the padding row (zero)
            staging_indices  [B], dtype of cache_indices: row b of the batch uses staging row
                             staging_indices[b] (b + 1, or 0 for a padded row)
        A request whose slot was reset (a new request) starts from zero; any other starts from
        its own state, decompressed (a later chunk of a chunked prefill).
        """
        pool = self.pools[layer_index]
        rows = cache_indices.shape[0]
        slot, real = self._rows(cache_indices)
        staging = torch.empty((rows + 1, self.value_heads, self.value_channels, self.key_channels),
                              dtype=torch.float32, device=self.device)
        staging[0].zero_()
        if rows > 0:
            fast_kernels.export(pool, slot, staging[1:])
            has_state = real & (self.valid[layer_index][slot.clamp(min=0).to(torch.long)] != 0)
            staging[1:].mul_(has_state.to(torch.float32)[:, None, None, None])     # new requests start from zero
        index = torch.arange(1, rows + 1, dtype=cache_indices.dtype, device=self.device)
        staging_indices = torch.where(real, index, torch.zeros_like(index))
        return staging, staging_indices

    def extend_end(self, layer_index, cache_indices, staging, staging_indices):
        """After the prefill kernel wrote the final states into `staging`: store them in the pool."""
        pool = self.pools[layer_index]
        slot, real = self._rows(cache_indices)
        if slot.shape[0] == 0:
            return
        fast_kernels.store(pool, slot, staging, source_index=staging_indices.to(torch.int32).contiguous())
        # The stored slots are in charge from now on. (Padded rows write the padding slot's flag; reset it.)
        self.valid[layer_index].index_fill_(0, slot.clamp(min=0).to(torch.long), 1)
        if self.padding_slot is not None:
            self.valid[layer_index][self.padding_slot] = 0


class CheckpointStore:
    """
    Prefix-cache checkpoints in the LeapQuant boundary format (task T11).

    It has the interface of SGLang's Int8CheckpointStore, so SGLang's
    MambaCheckpointPool can use it in its place:

        store_from_pool(src_temporal, src_slots, dst_slots)   cache store / donate
        copy_to_pool(dst_temporal, src_slots, dst_slots)      cache hit

    `temporal` is the engine's fp32 state of all layers, [L, pool_slots, HV, V, K].

    Why a checkpoint made here equals the boundary state of the request it was
    taken from: at every fold the kernel writes the full-precision state, as it
    is before rounding, to the engine's fp32 tensor. The engine checkpoints at
    fold boundaries. Storing that matrix here runs the same fold on the same
    numbers, so the result is the same boundary state, bit for bit. It is 19,456
    bytes per head: the "14.9 MB int8 state, stored as is" of the plan.

    A cache hit writes the checkpoint back as fp32. The request that continues
    from it is adopted again at its first decode step, which rounds once more.

    With the fast kernels the checkpoint is a fresh store of that matrix: a valid
    boundary state of it, close to the request's own but not bit-identical (the
    request's fold started its fit from its previous compensators).
    """

    def __init__(self, *, num_layers, num_slots, num_heads, head_v_dim, head_k_dim, device,
                 scale_dtype=None, compensator_count=None, bits_per_entry=8, fast=None, scale_group=None):
        # scale_dtype is accepted for compatibility with Int8CheckpointStore and ignored.
        self.num_layers = num_layers
        self.num_slots = num_slots
        self.H = num_heads
        self.d_v = head_v_dim
        self.d_k = head_k_dim
        self.device = torch.device(device)
        if scale_group is None:
            scale_group = low_bit_scale_group(bits_per_entry)
        self.scale_group = int(scale_group)
        self.fast = choose_kernel(fast, self.device, self.scale_group, head_k_dim)
        self.kernel = Kernel(self.device)
        counts = resolve_compensator_counts(compensator_count, num_layers, bits_per_entry)
        self.pools = [
            StatePool(num_slots, num_heads, head_k_dim, head_v_dim, compensator_count=count,
                      bits_per_entry=bits_per_entry, device=self.device, with_window=False,
                      scale_group=self.scale_group)
            for count in counts
        ]

    def _slots(self, slots):
        return torch.as_tensor(slots, device=self.device).reshape(-1).to(torch.int32).contiguous()

    def store(self, slots, state):
        """state: [L, N, HV, V, K] for the N slots, or [L, HV, V, K] for one slot. It is not changed."""
        if state.dim() == 4:
            state = state.unsqueeze(1)
        slot = self._slots(slots)
        for layer, pool in enumerate(self.pools):
            if self.fast:
                source = state[layer].to(torch.float32).contiguous()
                entries = torch.arange(slot.shape[0], dtype=torch.int32, device=self.device)
                fast_kernels.store(pool, slot, source, source_index=entries)
            else:
                working_copy = state[layer].to(torch.float32).clone().contiguous()
                self.kernel.store(pool, slot, working_copy)

    def load(self, slots, out_dtype):
        """The checkpoints at `slots` in full precision: [L, N, HV, V, K] in out_dtype."""
        slot = self._slots(slots)
        result = torch.empty((self.num_layers, slot.shape[0], self.H, self.d_v, self.d_k), dtype=torch.float32,
                             device=self.device)
        for layer, pool in enumerate(self.pools):
            if self.fast:
                fast_kernels.export(pool, slot, result[layer])
            else:
                self.kernel.export(pool, slot, None, result[layer])
        return result.to(out_dtype)

    def copy_to_pool(self, dst_temporal, src_slots, dst_slots):
        dst_temporal[:, dst_slots] = self.load(src_slots, dst_temporal.dtype)

    def store_from_pool(self, src_temporal, src_slots, dst_slots):
        self.store(dst_slots, src_temporal[:, src_slots])

    def mem_usage_bytes(self):
        return sum(getattr(pool, name).numel() * getattr(pool, name).element_size()
                   for pool in self.pools for name in StatePool.BOUNDARY_FIELDS)

    def bytes_per_slot(self):
        return self.mem_usage_bytes() // max(1, self.num_slots)


class BlockEngineState:
    """
    LeapQuant decode for an engine that keeps the recurrent state in a large pool
    of blocks and never says when a block changes owner. That is vLLM: the state
    of a request lives in a KV-cache block, the "slot" is a block id out of the
    whole block pool, and nothing is called when a request ends.

    Supported: one fixed block per request for its whole life (vLLM's mamba cache
    mode "none", that is, prefix caching off) and no speculative decoding.

    How it works, per layer
        - A small LeapQuant pool with `lq_slots` slots, and a table from block id
          to LeapQuant slot. Entry 0 means "no LeapQuant state".
        - assign(blocks): the engine calls it when prefill has written the fp32
          state of these blocks. Each block gets a LeapQuant slot (it keeps the
          one it has) and is marked for adoption. When the pool is full, the
          slots that have not decoded for the longest time are taken over: those
          belong to requests that ended.
        - decode(...): looks the slots up, adopts the marked ones from the fp32
          blocks, decodes, folds. It keeps its own window cursor per slot. The
          fp32 block is not written again: with a fixed block per request nothing
          reads it once the request decodes.

    Everything is tensor operations and kernel launches without a branch on
    tensor values, so decode() can be captured in a CUDA graph. The one exception
    is the safety check in assign(), which reads two numbers back from the device.

    What can go wrong, and how it shows
        If more requests decode at once than the pool has slots, a live request
        loses its slot and its rows decode as zeros. assign() raises before that
        happens (a slot that decoded within the last `idle_steps` steps would be
        taken), and `unmapped_rows` counts decode rows that had a block but no slot.
    """

    def __init__(self, layers, blocks, value_heads, key_channels, value_channels, lq_slots, max_rows,
                 window_length=16, compensator_count=None, bits_per_entry=8, device="cuda", idle_steps=4,
                 check_on_assign=True, scratch=None, fused_gates=True, fast=None, scale_group=None):
        self.layers = layers
        self.blocks = blocks
        self.lq_slots = lq_slots            # including slot 0, which stands for "none"
        self.window_length = window_length
        self.idle_steps = idle_steps
        self.check_on_assign = check_on_assign
        self.device = torch.device(device)
        self.engine = EngineState(layers, lq_slots, value_heads, None, key_channels, value_channels, max_rows,
                                  window_length=window_length, compensator_count=compensator_count,
                                  bits_per_entry=bits_per_entry, device=device, cursor=None, padding_slot=0,
                                  scratch=scratch, fused_gates=fused_gates, fast=fast, scale_group=scale_group)
        self.kernel = self.engine.kernel
        self.slot_of_block = torch.zeros((layers, blocks), dtype=torch.int32, device=self.device)
        self.block_of_slot = torch.zeros((layers, lq_slots), dtype=torch.int64, device=self.device)
        self.cursor = torch.zeros((layers, lq_slots), dtype=torch.int32, device=self.device)
        # Decode step at which a slot was last used; the layer's decode step counter.
        self.last_used = torch.full((layers, lq_slots), -(1 << 40), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((layers,), dtype=torch.int64, device=self.device)
        self.unmapped_rows = torch.zeros((layers,), dtype=torch.int64, device=self.device)

    def assign(self, layer_index, block_indices):
        """
        Prefill wrote the fp32 state of these blocks (one entry per request; block 0 is ignored).
        Gives each a LeapQuant slot and marks it for adoption at its first decode step.
        """
        table = self.slot_of_block[layer_index]
        owner = self.block_of_slot[layer_index]
        last_used = self.last_used[layer_index]
        blocks = block_indices.to(device=self.device, dtype=torch.long).reshape(-1)
        count = blocks.shape[0]
        if count == 0:
            return
        if count > self.lq_slots - 1:
            raise RuntimeError(f"LeapQuant: {count} requests finish prefill in one step but the pool has "
                               f"{self.lq_slots - 1} slots")
        current = table[blocks].to(torch.long)
        needs_slot = (current == 0) & (blocks > 0)

        # Candidates: the `count` slots that decoded longest ago. A slot that a block of this
        # call already holds is not a candidate (it gets the newest stamp for the selection).
        age = last_used.clone()
        age[0] = 1 << 60                                   # slot 0 is never handed out
        age.index_fill_(0, current, 1 << 60)
        candidates = torch.topk(age, count, largest=False).indices
        # The i-th block that needs a slot takes the i-th candidate.
        rank = torch.cumsum(needs_slot.to(torch.long), dim=0) - 1
        taken = candidates[rank.clamp(min=0)]
        slots = torch.where(needs_slot, taken, current)

        if self.check_on_assign:
            # Would a slot be taken from a request that is still decoding?
            newest_taken = torch.where(needs_slot, last_used[taken], torch.full_like(taken, -(1 << 40))).max()
            unmapped = int(self.unmapped_rows[layer_index])
            if int(newest_taken) > int(self.step[layer_index]) - self.idle_steps:
                raise RuntimeError("LeapQuant: the pool is too small. A slot that decoded within the last "
                                   f"{self.idle_steps} steps would be reused. Raise lq_slots (now {self.lq_slots}).")
            if unmapped:
                raise RuntimeError(f"LeapQuant: {unmapped} decode rows of layer {layer_index} had no slot.")

        # Blocks that lose their slot go back to "none". (Scatter the zeros first, the new
        # entries second, so a block that both loses and gains in one call ends up right.)
        evicted_owner = owner[taken]
        still_points_here = table[evicted_owner].to(torch.long) == taken
        table[evicted_owner[needs_slot & still_points_here]] = 0
        table[blocks] = slots.to(torch.int32)
        table[0] = 0
        owner[slots] = blocks
        # Marked for adoption; the window is empty; counts as used now.
        self.engine.valid[layer_index].index_fill_(0, slots, 0)
        self.cursor[layer_index].index_fill_(0, slots, 0)
        last_used.index_copy_(0, slots, self.step[layer_index].expand(count).clone())

    def decode(self, layer_index, mixed_qkv, a, b, A_log, dt_bias, scale, block_state, block_indices):
        """
        One token for each row, for one layer.

            mixed_qkv, a, b, A_log, dt_bias, scale   as vLLM's packed decode kernel takes them
            block_state     float32 [blocks, HV, V, K]; entries may be padded (its first stride is used)
            block_indices   int [B]   block of each row; 0 = padded row

        Returns the output, float32 [B, HV, V]. Rows without a slot give zeros.
        """
        engine = self.engine
        pool = engine.pools[layer_index]
        valid = engine.valid[layer_index]
        cursor = self.cursor[layer_index]
        rows = mixed_qkv.shape[0]

        blocks = block_indices.to(torch.long)
        slot = self.slot_of_block[layer_index][blocks]                    # int32 [B], 0 = none
        long_slot = slot.to(torch.long)
        kernel_slot = torch.where(slot > 0, slot, torch.full_like(slot, -1)).contiguous()
        self.unmapped_rows[layer_index] += ((blocks > 0) & (slot == 0)).sum()

        # 1. Adopt the slots marked by assign() from their fp32 blocks.
        engine._adopt(pool, kernel_slot, block_state, valid, source_index=block_indices.to(torch.int32).contiguous())
        valid.index_fill_(0, long_slot, 1)

        # 2. The token.
        write_pos = cursor[long_slot].contiguous()
        out = engine._token(pool, kernel_slot, write_pos, mixed_qkv, a, b, A_log, dt_bias, scale)

        # 3. Fold the full windows; advance this layer's cursor and stamps.
        engine._fold(pool, kernel_slot, write_pos, None, None)
        records = write_pos + 1
        next_pos = torch.where(records >= self.window_length, torch.zeros_like(records), records)
        cursor.index_copy_(0, long_slot, next_pos)
        cursor[0] = 0
        self.step[layer_index] += 1
        self.last_used[layer_index].index_copy_(0, long_slot, self.step[layer_index].expand(rows).clone())
        return out
