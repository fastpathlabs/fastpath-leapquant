# LeapQuant state: layout and numerical contract

Plan v6 task N01 (backlog B21). What every kernel in this repository (reference core `leapquant.cpp`, exact kernels
`kernel/leapquant_kernel_*.{h,cu,cpp}`, fast kernels `kernel/leapquant_triton.py`, `kernel/leapquant_fast_fold.cu`,
`kernel/leapquant_fast_decode.cu`) reads and writes, and what each step means. One value head, 8-bit policy.

## Orientation

The state of a head is the matrix S with one row per value channel and one column per key channel: S is V × K
(V = K = 128 for Qwen3.5-9B), the recurrence's `S = Σ v kᵀ` read as `o = S q`. The reference core stores the
transpose (K × V, `state_matrix[key][value]`); its API documents that and every conversion goes through
`copy_state_to_full_precision` / `load_full_precision_state`. All pool tensors below are value-major.

## Pool (one entry per slot, per value head)

| Tensor | Shape | dtype | Meaning |
|---|---|---|---|
| `rounded_residual` Z | [V][K] | int8 | integer levels, within ±LEVELS (127 at 8 bits) |
| `value_channel_step` B | [V] (or [V][G] with scale groups) | fp32 | step of value channel v: one level = B[v] |
| `key_row_scale` C | [K] | fp32 | smoothing scale of key channel k |
| `compensator_corrections` U_c | [R][V] | fp16 | value sides of the R compensators |
| `compensator_keys` U_k | [R][K] | fp16 | key sides of the R compensators |
| `window_corrections` | [L][V] | fp16 | record j: correction u_j = v_j − α_j · (S_{j−1} k_j) as stored |
| `window_keys` | [L][K] | fp16 | record j: β_j · k_j (beta times the normalized key), rounded to fp16 |
| `window_log_decays` | [L] | fp16 | record j: log α_j (one decay per token, Gated DeltaNet) |

Boundary state (what a prefix-cache checkpoint keeps): Z, B, C, U_c, U_k: 19,456 bytes per 128 × 128 head at rank 4.
Window: 8,224 bytes per head at L = 16. The cursor (records in the window, 0..L−1) belongs to the engine, per slot.

## Value of the state

Boundary state:  S_b[v][k] = B[v] · Z[v][k] · C[k] + Σ_r U_c[r][v] · U_k[r][k]
(steps and scales multiply the integer levels; the compensators are added as stored, in fp16, not again scaled).

With n records in the window, the state after them:
  S[v][k] = F · S_b[v][k] + Σ_{j<n} f_j · u_j[v] · (β_j k_j)[k],   F = Π_{j<n} α_j,   f_j = Π_{j<i<n} α_i
(f_j is the decay record j received from the records after it; F the decay of the whole window).

## Token step (decode)

Inputs: q, k (normalized when the model asks: x / sqrt(Σx² + 1e-6)), q scaled by 1/sqrt(K), value v, gates.
Gates (raw): β = sigmoid(b) rounded to b's dtype; g = −exp(A_log) · softplus(a + dt_bias) (softplus(x) = x above 20);
α = exp(g). The decay is STORED as fp16(g), and the output uses α_stored = exp(fp16(g)).

1. held = S k (the state before this token, with α applied: α · (S k)); correction u = fp16(v − α · S k).
2. Append record (u, fp16(β k), fp16(g)) at the cursor. The boundary state is not touched.
3. Output o = u · (fp16(β k) · q) + α_stored · (S q)  (the new record enters exactly as stored).
4. When the window is full (cursor reaches L) the fold runs; the engine resets the cursor.

## Fold (window → boundary)

1. M = the state above with all n records (fp32; the records' and compensators' products are exact in fp32: each
   weighted factor is split into two fp16 parts before a tensor-core product).
2. Fit R compensators: block iteration from the previous value sides (a missing one starts from a newest record):
   U = Mᵀ X, U orthonormal (Gram-Schmidt; a column whose remaining squared norm is at most 1e-6 of its starting
   one becomes zero), X = M U; `sweeps` times (2 by default; 0 = keep the key sides, refit the value sides).
3. New compensators AS STORED: balance b_r = (Σ X_r²)^(1/4); U_k = fp16(U_r · b_r), U_c = fp16(X_r / b_r).
4. Remainder E = M − U_c U_kᵀ (from the stored fp16 values).
5. Smoothing: C[k] = max(sqrt(Σ_v |E[v][k]| / V), 1e-6) (1 when smoothing is off).
6. Steps: B[v] = max_k |E[v][k] / C[k]| / LEVELS.
7. Levels: Z[v][k] = round-half-even(E[v][k] / C[k] / B[v]), 0 where B[v] = 0.

Store (adoption of a full-precision state, e.g. at the end of a prefill): the same steps 2-7 on the given M, the fit
starting from key rows chosen by a pivoted QR (the longest, then the one with most length outside those chosen),
with sweeps + 2 sweeps.

## Publication and ownership

A fold or store rewrites one head's boundary state in place; one program (Triton) or one block (CUDA) owns a head
for the whole rewrite, and no other program reads that head during the launch. The engine orders launches on one
stream (CUDA graphs keep the order): decode step N's appends, then the fold for the rows due, then step N+1. The
pool is reused across requests by slot; the engine clears a slot (`reset_slots`) before a new request uses it.
There is no slot generation or state epoch yet (backlog B13).

## Exactness classes

- Reference core and exact kernels: bit-identical (the exact GPU kernel reproduces the reference's every fold and
  int8 entry; `kernel/verify_kernel.py`, criterion B "bit-identical").
- Fast kernels (Triton, CUDA): same contract, other summation orders, the fit in reduced precision (it only chooses
  directions; the remainder is formed exactly), reciprocals instead of divisions in the CUDA rounding. Judged by
  tolerance: in-window outputs (A), fold loss within 5% of the reference's (B), free-run output error over the trace
  set within 10% of the reference's (C), plus fixture tests against the Triton kernels (`kernel/fast_fold_test.py`).

## The independent oracle

`reference_heads.cpp` (`FullPrecisionHead`): the same recurrence in fp32 on the raw inputs of a recorded trace; every
replay (`leapquant_replay`) and verification reports errors against it, never against a reconstruction of a
quantized state.
