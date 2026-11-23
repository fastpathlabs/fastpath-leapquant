#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
fast_fold_test.py

The CUDA fold (kernel/leapquant_fast_fold.cu) against the Triton fold on the same pools:
    - every case folds the same boundary state plus window with both backends;
    - the fold loss of each (new boundary state against the full-precision state, which the Triton
      fold exports) must agree: CUDA mean within 3% of Triton's, worst within 10%;
    - rows that do not fold (records 0) and padded rows (slot -1) must be left bit for bit as they were;
    - steps, scales and levels must be finite, levels within +-LEVELS.

Cases: forced folds with fewer records, 1/3/4/6/8 compensators, 0/1/2 sweeps, smoothing off, window 8.

    python3 kernel/fast_fold_test.py [--quick]      (needs a GPU and build/libleapquant_fastfold.so)
Exit status 1 on any failure.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leapquant_triton as fast  # noqa: E402
from leapquant_kernel import StatePool  # noqa: E402


def make_pool(rows, HV, R, L, sweeps, smoothing, seed, device):
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (scale * torch.randn(shape, generator=generator)).to(device)

    pool = StatePool(rows + 2, HV, 128, 128, window_length=L, compensator_count=R, bits_per_entry=8,
                     power_iteration_sweeps=sweeps, use_smoothing=smoothing, device=device, scale_group=0)
    slots = rows + 2
    state = randn(slots, HV, 128, 128, scale=0.05)
    for p in range(3):            # a few strong rank-1 directions (outliers), as real states have
        state += randn(slots, HV, 128, 1) * randn(slots, HV, 1, 128) / (p + 1)
    all_slots = torch.arange(slots, dtype=torch.int32, device=device)
    fast.store(pool, all_slots, state)
    pool.window_keys.copy_(randn(slots, HV, L, 128, scale=0.1).half())
    pool.window_corrections.copy_(randn(slots, HV, L, 128, scale=0.3).half())
    pool.window_log_decays.copy_((-0.08 * torch.rand(slots, HV, L, generator=generator)).half().to(device))
    return pool


def clone_pool(pool):
    twin = StatePool.__new__(StatePool)
    twin.__dict__.update(pool.__dict__)
    for name in StatePool.BOUNDARY_FIELDS + StatePool.WINDOW_FIELDS:
        setattr(twin, name, getattr(pool, name).clone())
    return twin


def head_loss(pool, slot, full):
    """Relative Frobenius error of the boundary state of each head against `full` [B, HV, V, K]."""
    out = torch.zeros_like(full)
    fast.export(pool, slot, out)
    num = (out - full).flatten(2).norm(dim=2)
    den = full.flatten(2).norm(dim=2).clamp_min(1e-30)
    return num / den


def run_case(name, rows, HV, R, L, sweeps, smoothing, records_list, seed, device):
    pool = make_pool(rows, HV, R, L, sweeps, smoothing, seed, device)
    slot = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    slot[-1] = -1                                       # a padded row
    records = torch.tensor(records_list, dtype=torch.int32, device=device)
    assert records.numel() == rows
    triton_pool, cuda_pool = clone_pool(pool), clone_pool(pool)
    full = torch.zeros(pool.slots, HV, 128, 128, device=device)
    fast.FOLD_BACKEND = "triton"
    fast.fold(triton_pool, slot, records, export_by_slot=full)
    fast.FOLD_BACKEND = "cuda"
    fast.fold(cuda_pool, slot, records)
    torch.cuda.synchronize()
    failures = []
    due = (records > 0) & (slot >= 0)
    idle = ~due
    due_slots = slot[due]
    if due_slots.numel():
        full_due = full[due_slots.long()]
        loss_t = head_loss(triton_pool, due_slots, full_due)
        loss_c = head_loss(cuda_pool, due_slots, full_due)
        ratio_mean = float(loss_c.mean() / loss_t.mean())
        ratio_worst = float(loss_c.max() / loss_t.max())
        if not (ratio_mean <= 1.03 and ratio_worst <= 1.10):
            failures.append(f"fold loss cuda/triton: mean {ratio_mean:.4f} (<= 1.03), worst {ratio_worst:.4f} (<= 1.10)")
        detail = (f"loss triton mean {float(loss_t.mean()):.5f} worst {float(loss_t.max()):.5f} | cuda mean "
                  f"{float(loss_c.mean()):.5f} worst {float(loss_c.max()):.5f} | ratio {ratio_mean:.4f} / {ratio_worst:.4f}")
    else:
        detail = "no row folds"
    # Every slot that no folding row names (rows with records 0, the padded row's slot, unused slots):
    # unchanged, bit for bit.
    folded = set(int(s) for s in due_slots.tolist())
    untouched = [s for s in range(pool.slots) if s not in folded]
    assert len(untouched) > 2 and idle.any()
    for field in StatePool.BOUNDARY_FIELDS:
        before = getattr(pool, field)[untouched]
        after = getattr(cuda_pool, field)[untouched]
        if not torch.equal(before, after):
            failures.append(f"{field} changed on a row that does not fold")
    for field in ("value_channel_step", "key_row_scale", "compensator_keys", "compensator_corrections"):
        if not torch.isfinite(getattr(cuda_pool, field).float()).all():
            failures.append(f"{field} not finite")
    levels = cuda_pool.rounded_residual.int()
    if int(levels.abs().max()) > 127:
        failures.append("a level outside +-127")
    if not smoothing and due_slots.numel() and not torch.equal(
            cuda_pool.key_row_scale[due_slots.long()], torch.ones_like(cuda_pool.key_row_scale[due_slots.long()])):
        failures.append("scales are not 1 with smoothing off")
    verdict = "PASS" if not failures else "FAIL"
    print(f"{verdict}  {name:<38} {detail}")
    for failure in failures:
        print(f"      {failure}")
    return not failures


def run_store_case(name, rows, HV, R, sweeps, smoothing, seed, device, with_index=False, with_skip=False):
    """The CUDA store against the Triton store: the same full-precision states into two copies of a pool."""
    generator = torch.Generator().manual_seed(seed)
    pool = make_pool(rows, HV, R, 16, sweeps, smoothing, seed, device)
    entries = rows + 3
    source = 0.05 * torch.randn((entries, HV, 128, 128), generator=generator)
    for p in range(3):
        source += torch.randn((entries, HV, 128, 1), generator=generator) * torch.randn((entries, HV, 1, 128), generator=generator) / (p + 1)
    source = source.to(device)
    slot = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
    slot[-1] = -1
    source_index = torch.randint(0, entries, (rows,), generator=generator, dtype=torch.int32).to(device) if with_index else None
    skip = None
    if with_skip:
        skip = torch.zeros(pool.slots, dtype=torch.int32, device=device)
        skip[2] = 1
    triton_pool, cuda_pool = clone_pool(pool), clone_pool(pool)
    fast.FOLD_BACKEND = "triton"
    fast.store(triton_pool, slot, source, skip_slot=skip, source_index=source_index)
    fast.FOLD_BACKEND = "cuda"
    fast.store(cuda_pool, slot, source, skip_slot=skip, source_index=source_index)
    torch.cuda.synchronize()
    failures = []
    stored = [b for b in range(rows) if int(slot[b]) >= 0 and not (with_skip and int(slot[b]) == 2)]
    stored_slots = slot[stored]
    entry = (source_index[stored] if with_index else stored_slots).long()
    truth = source[entry]
    loss_t = head_loss(triton_pool, stored_slots, truth)
    loss_c = head_loss(cuda_pool, stored_slots, truth)
    ratio_mean = float(loss_c.mean() / loss_t.mean())
    ratio_worst = float(loss_c.max() / loss_t.max())
    if not (ratio_mean <= 1.03 and ratio_worst <= 1.10):
        failures.append(f"store loss cuda/triton: mean {ratio_mean:.4f} (<= 1.03), worst {ratio_worst:.4f} (<= 1.10)")
    stored_set = set(int(s) for s in stored_slots.tolist())
    untouched = [s for s in range(pool.slots) if s not in stored_set]
    for field in StatePool.BOUNDARY_FIELDS:
        if not torch.equal(getattr(pool, field)[untouched], getattr(cuda_pool, field)[untouched]):
            failures.append(f"{field} changed on a slot that is not stored")
    for field in ("value_channel_step", "key_row_scale", "compensator_keys", "compensator_corrections"):
        if not torch.isfinite(getattr(cuda_pool, field).float()).all():
            failures.append(f"{field} not finite")
    verdict = "PASS" if not failures else "FAIL"
    print(f"{verdict}  store: {name:<31} loss triton mean {float(loss_t.mean()):.5f} worst {float(loss_t.max()):.5f} | "
          f"cuda mean {float(loss_c.mean()):.5f} worst {float(loss_c.max()):.5f} | ratio {ratio_mean:.4f} / {ratio_worst:.4f}")
    for failure in failures:
        print(f"      {failure}")
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--quick", action="store_true", help="few heads (for compute-sanitizer)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    device = torch.device("cuda")
    HV = 2 if args.quick else 32
    rows = 6
    base = [16, 16, 7, 0, 1, 16]                        # last row is padded
    cases = [
        ("4 compensators, 2 sweeps", dict(R=4, L=16, sweeps=2, smoothing=True)),
        ("4 compensators, 1 sweep", dict(R=4, L=16, sweeps=1, smoothing=True)),
        ("4 compensators, half sweep", dict(R=4, L=16, sweeps=0, smoothing=True)),
        ("4 compensators, no smoothing", dict(R=4, L=16, sweeps=2, smoothing=False)),
        ("1 compensator", dict(R=1, L=16, sweeps=2, smoothing=True)),
        ("3 compensators", dict(R=3, L=16, sweeps=2, smoothing=True)),
        ("6 compensators", dict(R=6, L=16, sweeps=2, smoothing=True)),
        ("8 compensators", dict(R=8, L=16, sweeps=2, smoothing=True)),
        ("window 8", dict(R=4, L=8, sweeps=2, smoothing=True)),
    ]
    ok = True
    for index, (name, config) in enumerate(cases):
        records = [min(r, config["L"]) for r in base]
        ok &= run_case(name, rows, HV, config["R"], config["L"], config["sweeps"], config["smoothing"], records,
                       seed=index, device=device)
    import leapquant_fast_cuda
    if hasattr(leapquant_fast_cuda, "store"):
        store_cases = [("4 compensators, 2 sweeps", dict(R=4, sweeps=2, smoothing=True)),
                       ("4 compensators, source index, skip", dict(R=4, sweeps=2, smoothing=True, with_index=True,
                                                                  with_skip=True)),
                       ("1 compensator", dict(R=1, sweeps=2, smoothing=True)),
                       ("8 compensators, no smoothing", dict(R=8, sweeps=1, smoothing=False))]
        for index, (name, config) in enumerate(store_cases):
            ok &= run_store_case(name, rows, HV, config["R"], config["sweeps"], config["smoothing"], seed=100 + index,
                                 device=device, with_index=config.get("with_index", False),
                                 with_skip=config.get("with_skip", False))
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
