# prof_triton.py: one decode step (8 records in the window, the 17th decode launch) and one fold of every row at
# batch B (default 512), for Nsight Compute. The backends follow LEAPQUANT_FOLD_BACKEND / LEAPQUANT_DECODE_BACKEND.
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leapquant_triton as fast
from leapquant_kernel import StatePool
dev = torch.device("cuda")
B, HV, H, K, V, L = int(os.environ.get("B", 512)), 32, 16, 128, 128, 16
pool = StatePool(B + 1, HV, K, V, window_length=L, device=dev)
slot = torch.arange(1, B + 1, dtype=torch.int32, device=dev)
g = torch.Generator().manual_seed(0)
state = torch.zeros(B + 1, HV, V, K, device=dev)
state[1:] = (0.05 * torch.randn(B, HV, V, K, generator=g)).to(dev)
for p in range(3):
    state[1:] += (torch.randn(B, HV, V, 1, generator=g) * torch.randn(B, HV, 1, K, generator=g)).to(dev) / (p + 1)
fast.store(pool, slot, state)
mixed = torch.randn(B, 2 * H * K + HV * V, generator=g).to(dev, torch.bfloat16)
a = (0.5 * torch.randn(B, HV, generator=g)).to(dev, torch.bfloat16)
b = torch.randn(B, HV, generator=g).to(dev, torch.bfloat16)
A_log = (0.3 * torch.randn(HV, generator=g) - 4).to(dev)
dt_bias = (0.1 * torch.randn(HV, generator=g)).to(dev)
out = torch.empty(B, HV, V, device=dev)
wp = torch.zeros(B, dtype=torch.int32, device=dev)
for pos in range(L):                      # 16 decode launches: fill the window
    wp.fill_(pos)
    fast.decode_step_raw(pool, slot, wp, mixed, a, b, A_log, dt_bias, K ** -0.5, H, out)
wp.fill_(8)                               # launch 17: the profiled decode step (8 records in the window)
fast.decode_step_raw(pool, slot, wp, mixed, a, b, A_log, dt_bias, K ** -0.5, H, out)
full = torch.full((B,), L, dtype=torch.int32, device=dev)
fast.fold(pool, slot, full)               # the profiled fold
torch.cuda.synchronize()
print("done")
