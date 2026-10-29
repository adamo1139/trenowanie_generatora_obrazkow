"""Sharded/stacked Muon == plain per-matrix Muon, and weights stay identical across ranks.

  python tests/check_muon.py                                         # 1 process
  DIST_BACKEND=gloo torchrun --nproc_per_node 2 tests/check_muon.py  # 2 ranks on one GPU
"""
import os, sys, torch
import torch.distributed as dist
sys.path.insert(0, ".")
from aurora.muon import Muon, zeropower_via_newtonschulz5
from aurora.model import SIZES, build_models

if "RANK" in os.environ:
    dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"))
dev = "cuda"
torch.manual_seed(0)
_, den = build_models(SIZES["50m"])
mats = [p for n, p in den.named_parameters() if n.startswith("blocks.") and p.ndim == 2 and "mod_table" not in n]
params = [torch.nn.Parameter(torch.randn_like(p).to(dev) * 0.02) for p in mats]
ref = [p.detach().clone() for p in params]
init = [p.detach().clone() for p in params]
opt = Muon(params, lr=1e-2, weight_decay=0.01)
bufs = [torch.zeros_like(p) for p in ref]
for step in range(3):
    g = torch.Generator(device=dev).manual_seed(step)  # same grads on every rank (as after DDP all-reduce)
    grads = [torch.randn(p.shape, generator=g, device=dev) for p in params]
    for p, gr in zip(params, grads):
        p.grad = gr.clone()
    opt.step()
    for r, b, gr in zip(ref, bufs, grads):  # reference: one matrix at a time
        b.lerp_(gr, 0.05)
        u = zeropower_via_newtonschulz5(gr.lerp(b, 0.95), 5).float()
        r.mul_(1 - 1e-2 * 0.01).add_(u, alpha=-1e-2 * max(1, r.size(0) / r.size(1)) ** 0.5)
# relative error of the accumulated update; NS runs in bf16, so batched vs single matmuls differ at bf16 precision
err = max(((p - r).norm() / (r - i).norm()).item() for p, r, i in zip(params, ref, init))
def ns_fp32(X, steps=5):  # same iteration without the bf16 cast, to measure bf16's inherent noise
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = X.mT / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT
x = torch.randn(1792, 768, device=dev)
noise = ((zeropower_via_newtonschulz5(x).float() - ns_fp32(x)).norm() / ns_fp32(x).norm()).item()
rank = dist.get_rank() if dist.is_initialized() else 0
print(f"rank {rank}: {len(params)} matrices, max relative |sharded - reference| = {err:.2e} "
      f"(bf16 NS noise level for scale: {noise:.2e})")
assert err < noise, (err, noise)  # stacking/sharding must not add more than bf16 NS itself does
if dist.is_initialized():
    flat = torch.cat([p.detach().flatten() for p in params])
    other = flat.clone(); dist.broadcast(other, 0)
    assert torch.equal(flat, other), "weights diverged across ranks"
    print(f"rank {rank}: weights identical to rank 0")
    dist.destroy_process_group()
