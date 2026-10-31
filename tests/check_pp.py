"""Pipeline stages (train_pp.Stage + Schedule1F1B) give the same loss and gradients as the unsplit model,
including the REPA auxiliary loss injected in a middle stage (compared against Denoiser.forward(hidden_at=...)).

  DIST_BACKEND=gloo torchrun --nproc_per_node 2 tests/check_pp.py
  DIST_BACKEND=gloo torchrun --nproc_per_node 4 tests/check_pp.py [conv]   # conv = iREPA projector

Runs on CPU (gloo), fp32, so the comparison is exact up to float rounding.
"""
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

sys.path.insert(0, ".")
from aurora.model import build_models, make_configs
from aurora.repa import make_projector
from train_pp import Stage, owned_den_names, stage_ranges

dist.init_process_group("gloo")
rank, pp = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)
den_cfg, te_cfg = make_configs("50m", "13m")
te, den = build_models(den_cfg, te_cfg)
# perturb the zero-initialised gates / heads so every path carries signal
with torch.no_grad():
    for p in den.parameters():
        p.add_(torch.randn_like(p) * 0.02)
B, M = 8, 4
z, t = torch.randn(B, 128, 16, 16), torch.rand(B)
ids = torch.randint(6, 8192, (B, 64))
ids[:, 0], ids[:, 40:] = 2, 0
target = torch.randn(B, 128, 16, 16)
# REPA after block 5 (lands in a middle stage for pp >= 3), first 1 sample of each micro-batch, weight 0.5
REPA_LAYER, N_REPA, W = 5, 1, 0.5
proj = make_projector(den_cfg.dim, 32, hidden=64, conv=len(sys.argv) > 1 and sys.argv[1] == "conv")
repa_tgts = [torch.randn(N_REPA, 256, 32) for _ in range(M)]

# reference: the plain models (Denoiser.forward with hidden_at), mean over micro-batches of fm + W * repa
loss_ref = 0
for k, (zc, tc, ic, gc) in enumerate(zip(z.chunk(M), t.chunk(M), ids.chunk(M), target.chunk(M))):
    text, valid = te(ic)
    v, h = den(zc, tc, text, valid, hidden_at=REPA_LAYER)
    repa = -F.cosine_similarity(proj(h[:N_REPA]), repa_tgts[k], dim=-1).mean()
    loss_ref = loss_ref + (F.mse_loss(v, gc) + W * repa) / M
loss_ref.backward()
ref_proj = {n: p.grad.clone() for n, p in proj.named_parameters()}
for p in proj.parameters():
    p.grad = None
ref_grads = {("den", n): p.grad.clone() for n, p in den.named_parameters()}
ref_grads.update({("te", n): p.grad.clone() for n, p in te.named_parameters()})
for p in list(den.parameters()) + list(te.parameters()):
    p.grad = None

ranges = stage_ranges(den_cfg.depth, pp)
lo, hi = ranges[rank]
first, last = rank == 0, rank == pp - 1
b = REPA_LAYER - 1
repa_here = lo <= b < hi
stage = Stage(den, den_cfg, lo, hi, first, last, te if first else None, proj if repa_here else None,
              b - lo if repa_here else -1, W)
stage.repa_targets = repa_tgts if repa_here else None
pstage = PipelineStage(stage, rank, pp, torch.device("cpu"))
_fwd = pstage.forward_one_chunk


def _f(i, *a, **k):
    stage._mb = i
    return _fwd(i, *a, **k)
pstage.forward_one_chunk = _f
sched = Schedule1F1B(pstage, M,
                     loss_fn=lambda v, g: F.mse_loss(v, g))  # needed on every stage to enable backward
losses = []
if first and last:
    sched.step(z, t, ids, target=target, losses=losses)
elif first:
    sched.step(z, t, ids)
elif last:
    sched.step(target=target, losses=losses)
else:
    sched.step()
if last:
    print(f"fm loss: pipeline {torch.stack(losses).mean().item():.6f}")
mine = [("den", n) for n in owned_den_names(den, rank, pp, ranges)]
if first:
    mine += [("te", n) for n, _ in te.named_parameters()]
params = dict([(("den", n), p) for n, p in den.named_parameters()] + [(("te", n), p) for n, p in te.named_parameters()])
missing = [k for k in mine if params[k].grad is None]
err = max((params[k].grad - ref_grads[k]).abs().max().item() / (ref_grads[k].abs().max().item() + 1e-12)
          for k in mine if params[k].grad is not None)
print(f"stage {rank}/{pp} blocks {lo}-{hi}: {len(mine)} params, max relative grad diff {err:.1e}, missing {missing}")
assert not missing and err < 1e-4
if repa_here:
    perr = max((p.grad - ref_proj[n]).abs().max().item() / (ref_proj[n].abs().max().item() + 1e-12)
               for n, p in proj.named_parameters())
    print(f"stage {rank}: REPA projector max relative grad diff {perr:.1e}")
    assert perr < 1e-4
dist.barrier()
if rank == 0:
    print("OK")
dist.destroy_process_group()
