"""Speed / memory of activation checkpointing for one pipeline stage (4 blocks of the 2B model) in a 1F1B pattern.

  python tests/bench_ckpt.py none|full|selective <micro-batch> <micro-batches in flight>
  python tests/bench_ckpt.py selective 16 8      # stage 0 of --pp 8 at --batch 2048 --microbatches 128
"""
import sys, time, torch
sys.path.insert(0, ".")
import torch._inductor.config as ic
ic.split_reductions = False
from aurora.model import SIZES, DenoiserBlock, build_rope_2d
from train_pp import checkpointed
mode = None if sys.argv[1] == "none" else sys.argv[1]
mb, inflight, M = int(sys.argv[2]), int(sys.argv[3]), 24
cfg = SIZES["2b"]; dev = "cuda"
blocks = torch.nn.ModuleList([DenoiserBlock(cfg.dim, cfg.heads, 768, cfg.mlp_hidden) for _ in range(4)]).to(dev)
with torch.no_grad():
    for p in blocks.parameters(): p.normal_(0, 0.02)
fns = [torch.compile(checkpointed(b, mode) if mode else b, dynamic=False) for b in blocks]
if not mode:
    fns = []
    for b in blocks: b.compile(dynamic=False); fns.append(b)
cos, sin = (u.to(dev) for u in build_rope_2d(16, cfg.dim // cfg.heads))
t6 = torch.randn(mb, 6, cfg.dim, device=dev, dtype=torch.bfloat16)
text = torch.randn(mb, 96, 768, device=dev, dtype=torch.bfloat16)
valid = torch.ones(mb, 96, dtype=torch.bool, device=dev); valid[:, 50:] = False
def fwd():
    x = torch.randn(mb, 256, cfg.dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cuda", torch.bfloat16):
        h = x
        for f in fns: h = f(h, t6, text, valid, cos, sin)
    return h
def run():
    q = []
    for k in range(M):          # 1F1B steady state: keep `inflight` forwards outstanding
        q.append(fwd())
        if len(q) == inflight:
            q.pop(0).float().pow(2).mean().backward()
    while q: q.pop(0).float().pow(2).mean().backward()
run(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
t = time.time(); run(); torch.cuda.synchronize()
print(f"{sys.argv[1]:>9} mb {mb} inflight {inflight}: {(time.time()-t)/M*1000:.1f} ms per micro-batch, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
