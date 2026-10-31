"""Which denoiser shape gets the most TFLOPS out of a pipeline on this machine? Measures one block's fwd+bwd per shape
(compiled, bf16, on the current GPU) and estimates a pp-stage pipeline stage: blocks per stage x block time + sending
the activations / text tokens (and their gradients) over the inter-GPU link, assumed not overlapped (upper bound).

  python tests/bench_shapes.py                      # defaults: --pp 8, --mb 8 16, --gbps 3.0 (PCIe 3.0 x4)
  python tests/bench_shapes.py --mb 16 --gbps 6     # x8 links

Depths are 8k - 1 (stage 0 also runs the text encoder, ~1 block); text_dim = width (train.py's default refiner).
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
import torch._inductor.config as ic

ic.split_reductions = False
from aurora.model import DenoiserBlock, build_rope_2d

# name: (dim, heads, mlp_hidden, depth)
SHAPES = {
    "2048x31 hd64 (2b)": (2048, 32, 4864, 31),
    "2048x31 hd128": (2048, 16, 4864, 31),
    "2560x23 hd128": (2560, 20, 6912, 23),
    "2560x31 hd128": (2560, 20, 6912, 31),
    "3072x15 hd128": (3072, 24, 8192, 15),
    "3072x23 hd128": (3072, 24, 8192, 23),
}

p = argparse.ArgumentParser()
p.add_argument("--pp", type=int, default=8)
p.add_argument("--mb", type=int, nargs="+", default=[8, 16])
p.add_argument("--gbps", type=float, default=3.0, help="effective GPU-to-GPU bandwidth per direction, GB/s")
p.add_argument("--peak", type=float, default=153.0, help="GPU peak TFLOPS, for MFU (3090 Ti, fp32-acc unlocked)")
p.add_argument("--only", nargs="*", default=None, help="substrings of shape names to run")
args = p.parse_args()
dev, T, Tt = "cuda", 256, 96


def block_flops(dim, heads, mlp, text_dim, mb):
    img = 4 * dim * dim + 2 * dim * dim + 3 * dim * mlp  # self-attn qkvo, cross q/o, SwiGLU (per image token)
    txt = 2 * text_dim * dim                             # cross k/v (per text token)
    attn = 4 * T * T * dim + 4 * T * Tt * dim            # QK^T + AV, self and cross (per image)
    return 3 * mb * (2 * img * T + 2 * txt * Tt + attn)  # fwd + bwd = 3x fwd


print(f"GPU {torch.cuda.get_device_name()}, pp {args.pp}, link {args.gbps} GB/s, peak {args.peak} TFLOPS\n")
print(f"{'shape':<20} {'params':>7} {'mb':>3} {'block ms':>9} {'block TF':>9} {'comm ms':>8} {'stage TF':>9} {'MFU':>5}")
for name, (dim, heads, mlp, depth) in SHAPES.items():
    if args.only and not any(o in name for o in args.only):
        continue
    text_dim = dim
    torch.manual_seed(0)
    blk = DenoiserBlock(dim, heads, text_dim, mlp).to(dev)
    with torch.no_grad():
        for q in blk.parameters():
            q.normal_(0, 0.02)
    n_block = sum(q.numel() for q in blk.parameters())
    cblk = torch.compile(blk, dynamic=False)
    cos, sin = (u.to(dev) for u in build_rope_2d(16, dim // heads))
    bps = (depth + 1) // args.pp  # blocks per stage (stage 0: one fewer + text encoder)
    for mb in args.mb:
        x = torch.randn(mb, T, dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
        t6 = torch.randn(mb, 6, dim, device=dev, dtype=torch.bfloat16)
        text = torch.randn(mb, Tt, text_dim, device=dev, dtype=torch.bfloat16)
        valid = torch.ones(mb, Tt, dtype=torch.bool, device=dev)
        valid[:, 50:] = False

        def step():
            with torch.autocast("cuda", torch.bfloat16):
                y = cblk(x, t6, text, valid, cos, sin)
            y.float().square().mean().backward()
        try:
            for _ in range(3):
                step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(10):
                step()
            torch.cuda.synchronize()
        except torch.OutOfMemoryError:
            print(f"{name:<20} mb {mb}: OOM")
            continue
        ms = (time.time() - t0) / 10 * 1000
        fl = block_flops(dim, heads, mlp, text_dim, mb)
        # per hop: x, t6, text, mask forward + their gradients back, bf16
        comm = 2 * 2 * mb * (T * dim + 6 * dim + Tt * text_dim + Tt) / (args.gbps * 1e9) * 1000
        stage_tf = bps * fl / ((bps * ms + comm) / 1000) / 1e12
        print(f"{name:<20} {n_block * depth / 1e9:6.2f}B {mb:>3} {ms:9.1f} {fl / ms / 1e9:9.1f} {comm:8.1f} "
              f"{stage_tf:9.1f} {stage_tf / args.peak:5.0%}", flush=True)
    del blk, cblk
    torch.cuda.empty_cache()
