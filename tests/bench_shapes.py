"""Which denoiser shape gets the most TFLOPS out of a pipeline on this machine? Measures one block's fwd+bwd per shape
(compiled, bf16, on the current GPU) and estimates a pp-stage pipeline stage: blocks per stage x block time + sending
the activations / text tokens (and their gradients) over the inter-GPU link, assumed not overlapped (upper bound).

  python tests/bench_shapes.py                      # defaults: --pp 8, --mb 8 16, --gbps 3.0 (PCIe 3.0 x4)
  python tests/bench_shapes.py --mb 16 --gbps 6     # x8 links
  python tests/bench_shapes.py --size 1.6b 1b --pp 4   # presets from aurora/model.py SIZES
  python tests/bench_shapes.py --pp 1 --size 750m 750m-w2048 750m-w1792 750m-w1536 750m-w1280   # train.py, no pipeline
  python tests/bench_shapes.py --sweep              # train.py: widths x micro-batches, depth / memory for a param budget

Depths are 8k - 1 (stage 0 also runs the text encoder, ~1 block); text_dim = width / 2 (train.py's default refiner).
--pp 1 = plain train.py (whole model per GPU): no link cost, stage = the whole stack.
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, ".")
import torch._inductor.config as ic

ic.split_reductions = False
from aurora.model import SIZES, DenoiserBlock, build_rope_2d

# name: (dim, heads, mlp_hidden, depth)
SHAPES = {
    "2048x31 hd64 (2b)": (2048, 32, 4864, 31),
    "2048x31 hd128": (2048, 16, 4864, 31),
    "2048x15 hd64 (1b)": (2048, 32, 4864, 15),
    "2560x7 hd128": (2560, 20, 6912, 7),
    "2560x15 hd128": (2560, 20, 6912, 15),
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
p.add_argument("--size", nargs="+", choices=sorted(SIZES), default=None, help="benchmark these presets instead")
p.add_argument("--sweep", action="store_true",
               help="train.py sweep: every width x micro-batch (block speed + activation memory), then the depth that "
                    "hits each --params budget and whether it fits in --mem-gb with --offload")
p.add_argument("--widths", type=int, nargs="+", default=[1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072])
p.add_argument("--sweep-mb", type=int, nargs="+", default=[8, 16, 24, 32, 48, 64])
p.add_argument("--params", type=float, nargs="+", default=[0.6, 0.75], help="denoiser budgets in B")
p.add_argument("--mem-gb", type=float, default=24.0)
p.add_argument("--overhead-gb", type=float, default=3.5,
               help="CUDA context, frozen LM + refiner, iREPA decoder + DINOv2, compile buffers, allocator slack")
args = p.parse_args()
if args.size:
    SHAPES = {f"{s} ({SIZES[s].dim}x{SIZES[s].depth})": (SIZES[s].dim, SIZES[s].heads, SIZES[s].mlp_hidden,
                                                          SIZES[s].depth) for s in args.size}
dev, T, Tt = "cuda", 256, 96


def block_flops(dim, heads, mlp, text_dim, mb):
    img = 4 * dim * dim + 2 * dim * dim + 3 * dim * mlp  # self-attn qkvo, cross q/o, SwiGLU (per image token)
    txt = 2 * text_dim * dim                             # cross k/v (per text token)
    attn = 4 * T * T * dim + 4 * T * Tt * dim            # QK^T + AV, self and cross (per image)
    return 3 * mb * (2 * img * T + 2 * txt * Tt + attn)  # fwd + bwd = 3x fwd


if args.sweep:
    SHAPES = {}  # the sweep (below) replaces the shape table
print(f"GPU {torch.cuda.get_device_name()}, pp {args.pp}, link {args.gbps} GB/s, peak {args.peak} TFLOPS\n")
print(f"{'shape':<20} {'params':>7} {'blk/st':>6} {'mb':>3} {'block ms':>9} {'block TF':>9} {'comm ms':>8} {'stage TF':>9} {'MFU':>5}")
for name, (dim, heads, mlp, depth) in SHAPES.items():
    if args.only and not any(o in name for o in args.only):
        continue
    text_dim = dim // 2
    torch._dynamo.reset()  # every shape compiles DenoiserBlock.forward again; don't hit the recompile limit
    torch.manual_seed(0)
    blk = DenoiserBlock(dim, heads, text_dim, mlp).to(dev)
    with torch.no_grad():
        for q in blk.parameters():
            q.normal_(0, 0.02)
    n_block = sum(q.numel() for q in blk.parameters())
    cblk = torch.compile(blk, dynamic=False)
    cos, sin = (u.to(dev) for u in build_rope_2d(16, dim // heads))
    bps = depth if args.pp == 1 else max(1, (depth + 1) // args.pp)  # blocks per stage (stage 0: one fewer + TE)
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
        comm = 0.0 if args.pp == 1 else 2 * 2 * mb * (T * dim + 6 * dim + Tt * text_dim + Tt) / (args.gbps * 1e9) * 1000
        stage_tf = bps * fl / ((bps * ms + comm) / 1000) / 1e12
        print(f"{name:<20} {n_block * depth / 1e9:6.2f}B {bps:>6} {mb:>3} {ms:9.1f} {fl / ms / 1e9:9.1f} {comm:8.1f} "
              f"{stage_tf:9.1f} {stage_tf / args.peak:5.0%}", flush=True)
    del blk, cblk
    torch.cuda.empty_cache()


def sweep():
    """train.py (one model per GPU, MuLoCo + --offload): GPU memory = weights + grads + Muon momentum (3 x 4 B/param)
    + activations (depth x measured per block, x1.1 for the backward peak) + overhead."""
    print(f"\nsweep: GPU {torch.cuda.get_device_name()}, budget {args.mem_gb} GB, overhead {args.overhead_gb} GB, "
          f"state 12 B/param (--offload)\n")
    hdr = f"{'width':>5} {'heads':>5} {'mlp':>5} {'mb':>3} {'block ms':>8} {'TFLOPS':>7} {'MFU':>4} {'act/blk':>8}"
    for b_ in args.params:
        hdr += f" | {b_:.2f}B: {'depth':>5} {'ratio':>5} {'GB':>5} fit"
    print(hdr)
    for dim in args.widths:
        heads = dim // 128
        mlp = -(-dim * 8 // 3 // 128) * 128
        text_dim = dim // 2
        torch._dynamo.reset()
        torch.manual_seed(0)
        blk = DenoiserBlock(dim, heads, text_dim, mlp).to(dev)
        with torch.no_grad():
            for q in blk.parameters():
                q.normal_(0, 0.02)
        n_block = sum(q.numel() for q in blk.parameters())
        cblk = torch.compile(blk, dynamic=False)
        cos, sin = (u.to(dev) for u in build_rope_2d(16, dim // heads))
        for mb in args.sweep_mb:
            x = torch.randn(mb, T, dim, device=dev, dtype=torch.bfloat16, requires_grad=True)
            t6 = torch.randn(mb, 6, dim, device=dev, dtype=torch.bfloat16)
            text = torch.randn(mb, Tt, text_dim, device=dev, dtype=torch.bfloat16)
            valid = torch.ones(mb, Tt, dtype=torch.bool, device=dev)
            valid[:, 50:] = False

            def fwd():
                with torch.autocast("cuda", torch.bfloat16):
                    return cblk(x, t6, text, valid, cos, sin)
            try:
                for _ in range(3):
                    fwd().float().square().mean().backward()
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                y = fwd()
                torch.cuda.synchronize()
                act = (torch.cuda.memory_allocated() - base) / 2**30  # saved for backward, per block
                y.float().square().mean().backward()
                del y
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    fwd().float().square().mean().backward()
                torch.cuda.synchronize()
            except torch.OutOfMemoryError:
                print(f"{dim:>5} {heads:>5} {mlp:>5} {mb:>3}  OOM")
                torch.cuda.empty_cache()
                continue
            ms = (time.time() - t0) / 10 * 1000
            tf = block_flops(dim, heads, mlp, text_dim, mb) / ms / 1e9
            row = f"{dim:>5} {heads:>5} {mlp:>5} {mb:>3} {ms:8.1f} {tf:7.1f} {tf / args.peak:4.0%} {act * 1024:6.0f}MB"
            for b_ in args.params:
                depth = max(1, round(b_ * 1e9 / n_block))
                gb = depth * n_block * 12 / 2**30 + depth * act * 1.1 + args.overhead_gb
                row += f" | {'':>7}{depth:>5} {dim / depth:5.0f} {gb:5.1f} {'yes' if gb <= args.mem_gb else 'NO':>3}"
            print(row, flush=True)
            del x, t6, text
        del blk, cblk
        torch.cuda.empty_cache()


if args.sweep:
    sweep()
    sys.exit()
