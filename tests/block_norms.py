"""How far grown blocks have moved from zero: per denoiser block, the RMS of the three output projections that
train_pp.py --init zero-initialises in new blocks (attn.out, cross.out, mlp.out_proj), raw and EMA weights, and each
block's size relative to the mean of the original (old) blocks. Reads a train.py / train_pp.py checkpoint lazily.

  python tests/block_norms.py runs/poziomka_grow12_3ep/latest.pt --old 6
"""
import argparse

import torch

OUTS = ("attn.out.weight", "cross.out.weight", "mlp.out_proj.weight")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--old", type=int, default=6, help="blocks that came from the --init checkpoint")
    a = p.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    den = ck["den"]
    ema = None
    if ck.get("ema") is not None:  # train.py's EMA list order: te parameters, then denoiser parameters
        ema = dict(zip(den.keys(), ck["ema"][len(ck["te"]):]))
    depth = 1 + max(int(k.split(".")[1]) for k in den if k.startswith("blocks."))
    print(f"{a.ckpt}: step {ck.get('step')}, {depth} blocks (0-{a.old - 1} old, {a.old}-{depth - 1} grown)")
    for src, sd in (("raw", den), ("EMA", ema)):
        if sd is None:
            continue
        rms = {(i, o): sd[f"blocks.{i}.{o}"].float().pow(2).mean().sqrt().item() for i in range(depth) for o in OUTS}
        old = {o: sum(rms[i, o] for i in range(a.old)) / a.old for o in OUTS}
        print(f"\n{src} weights: RMS of the output projections (x 1e-3), and relative to the old blocks' mean")
        print("block  " + "  ".join(f"{o.split('.weight')[0]:>22s}" for o in OUTS))
        for i in range(depth):
            tag = "old" if i < a.old else "new"
            cells = [f"{rms[i, o] * 1e3:9.3f} ({rms[i, o] / old[o]:5.1%})" for o in OUTS]
            print(f"{i:3d} {tag}  " + "  ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
