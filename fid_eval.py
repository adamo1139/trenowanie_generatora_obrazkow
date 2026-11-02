"""FD-DINOv2 (aurora/fid.py) of saved checkpoints: latest.pt / snap_*.pt (EMA weights) and merge/*.pt / merged.pt
(raw weights). Same prompts, seeds and per-rank split as train.py --fid-every, so the numbers are comparable.

  torchrun --nproc_per_node 8 fid_eval.py --data data/aurora_full runs/run/latest.pt runs/run/snap_*.pt runs/run/merge/*.pt
  python fid_eval.py --data data/aurora_full runs/run/latest.pt --samples 1024      # one GPU, quicker
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from aurora.data import load_val, select_shards
from aurora.fid import frechet, gather_features, sample, stats
from aurora.model import build_models, configs_from_ckpt
from aurora.repa import REPATarget


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpts", nargs="+")
    p.add_argument("--data", required=True)
    p.add_argument("--vae", default="AuroraIMG2-50M-Preview/vae")
    p.add_argument("--encoder", default="vit_base_patch14_reg4_dinov2.lvd142m")
    p.add_argument("--samples", type=int, default=4096)
    p.add_argument("--val-samples", type=int, default=4096, help="held-out set size, as in train.py")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0, help="train.py --seed (same noise as the in-training eval)")
    p.add_argument("--raw", action="store_true", help="raw weights even where the checkpoint has EMA weights")
    p.add_argument("--out", default=None, help="append one JSON line per checkpoint to this file")
    a = p.parse_args()

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    enc = REPATarget(a.vae, a.encoder, device)
    feats = lambda xs: torch.cat([enc.cls_features(xs[i:i + 64].to(device)) for i in range(0, len(xs), 64)])
    ref = ids = None
    for path in a.ckpts:
        t0 = time.time()
        ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        den_cfg, te_cfg = configs_from_ckpt(ck)
        te, den = build_models(den_cfg, te_cfg)
        te.load_state_dict(ck["te"]), den.load_state_dict(ck["den"])
        ema = not a.raw and ck.get("ema") is not None
        if ema:  # train.py's EMA list order: te parameters, then denoiser parameters
            with torch.no_grad():
                for q, e in zip(list(te.parameters()) + list(den.parameters()), ck["ema"]):
                    q.copy_(e)
        step = ck.get("step")
        del ck
        te.to(device).eval(), den.to(device).eval()
        if ref is None:  # held-out set and the real images' statistics, once
            tok_suffix = getattr(te_cfg, "tok_suffix", "")
            _, val_names = select_shards(a.data, 1.0, tok_suffix)
            lat, all_ids = load_val(a.data, val_names, a.val_samples, tok_suffix, te.null_ids())
            n = min(a.samples, len(all_ids))
            mine = list(range(rank, n, world))
            ids = all_ids[mine]
            ref = stats(gather_features(feats(lat[mine]), n))
        null_row = torch.from_numpy(te.null_ids()).to(device)
        z = sample(te, den, ids, null_row, a.steps, a.cfg, a.seed, device)
        fd = frechet(*stats(gather_features(feats(z), n)), *ref)
        if rank == 0:
            res = dict(ckpt=path, step=step, weights="ema" if ema else "raw", fd_dinov2=round(fd, 3), samples=n,
                       steps=a.steps, cfg=a.cfg)
            print(f"{path}  step {step}  {res['weights']}  FD-DINOv2 {fd:.2f}  ({time.time() - t0:.0f} s)", flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
        del te, den, z
        torch.cuda.empty_cache()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
