"""36 FLUX-Reason prompts (tests/flux36_en.json + their Polish translations tests/flux36_pl.json) as one 6x6 grid per
resolution and language: CFG 3, 50 steps, seed 0. The noise is drawn in batches of 12 as in the earlier grids (96 /
128 / 256 px), so the same prompt gets the same starting noise at a given resolution, whatever the model.

  python tests/flux36_grid.py runs/poziomka_1024px/latest.pt --vae exports/HomeIMG-1.2B-PL-EN-epoch47.5/vae \
      --name epoch47.75 --res 256 512 768 1024 --out-dir exports/grids36
(a training checkpoint; exported folders made before the current text encoder format don't load)
"""
import argparse
import gc
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aurora import fid as fidlib  # noqa: E402
from aurora.repa import REPATarget  # noqa: E402
from sample_ckpt import load, tokenize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="latest.pt, an export folder or a Hugging Face repo")
    p.add_argument("--res", type=int, nargs="+", default=[256, 512, 768, 1024])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", default="", help="file name prefix (default: the model folder's name)")
    p.add_argument("--vae", default=None, help="FLUX.2 decoder folder (default: the model's vae/)")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--attn-scale", type=float, default=1.0)
    p.add_argument("--t-shift", type=float, default=1.0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.name or os.path.basename(os.path.normpath(a.model if not a.model.endswith(".pt")
                                                       else os.path.dirname(a.model)))
    prompts = {"en": json.load(open(os.path.join(HERE, "flux36_en.json"))),
               "pl": json.load(open(os.path.join(HERE, "flux36_pl.json")))}
    vae = None
    for res in a.res:
        te, den, info = load(a.model, False, res, None)
        te.to("cuda").eval(), den.to("cuda").eval()
        for m in den.modules():
            if hasattr(m, "logit_mult"):
                m.logit_mult = a.attn_scale
        if vae is None:
            path = a.vae or (info["root"] and os.path.join(info["root"], "vae"))
            assert path and os.path.isdir(path), "no vae/ next to the model: pass --vae"
            vae = REPATarget(path, "vit_base_patch14_reg4_dinov2.lvd142m", "cuda")
        null_row = torch.from_numpy(te.null_ids()).to("cuda")[None]
        dec = max(1, int(12 * (256 / res) ** 2))  # decoder images per call: 12 at 256 px, 1 at 1024 px
        for lang, ps in prompts.items():
            with torch.no_grad():
                # noise in batches of 12 (as the earlier grids), denoised 12 / 5 / 3 at a time at <= 512 / 768 / 1024
                # px (24 GB). Poziomka's MoE routing depends slightly on the batch, so keep this fixed per resolution.
                cb = max(1, min(12, int(12 * (512 / res) ** 2)))
                z = fidlib.sample(te, den, tokenize(ps, te.config), null_row, a.steps, a.cfg, 0, "cuda", batch=12,
                                  shift=a.t_shift, compute_batch=cb)
                imgs = torch.cat([((vae.decode(z[i:i + dec]) + 1) / 2).clamp(0, 1).cpu()
                                  for i in range(0, len(z), dec)])
            arr = (imgs.permute(0, 2, 3, 1).float().numpy() * 255).round().astype(np.uint8)
            grid = Image.new("RGB", (6 * res, 6 * res))
            for k, im in enumerate(arr):
                grid.paste(Image.fromarray(im), (res * (k % 6), res * (k // 6)))
            path = os.path.join(a.out_dir, f"{name}_flux36_{res}px_{lang}.jpg")
            grid.save(path, quality=92)
            print("saved", path, flush=True)
        del te, den
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
