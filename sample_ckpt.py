"""Prompt grids straight from a train.py checkpoint (EMA weights if it has them), for text encoders the diffusers
pipeline does not know yet (frozen-LM encoders such as Poziomka): the README prompts in English and in Polish, with the
same noise for prompt k in both grids, so the two languages can be compared image by image. Optionally exports the
weights (EMA) as safetensors + configs for a Hub upload.

  python sample_ckpt.py runs/poziomka_stage1/latest.pt --out-dir exports/poziomka_stage1
  python sample_ckpt.py runs/poziomka_stage1/latest.pt --out-dir exports/poziomka_stage1 --export
"""
import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

from aurora import fid as fidlib
from aurora.model import build_models, configs_from_ckpt
from aurora.repa import REPATarget
from aurora.slayer import LM_PAD
from prepare_data import ENCODERS
from sample import PROMPTS

PROMPTS_PL = [
    "rudy lis siedzący w zaśnieżonym lesie o wschodzie słońca, fotografia",
    "futurystyczna panorama miasta nocą z neonami odbijającymi się w deszczu",
    "miska świeżych truskawek na drewnianym stole, miękkie naturalne światło",
    "akwarela przedstawiająca latarnię morską na skalistym wybrzeżu",
    "mały robot czytający książkę w przytulnej bibliotece",
    "astronauta jadący konno po Księżycu, grafika cyfrowa",
    "szyld z neonowym napisem OPEN",
    "dwoje dzieci puszczających latawiec na zielonym wzgórzu pod błękitnym niebem",
]


def tokenize(texts, te_cfg):
    enc = ENCODERS[te_cfg.tok_suffix]
    tok = Tokenizer.from_file(hf_hub_download(enc["repo"], "tokenizer.json"))
    rows = np.full((len(texts), te_cfg.max_tokens), enc["pad"], dtype=np.int64)
    for i, t in enumerate(texts):  # as prepare_data.py tokenize: collapsed whitespace, [BOS] ids[:max - 1]
        ids = tok.encode(" ".join(t.split()), add_special_tokens=False).ids[: te_cfg.max_tokens - 1]
        rows[i, 0] = enc["bos"]
        rows[i, 1:1 + len(ids)] = ids
    return torch.from_numpy(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="the raw weights instead of the EMA")
    p.add_argument("--vae", default="AuroraIMG2-50M-Preview/vae")
    p.add_argument("--export", action="store_true", help="also write text_encoder / transformer safetensors + configs")
    a = p.parse_args()
    device = torch.device("cuda")
    os.makedirs(a.out_dir, exist_ok=True)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    den_cfg, te_cfg = configs_from_ckpt(ck)
    assert getattr(te_cfg, "kind", "") == "lm", "for the released text encoders use sample.py on an export"
    te, den = build_models(den_cfg, te_cfg)
    te.load_state_dict(ck["te"]), den.load_state_dict(ck["den"])
    if ck.get("ema") and not a.raw:  # EMA list = te params, then den params, in parameters() order
        with torch.no_grad():
            for p_, e in zip(list(te.parameters()) + list(den.parameters()), ck["ema"], strict=True):
                p_.copy_(e)
    src = "raw" if a.raw or not ck.get("ema") else "EMA"
    print(f"{a.ckpt}: step {ck.get('step')}, {src} weights, text encoder {te_cfg.repo} "
          f"({te_cfg.experts or 'all'} experts{', ' + te_cfg.quant if te_cfg.quant else ''})")
    te.to(device).eval(), den.to(device).eval()
    vae = REPATarget(a.vae, "vit_base_patch14_reg4_dinov2.lvd142m", device)
    null_row = torch.from_numpy(te.null_ids()).to(device)[None]

    with torch.no_grad():
        for lang, prompts in (("en", PROMPTS), ("pl", PROMPTS_PL)):
            z = fidlib.sample(te, den, tokenize(prompts, te_cfg), null_row, a.steps, a.cfg, a.seed, device)
            imgs = ((vae.decode(z) + 1) / 2).clamp(0, 1)
            imgs = (imgs.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
            h, w = imgs.shape[1:3]
            grid = Image.new("RGB", (4 * w, 2 * h))
            for i, im in enumerate(imgs):
                grid.paste(Image.fromarray(im), ((i % 4) * w, (i // 4) * h))
            path = os.path.join(a.out_dir, f"grid_{lang}.png")
            grid.save(path)
            print(f"saved {path}")

    if a.export:
        from safetensors.torch import save_file
        for sub, mod, cfg in (("text_encoder", te, te_cfg), ("transformer", den, den_cfg)):
            os.makedirs(os.path.join(a.out_dir, sub), exist_ok=True)
            sd = {k: v.detach().cpu().contiguous() for k, v in mod.state_dict().items()}
            save_file(sd, os.path.join(a.out_dir, sub, "diffusion_pytorch_model.safetensors"))
            with open(os.path.join(a.out_dir, sub, "config.json"), "w") as f:
                json.dump(asdict(cfg), f, indent=1)
        with open(os.path.join(a.out_dir, "prompts.json"), "w") as f:
            json.dump(dict(en=PROMPTS, pl=PROMPTS_PL, steps=a.steps, cfg=a.cfg, seed=a.seed, step=ck.get("step"),
                           weights=src), f, ensure_ascii=False, indent=1)
        print(f"exported weights to {a.out_dir}/text_encoder, {a.out_dir}/transformer")


if __name__ == "__main__":
    main()
