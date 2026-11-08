"""Images from a model whose text encoder the diffusers pipeline does not know yet (frozen-LM encoders such as
Poziomka): a train.py checkpoint (EMA weights if it has them) or a model exported by this script (local dir or Hub
repo). By default the README prompts in English and in Polish, with the same noise for prompt k in both grids, so the
two languages can be compared image by image; --prompt for your own. --export writes the weights (EMA) as
safetensors + configs + the decoder, for a Hub upload.

  python sample_ckpt.py runs/poziomka_stage1/latest.pt --out-dir exports/HomeIMG-641M-PL-stage1 --export
  python sample_ckpt.py cpral/HomeIMG-641M-PL-stage1 --out-dir out --prompt "rudy kot na parapecie" "a red cat"
"""
import argparse
import json
import os
import shutil
from dataclasses import asdict, fields, replace

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

from aurora import fid as fidlib
from aurora.model import DenoiserConfig, LMEncoderConfig, build_models, configs_from_ckpt
from aurora.repa import REPATarget
from aurora.slayer import LM_PAD
from prepare_data import ENCODERS
from sample import PROMPTS, PROMPTS_PL



def tokenize(texts, te_cfg):
    enc = ENCODERS[te_cfg.tok_suffix]
    tok = Tokenizer.from_file(hf_hub_download(enc["repo"], "tokenizer.json"))
    rows = np.full((len(texts), te_cfg.max_tokens), enc["pad"], dtype=np.int64)
    for i, t in enumerate(texts):  # as prepare_data.py tokenize: collapsed whitespace, [BOS] ids[:max - 1]
        ids = tok.encode(" ".join(t.split()), add_special_tokens=False).ids[: te_cfg.max_tokens - 1]
        rows[i, 0] = enc["bos"]
        rows[i, 1:1 + len(ids)] = ids
    return torch.from_numpy(rows)


def load(src, raw=False, res=None, rope_ref=None):
    """(te, den, info) from a .pt checkpoint, or an exported dir / Hub repo (text_encoder/, transformer/)."""
    if src.endswith(".pt"):
        ck = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
        den_cfg, te_cfg = configs_from_ckpt(ck)
        if res:  # the weights are resolution-free: build the denoiser (RoPE) for this grid
            den_cfg = replace(den_cfg, latent_size=res // 16)
        if rope_ref is not None:
            den_cfg = replace(den_cfg, rope_ref=rope_ref)
        assert getattr(te_cfg, "kind", "") == "lm", "for the released text encoders use sample.py on an export"
        te, den = build_models(den_cfg, te_cfg)
        te.load_state_dict(ck["te"]), den.load_state_dict(ck["den"])
        ema = bool(ck.get("ema")) and not raw
        if ema:  # EMA list = te params, then den params, in parameters() order
            with torch.no_grad():
                for p_, e in zip(list(te.parameters()) + list(den.parameters()), ck["ema"], strict=True):
                    p_.copy_(e)
        return te, den, dict(step=ck.get("step"), weights="EMA" if ema else "raw", root=None)
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    root = src if os.path.isdir(src) else snapshot_download(src)
    def cfg(sub, cls):  # the exported configs also carry diffusers keys (_class_name, _diffusers_version, ...)
        d = json.load(open(os.path.join(root, sub, "config.json")))
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})
    den_cfg = cfg("transformer", DenoiserConfig)
    if res:
        den_cfg = replace(den_cfg, latent_size=res // 16)
    if rope_ref is not None:
        den_cfg = replace(den_cfg, rope_ref=rope_ref)
    te, den = build_models(den_cfg, cfg("text_encoder", LMEncoderConfig))
    for sub, mod in (("text_encoder", te), ("transformer", den)):
        mod.load_state_dict(load_file(os.path.join(root, sub, "diffusion_pytorch_model.safetensors")))
    info = json.load(open(os.path.join(root, "prompts.json"))) if os.path.exists(os.path.join(root, "prompts.json")) \
        else {}
    return te, den, dict(step=info.get("step"), weights=info.get("weights", "exported"), root=root)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="the raw weights instead of the EMA")
    p.add_argument("--vae", default=None, help="FLUX.2 decoder dir (default: the model's vae/, else "
                   "AuroraIMG2-50M-Preview/vae)")
    p.add_argument("--prompt", nargs="+", default=None, help="your own prompts (any language) instead of the grids")
    p.add_argument("--res", type=int, default=None,
                   help="sample at this square resolution (multiple of 16) instead of the checkpoint's; grids get a "
                        "_<res>px suffix")
    p.add_argument("--rope-ref", type=int, default=None,
                   help="override the RoPE position field (aurora.model.build_rope_2d ref), e.g. the trained grid side "
                        "to sample another resolution with the trained spatial scale; grids get a _rope<ref> suffix")
    p.add_argument("--attn-scale", type=float, default=1.0,
                   help="multiply the image self-attention logits (sharper attention when sampling more tokens than "
                        "trained; RPE-2D eq. 7: ln n / ln n_train for n image tokens, e.g. 96 -> 256 px 1.547, "
                        "128 -> 256 px 1.333); grids get an _as<x> suffix")
    p.add_argument("--t-shift", type=float, default=1.0,
                   help="time shift of the sampling grid (SD3; RPE-2D uses sqrt(n_tokens / n_trained_tokens) above "
                        "the trained resolution, e.g. 96 -> 256 px 2.67); grids get a _ts<x> suffix")
    p.add_argument("--batch", type=int, default=64,
                   help="prompts denoised (and decoded) at once; e.g. 2 for 1024 px on a 16 GB card")
    p.add_argument("--export", action="store_true", help="also write text_encoder / transformer safetensors + configs")
    a = p.parse_args()
    device = torch.device("cuda")
    os.makedirs(a.out_dir, exist_ok=True)

    te, den, info = load(a.ckpt, a.raw, a.res, a.rope_ref)
    te_cfg, den_cfg = te.config, den.config
    src = info["weights"]
    print(f"{a.ckpt}: step {info['step']}, {src} weights, text encoder {te_cfg.repo} "
          f"({te_cfg.experts or 'all'} experts{', ' + te_cfg.quant if te_cfg.quant else ''})")
    te.to(device).eval(), den.to(device).eval()
    for m in den.modules():
        if hasattr(m, "logit_mult"):
            m.logit_mult = a.attn_scale
    if a.vae is None:
        own = info["root"] and os.path.join(info["root"], "vae")
        a.vae = own if own and os.path.isdir(own) else "AuroraIMG2-50M-Preview/vae"
    vae = REPATarget(a.vae, "vit_base_patch14_reg4_dinov2.lvd142m", device)
    null_row = torch.from_numpy(te.null_ids()).to(device)[None]

    sets = [("en", PROMPTS), ("pl", PROMPTS_PL)] if a.prompt is None else [("prompt", a.prompt)]
    with torch.no_grad():
        for lang, prompts in sets:
            z = fidlib.sample(te, den, tokenize(prompts, te_cfg), null_row, a.steps, a.cfg, a.seed, device,
                              batch=a.batch, shift=a.t_shift)
            imgs = torch.cat([((vae.decode(z[i:i + a.batch]) + 1) / 2).clamp(0, 1).cpu()
                              for i in range(0, len(z), a.batch)])
            imgs = (imgs.permute(0, 2, 3, 1).float().cpu().numpy() * 255).round().astype(np.uint8)
            if a.prompt is not None:
                for i, im in enumerate(imgs):
                    path = os.path.join(a.out_dir, f"{i:02d}.png")
                    Image.fromarray(im).save(path)
                    print(f"saved {path}: {prompts[i]}")
                continue
            h, w = imgs.shape[1:3]
            grid = Image.new("RGB", (4 * w, 2 * h))
            for i, im in enumerate(imgs):
                grid.paste(Image.fromarray(im), ((i % 4) * w, (i // 4) * h))
            path = os.path.join(a.out_dir, f"grid_{lang}{f'_{a.res}px' if a.res else ''}{f'_rope{a.rope_ref}' if a.rope_ref is not None else ''}{f'_as{a.attn_scale:g}' if a.attn_scale != 1.0 else ''}{f'_ts{a.t_shift:g}' if a.t_shift != 1.0 else ''}.png")
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
            json.dump(dict(en=PROMPTS, pl=PROMPTS_PL, steps=a.steps, cfg=a.cfg, seed=a.seed, step=info["step"],
                           weights=src), f, ensure_ascii=False, indent=1)
        if os.path.abspath(a.vae) != os.path.abspath(os.path.join(a.out_dir, "vae")):
            shutil.copytree(a.vae, os.path.join(a.out_dir, "vae"), dirs_exist_ok=True)
        print(f"exported weights to {a.out_dir}/text_encoder, {a.out_dir}/transformer")


if __name__ == "__main__":
    main()
