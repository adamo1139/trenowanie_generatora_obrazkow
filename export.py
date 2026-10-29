"""Export a training checkpoint (EMA weights) into the AuroraIMG2 diffusers repo layout.

  python export.py runs/aurora/latest.pt exports/aurora-step247805
  python export.py runs/aurora/latest.pt exports/raw --no-ema      # live weights instead of EMA

The tokenizer, FLUX.2 small decoder, configs and pipeline.py are copied from the reference repo, so the result loads
with the released pipeline (needs a diffusers version that has AutoencoderKLFlux2):

  DiffusionPipeline.from_pretrained(out_dir, custom_pipeline=out_dir, trust_remote_code=True)
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from aurora.model import DenoiserConfig, build_models


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--ref", default="AuroraIMG2-50M-Preview")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    den_cfg = DenoiserConfig(**ck["den_config"])
    te, den = build_models(den_cfg)
    te.load_state_dict(ck["te"]), den.load_state_dict(ck["den"])
    if not args.no_ema:
        params = list(te.parameters()) + list(den.parameters())
        assert len(params) == len(ck["ema"])
        with torch.no_grad():
            for p_, e in zip(params, ck["ema"]):
                p_.copy_(e)

    shutil.copytree(args.ref, args.out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".cache", ".git*", "__pycache__", "samples", "*.safetensors", "README.md"))
    shutil.copy(os.path.join(args.ref, "vae", "diffusion_pytorch_model.safetensors"),
                os.path.join(args.out, "vae", "diffusion_pytorch_model.safetensors"))
    for sub, mod in (("text_encoder", te), ("transformer", den)):
        sd = {k: v.contiguous() for k, v in mod.state_dict().items()}
        save_file(sd, os.path.join(args.out, sub, "diffusion_pytorch_model.safetensors"))
    cfg_path = os.path.join(args.out, "transformer", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.update(ck["den_config"])  # our size, the reference's _class_name / _diffusers_version
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    kind = "live" if args.no_ema else ("WSM-merged" if "wsm" in ck else "EMA")
    print(f"exported step {ck['step']} ({kind} weights) to {args.out}")


if __name__ == "__main__":
    main()
