"""Same latents, two FLUX.2 decoders: top row the small decoder shipped with HomeIMG, bottom row the full one.
Both share the encoder (identical latent normalisation), so only the decoding differs. Saved 1:1 and 2x (nearest).

  python tests/decoder_grid.py --model cpral/HomeIMG-641M-PL-epoch19 --full kaiyuyue/FLUX.2-dev-vae
"""
import argparse
import os
import sys

import torch
from diffusers import AutoencoderKLFlux2
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from guidance_grid import FACES, GENERAL, load, sample  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cpral/HomeIMG-641M-PL-epoch19")
    p.add_argument("--full", default="kaiyuyue/FLUX.2-dev-vae", help="repo[:subfolder] of the full VAE")
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="decoder_grid")
    a = p.parse_args()
    device = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    pipe = load(a.model, device)
    small = pipe.vae
    repo, _, sub = a.full.partition(":")
    full = AutoencoderKLFlux2.from_pretrained(repo, subfolder=sub or None, torch_dtype=small.dtype).to(device).eval()
    # same encoder, so the same latent normalisation; some copies store it in bf16, so up to bf16 rounding.
    # Decode both with the exact statistics the model was trained with
    for name in ("running_mean", "running_var"):
        s, f = getattr(small.bn, name), getattr(full.bn, name)
        assert torch.allclose(s.float().cpu(), f.float().cpu(), rtol=1e-2, atol=1e-3), f"bn.{name} differs"
        f.copy_(s)
    prompts = GENERAL + FACES
    with torch.no_grad():
        z = sample(pipe, prompts, ("cfg", a.cfg, {}), {}, a.steps, a.seed, device, decode=False)
    rows = []
    for vae in (small, full):
        pipe.vae = vae
        with torch.no_grad():
            rows.append(((pipe.decode_latents(z) + 1) / 2).clamp(0, 1))
    grid = Image.new("RGB", (256 * len(prompts), 512))
    for r, imgs in enumerate(rows):
        arr = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
        for c, im in enumerate(arr):
            grid.paste(Image.fromarray(im), (256 * c, 256 * r))
    grid.save(os.path.join(a.out_dir, "decoders.png"))
    grid.resize((grid.width * 2, grid.height * 2), Image.NEAREST).save(os.path.join(a.out_dir, "decoders_2x.png"))
    print(f"saved {a.out_dir}/decoders.png and decoders_2x.png: top small decoder, bottom {a.full}", flush=True)


if __name__ == "__main__":
    main()
