"""Guidance variants side by side, same prompts and noise: rows = methods, columns = prompts.

  * cfg       : v = v_u + w (v_c - v_u), the released sampler
  * cfgzero   : CFG-Zero* (Fan et al. 2025, arXiv:2503.18886): the unconditional velocity is first scaled by
                s* = <v_c, v_u> / |v_u|^2 (per sample), v = s* v_u + w (v_c - s* v_u); the first step is skipped
                (zero velocity: "zero-init")
  * ag:<repo> : autoguidance (Karras et al. 2024, arXiv:2406.02507): a weaker model with the SAME prompt replaces the
                unconditional branch, v = v_bad + w (v_c - v_bad). Every HomeIMG model shares the FLUX.2 latent space
                and t convention, so any older one can guide; each encodes the prompt with its own text encoder
                (the older ones are English-only)

Options per method after "|": shift=S (time grid t = S u / (1 + (S - 1) u), u = linspace(1, 0): more steps at high
noise, as AuroraIMG2-100M samples with S = 1.5) and off=T (no guidance while t <= T: plain conditional velocity on the
last, lowest-noise steps; AuroraIMG2-100M uses 0.05).

  python tests/guidance_grid.py --model cpral/HomeIMG-641M-PL-stage1 --out-dir guidance \\
      --methods cfg:3 "cfg:3|shift=1.5|off=0.05" cfgzero:3 ag:cpral/HomeIMG-50M:2
"""
import argparse
import os

import torch
from diffusers import DiffusionPipeline
from PIL import Image

GENERAL = [
    "a red fox sitting in a snowy forest at sunrise, photograph",
    "a futuristic city skyline at night with neon lights reflected in the rain",
    "a bowl of fresh strawberries on a wooden table, soft natural light",
    "a watercolor painting of a lighthouse on a rocky coast",
    "a small robot reading a book in a cozy library",
    "an astronaut riding a horse on the moon, digital art",
    "a street sign that says OPEN in neon letters",
    "two children flying a kite on a green hill under a blue sky",
]
FACES = [
    "close-up portrait photo of a young woman, studio lighting",
    "close-up portrait of an old man with a grey beard, photograph",
    "a smiling girl with freckles, portrait, natural light",
    "a man walking down a city street, photograph",
]


def load(repo, device, vae=True):
    pipe = DiffusionPipeline.from_pretrained(repo, custom_pipeline=repo, trust_remote_code=True)
    if not vae:
        pipe.vae = None
    return pipe.to(device)


@torch.no_grad()
def sample(good, prompts, method, guides, steps, seed, device, decode=True):
    kind, w, opts = method[0], method[-2], method[-1]
    shift, off = float(opts.get("shift", 1.0)), float(opts.get("off", -1.0))
    b = len(prompts)
    tc = good.transformer.config
    g = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(b, tc.latent_channels, tc.latent_size, tc.latent_size, device=device, generator=g)
    if kind == "ag":
        bad = guides[method[1]]
        ctx, valid = good.encode_prompt(prompts, None, False, device)
        bctx, bvalid = bad.encode_prompt(prompts, None, False, device)
    else:
        ctx, valid = good.encode_prompt(prompts, None, True, device)  # [uncond; cond]
    u = torch.linspace(1.0, 0.0, steps + 1, device=device)
    ts = shift * u / (1 + (shift - 1) * u)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        with good._autocast():
            if t0 <= off:  # guidance switched off on the last steps: conditional velocity only
                if kind == "ag":
                    v = good.transformer(z, t0.expand(b), ctx, valid).float()
                else:
                    v = good.transformer(z, t0.expand(b), ctx[b:], valid[b:]).float()
            elif kind == "ag":
                vc = good.transformer(z, t0.expand(b), ctx, valid).float()
                vb = bad.transformer(z, t0.expand(b), bctx, bvalid).float()
                v = vb + w * (vc - vb)
            else:
                vu, vc = good.transformer(torch.cat([z, z]), t0.expand(2 * b), ctx, valid).float().chunk(2)
                if kind == "cfgzero":
                    if i == 0:
                        continue  # zero-init: no update on the first step
                    dot = (vc * vu).flatten(1).sum(1)
                    s = (dot / (vu.flatten(1).pow(2).sum(1) + 1e-8)).view(-1, 1, 1, 1)
                    vu = s * vu
                v = vu + w * (vc - vu)
        z = z + (t1 - t0) * v
    if not decode:
        return z
    return ((good.decode_latents(z) + 1) / 2).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cpral/HomeIMG-641M-PL-stage1")
    p.add_argument("--methods", nargs="+", default=["cfg:3", "cfgzero:3", "ag:cpral/HomeIMG-253M:2",
                                                     "ag:cpral/HomeIMG-50M:2"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="guidance")
    a = p.parse_args()
    device = "cuda"
    os.makedirs(a.out_dir, exist_ok=True)
    methods = []
    for m in a.methods:
        spec, *extra = m.split("|")
        opts = dict(e.split("=") for e in extra)
        parts = spec.split(":")
        methods.append((parts[0], ":".join(parts[1:-1]), float(parts[-1]), opts) if parts[0] == "ag"
                       else (parts[0], float(parts[1]), opts))
    good = load(a.model, device)
    good.vae.to(torch.bfloat16)
    guides = {m[1]: load(m[1], device, vae=False) for m in methods if m[0] == "ag"}
    for name, prompts in (("general", GENERAL), ("faces", FACES)):
        rows = [sample(good, prompts, m, guides, a.steps, a.seed, device) for m in methods]
        grid = Image.new("RGB", (256 * len(prompts), 256 * len(rows)))
        for r, imgs in enumerate(rows):
            arr = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype("uint8")
            for c, im in enumerate(arr):
                grid.paste(Image.fromarray(im), (256 * c, 256 * r))
        path = os.path.join(a.out_dir, f"{name}.png")
        grid.save(path)
        print(f"saved {path}: rows = {a.methods}", flush=True)


if __name__ == "__main__":
    main()
