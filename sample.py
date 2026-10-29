"""Generate the README's 8-prompt grid from an exported model dir (or the reference repo, for comparison).

  python sample.py exports/aurora-step247805 --out grid.png
  python sample.py AuroraIMG2-50M-Preview --out grid_ref.png
"""
import argparse

import torch
from diffusers import DiffusionPipeline
from PIL import Image

PROMPTS = [
    "a red fox sitting in a snowy forest at sunrise, photograph",
    "a futuristic city skyline at night with neon lights reflected in the rain",
    "a bowl of fresh strawberries on a wooden table, soft natural light",
    "a watercolor painting of a lighthouse on a rocky coast",
    "a small robot reading a book in a cozy library",
    "an astronaut riding a horse on the moon, digital art",
    "a street sign that says OPEN in neon letters",
    "two children flying a kite on a green hill under a blue sky",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--out", default="grid.png")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    pipe = DiffusionPipeline.from_pretrained(args.model, custom_pipeline=args.model, trust_remote_code=True).to("cuda")
    pipe.vae.to(torch.bfloat16)
    images = []
    for prompt in PROMPTS:  # one call per prompt so every image uses seed `args.seed`, like the README grid
        g = torch.Generator(device="cuda").manual_seed(args.seed)
        images.append(pipe(prompt, num_inference_steps=args.steps, guidance_scale=args.cfg, generator=g).images[0])
    w, h = images[0].size
    grid = Image.new("RGB", (4 * w, 2 * h))
    for i, im in enumerate(images):
        grid.paste(im, ((i % 4) * w, (i // 4) * h))
    grid.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
