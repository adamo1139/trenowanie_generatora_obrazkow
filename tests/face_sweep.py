"""Faces side by side for several models (e.g. before / after --opd): rows = models, columns = prompts, one grid per
seed; every model gets the same prompts and noise. Models load one at a time (fits a 16 GB GPU).

  python tests/face_sweep.py --models cpral/HomeIMG-1.2B-PL-EN-epoch22:3 cpral/HomeIMG-1.2B-PL-EN-epoch22-OPD300:1 \\
      --out-dir faces --seeds 0 1 --cache-dir /path/with/space
"""
import argparse
import gc
import os

import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw

FACES_PL = [
    "portret młodej kobiety w studiu, miękkie światło, fotografia",
    "zbliżenie twarzy starszego mężczyzny z siwą brodą, fotografia",
    "uśmiechnięta dziewczynka z piegami, portret, naturalne światło",
    "30 letni Koreańczyk, bliski rzut na twarz, widać teksturę skóry",
    "portret starszej kobiety w chustce na wsi, fotografia",
    "zbliżenie twarzy płaczącego dziecka, fotografia",
    "żołnierz w mundurze, czarno-biały portret",
    "mężczyzna idący ulicą miasta, fotografia",
    "grupa przyjaciół śmiejących się przy stole w kawiarni",
    "piłkarz cieszący się z gola na stadionie",
    "kobieta czytająca książkę na ławce w parku, zdjęcie z daleka",
    "para młoda tańcząca na weselu",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="repo_or_dir:cfg, e.g. cpral/HomeIMG-1.2B-PL-EN-epoch22:3")
    p.add_argument("--prompts", nargs="+", default=FACES_PL)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--out-dir", default="faces")
    p.add_argument("--cache-dir", default=None, help="HF cache for the downloads (default: the usual one)")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = "cuda"
    rows = {}  # (seed, model) -> images
    for spec in a.models:
        repo, cfg = spec.rsplit(":", 1)
        pipe = DiffusionPipeline.from_pretrained(repo, custom_pipeline=repo, trust_remote_code=True,
                                                 cache_dir=a.cache_dir).to(device)
        pipe.vae.to(torch.bfloat16)
        pipe.set_progress_bar_config(disable=True)
        for seed in a.seeds:
            g = torch.Generator(device=device).manual_seed(seed)  # one batch per seed: the same noise for every model
            rows[seed, spec] = pipe(a.prompts, num_inference_steps=a.steps, guidance_scale=float(cfg),
                                    generator=g).images
        print(f"done {spec}", flush=True)
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    label_w = 220
    for seed in a.seeds:
        grid = Image.new("RGB", (label_w + 256 * len(a.prompts), 256 * len(a.models)), "white")
        draw = ImageDraw.Draw(grid)
        for r, spec in enumerate(a.models):
            draw.multiline_text((8, 256 * r + 110), spec.replace(":", "\ncfg "), fill="black")
            for c, im in enumerate(rows[seed, spec]):
                grid.paste(im.resize((256, 256)), (label_w + 256 * c, 256 * r))
        path = os.path.join(a.out_dir, f"faces_seed{seed}.png")
        grid.save(path)
        print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
