"""Is FLUX.2 [klein] base 4B a usable teacher at 256 px? (aurora/klein.py)

  1. --check: our Euler sampler with KleinTeacher.velocity vs the official Flux2KleinPipeline, same prompts and noise
     (validates the wrapper's conventions; the difference should be bf16 noise).
  2. grids of sample.py's 8 README prompts in English and Polish, with the noise our sample_ckpt.py grids use (seed 0),
     so they compare image by image with a HomeIMG release's grid_en.png / grid_pl.png.
  3. FD-DINOv2 on the 4,096 held-out captions exactly as fid_eval.py (same captions, same per-rank noise for the same
     number of GPUs, same reference, same decoder), so the number sits next to our models' FD.

  python tests/klein_teacher_check.py --data data/aurora_full --out-dir klein_check --check --samples 0
  NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 tests/klein_teacher_check.py --data data/aurora_full --out-dir klein_check
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aurora.data import load_val, select_shards  # noqa: E402
from aurora.fid import frechet, gather_features, stats  # noqa: E402
from aurora.klein import REPO, KleinTeacher  # noqa: E402
from aurora.repa import REPATarget, unpatchify_flux2  # noqa: E402
from sample import PROMPTS, PROMPTS_PL  # noqa: E402


def val_captions(data, names, n, lang):
    """First caption of the first images of each held-out shard, in load_val's order."""
    per = -(-n // len(names))
    out = []
    for name in names:
        caps = json.load(open(os.path.join(data, f"{name}.caps{'_' + lang if lang else ''}.json")))
        out += [c[0] if c else "" for c in caps[:per]]
    return out[:n]


def latents(teacher, prompts, null, seed, cfg, steps, batch=64, micro=16):
    """fid.sample's noise (seed * 1_000_003 + batch start, per batch of 64) through klein."""
    out = []
    for i in range(0, len(prompts), batch):
        ps = prompts[i:i + batch]
        g = torch.Generator(device=teacher.device).manual_seed(seed * 1_000_003 + i)
        z = torch.randn(len(ps), 128, 16, 16, device=teacher.device, generator=g)
        for j in range(0, len(ps), micro):
            out.append(teacher.sample(teacher.encode(ps[j:j + micro]), null, z[j:j + micro], steps, cfg))
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", default="klein_check")
    p.add_argument("--vae", default="AuroraIMG2-50M-Preview/vae", help="decoder, as fid_eval.py")
    p.add_argument("--klein", default=REPO, help="klein base repo id or a local download of it")
    p.add_argument("--cfg", type=float, default=4.0, help="klein base's recommended guidance")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--samples", type=int, default=4096, help="FD prompts (0 = skip FD)")
    p.add_argument("--lang", default="", help="'' = English captions, 'pl' = the Polish translations")
    p.add_argument("--check", action="store_true", help="compare with the official pipeline first")
    a = p.parse_args()
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    os.makedirs(a.out_dir, exist_ok=True)
    teacher = KleinTeacher(device, a.klein)
    null = teacher.encode([""])
    enc = REPATarget(a.vae, "vit_base_patch14_reg4_dinov2.lvd142m", device)

    if a.check and rank == 0:
        from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler, Flux2KleinPipeline
        sched = FlowMatchEulerDiscreteScheduler.from_pretrained(a.klein, subfolder="scheduler")
        pipe = Flux2KleinPipeline(scheduler=sched, vae=AutoencoderKLFlux2.from_pretrained(a.klein, subfolder="vae").to(device),
                                  text_encoder=teacher.te, tokenizer=teacher.tok, transformer=teacher.transformer)
        g = torch.Generator(device=device).manual_seed(123)
        noise = torch.randn(2, 128, 16, 16, device=device, generator=g).to(torch.bfloat16)  # same noise for both
        for steps in (1, a.steps):  # 1 step: the velocity alone; many steps: plus the ODE amplifying bf16 rounding
            ref = pipe(prompt=PROMPTS[:2], height=256, width=256, latents=noise, num_inference_steps=steps,
                       guidance_scale=a.cfg, output_type="latent").images.float()  # denormalised, unpatchified
            ours = teacher.sample(teacher.encode(PROMPTS[:2]), null, noise.float(), steps, a.cfg)
            ours = unpatchify_flux2(ours * enc.std + enc.mean)
            print(f"check vs Flux2KleinPipeline, {steps} steps: relative difference "
                  f"{((ours - ref).norm() / ref.norm()).item():.4f}", flush=True)
        del pipe

    if rank == 0:
        for lang, prompts in (("en", PROMPTS), ("pl", PROMPTS_PL)):
            z = latents(teacher, prompts, null, 0, a.cfg, a.steps)
            imgs = ((enc.decode(z) + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            grid = Image.new("RGB", (4 * 256, 2 * 256))
            for i, im in enumerate((imgs * 255).round().astype(np.uint8)):
                grid.paste(Image.fromarray(im), ((i % 4) * 256, (i // 4) * 256))
            grid.save(os.path.join(a.out_dir, f"grid_{lang}.png"))
            print(f"saved {a.out_dir}/grid_{lang}.png", flush=True)

    if a.samples:
        _, val_names = select_shards(a.data, 1.0, "poziomka")
        lat, _ = load_val(a.data, val_names, a.samples, "poziomka")
        caps = val_captions(a.data, val_names, a.samples, a.lang)
        n = min(a.samples, len(caps))
        mine = list(range(rank, n, world))
        feats = lambda xs: torch.cat([enc.cls_features(xs[i:i + 64].to(device)) for i in range(0, len(xs), 64)])
        ref = stats(gather_features(feats(lat[mine]), n))
        z = latents(teacher, [caps[i] for i in mine], null, 0, a.cfg, a.steps)
        fd = frechet(*stats(gather_features(feats(z), n)), *ref)
        if rank == 0:
            print(f"FLUX.2-klein-base-4B at 256 px: FD-DINOv2 {fd:.2f} ({n} prompts, {a.lang or 'en'}, cfg {a.cfg}, "
                  f"{a.steps} steps)", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
