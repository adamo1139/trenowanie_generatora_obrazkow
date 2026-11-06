"""Lower-resolution latents from the 256 px ones on disk: decode (full FLUX.2 decoder) -> 256 px image -> antialiased
bicubic downscale to each --sizes -> FLUX.2 encoder (posterior mode, as the FLUX.2 pipeline encodes images) -> packed
2x2 and normalised with the VAE's batch-norm statistics, i.e. the same layout as <shard>.lat.npy.
Writes <shard>.lat<size>.npy, (N, 128, size/16, size/16) float16, next to each shard; shards already done are skipped,
so an interrupted run resumes. Shards are split over the ranks; no collectives.

  torchrun --nproc_per_node 8 prepare_lowres.py --data data/aurora_full --vae FLUX.2-dev-vae --sizes 128 64 32
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from aurora.repa import unpatchify_flux2


def patchify_flux2(z):
    """(B, 32, 2h, 2w) -> (B, 128, h, w); inverse of unpatchify_flux2 (channel = c*4 + dh*2 + dw)."""
    b, c, h2, w2 = z.shape
    return z.view(b, c, h2 // 2, 2, w2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2 // 2, w2 // 2)


@torch.no_grad()
def convert(lat, vae, mean, std, sizes, device):
    """(B, 128, 16, 16) normalised latents -> {size: (B, 128, size/16, size/16) normalised latents}, float16."""
    x = torch.from_numpy(np.ascontiguousarray(lat)).to(device).float()
    img = vae.decode(unpatchify_flux2(x * std + mean).to(torch.bfloat16)).sample.float().clamp(-1, 1)
    out = {}
    for s in sizes:
        small = F.interpolate(img, size=(s, s), mode="bicubic", antialias=True, align_corners=False).clamp(-1, 1)
        z = patchify_flux2(vae.encode(small.to(torch.bfloat16)).latent_dist.mode().float())
        out[s] = ((z - mean) / std).half().cpu().numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--vae", default="FLUX.2-dev-vae", help="diffusers AutoencoderKLFlux2 with encoder and full decoder")
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--batch", type=int, default=128)
    a = p.parse_args()
    assert all(s % 16 == 0 and s < 256 for s in a.sizes), "sizes must be multiples of 16, below 256"
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(a.vae, torch_dtype=torch.bfloat16).to(device).eval()
    eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
    mean = vae.bn.running_mean.float().view(1, -1, 1, 1).to(device)
    std = torch.sqrt(vae.bn.running_var.float() + eps).view(1, -1, 1, 1).to(device)
    shards = sorted(glob.glob(os.path.join(a.data, "*.lat.npy")))[rank::world]
    t0, done = time.time(), 0
    for path in shards:
        stem = path[: -len(".lat.npy")]
        outs = {s: f"{stem}.lat{s}.npy" for s in a.sizes}
        if all(os.path.exists(o) for o in outs.values()):
            continue
        lat = np.load(path, mmap_mode="r")
        res = {s: np.empty((len(lat), 128, s // 16, s // 16), dtype=np.float16) for s in a.sizes}
        for i in range(0, len(lat), a.batch):
            for s, z in convert(lat[i:i + a.batch], vae, mean, std, a.sizes, device).items():
                res[s][i:i + len(z)] = z
        for s, o in outs.items():
            np.save(o + ".tmp.npy", res[s])
            os.replace(o + ".tmp.npy", o)
        done += len(lat)
        print(f"rank {rank}: {os.path.basename(stem)} ({len(lat)} images), {done / (time.time() - t0):.0f} img/s",
              flush=True)


if __name__ == "__main__":
    main()
