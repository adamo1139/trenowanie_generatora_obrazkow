"""Lower-resolution latents from the 256 px ones on disk: decode (full FLUX.2 decoder) -> 256 px image -> antialiased
bicubic downscale to each --sizes -> FLUX.2 encoder (posterior mode, as the FLUX.2 pipeline encodes images) -> packed
2x2 and normalised with the VAE's batch-norm statistics, i.e. the same layout as <shard>.lat.npy.
Writes <shard>.lat<size>.npy, (N, 128, size/16, size/16) float16, next to each shard; shards already done are skipped,
so an interrupted run resumes. Shards are split over the ranks; no collectives.

  torchrun --nproc_per_node 8 prepare_lowres.py --data data/aurora_full --vae FLUX.2-dev-vae --sizes 128 64 32
  python prepare_lowres.py --data data/aurora_full --vae FLUX.2-dev-vae --sizes 128 64 --bench 20 --batch 64
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


class Phases:
    """CUDA-event timing of named phases (--bench); a no-op when disabled."""

    def __init__(self, on: bool):
        self.on, self.ev, self.ms = on, [], {}

    def __call__(self, name):
        if self.on:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev.append((name, e))

    def flush(self):
        if self.on and self.ev:
            torch.cuda.synchronize()
            for (_, a), (name, b) in zip(self.ev[:-1], self.ev[1:]):
                self.ms[name] = self.ms.get(name, 0.0) + a.elapsed_time(b)
        self.ev = []


@torch.no_grad()
def convert(lat, vae, mean, std, sizes, device, phases=None, channels_last=False):
    """(B, 128, 16, 16) normalised latents -> {size: (B, 128, size/16, size/16) normalised latents}, float16."""
    ph = phases or Phases(False)
    ph("start")
    x = torch.from_numpy(np.ascontiguousarray(lat)).to(device).float()
    z = unpatchify_flux2(x * std + mean).to(torch.bfloat16)
    if channels_last:
        z = z.contiguous(memory_format=torch.channels_last)
    ph("upload")
    img = vae.decode(z).sample.float().clamp(-1, 1)
    ph("decode 256")
    out = {}
    for s in sizes:
        small = F.interpolate(img, size=(s, s), mode="bicubic", antialias=True, align_corners=False).clamp(-1, 1)
        small = small.to(torch.bfloat16)
        if channels_last:
            small = small.contiguous(memory_format=torch.channels_last)
        ph(f"resize {s}")
        z = patchify_flux2(vae.encode(small).latent_dist.mode().float())
        ph(f"encode {s}")
        out[s] = ((z - mean) / std).half().cpu().numpy()
        ph(f"download {s}")
    ph.flush()
    return out


def bench(paths, vae, mean, std, a, device, rank):
    """--bench: --bench batches of the first shard (after 2 warm-up batches), nothing written."""
    lat = np.load(paths[0], mmap_mode="r")
    n = a.batch * (a.bench + 2)
    lat = np.concatenate([lat] * (-(-n // len(lat))))[:n]
    for i in range(2):
        convert(lat[i * a.batch:(i + 1) * a.batch], vae, mean, std, a.sizes, device, None, a.channels_last)
    ph = Phases(True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(2, a.bench + 2):
        convert(lat[i * a.batch:(i + 1) * a.batch], vae, mean, std, a.sizes, device, ph, a.channels_last)
    dt = time.perf_counter() - t0
    imgs = a.batch * a.bench
    parts = ", ".join(f"{k} {v / a.bench:.0f} ms" for k, v in ph.ms.items())
    print(f"rank {rank}: {imgs / dt:.1f} img/s (batch {a.batch}, sizes {a.sizes}, decoder {a.decoder or a.vae}, "
          f"compile {a.compile or 'off'}, cudnn.benchmark {a.cudnn_benchmark}, channels_last "
          f"{a.channels_last}, peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GB) | per batch: {parts}",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--vae", default="FLUX.2-dev-vae", help="diffusers AutoencoderKLFlux2 with encoder and full decoder")
    p.add_argument("--decoder", default="", help="another AutoencoderKLFlux2 whose decoder makes the 256 px images, e.g. "
                   "the small decoder (~1.8x cheaper; its differences mostly vanish after a 2-4x downscale); default --vae")
    p.add_argument("--out-dir", default="", help="write <out-dir>/<size>px/<shard>.lat<size>.npy (one folder per size, "
                   "e.g. for a Hub upload; train_pp.py --lat-dir <out-dir>/<size>px); default: next to the shards")
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 64, 32])
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--bench", type=int, default=0,
                   help="time this many batches of the first shard per rank (img/s and per-phase ms), write nothing")
    p.add_argument("--compile", nargs="?", const="default", default="",
                   help="torch.compile the decoder and the encoder (optionally a mode, e.g. max-autotune)")
    p.add_argument("--cudnn-benchmark", action="store_true", help="let cuDNN pick its fastest conv algorithms")
    p.add_argument("--cpu-threads", type=int, default=3,
                   help="torch CPU threads per rank (torchrun sets 1); 3 x 8 ranks = 24")
    p.add_argument("--channels-last", action="store_true", help="NHWC activations (often faster convs on Ampere)")
    a = p.parse_args()
    assert all(s % 16 == 0 and s < 256 for s in a.sizes), "sizes must be multiples of 16, below 256"
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.set_num_threads(a.cpu_threads)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    from diffusers import AutoencoderKLFlux2
    vae = AutoencoderKLFlux2.from_pretrained(a.vae, torch_dtype=torch.bfloat16).to(device).eval()
    eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
    mean = vae.bn.running_mean.float().view(1, -1, 1, 1).to(device)
    std = torch.sqrt(vae.bn.running_var.float() + eps).view(1, -1, 1, 1).to(device)
    if a.decoder:  # same encoder and latent statistics (FLUX.2), only the decoding differs
        vae.decoder = AutoencoderKLFlux2.from_pretrained(a.decoder, torch_dtype=torch.bfloat16).decoder.to(device).eval()
    if a.channels_last:
        vae.to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = a.cudnn_benchmark
    if a.compile:
        vae.decoder, vae.encoder = (torch.compile(m, mode=None if a.compile == "default" else a.compile)
                                    for m in (vae.decoder, vae.encoder))
    shards = sorted(glob.glob(os.path.join(a.data, "*.lat.npy")))[rank::world]
    if a.bench:
        bench(shards, vae, mean, std, a, device, rank)
        return
    def outs_of(path):
        name = os.path.basename(path)[: -len(".lat.npy")]
        return {s: (os.path.join(a.out_dir, f"{s}px", f"{name}.lat{s}.npy") if a.out_dir
                    else f"{path[: -len('.lat.npy')]}.lat{s}.npy") for s in a.sizes}

    todo = [p_ for p_ in shards if not all(os.path.exists(o) for o in outs_of(p_).values())]
    print(f"rank {rank}: {len(todo)} of {len(shards)} shards to convert", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    io = ThreadPoolExecutor(2)  # the next shard is read and the last one written while the GPU converts

    def save(outs, res):
        for s, o in outs.items():
            np.save(o + ".tmp.npy", res[s])
            os.replace(o + ".tmp.npy", o)

    t0, done, writes = time.time(), 0, []
    nxt = io.submit(np.load, todo[0]) if todo else None
    for k, path in enumerate(todo):
        outs = outs_of(path)
        for o in outs.values():
            os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
        lat = nxt.result()
        nxt = io.submit(np.load, todo[k + 1]) if k + 1 < len(todo) else None
        res = {s: np.empty((len(lat), 128, s // 16, s // 16), dtype=np.float16) for s in a.sizes}
        for i in range(0, len(lat), a.batch):
            chunk = lat[i:i + a.batch]
            n = len(chunk)
            if n < a.batch:  # pad the shard's last batch: one shape for the compiled models, no recompile
                chunk = np.concatenate([chunk, np.repeat(chunk[-1:], a.batch - n, axis=0)])
            for s, z in convert(chunk, vae, mean, std, a.sizes, device, None, a.channels_last).items():
                res[s][i:i + n] = z[:n]
        writes.append(io.submit(save, outs, res))
        done += len(lat)
        el = time.time() - t0
        eta = el / (k + 1) * (len(todo) - k - 1)
        print(f"rank {rank}: [{k + 1}/{len(todo)}] {os.path.basename(path)[: -len('.lat.npy')]} ({len(lat)} images), "
              f"{done / el:.0f} img/s, eta {eta / 3600:.1f} h", flush=True)
    for w in writes:
        w.result()
    io.shutdown()


if __name__ == "__main__":
    main()
