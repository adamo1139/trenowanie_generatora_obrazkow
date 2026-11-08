"""FD-DINOv2 (Stein et al. 2023, arXiv:2306.04675): Frechet distance between DINOv2 CLS features of generated and real
held-out images; it tracks human judgement better than Inception FID and reuses the REPA encoder (no extra weights).

Generation follows the HF pipeline exactly: Euler from t=1 (noise) to t=0 over a uniform grid, classifier-free
guidance against the null prompt. Real images are the held-out latents through the same frozen decoder, so the
decoder's own error cancels out. Every rank generates its slice of the prompts; features are gathered on rank 0.
"""
import torch
import torch.distributed as dist


@torch.no_grad()
def sample(te, den, ids, null_row, steps, cfg, seed, device, batch=64, shift=1.0):
    """Normalised latents (N, C, H, W) for the prompts `ids` (N, T); noise is fixed by (seed, prompt index).
    shift: t -> s t / (1 + (s - 1) t) on the time grid (s > 1: more steps at high noise, SD3)."""
    c = getattr(den, "_orig_mod", den).config
    out = []
    u = torch.linspace(1.0, 0.0, steps + 1, device=device)
    ts = shift * u / (1 + (shift - 1) * u)
    for i in range(0, len(ids), batch):
        b_ids = ids[i:i + batch].to(device)
        b = len(b_ids)
        g = torch.Generator(device=device).manual_seed(seed * 1_000_003 + i)
        z = torch.randn(b, c.latent_channels, c.latent_size, c.latent_size, device=device, generator=g)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text, valid = te(torch.cat([null_row.expand(b, -1), b_ids]))
            for k in range(steps):
                t0, t1 = ts[k], ts[k + 1]
                v = den(torch.cat([z, z]), t0.expand(2 * b), text, valid).float()
                v_uncond, v_cond = v.chunk(2)
                z = z + (t1 - t0) * (v_uncond + cfg * (v_cond - v_uncond))
        out.append(z)
    return torch.cat(out)


def frechet(mu1, s1, mu2, s2):
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1^1/2 S2 S1^1/2)^1/2), in float64 via eigendecompositions."""
    mu1, s1, mu2, s2 = (x.double() for x in (mu1, s1, mu2, s2))
    w, q = torch.linalg.eigh(s1)
    r1 = (q * w.clamp(min=0).sqrt()) @ q.T
    covmean = torch.linalg.eigvalsh(r1 @ s2 @ r1).clamp(min=0).sqrt().sum()
    return ((mu1 - mu2).square().sum() + s1.trace() + s2.trace() - 2 * covmean).item()


def stats(f):
    f = f.double()
    return f.mean(0), torch.cov(f.T)


def gather_features(local, n_total):
    """Concatenate every rank's (n_i, d) features on all ranks (slices may differ in length by one)."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return local
    world = dist.get_world_size()
    per = -(-n_total // world)
    pad = torch.zeros(per, local.shape[1], device=local.device, dtype=local.dtype)
    pad[: len(local)] = local
    parts = [torch.empty_like(pad) for _ in range(world)]
    dist.all_gather(parts, pad)
    sizes = [len(range(r, n_total, world)) for r in range(world)]
    return torch.cat([p[:s] for p, s in zip(parts, sizes)])
