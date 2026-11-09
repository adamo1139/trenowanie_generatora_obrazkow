"""Images -> normalised, 2x2-packed FLUX.2 latents on the GPU, for training from images (train_pp.py --img-dir):
the same encoding as prepare_lowres.py (posterior mode, then (z - bn mean) / sqrt(bn var + eps)), so latents made here
match the stored ones. Only the encoder is kept in memory."""
import torch
import torch.nn as nn


def patchify_flux2(z):
    """(B, 32, 2h, 2w) -> (B, 128, h, w); channel = c*4 + dh*2 + dw (the FLUX.2 pipeline's packing)."""
    b, c, h2, w2 = z.shape
    return z.view(b, c, h2 // 2, 2, w2 // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2 // 2, w2 // 2)


class LatentEncoder:
    def __init__(self, path: str, device, chunk: int = 8):
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(path, torch_dtype=torch.bfloat16)
        vae.decoder = nn.Identity()  # never used here
        self.vae = vae.to(device).eval().requires_grad_(False)
        eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
        self.mean = vae.bn.running_mean.float().view(1, -1, 1, 1).to(device)
        self.std = torch.sqrt(vae.bn.running_var.float() + eps).view(1, -1, 1, 1).to(device)
        self.chunk = chunk

    @torch.no_grad()
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """uint8 (B, 3, H, W) on the GPU -> float32 (B, 128, H/16, W/16), in chunks (the encoder's activations at
        full resolution are large)."""
        out = []
        for i in range(0, len(img), self.chunk):
            x = (img[i:i + self.chunk].float() / 127.5 - 1).to(torch.bfloat16)
            z = patchify_flux2(self.vae.encode(x).latent_dist.mode().float())
            out.append((z - self.mean) / self.std)
        return torch.cat(out)
