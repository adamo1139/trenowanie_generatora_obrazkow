"""REPA (representation alignment, Yu et al. 2024, arXiv:2410.06940) with Meta's Perception Encoder as the target.

The dataset only has latents, so targets are computed on the fly: latent -> FLUX.2 small decoder -> 256x256 image ->
PE-Spatial-B16 (timm `vit_pe_spatial_base_patch16_512.fb`, Apache-2.0) -> 16x16 patch features, which line up 1:1 with
the DiT's 16x16 latent tokens (one packed latent pixel = 16x16 image pixels = one PE patch, both row-major).
The decoder is the expensive part (~8 ms/image on an RTX 5080), so train.py only runs it for a fraction of each
micro-batch and only for the first part of training (--repa-frac, --repa-until).
"""
import torch
import torch.nn as nn


def unpatchify_flux2(z):
    """(B, 128, h, w) packed latents -> (B, 32, 2h, 2w); FLUX.2 pipeline layout, channel = c*4 + dh*2 + dw."""
    b, c4, h, w = z.shape
    return z.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c4 // 4, h * 2, w * 2)


class REPATarget:
    """Frozen decoder + vision encoder (plain attributes, not an nn.Module: never trained, never checkpointed)."""

    def __init__(self, vae_dir: str, encoder: str, device):
        import timm
        from diffusers import AutoencoderKLFlux2
        vae = AutoencoderKLFlux2.from_pretrained(vae_dir)
        eps = float(getattr(vae.config, "batch_norm_eps", 1e-4))
        self.mean = vae.bn.running_mean.detach().float().view(1, -1, 1, 1).to(device)
        self.std = torch.sqrt(vae.bn.running_var.detach().float() + eps).view(1, -1, 1, 1).to(device)
        vae.encoder = None  # only the decoder is needed
        self.vae = vae.to(device, torch.bfloat16).eval().requires_grad_(False)
        self.enc = timm.create_model(encoder, pretrained=True, dynamic_img_size=True)
        self.enc = self.enc.to(device, torch.bfloat16).eval().requires_grad_(False)
        cfg = self.enc.pretrained_cfg
        self.img_mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.img_std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.prefix = getattr(self.enc, "num_prefix_tokens", 0)
        self.dim = self.enc.num_features

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        """Normalised packed latents (B, 128, 16, 16) -> encoder patch features (B, 256, dim), fp32."""
        z = unpatchify_flux2(x0.float() * self.std + self.mean)
        img = self.vae.decode(z.to(torch.bfloat16)).sample.float().clamp(-1, 1)
        img = ((img + 1) / 2 - self.img_mean) / self.img_std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.enc.forward_features(img)
        return feats[:, self.prefix:].float()


def make_projector(dim: int, target_dim: int, hidden: int = 2048) -> nn.Module:
    """REPA's 3-layer MLP from DiT hidden states to the target feature space (training only, not exported)."""
    return nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, target_dim))
