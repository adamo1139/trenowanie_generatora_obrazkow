"""REPA (representation alignment, Yu et al. 2024, arXiv:2410.06940) with a frozen timm vision encoder as the target.

The dataset only has latents, so targets are computed on the fly: latent -> FLUX.2 small decoder -> 256x256 image ->
resized to patch x 16 (224 for patch-14 encoders like DINOv2) -> 16x16 patch features, which line up 1:1 with the DiT's
16x16 latent tokens (both row-major). Default target: DINOv2-B/14 with registers (timm
`vit_base_patch14_reg4_dinov2.lvd142m`, Apache-2.0); Meta PE-Spatial-B16 (`vit_pe_spatial_base_patch16_512.fb`) works too.
iREPA (arXiv:2512.10794, --irepa): a 3x3 conv projector instead of the MLP, and targets normalised over the 16x16
tokens (per image and channel), which removes their shared global component so only spatial structure is aligned.
The decoder is the expensive part (~8 ms/image on an RTX 5080), so train.py only runs it for a fraction of each
micro-batch and only for the first part of training (--repa-frac, --repa-until).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def unpatchify_flux2(z):
    """(B, 128, h, w) packed latents -> (B, 32, 2h, 2w); FLUX.2 pipeline layout, channel = c*4 + dh*2 + dw."""
    b, c4, h, w = z.shape
    return z.reshape(b, c4 // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c4 // 4, h * 2, w * 2)


class REPATarget:
    """Frozen decoder + vision encoder (plain attributes, not an nn.Module: never trained, never checkpointed)."""

    def __init__(self, vae_dir: str, encoder: str, device, spatial_norm: bool = False, grid: int = 16):
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
        patch = self.enc.patch_embed.patch_size
        self.size = (patch[0] * grid, patch[1] * grid)  # one encoder patch per latent token
        self.spatial_norm = spatial_norm

    @torch.no_grad()
    def __call__(self, x0: torch.Tensor) -> torch.Tensor:
        """Normalised packed latents (B, 128, 16, 16) -> encoder patch features (B, 256, dim), fp32."""
        z = unpatchify_flux2(x0.float() * self.std + self.mean)
        img = self.vae.decode(z.to(torch.bfloat16)).sample.float().clamp(-1, 1)
        if tuple(img.shape[-2:]) != self.size:
            img = F.interpolate(img, size=self.size, mode="bicubic", antialias=True, align_corners=False)
        img = ((img + 1) / 2 - self.img_mean) / self.img_std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = self.enc.forward_features(img)
        feats = feats[:, self.prefix:].float()
        if self.spatial_norm:  # iREPA: per image and channel, over the tokens
            feats = (feats - feats.mean(1, keepdim=True)) / (feats.std(1, keepdim=True) + 1e-6)
        return feats


class ConvProjector(nn.Module):
    """iREPA projector: one 3x3 conv over the token grid (keeps neighbourhood structure)."""

    def __init__(self, dim: int, target_dim: int, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.conv = nn.Conv2d(dim, target_dim, 3, padding=1)

    def forward(self, h):  # (N, grid*grid, dim) -> (N, grid*grid, target_dim)
        n, t, d = h.shape
        x = h.transpose(1, 2).reshape(n, d, self.grid, self.grid)
        return self.conv(x).flatten(2).transpose(1, 2)


def make_projector(dim: int, target_dim: int, hidden: int = 2048, conv: bool = False) -> nn.Module:
    """REPA's 3-layer MLP (or iREPA's 3x3 conv) from DiT hidden states to the target features (training only)."""
    if conv:
        return ConvProjector(dim, target_dim)
    return nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, target_dim))
