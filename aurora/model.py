"""AuroraIMG2 text encoder + denoiser as plain nn.Modules.

Parameter names match AuroraIMG2-50M-Preview/{text_encoder,transformer}/diffusion_pytorch_model.safetensors exactly,
so released weights load with strict=True and our checkpoints export to that layout unchanged.
The building blocks are copied from the reference pipeline.py; only the init scheme is ours (not published).
"""
import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def masked_attention(q, k, v, mask, backend):
    if backend == "cudnn":
        # Keep the backend choice inside the compiled graph, including backward.
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


# =============================================================================
# Building blocks (identical to the reference pipeline.py)
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def _angles(positions: torch.Tensor, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    return torch.outer(positions.float(), inv_freq)


def build_rope_1d(n: int, d: int):
    ang = _angles(torch.arange(n), d)
    return ang.cos().repeat_interleave(2, dim=-1)[None, None], ang.sin().repeat_interleave(2, dim=-1)[None, None]


def build_rope_2d(side: int, d: int, ref: int = 0):
    """First half of the head dim encodes the row, second half the column. Tokens are row-major.
    ref = 0: positions 0 .. side-1 (one unit per token, the grid grows with the resolution). ref > 0: the side tokens
    are spread over the same field as a ref x ref grid (centres (i + 0.5) ref / side - 0.5; at side == ref exactly
    0 .. ref-1), so every resolution sees the same spatial scale (position interpolation)."""
    coords = torch.arange(side).float()
    if ref and ref != side:
        coords = (coords + 0.5) * (ref / side) - 0.5
    return rope_2d_from_coords(coords[None], coords[None], d)


def rope_2d_from_coords(rows: torch.Tensor, cols: torch.Tensor, d: int):
    """2D RoPE for (B, h) row and (B, w) column positions (the grid is their Cartesian product, row-major)
    -> cos, sin of shape (B, 1, h * w, d)."""
    b, h, w = rows.shape[0], rows.shape[1], cols.shape[1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, d // 2, 2, dtype=torch.float32, device=rows.device) / (d // 2)))
    ar = (rows.float()[:, :, None] * inv_freq)[:, :, None].expand(b, h, w, -1)
    ac = (cols.float()[:, :, None] * inv_freq)[:, None].expand(b, h, w, -1)
    ang = torch.cat([ar, ac], dim=-1).reshape(b, 1, h * w, -1)
    return ang.cos().repeat_interleave(2, dim=-1), ang.sin().repeat_interleave(2, dim=-1)


def random_rope_coords(b: int, side: int, field: int, device=None, generator=None):
    """RPE-2D (arXiv:2503.18719): per sample and axis, `side` distinct integer positions of 0 .. field-1, sorted
    -> (B, side) rows, (B, side) columns. Sampling uses the even grid of build_rope_2d(side, d, field) instead."""
    def axis():
        return torch.rand(b, field, device=device, generator=generator).topk(side, dim=1).indices.sort(dim=1).values
    return axis(), axis()


def micro_cond_embedding(cond: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """(B, 8) micro-conditions in pixels -> (B, 8 * dim) sinusoids (as SDXL's size / crop conditioning)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=cond.device).float() / half)
    ang = cond.float()[:, :, None] * freqs
    return torch.cat([ang.sin(), ang.cos()], dim=-1).flatten(1)


def default_micro_cond(base: int, res: int, b: int, device=None) -> torch.Tensor:
    """The micro-conditions of a whole, uncropped image: original size max(base, res), crop = the whole original,
    target size res -> (B, 8) [orig_h, orig_w, top, left, bottom, right, target_h, target_w]."""
    o = max(base, res)
    return torch.tensor([o, o, 0, 0, o, o, res, res], dtype=torch.float32, device=device).expand(b, 8)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos.to(x.dtype) + rotate_half_pairs(x) * sin.to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.in_proj = nn.Linear(dim, hidden * 2, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(F.silu(a) * b)


class TextAttention(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, head_dim
        self.sdpa_backend = "auto"
        self.qkv = nn.Linear(dim, (heads + 2 * kv_heads) * head_dim, bias=False)
        self.out = nn.Linear(heads * head_dim, dim, bias=False)

    def forward(self, x, valid_mask, cos, sin):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split(
            [self.heads * self.head_dim, self.kv_heads * self.head_dim, self.kv_heads * self.head_dim], dim=-1)
        q = q.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = self.heads // self.kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        h = masked_attention(q, k, v, valid_mask[:, None, None, :].bool(), self.sdpa_backend)
        return self.out(h.transpose(1, 2).reshape(b, t, -1))


class TextBlock(nn.Module):
    def __init__(self, dim: int, heads: int, kv_heads: int, head_dim: int, mlp_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = TextAttention(dim, heads, kv_heads, head_dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)

    def forward(self, x, valid_mask, cos, sin):
        x = x + self.attn(self.norm1(x), valid_mask, cos, sin)
        return x + self.mlp(self.norm2(x))


class SelfAttention(nn.Module):
    """Global self-attention over the 16x16 latent tokens with 2D RoPE and QK-norm."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.logit_mult = 1.0  # sampling only: multiplies the attention logits (e.g. sqrt(log n / log n_train))

    def forward(self, x, cos, sin):
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        h = F.scaled_dot_product_attention(
            q, k, v, scale=None if self.logit_mult == 1.0 else self.logit_mult / self.head_dim ** 0.5)
        return self.out(h.transpose(1, 2).reshape(b, t, c))


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, text_dim: int):
        super().__init__()
        self.heads = heads
        self.sdpa_backend = "auto"
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(text_dim, 2 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, text, text_valid):
        b, t, c = x.shape
        n = text.shape[1]
        q = self.q(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(text).chunk(2, dim=-1)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        h = masked_attention(q, k, v, text_valid[:, None, None, :].bool(), self.sdpa_backend)
        return self.out(h.transpose(1, 2).reshape(b, t, c))


class DenoiserBlock(nn.Module):
    """DiT block with PixArt-style adaLN-single: a shared time->(6*dim) projection plus a per-block table."""

    def __init__(self, dim: int, heads: int, text_dim: int, mlp_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.cross = CrossAttention(dim, heads, text_dim)
        self.norm3 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_hidden)
        self.mod_table = nn.Parameter(torch.zeros(6, dim))

    def forward(self, x, t6, text, text_valid, cos, sin):
        shift1, scale1, gate1, shift2, scale2, gate2 = (t6 + self.mod_table[None]).to(x.dtype).unbind(1)
        h = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None] * self.attn(h, cos, sin)
        x = x + self.cross(self.norm2(x), text, text_valid)
        h = self.norm3(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(h)


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / max(half - 1, 1))
    angles = t[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


# =============================================================================
# Models
# =============================================================================
@dataclass
class TextEncoderConfig:
    vocab_size: int = 8192
    dim: int = 384
    depth: int = 6
    heads: int = 6
    kv_heads: int = 2
    head_dim: int = 64
    mlp_hidden: int = 1024
    max_tokens: int = 64
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    null_id: int = 4
    mask_id: int = 5


@dataclass
class LMEncoderConfig:
    """Frozen pretrained LM (aurora/slayer.py) + trainable refiner."""
    kind: str = "lm"
    repo: str = "SlayerLab/Slayer149"
    tok_suffix: str = "slayer149"  # captions in <shard>.tok_slayer149.npy (prepare_data.py tokenize --encoder slayer149)
    lm_dim: int = 768
    skip_last: int = 1  # use the residual stream before the last LM block
    dim: int = 768      # refiner width = denoiser text_dim
    depth: int = 2
    heads: int = 12
    kv_heads: int = 4
    head_dim: int = 64
    mlp_hidden: int = 2048
    max_tokens: int = 96
    backend: str = "slayer"  # "slayer" = aurora/slayer.py GPT, "hf" = transformers causal LM (aurora/hf_compat.py)
    experts: int = 0         # hf MoE: active experts per token (0 = the model's own)
    quant: str = ""          # hf MoE: "int8" = expert weights stored as int8 (half the memory, bf16 matmuls)
    out_dim: int = 0         # != 0: refiner output projected to this width (= denoiser text_dim), e.g. a new refiner
                             # of another size in front of a pretrained denoiser


@dataclass
class DenoiserConfig:
    latent_channels: int = 128
    latent_size: int = 16
    dim: int = 640
    depth: int = 8
    heads: int = 10
    mlp_hidden: int = 1536
    text_dim: int = 384
    latent_layout: str = "flux2_pipeline"
    rope_ref: int = 0  # build_rope_2d's ref: 0 = integer positions per grid, 16 = every grid on the 256 px field
    # RPE-2D micro-conditioning (arXiv:2503.18719): 0 = off; else the original image size in px of the training data
    # (crops and resizes are described relative to it); adds cond_mlp, whose output is added to the time embedding
    micro_cond: int = 0


class TextEncoder(nn.Module):
    """Small from-scratch text encoder: GQA transformer with 1D RoPE, SwiGLU and RMSNorm."""

    def __init__(self, cfg: TextEncoderConfig = TextEncoderConfig()):
        super().__init__()
        self.config = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([TextBlock(cfg.dim, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.mlp_hidden)
                                     for _ in range(cfg.depth)])
        self.norm = RMSNorm(cfg.dim)
        cos, sin = build_rope_1d(cfg.max_tokens, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def null_ids(self):
        from aurora.data import null_ids
        return null_ids()

    def forward(self, ids: torch.Tensor):
        valid = ids.ne(self.config.pad_id)
        h = self.token_embedding(ids)
        for block in self.blocks:
            h = block(h, valid, self.rope_cos, self.rope_sin)
        return self.norm(h), valid

    @torch.no_grad()
    def init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        for block in self.blocks:
            for lin in (block.attn.qkv, block.mlp.in_proj):
                nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
            # residual output projections start at zero (Muon still moves them from step 1)
            nn.init.zeros_(block.attn.out.weight)
            nn.init.zeros_(block.mlp.out_proj.weight)


class Denoiser(nn.Module):
    """Flow-matching DiT: velocity predictor on (B, 128, 16, 16) FLUX.2 latents, one latent pixel per token."""

    def __init__(self, cfg: DenoiserConfig = DenoiserConfig()):
        super().__init__()
        self.config = cfg
        dim = cfg.dim
        self.patch_in = nn.Linear(cfg.latent_channels, dim, bias=False)
        self.time_mlp = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        if cfg.micro_cond:
            self.cond_mlp = nn.Sequential(nn.Linear(8 * 256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([DenoiserBlock(dim, cfg.heads, cfg.text_dim, cfg.mlp_hidden)
                                     for _ in range(cfg.depth)])
        self.norm = RMSNorm(dim)
        self.patch_out = nn.Linear(dim, cfg.latent_channels, bias=False)
        cos, sin = build_rope_2d(cfg.latent_size, dim // cfg.heads, cfg.rope_ref)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def time_embedding(self, t: torch.Tensor, cond: torch.Tensor = None, res: int = 0):
        """Time embedding, plus the micro-conditions (B, 8) with micro_cond (default: a whole image at res px)."""
        t_emb = self.time_mlp(timestep_embedding(t, 256))
        if self.config.micro_cond:
            if cond is None:
                cond = default_micro_cond(self.config.micro_cond, res, t.shape[0], t.device)
            t_emb = t_emb + self.cond_mlp(micro_cond_embedding(cond).to(t_emb.dtype))
        return t_emb

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor,
                hidden_at: int = 0, cond: torch.Tensor = None):
        """hidden_at > 0: also return the token states after that many blocks (for the REPA loss)."""
        b, c, h, w = z_t.shape
        dim = self.config.dim
        x = self.patch_in(z_t.flatten(2).transpose(1, 2))
        t_emb = self.time_embedding(t, cond, 16 * h)
        t6 = self.ada(F.silu(t_emb)).view(b, 6, dim)
        hidden = None
        for i, block in enumerate(self.blocks):
            x = block(x, t6, text, text_valid, self.rope_cos, self.rope_sin)
            if i + 1 == hidden_at:
                hidden = x
        out = self.patch_out(self.norm(x)).transpose(1, 2).reshape(b, c, h, w)
        return out if not hidden_at else (out, hidden)

    @torch.no_grad()
    def init_weights(self):
        self.init_stem()
        for block in self.blocks:
            self.init_block(block)
        self.init_head()

    # split so that pipeline stages can initialise only the parts they own (train_pp.py)
    @torch.no_grad()
    def init_stem(self):
        nn.init.normal_(self.patch_in.weight, std=self.patch_in.in_features ** -0.5)
        for lin in (self.time_mlp[0], self.time_mlp[2]):
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        # adaLN-Zero (single): zero time projection + zero tables -> all gates 0, every block starts as the identity.
        # Only ONE factor of each gated product may be zero: the gated output projections (attn.out, mlp.out_proj)
        # get a normal init. Zeroing both the gate and the projection makes each one's gradient proportional to the
        # other, so self-attention and the MLP would never train (the bug behind the ~1.09 plateau, see PROBLEM.md).
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.init_cond()

    @torch.no_grad()
    def init_cond(self):
        """Micro-conditioning starts as a no-op (zero output layer), also when added to a trained model."""
        if self.config.micro_cond:
            nn.init.normal_(self.cond_mlp[0].weight, std=0.02)
            nn.init.zeros_(self.cond_mlp[0].bias)
            nn.init.zeros_(self.cond_mlp[2].weight)
            nn.init.zeros_(self.cond_mlp[2].bias)

    @staticmethod
    @torch.no_grad()
    def init_block(block):
        for lin in (block.attn.qkv, block.attn.out, block.cross.q, block.cross.kv, block.mlp.in_proj,
                    block.mlp.out_proj):
            nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
        for norm in (block.norm1, block.norm2, block.norm3, block.attn.q_norm, block.attn.k_norm):
            nn.init.ones_(norm.weight)
        nn.init.zeros_(block.mod_table)
        nn.init.zeros_(block.cross.out.weight)  # cross-attn is ungated: its zero output still gets a gradient

    @torch.no_grad()
    def init_head(self):
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.patch_out.weight)


# Denoiser sizes (the text encoder stays at 12.6M). "50m" is the released model.
SIZES = {
    "50m": DenoiserConfig(),
    "116m": DenoiserConfig(dim=768, depth=12, heads=12, mlp_hidden=1792),
    "253m": DenoiserConfig(dim=1024, depth=16, heads=16, mlp_hidden=2432),
    # wide models for pipeline-parallel training (train_pp.py): width 2048, 32 heads of 64, MLP 2.375x
    # (~58M params per block with a 768-wide text encoder)
    # depth = 8k - 1: in train_pp.py stage 0 also runs the text encoder (~1 block of work) and gets one block fewer,
    # which only balances when depth + 1 divides by the number of stages (8, 4, 2)
    "1b": DenoiserConfig(dim=2048, depth=15, heads=32, mlp_hidden=4864),
    "1.5b": DenoiserConfig(dim=2048, depth=23, heads=32, mlp_hidden=4864),
    "2b": DenoiserConfig(dim=2048, depth=31, heads=32, mlp_hidden=4864),
    # width 2560 (head dim 128): bigger GEMMs, best TFLOPS per stage on the 3090 Ti PCIe box (tests/bench_shapes.py)
    "750m": DenoiserConfig(dim=2560, depth=7, heads=20, mlp_hidden=6912),  # fits one GPU: train.py + MuLoCo
    # ~0.75B at less extreme width / depth ratios (head dim 128, SwiGLU 8/3 x width); 2560x7 above is 366
    "750m-w2048": DenoiserConfig(dim=2048, depth=11, heads=16, mlp_hidden=5504),  # ratio 186
    "750m-w1792": DenoiserConfig(dim=1792, depth=15, heads=14, mlp_hidden=4864),  # ratio 119
    "750m-w1536": DenoiserConfig(dim=1536, depth=20, heads=12, mlp_hidden=4096),  # ratio 77 (like 253m)
    "750m-w1280": DenoiserConfig(dim=1280, depth=30, heads=10, mlp_hidden=3456),  # ratio 43 (like DiT-XL)
    "1.6b": DenoiserConfig(dim=2560, depth=15, heads=20, mlp_hidden=6912),
    "2.4b": DenoiserConfig(dim=2560, depth=23, heads=20, mlp_hidden=6912),
}


# Text encoder sizes. "13m" is the released one (12.6M); "27m" (512 x 8) goes with the larger denoisers.
TE_SIZES = {
    "13m": TextEncoderConfig(),
    "27m": TextEncoderConfig(dim=512, depth=8, heads=8, kv_heads=2, head_dim=64, mlp_hidden=1408),
    "slayer149": LMEncoderConfig(),  # frozen SlayerLab/Slayer149 + 2-block refiner (~12M trainable)
    # frozen cpral/Poziomka-Baza (Polish/English 4B MoE, 16 layers x 2048, 128 experts) with 8 of its 32 active
    # experts (BLiMP: 32 -> 68.5%, 8 -> 68.3%, 4 -> 67.2%); captions [EOS-as-BOS] ids[:255] (prepare_data.py tokenize --encoder poziomka [--lang pl])
    "poziomka": LMEncoderConfig(repo="cpral/Poziomka-Baza-2026-09-05", tok_suffix="poziomka", lm_dim=2048,
                                skip_last=0, max_tokens=256, backend="hf", experts=8),
}


def make_configs(size: str, te_size: str, te_dim: int = None, te_depth: int = None, depth: int = None,
                 te_experts: int = None, te_quant: str = None):
    """te_dim / te_depth resize the trainable refiner of an LM text encoder (heads of 64, GQA, SwiGLU 8/3 x dim);
    te_experts sets the active experts of an MoE LM."""
    te_cfg = TE_SIZES[te_size]
    if te_experts:
        assert getattr(te_cfg, "backend", "") == "hf", "--te-experts only applies to transformers (MoE) LM encoders"
        te_cfg = replace(te_cfg, experts=te_experts)
    if te_quant is not None:
        assert getattr(te_cfg, "backend", "") == "hf", "--te-quant only applies to transformers (MoE) LM encoders"
        te_cfg = replace(te_cfg, quant=te_quant)
    if te_dim or te_depth:
        assert isinstance(te_cfg, LMEncoderConfig), "--te-dim / --te-depth only apply to LM text encoders"
        dim = te_dim or te_cfg.dim
        assert dim % 64 == 0, "refiner width must be a multiple of 64 (heads of 64)"
        heads = dim // 64
        kv_heads = max(d for d in range(1, max(1, heads // 3) + 1) if heads % d == 0)  # GQA, >= 3 q heads per kv
        te_cfg = replace(te_cfg, dim=dim, depth=te_depth or te_cfg.depth, heads=heads, kv_heads=kv_heads,
                         mlp_hidden=-(-dim * 8 // 3 // 128) * 128)
    den_cfg = replace(SIZES[size], text_dim=te_cfg.dim)
    return (replace(den_cfg, depth=depth) if depth else den_cfg), te_cfg


def te_config_from_dict(d):
    """Checkpoints from before te_config existed used the released text encoder."""
    return LMEncoderConfig(**d) if d.get("kind") == "lm" else TextEncoderConfig(**d)


def configs_from_ckpt(ck):
    return DenoiserConfig(**ck["den_config"]), te_config_from_dict(ck.get("te_config", {}))


def build_models(den_cfg: DenoiserConfig = DenoiserConfig(), te_cfg=TextEncoderConfig()):
    te_out = getattr(te_cfg, "out_dim", 0) or te_cfg.dim
    assert den_cfg.text_dim == te_out, f"denoiser text_dim {den_cfg.text_dim} != text encoder output {te_out}"
    if isinstance(te_cfg, LMEncoderConfig):
        from aurora.slayer import LMTextEncoder
        te = LMTextEncoder(te_cfg)
    else:
        te = TextEncoder(te_cfg)
    den = Denoiser(den_cfg)
    te.init_weights()
    den.init_weights()
    return te, den
