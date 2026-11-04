"""Frozen pretrained LM as the text encoder (default: SlayerLab/Slayer149), plus a small trainable refiner.

The GPT definition below is the inference architecture from SlayerLab/Slayer149 (`modeling_gollem.py`, Apache-2.0,
see that repo's README and NOTICE), with an added `hidden()` that returns the residual stream after the first n blocks.

The LM is frozen and deliberately NOT registered as a submodule: it is not in parameters() / state_dict(), so the
optimizer, DDP, DiLoCo syncs, EMA and checkpoints only see the refiner. It is moved along with the module by `_apply`.

Captions are stored as uint16 rows (prepare_data.py tokenize --encoder slayer149): [<|endoftext|>] ids... PAD...,
PAD = 65535. The CFG null prompt is the row [NULL, PAD, ...] (NULL = 65534) and maps to one learned vector.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.model import RMSNorm as _RMSNorm, TextBlock, build_rope_1d

LM_PAD, LM_NULL = 65535, 65534


# ---------------------------------------------------------------------------------------------------------------------
# GoLLeM / Slayer GPT (inference architecture, Apache-2.0)
# ---------------------------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def make_norm(d, cfg):
    return RMSNorm(d, cfg.norm_eps) if cfg.norm == "rmsnorm" else nn.LayerNorm(d)


def apply_rope(x, base=100000.0):
    """Parameter-free RoPE on [B, H, T, D], interleaved convention (even/odd channel pairs)."""
    _, _, T, dim = x.shape
    pos = torch.arange(T, device=x.device, dtype=torch.float32)
    freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    ang = torch.outer(pos, freq)
    cos, sin = ang.cos().to(x.dtype)[None, None], ang.sin().to(x.dtype)[None, None]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up = nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d, nh, block, cfg, is_first=False):
        super().__init__()
        self.ln1 = make_norm(d, cfg)
        self.ln2 = make_norm(d, cfg)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        if cfg.ffn == "swiglu":
            self.mlp = SwiGLU(d, int(round(cfg.ffn_mult * d)))
        else:
            self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.nh, self.d, self.cfg, self.is_first = nh, d, cfg, is_first
        if cfg.value_residual and not is_first:
            self.vr_lambda = nn.Parameter(torch.zeros(1))
        if cfg.qk_norm:
            hd = d // nh
            self.q_norm = RMSNorm(hd, cfg.norm_eps)
            self.k_norm = RMSNorm(hd, cfg.norm_eps)

    def forward(self, x, v0=None):
        B, T, D = x.size()
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(self.d, dim=2)
        hd = D // self.nh
        q = q.view(B, T, self.nh, hd).transpose(1, 2)
        k = k.view(B, T, self.nh, hd).transpose(1, 2)
        v = v.view(B, T, self.nh, hd).transpose(1, 2)
        if self.cfg.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if self.cfg.pos == "rope":
            q, k = apply_rope(q, self.cfg.rope_theta), apply_rope(k, self.cfg.rope_theta)
        if self.cfg.value_residual:
            if self.is_first:
                v0 = v
            else:
                v = v + self.vr_lambda * v0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x, v0


class GPT(nn.Module):
    def __init__(self, vocab, n_layer, n_embd, n_head, block, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab, n_embd)
        self.use_rope = cfg.pos == "rope"
        if not self.use_rope:
            self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block, cfg, is_first=(i == 0)) for i in range(n_layer)])
        self.lnf = make_norm(n_embd, cfg)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied

    def hidden(self, idx, n_blocks):
        """Residual stream after the first n_blocks blocks (no final norm, no head)."""
        x = self.tok(idx)
        if not self.use_rope:
            x = x + self.pos(torch.arange(idx.shape[1], device=idx.device))[None]
        v0 = None
        for b in self.blocks[:n_blocks]:
            x, v0 = b(x, v0)
        return x


def load_lm(repo: str):
    """GPT + tokenizer path from a Hub repo or a local dir with config.json, model.safetensors, tokenizer.json."""
    from safetensors.torch import load_model
    if os.path.isdir(repo):
        path = repo
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, allow_patterns=["config.json", "model.safetensors", "tokenizer.json"])
    with open(os.path.join(path, "config.json")) as f:
        cfg = SimpleNamespace(**json.load(f))
    model = GPT(cfg.vocab, cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.block, cfg)
    load_model(model, os.path.join(path, "model.safetensors"), strict=True)
    model.requires_grad_(False).eval()
    return model, os.path.join(path, "tokenizer.json")


# ---------------------------------------------------------------------------------------------------------------------
# Frozen-LM text encoder
# ---------------------------------------------------------------------------------------------------------------------
class LMTextEncoder(nn.Module):
    """Frozen LM hidden states (causal) -> trainable refiner (bidirectional TextBlocks) -> denoiser text tokens."""

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        if getattr(cfg, "backend", "slayer") == "hf":
            lm = load_hf_lm(cfg)
            self.n_blocks = len(lm.model.layers)
        else:
            lm, _ = load_lm(cfg.repo)
            self.n_blocks = len(lm.blocks) - cfg.skip_last
        self.__dict__["lm"] = lm  # frozen, unregistered (see module docstring)
        self.in_norm = _RMSNorm(cfg.lm_dim)
        self.null = nn.Parameter(torch.zeros(1, 1, cfg.lm_dim))  # learned CFG null prompt
        self.in_proj = nn.Linear(cfg.lm_dim, cfg.dim, bias=False) if cfg.dim != cfg.lm_dim else nn.Identity()
        self.blocks = nn.ModuleList([TextBlock(cfg.dim, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.mlp_hidden)
                                     for _ in range(cfg.depth)])
        self.norm = _RMSNorm(cfg.dim)
        out = getattr(cfg, "out_dim", 0)
        self.out_proj = nn.Linear(cfg.dim, out, bias=False) if out and out != cfg.dim else nn.Identity()
        cos, sin = build_rope_1d(cfg.max_tokens, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.lm._apply(fn)
        return self

    def train(self, mode=True):
        super().train(mode)
        self.lm.eval()
        return self

    @torch.no_grad()
    def init_weights(self):
        for block in self.blocks:
            for lin in (block.attn.qkv, block.mlp.in_proj):
                nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
            nn.init.zeros_(block.attn.out.weight)  # ungated residual branches: zero output still gets a gradient
            nn.init.zeros_(block.mlp.out_proj.weight)
        for lin in (self.in_proj, self.out_proj):
            if isinstance(lin, nn.Linear):
                nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)

    def null_ids(self) -> np.ndarray:
        row = np.full(self.config.max_tokens, LM_PAD, dtype=np.int64)
        row[0] = LM_NULL
        return row

    @torch.no_grad()
    def lm_states(self, ids: torch.Tensor) -> torch.Tensor:
        """Frozen LM hidden states (no gradient); train_pp.py computes these on other GPUs and passes them in."""
        # pads / null marker -> any real id; attention is causal and padding is on the right, so real tokens never
        # see them, and their positions are masked out downstream
        ids = ids.masked_fill(ids.eq(LM_PAD) | ids.eq(LM_NULL), 0)
        if getattr(self.config, "backend", "slayer") == "hf":
            return _hf_states(self.lm, ids)
        return self.lm.hidden(ids, self.n_blocks)

    def forward(self, ids: torch.Tensor, lm_states: torch.Tensor = None):
        valid = ids.ne(LM_PAD)
        is_null = ids[:, 0].eq(LM_NULL)
        h = self.lm_states(ids) if lm_states is None else lm_states
        h = self.in_norm(h)
        h = torch.where(is_null[:, None, None], self.null.to(h.dtype).expand_as(h), h)
        h = self.in_proj(h)
        T = ids.shape[1]  # rows may be trimmed below max_tokens (trim_ids)
        cos, sin = self.rope_cos[..., :T, :], self.rope_sin[..., :T, :]
        for block in self.blocks:
            h = block(h, valid, cos, sin)
        return self.out_proj(self.norm(h)), valid


def load_hf_lm(cfg):
    """Transformers causal LM (e.g. cpral/Poziomka-Baza, BailingMoeV2) through aurora/hf_compat.py: first
    num_layers - skip_last decoder layers only, cfg.experts active MoE experts (0 = the model's own), batched experts.
    Loaded straight onto the current GPU (8 ranks x a few GB of fp32 staging in RAM would not fit)."""
    import json as _json

    from huggingface_hub import hf_hub_download

    from aurora.hf_compat import load_causal_lm
    total = _json.load(open(hf_hub_download(cfg.repo, "config.json")))["num_hidden_layers"]
    overrides = {"num_experts_per_tok": cfg.experts} if cfg.experts else {}
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    lm, _ = load_causal_lm(cfg.repo, device, num_layers=total - cfg.skip_last, fused_moe=True,
                           quant=getattr(cfg, "quant", ""), **overrides)
    return lm


@torch.compiler.disable  # remote code + data-dependent MoE routing: stays eager when the refiner is compiled
def _hf_states(lm, ids):
    with torch.autocast("cuda", enabled=False):  # bf16 weights already; autocast on top gives NaN in BailingMoeV2
        return lm.model(input_ids=ids, use_cache=False).last_hidden_state


def trim_ids(ids: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    """Drop the all-padding tail columns of a batch of caption rows (rounded up to `multiple`, so a compiled model
    sees few distinct shapes). Rows are right-padded with LM_PAD; the null prompt keeps its first column."""
    n = int(ids.ne(LM_PAD).sum(1).max()) if ids.numel() else 1
    return ids[:, : min(ids.shape[1], max(multiple, -(-n // multiple) * multiple))]


def length_sorted(lat: torch.Tensor, ids: torch.Tensor, parts: int, generator=None):
    """Split one GPU's samples of a step into `parts` micro-batches of similar caption length (sorted by length, then
    shuffled inside each part), each trimmed with trim_ids: the frozen LM and the text attention then run on ~the real
    caption lengths instead of max_tokens. Gradient accumulation sums over all of them, so the step is unchanged."""
    order = ids.ne(LM_PAD).sum(1).argsort(stable=True)
    out = []
    for part in order.chunk(parts):
        part = part[torch.randperm(len(part), generator=generator)]
        out.append((lat[part], trim_ids(ids[part])))
    return out

