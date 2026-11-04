"""Loading Hugging Face causal LMs whose remote code (trust_remote_code) was written for transformers 4, under
transformers 5 — e.g. cpral/Poziomka-Baza (BailingMoeV2). Without these fixes such a model loads without errors but
computes garbage (NaN logits or chance-level scores):

  * `transformers.utils.import_utils.is_torch_fx_available` was removed -> stub it;
  * ROPE_INIT_FUNCTIONS has no "default" entry -> add transformers 4's plain RoPE, reading rope_theta and
    partial_rotary_factor also from rope_scaling / rope_parameters, where transformers 5 moves them;
  * models are built on the meta device and old remote code's non-persistent RoPE buffers are never recomputed
    (they hold uninitialised memory) -> recompute them after loading;
  * load with an explicit dtype (bf16) and attn_implementation="sdpa", and run without autocast.

`fuse_moe` replaces DeepSeek/Bailing-style `moe_infer` (a Python loop over 128 experts plus a CPU sync per layer) by
three batched matmuls over stacked expert weights, for frozen inference.
"""
import types

import torch
import torch.nn.functional as F


def patch_transformers():
    import transformers.modeling_rope_utils as ru
    import transformers.utils.import_utils as iu
    if not hasattr(iu, "is_torch_fx_available"):
        iu.is_torch_fx_available = lambda: False
    if "default" not in ru.ROPE_INIT_FUNCTIONS:
        def default_rope(config, device=None, seq_len=None):
            extra = {**(getattr(config, "rope_parameters", None) or {}), **(getattr(config, "rope_scaling", None) or {})}
            base = getattr(config, "rope_theta", None) or extra["rope_theta"]
            factor = getattr(config, "partial_rotary_factor", None) or extra.get("partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * factor)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
            return inv, 1.0
        ru.ROPE_INIT_FUNCTIONS["default"] = default_rope


def load_causal_lm(repo, device, num_layers=None, dtype=torch.bfloat16, fused_moe=False, **overrides):
    """(model, tokenizer). num_layers keeps only the first N decoder layers (the rest are never loaded: less memory);
    overrides go to the config, e.g. num_experts_per_tok=16 for fewer active MoE experts."""
    patch_transformers()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if num_layers:
        overrides["num_hidden_layers"] = num_layers
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, dtype=dtype,
                                                 attn_implementation="sdpa", **overrides).to(device)
    model.eval().requires_grad_(False)
    for mod in model.modules():
        if hasattr(mod, "rope_init_fn") and hasattr(mod, "inv_freq"):
            inv, scale = mod.rope_init_fn(mod.config, device)
            mod.inv_freq, mod.original_inv_freq, mod.attention_scaling = inv, inv, scale
    if fused_moe:
        fuse_moe(model)
    return model, tok


@torch.no_grad()
def _fused_moe_infer(self, x, topk_ids, topk_weight):
    """Same result as the reference moe_infer (checked to ~1e-6 relative in fp32; in bf16 cos 0.9999 to the loop, differences from routing flips): token copies sorted by expert, every
    expert applied at once, outputs put back in place and combined with the routing weights. torch._grouped_mm takes
    the per-expert groups as they are (no padding, no CPU sync); otherwise an (E, capacity, H) buffer and bmm."""
    n, k = topk_ids.shape
    flat = topk_ids.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=self._wg.shape[0])
    xs = x[order // k]
    if self._grouped:
        offs = counts.cumsum(0).to(torch.int32)
        h = self._act(torch._grouped_mm(xs, self._wg, offs=offs)) * torch._grouped_mm(xs, self._wu, offs=offs)
        ys = torch._grouped_mm(h, self._wd, offs=offs)
    else:
        expert = flat[order]
        slot = torch.arange(n * k, device=x.device) - (counts.cumsum(0) - counts)[expert]
        buf = x.new_zeros(self._wg.shape[0], int(counts.max()), x.shape[-1])
        buf[expert, slot] = xs
        h = self._act(torch.bmm(buf, self._wg)) * torch.bmm(buf, self._wu)
        ys = torch.bmm(h, self._wd)[expert, slot]
    # routing weight applied in the sorted order, back to (n, k) order by a gather, sum over k: all in x.dtype (the
    # fp32 (n*k, H) temporaries of the reference combine cost about as much as the expert matmuls themselves)
    ys = ys * topk_weight.reshape(-1)[order].to(ys.dtype).unsqueeze(-1)
    return ys[order.argsort()].view(n, k, -1).sum(1)


def _grouped_mm_works(device, dtype):
    if not hasattr(torch, "_grouped_mm"):
        return False
    try:
        a = torch.randn(48, 64, device=device, dtype=dtype)
        b = torch.randn(3, 64, 32, device=device, dtype=dtype)
        offs = torch.tensor([16, 16, 48], device=device, dtype=torch.int32)
        ref = torch.cat([a[:16] @ b[0], a[16:16] @ b[1], a[16:] @ b[2]])
        return torch.allclose(torch._grouped_mm(a, b, offs=offs).float(), ref.float(), atol=1e-1, rtol=5e-2)
    except Exception:  # noqa: BLE001 (unsupported GPU / dtype)
        return False


def fuse_moe(model):
    """Swap every MoE block's per-expert loop for the batched version; the per-expert modules are dropped (the stacked
    weights replace them, so memory does not double). Inference only. Returns the number of blocks changed."""
    n = 0
    for mod in model.modules():
        experts = getattr(mod, "experts", None)
        if experts is None or not hasattr(mod, "moe_infer") or not isinstance(experts, torch.nn.ModuleList):
            continue
        # (E, in, out) and contiguous: the layout both grouped_mm and bmm(x, w) want
        mod._wg = torch.stack([e.gate_proj.weight.t() for e in experts]).contiguous()
        mod._wu = torch.stack([e.up_proj.weight.t() for e in experts]).contiguous()
        mod._wd = torch.stack([e.down_proj.weight.t() for e in experts]).contiguous()
        mod._act = experts[0].act_fn
        mod._grouped = _grouped_mm_works(mod._wg.device, mod._wg.dtype)
        mod.moe_infer = types.MethodType(_fused_moe_infer, mod)
        del mod.experts
        mod.experts = list(range(mod._wg.shape[0]))  # len(self.experts) still works for any remaining code
        n += 1
    torch.cuda.empty_cache()
    return n
