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
    """Same result as the reference moe_infer (checked to ~1e-6 relative in fp32; in bf16 cos 0.9999 to the loop, the
    differences come from routing flips): token copies sorted by expert, every expert applied at once, outputs put
    back in place and combined with the routing weights. Expert matmuls, fastest first: one Triton launch for all
    experts (aurora/moe_triton.py), torch._grouped_mm (one kernel per expert before Hopper), an (E, capacity, H)
    buffer and bmm. Gate and up projections are one matmul."""
    n, k = topk_ids.shape
    flat = topk_ids.reshape(-1)
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=self._wgu.shape[0])
    if self._mode is None:  # decided on the device the weights ended up on
        self._mode = _pick_mode(x.device, x.dtype)
    if self._mode == "triton":  # gather, routing weight and scatter back happen inside the two kernels
        from aurora.moe_triton import grouped_mm, make_schedule
        sched = make_schedule(counts, n * k)
        a_idx, y_idx = (order // k).to(torch.int32), order.to(torch.int32)
        g, u = grouped_mm(x, self._wgu, counts, sched, a_idx=a_idx).chunk(2, dim=-1)
        w = topk_weight.reshape(-1)[order].to(x.dtype)
        y = grouped_mm(self._act(g) * u, self._wd, counts, sched, y_idx=y_idx, scale=w)
        return y.view(n, k, -1).sum(1)
    xs = x[order // k]
    if self._mode == "grouped":
        offs = counts.cumsum(0).to(torch.int32)
        g, u = torch._grouped_mm(xs, self._wgu, offs=offs).chunk(2, dim=-1)
        ys = torch._grouped_mm(self._act(g) * u, self._wd, offs=offs)
    else:
        expert = flat[order]
        slot = torch.arange(n * k, device=x.device) - (counts.cumsum(0) - counts)[expert]
        buf = x.new_zeros(self._wgu.shape[0], int(counts.max()), x.shape[-1])
        buf[expert, slot] = xs
        g, u = torch.bmm(buf, self._wgu).chunk(2, dim=-1)
        ys = torch.bmm(self._act(g) * u, self._wd)[expert, slot]
    # routing weight applied in the sorted order, back to (n, k) order by a gather, sum over k: all in x.dtype (the
    # fp32 (n*k, H) temporaries of the reference combine cost about as much as the expert matmuls themselves)
    ys = ys * topk_weight.reshape(-1)[order].to(ys.dtype).unsqueeze(-1)
    return ys[order.argsort()].view(n, k, -1).sum(1)


def _self_test(fn, device, dtype):
    a = torch.randn(48, 64, device=device, dtype=dtype)
    b = torch.randn(3, 64, 32, device=device, dtype=dtype)
    counts = torch.tensor([16, 0, 32], device=device)
    ref = torch.cat([a[:16] @ b[0], a[16:] @ b[2]])
    return torch.allclose(fn(a, b, counts).float(), ref.float(), atol=1e-1, rtol=5e-2)


def _pick_mode(device, dtype):
    if device.type == "cuda":
        try:
            from aurora.moe_triton import grouped_mm
            if _self_test(grouped_mm, device, dtype):
                return "triton"
        except Exception:  # noqa: BLE001 (no triton / unsupported GPU)
            pass
    if hasattr(torch, "_grouped_mm"):
        try:
            if _self_test(lambda a, b, c: torch._grouped_mm(a, b, offs=c.cumsum(0).to(torch.int32)), device, dtype):
                return "grouped"
        except Exception:  # noqa: BLE001
            pass
    return "bmm"


def fuse_moe(model):
    """Swap every MoE block's per-expert loop for the batched version; the per-expert modules are dropped (the stacked
    weights replace them, so memory does not double). Inference only. Returns the number of blocks changed."""
    n = 0
    for mod in model.modules():
        experts = getattr(mod, "experts", None)
        if experts is None or not hasattr(mod, "moe_infer") or not isinstance(experts, torch.nn.ModuleList):
            continue
        # (E, in, out) and contiguous: the layout all three matmul paths want; buffers, so .to() moves them
        wgu = torch.stack([torch.cat([e.gate_proj.weight, e.up_proj.weight]).t() for e in experts]).contiguous()
        mod.register_buffer("_wgu", wgu, persistent=False)
        mod.register_buffer("_wd", torch.stack([e.down_proj.weight.t() for e in experts]).contiguous(),
                            persistent=False)
        mod._act = experts[0].act_fn
        mod._mode = None
        mod.moe_infer = types.MethodType(_fused_moe_infer, mod)
        del mod.experts
        mod.experts = list(range(mod._wgu.shape[0]))  # len(self.experts) still works for any remaining code
        n += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return n
