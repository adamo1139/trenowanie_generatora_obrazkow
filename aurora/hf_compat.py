"""Loading Hugging Face causal LMs whose remote code (trust_remote_code) was written for transformers 4, under
transformers 5 — e.g. cpral/Poziomka-Baza (BailingMoeV2). Without these fixes such a model loads without errors but
computes garbage (NaN logits or chance-level scores):

  * `transformers.utils.import_utils.is_torch_fx_available` was removed -> stub it;
  * ROPE_INIT_FUNCTIONS has no "default" entry -> add transformers 4's plain RoPE, reading rope_theta and
    partial_rotary_factor also from rope_scaling / rope_parameters, where transformers 5 moves them;
  * models are built on the meta device and old remote code's non-persistent RoPE buffers are never recomputed
    (they hold uninitialised memory) -> recompute them after loading;
  * load with an explicit dtype (bf16) and attn_implementation="sdpa", and run without autocast.
"""
import torch


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


def load_causal_lm(repo, device, num_layers=None, dtype=torch.bfloat16, **overrides):
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
    return model, tok
