"""Export a training checkpoint (EMA weights) into the AuroraIMG2 diffusers repo layout.

  python export.py runs/aurora/latest.pt exports/aurora-step247805
  python export.py runs/aurora/latest.pt exports/raw --no-ema      # live weights instead of EMA

The FLUX.2 small decoder and configs come from the reference repo; pipeline.py is aurora/hf_pipeline.py (the released
pipeline + frozen-LM text encoder support). For --te-size slayer149 models the used LM layers and its tokenizer are
packed into text_encoder/ and tokenizer/, and likewise Poziomka for --te-size poziomka models (bf16, int8 experts, as
trained; ~4.4 GB), so the export is self-contained (needs a diffusers with AutoencoderKLFlux2):

  DiffusionPipeline.from_pretrained(out_dir, custom_pipeline=out_dir, trust_remote_code=True)
"""
import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file

from dataclasses import asdict

from aurora.model import build_models, configs_from_ckpt
from aurora.slayer import LM_NULL, LM_PAD
from prepare_data import ENCODERS


def export_lm_encoder(te, te_cfg, out):
    """text_encoder/ = refiner (trained) + the used LM blocks (frozen) as one AuroraLMTextEncoder; tokenizer/ = the LM's."""
    lm, lm_cfg = te.lm, te.lm.cfg
    assert lm_cfg.pos == "rope" and lm_cfg.ffn == "swiglu" and lm_cfg.norm == "rmsnorm", "unsupported LM variant"
    n = te.n_blocks
    sd = {k: v.contiguous() for k, v in te.state_dict().items()}
    for k, v in lm.state_dict().items():
        if k == "tok.weight" or (k.startswith("blocks.") and int(k.split(".")[1]) < n):
            sd["lm." + k] = v.detach().float().contiguous()
    save_file(sd, os.path.join(out, "text_encoder", "diffusion_pytorch_model.safetensors"))
    cfg = {"_class_name": "AuroraLMTextEncoder", "_diffusers_version": "0.42.1", "kind": "lm",
           "vocab_size": lm_cfg.vocab, "lm_blocks": n, "lm_dim": lm_cfg.n_embd, "lm_heads": lm_cfg.n_head,
           "lm_ffn_mult": lm_cfg.ffn_mult, "lm_norm_eps": lm_cfg.norm_eps, "lm_rope_theta": lm_cfg.rope_theta,
           "lm_value_residual": bool(lm_cfg.value_residual), "lm_qk_norm": bool(lm_cfg.qk_norm),
           "dim": te_cfg.dim, "depth": te_cfg.depth, "heads": te_cfg.heads, "kv_heads": te_cfg.kv_heads,
           "head_dim": te_cfg.head_dim, "mlp_hidden": te_cfg.mlp_hidden, "max_tokens": te_cfg.max_tokens,
           "bos_id": int(ENCODERS[te_cfg.tok_suffix]["bos"]), "pad_id": LM_PAD, "null_id": LM_NULL}
    with open(os.path.join(out, "text_encoder", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    from huggingface_hub import hf_hub_download
    tok_dir = os.path.join(out, "tokenizer")
    shutil.copy(hf_hub_download(te_cfg.repo, "tokenizer.json"), os.path.join(tok_dir, "tokenizer.json"))
    special = {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>"}
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 1024, **special}, f, indent=2)
    with open(os.path.join(tok_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special, f, indent=2)
    with open(os.path.join(out, "model_index.json")) as f:
        index = json.load(f)
    index["text_encoder"] = ["pipeline", "AuroraLMTextEncoder"]
    with open(os.path.join(out, "model_index.json"), "w") as f:
        json.dump(index, f, indent=2)


def export_poziomka_encoder(te, te_cfg, out):
    """text_encoder/ = refiner (trained) + Poziomka (frozen, bf16, experts int8 exactly as in training) as one
    AuroraLMTextEncoder with lm_arch="poziomka"; tokenizer/ = Poziomka's."""
    lm = te.lm.model  # BailingMoeV2Model with aurora/hf_compat.fuse_moe applied
    lc = te.lm.config
    assert getattr(te_cfg, "quant", "") == "int8", "the pipeline's Poziomka trunk stores the experts as int8"
    sd = {k: v.detach().contiguous() for k, v in te.state_dict().items()}
    for name, t in list(lm.named_parameters()) + list(lm.named_buffers()):
        if "inv_freq" not in name:
            sd["lm." + name] = t.detach().cpu().contiguous()
    save_file(sd, os.path.join(out, "text_encoder", "diffusion_pytorch_model.safetensors"))
    cfg = {"_class_name": "AuroraLMTextEncoder", "_diffusers_version": "0.42.1", "kind": "lm", "lm_arch": "poziomka",
           "vocab_size": lc.vocab_size, "lm_blocks": len(lm.layers), "lm_dim": lc.hidden_size,
           "lm_heads": lc.num_attention_heads, "lm_kv_heads": lc.num_key_value_heads, "lm_head_dim": lc.head_dim,
           "lm_rotary_dim": int(lc.head_dim * (getattr(lc, "partial_rotary_factor", None)
                                               or (lc.rope_scaling or {}).get("partial_rotary_factor", 1.0))),
           "lm_rope_theta": float(getattr(lc, "rope_theta", None) or (lc.rope_scaling or {})["rope_theta"]),
           "lm_norm_eps": lc.rms_norm_eps, "lm_dense_layers": lc.first_k_dense_replace,
           "lm_intermediate": lc.intermediate_size, "lm_moe_intermediate": lc.moe_intermediate_size,
           "lm_experts": lc.num_experts, "lm_top_k": lc.num_experts_per_tok, "lm_n_group": lc.n_group,
           "lm_topk_group": lc.topk_group, "lm_routed_scaling": lc.routed_scaling_factor,
           "lm_shared_intermediate": lc.moe_intermediate_size * lc.num_shared_experts,
           "dim": te_cfg.dim, "depth": te_cfg.depth, "heads": te_cfg.heads, "kv_heads": te_cfg.kv_heads,
           "head_dim": te_cfg.head_dim, "mlp_hidden": te_cfg.mlp_hidden, "max_tokens": te_cfg.max_tokens,
           "out_dim": te_cfg.out_dim, "bos_id": int(ENCODERS[te_cfg.tok_suffix]["bos"]), "pad_id": LM_PAD,
           "null_id": LM_NULL}
    with open(os.path.join(out, "text_encoder", "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    from huggingface_hub import hf_hub_download
    tok_dir = os.path.join(out, "tokenizer")
    shutil.copy(hf_hub_download(te_cfg.repo, "tokenizer.json"), os.path.join(tok_dir, "tokenizer.json"))
    special = {"bos_token": "<s>", "eos_token": "<|im_end|>", "pad_token": "</s>", "unk_token": "<unk>"}
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": 8192, **special}, f, indent=2)
    with open(os.path.join(tok_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special, f, indent=2)
    with open(os.path.join(out, "model_index.json")) as f:
        index = json.load(f)
    index["text_encoder"] = ["pipeline", "AuroraLMTextEncoder"]
    with open(os.path.join(out, "model_index.json"), "w") as f:
        json.dump(index, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--ref", default="AuroraIMG2-50M-Preview")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    den_cfg, te_cfg = configs_from_ckpt(ck)
    is_lm = getattr(te_cfg, "kind", "") == "lm"
    te, den = build_models(den_cfg, te_cfg)
    te.load_state_dict(ck["te"]), den.load_state_dict(ck["den"])
    if not args.no_ema:
        params = list(te.parameters()) + list(den.parameters())
        assert len(params) == len(ck["ema"])
        with torch.no_grad():
            for p_, e in zip(params, ck["ema"]):
                p_.copy_(e)

    shutil.copytree(args.ref, args.out, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(".cache", ".git*", "__pycache__", "samples", "*.safetensors", "README.md"))
    shutil.copy(os.path.join(args.ref, "vae", "diffusion_pytorch_model.safetensors"),
                os.path.join(args.out, "vae", "diffusion_pytorch_model.safetensors"))
    here = os.path.dirname(os.path.abspath(__file__))
    for dst in ("", "text_encoder", "transformer"):
        shutil.copy(os.path.join(here, "aurora", "hf_pipeline.py"), os.path.join(args.out, dst, "pipeline.py"))
    den_sd = {k: v.contiguous() for k, v in den.state_dict().items()}
    save_file(den_sd, os.path.join(args.out, "transformer", "diffusion_pytorch_model.safetensors"))
    if is_lm and getattr(te_cfg, "backend", "") == "hf":
        export_poziomka_encoder(te, te_cfg, args.out)
    elif is_lm:
        export_lm_encoder(te, te_cfg, args.out)
    else:
        te_sd = {k: v.contiguous() for k, v in te.state_dict().items()}
        save_file(te_sd, os.path.join(args.out, "text_encoder", "diffusion_pytorch_model.safetensors"))
    for sub, model_cfg in (("transformer", den_cfg),) + ((("text_encoder", te_cfg),) if not is_lm else ()):
        cfg_path = os.path.join(args.out, sub, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg.update(asdict(model_cfg))  # our sizes, the reference's _class_name / _diffusers_version
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
    kind = "live" if args.no_ema else ("WSM-merged" if "wsm" in ck else "EMA")
    print(f"exported step {ck['step']} ({kind} weights) to {args.out}")


if __name__ == "__main__":
    main()
