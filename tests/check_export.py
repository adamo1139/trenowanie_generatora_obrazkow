"""An exported checkpoint loads with diffusers and matches the training modules exactly.

  python tests/check_export.py <checkpoint.pt> <export_dir>

Checks: the pipeline's tokenisation equals prepare_data.py's, its text encoder and denoiser outputs equal the training
modules' (EMA weights) for real prompts and for the CFG null prompt, and one image is generated.
"""
import sys

import numpy as np
import torch
from diffusers import DiffusionPipeline

sys.path.insert(0, ".")
from aurora.model import build_models, configs_from_ckpt

ckpt, export_dir = sys.argv[1], sys.argv[2]
dev = "cuda"
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
te, den = build_models(*configs_from_ckpt(ck))
with torch.no_grad():
    for p, e in zip(list(te.parameters()) + list(den.parameters()), ck["ema"]):
        p.copy_(e)
te.to(dev).eval(), den.to(dev).eval()

pipe = DiffusionPipeline.from_pretrained(export_dir, custom_pipeline=export_dir, trust_remote_code=True).to(dev)
prompts = ["a red fox sitting in a snowy forest at sunrise, photograph", "a small robot reading a book"]
ids = pipe._tokenize(prompts).to(dev)

# tokenisation must match the training data (prepare_data.py)
if getattr(te.config, "kind", "") == "lm":
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    from prepare_data import ENCODERS
    enc = ENCODERS[te.config.tok_suffix]
    tok = Tokenizer.from_file(hf_hub_download(te.config.repo, "tokenizer.json"))
    for row, text in zip(ids.cpu().numpy(), prompts):
        ref = np.full(enc["width"], enc["pad"], dtype=np.int64)
        toks = tok.encode(" ".join(text.split()), add_special_tokens=False).ids[: enc["width"] - 1]
        ref[0], ref[1:1 + len(toks)] = enc["bos"], toks
        assert (row == ref).all(), (row[:12], ref[:12])
null = pipe._null_ids(1).to(dev)
assert (null[0].cpu().numpy() == te.null_ids()).all(), "null prompt differs from training"

with torch.no_grad():
    for name, x in (("prompts", ids), ("null", null)):
        a, av = te(x)
        b, bv = pipe.text_encoder(x)
        print(f"text encoder ({name}): max |diff| {(a - b).abs().max().item():.2e}")
        assert torch.equal(av, bv) and (a - b).abs().max().item() < 1e-4
    z = torch.randn(2, 128, 16, 16, device=dev)
    t = torch.rand(2, device=dev)
    text, valid = te(ids)
    va, vb = den(z, t, text, valid), pipe.transformer(z, t, text, valid)
    print(f"denoiser: max |diff| {(va - vb).abs().max().item():.2e}")
    assert (va - vb).abs().max().item() < 1e-4

img = pipe(prompts[0], num_inference_steps=4, guidance_scale=3.0,
           generator=torch.Generator(dev).manual_seed(0)).images[0]
print("generated", img.size)
print("OK")
