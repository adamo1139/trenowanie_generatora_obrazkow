"""RPE-2D pieces (arXiv:2503.18719), on the CPU with a tiny denoiser:
  * random positions: distinct, sorted, inside the field; rope_2d_from_coords on the even grid == build_rope_2d
  * micro-conditioning: a fresh cond_mlp is a no-op; aurora.model.Denoiser == hf_pipeline.AuroraDenoiser with it on
  * train_pp.Stage: the even grid passed as mb_rope == its buffers; default micro-conditions == whole-image ones
  * ShardStore crops: windows of the larger latents with their crop boxes, the rest whole resized images

  python tests/check_rpe2d.py
"""
import os
import sys
import tempfile
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from aurora.data import ShardStore  # noqa: E402
from aurora.model import (Denoiser, DenoiserConfig, build_rope_2d, default_micro_cond,  # noqa: E402
                          random_rope_coords, rope_2d_from_coords)


def check_positions():
    g = torch.Generator().manual_seed(0)
    rows, cols = random_rope_coords(64, 8, 64, generator=g)
    for x in (rows, cols):
        assert x.shape == (64, 8) and x.min() >= 0 and x.max() < 64
        assert (x[:, 1:] > x[:, :-1]).all(), "positions must be distinct and sorted"
    assert not torch.equal(rows, cols), "rows and columns are sampled independently"
    for side, ref in ((8, 64), (16, 0), (6, 64)):
        even = torch.arange(side).float()
        if ref:
            even = (even + 0.5) * (ref / side) - 0.5
        a, b = rope_2d_from_coords(even[None].expand(3, -1), even[None].expand(3, -1), 128), build_rope_2d(side, 128, ref)
        assert torch.allclose(a[0], b[0].expand_as(a[0])) and torch.allclose(a[1], b[1].expand_as(a[1]))
    print("positions: ok")


def tiny(micro_cond=0, side=8):
    return DenoiserConfig(latent_channels=16, latent_size=side, dim=64, depth=2, heads=2, mlp_hidden=96, text_dim=32,
                          rope_ref=64, micro_cond=micro_cond)


def inputs(cfg, b=3):
    g = torch.Generator().manual_seed(1)
    return (torch.randn(b, cfg.latent_channels, cfg.latent_size, cfg.latent_size, generator=g), torch.rand(b, generator=g),
            torch.randn(b, 5, cfg.text_dim, generator=g), torch.ones(b, 5, dtype=torch.bool))


def check_micro_cond():
    torch.manual_seed(0)
    off = Denoiser(tiny(0))
    off.init_weights()
    on = Denoiser(tiny(256))
    on.init_weights()
    missing = on.load_state_dict(off.state_dict(), strict=False).missing_keys
    assert all(k.startswith("cond_mlp.") for k in missing) and missing
    on.init_cond()
    with torch.no_grad():  # a trained-looking model: every gate open
        for m in (off, on):
            m.ada.weight.normal_(std=0.05)
            m.ada.bias.normal_()
            for blk in m.blocks:
                blk.cross.out.weight.normal_(std=0.1)
            m.patch_out.weight.normal_(std=0.1)
        on.load_state_dict({**on.state_dict(), **off.state_dict()})
    x = inputs(on.config)
    crop = torch.tensor([[256, 256, 64, 32, 192, 160, 128, 128]] * 3, dtype=torch.float32)
    with torch.no_grad():
        ref = off(*x)
        assert ref.abs().mean() > 1e-2, "the test model must not output zeros"
        assert torch.equal(on(*x), ref), "a fresh cond_mlp must not change the output"
        assert torch.equal(on(*x, cond=crop), ref)
        on.cond_mlp[2].weight.normal_(std=0.05)
        assert not torch.allclose(on(*x, cond=crop), on(*x)), "a trained cond_mlp must see the crop box"
        assert torch.equal(on(*x), on(*x, cond=default_micro_cond(256, 128, 3)))
    print("micro-conditioning: ok")
    return on


def check_pipeline(den):
    from aurora.hf_pipeline import AuroraDenoiser
    c = den.config
    hf = AuroraDenoiser(latent_channels=c.latent_channels, latent_size=c.latent_size, dim=c.dim, depth=c.depth,
                        heads=c.heads, mlp_hidden=c.mlp_hidden, text_dim=c.text_dim, rope_ref=c.rope_ref,
                        micro_cond=c.micro_cond)
    hf.load_state_dict(den.state_dict(), strict=True)
    x = inputs(c)
    with torch.no_grad():
        assert torch.allclose(hf(*x), den(*x), atol=1e-5), "pipeline denoiser differs from aurora.model"
        for m in hf.modules():
            if type(m).__name__ == "SelfAttention":
                m.logit_mult = 1.5
        assert not torch.allclose(hf(*x), den(*x), atol=1e-5), "attn_scale has no effect"
        for m in hf.modules():
            if type(m).__name__ == "SelfAttention":
                m.logit_mult = 1.0
        # another grid with another rope_ref, set on the buffers as the pipeline's __call__ does
        side, ref = 2 * c.latent_size, 4
        big = Denoiser(replace(c, latent_size=side, rope_ref=ref))
        big.load_state_dict(den.state_dict())
        hf.rope_cos, hf.rope_sin = build_rope_2d(side, c.dim // c.heads, ref)
        xb = inputs(big.config)
        assert torch.allclose(hf(*xb), big(*xb), atol=1e-5), "pipeline ignored its rope_ref override at another size"
    print("pipeline parity: ok")


def check_stage(den):
    import train_pp
    stage = train_pp.Stage(den, den.config, 0, den.config.depth, True, True, te=None)
    z, t, text, valid = inputs(den.config)
    stage.text_cache = (text, valid)
    with torch.no_grad():
        ref = stage(z, t, None)
        assert torch.allclose(ref, den(z, t, text, valid), atol=1e-5), "Stage differs from Denoiser"
        side, ref_f = den.config.latent_size, den.config.rope_ref
        even = ((torch.arange(side).float() + 0.5) * (ref_f / side) - 0.5)[None].expand(3, -1)
        stage.mb_rope = rope_2d_from_coords(even, even, den.config.dim // den.config.heads)
        assert torch.allclose(stage(z, t, None), ref, atol=1e-5), "per-sample RoPE on the even grid != buffers"
        stage.mb_rope = rope_2d_from_coords(*random_rope_coords(3, side, ref_f), den.config.dim // den.config.heads)
        assert not torch.allclose(stage(z, t, None), ref, atol=1e-4)
        stage.mb_rope = None
        stage.mb_cond = default_micro_cond(den.config.micro_cond, 16 * side, 3)
        assert torch.allclose(stage(z, t, None), ref, atol=1e-6)
    print("train_pp.Stage: ok")


def check_crops():
    with tempfile.TemporaryDirectory() as d:
        n_img = 64
        big = np.random.default_rng(0).standard_normal((n_img, 4, 16, 16)).astype(np.float16)
        small = np.random.default_rng(1).standard_normal((n_img, 4, 8, 8)).astype(np.float16)
        os.makedirs(os.path.join(d, "128px"))
        np.save(os.path.join(d, "s0.lat.npy"), big)
        np.save(os.path.join(d, "128px", "s0.lat128.npy"), small)
        np.save(os.path.join(d, "s0.off.npy"), np.arange(n_img + 1, dtype=np.int64))
        np.save(os.path.join(d, "s0.tok.npy"), np.ones((n_img, 8), dtype=np.uint16))
        store = ShardStore(d, ["s0"], lat_kind="lat128", lat_root=os.path.join(d, "128px"), crops=[("lat", "")],
                           crop_prob=0.5, cond_base=256)
        idx = np.arange(n_img)
        lat, ids, cond = store.fetch(idx, np.random.default_rng(2))
        assert lat.shape == (n_img, 4, 8, 8) and cond.shape == (n_img, 8)
        crops = 0
        for j in idx:
            top, left, bottom, right = (cond[j, 2:6] / 16).astype(int)
            assert list(cond[j, [0, 1, 6, 7]]) == [256, 256, 128, 128]
            if (top, left, bottom, right) == (0, 0, 16, 16):
                assert np.array_equal(lat[j], small[j]), "a whole image must be the resized latent"
            else:
                crops += 1
                assert bottom - top == 8 and right - left == 8
                assert np.array_equal(lat[j], big[j, :, top:bottom, left:right]), "crop window / box mismatch"
        assert 16 < crops < 48, crops
        lat, ids, cond = store.fetch(idx, None)  # validation: no crops
        assert np.array_equal(lat, small) and (cond == default_micro_cond(256, 128, 1).numpy()).all()
    print(f"crops: ok ({crops} of {n_img} cropped)")


if __name__ == "__main__":
    check_positions()
    den = check_micro_cond()
    check_pipeline(den)
    check_stage(den)
    check_crops()
