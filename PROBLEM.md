# PROBLEM: from-scratch training plateaus at val/loss ~1.09, the released AuroraIMG2-50M reaches 0.693

## TL;DR, likely root cause (found 2026-10-30, not yet fixed)

**In every denoiser block, self-attention and the MLP are dead from step 0 and never train.**
`Denoiser.init_weights()` in `aurora/model.py` zero-initialises both:
- the adaLN gates: `ada.weight`, `ada.bias` and `mod_table` are all zero, so `gate1 = gate2 = 0`;
- the output projections behind those gates: `attn.out.weight` and `mlp.out_proj.weight` are zero.

The block computes `x + gate * out(...)`. The gradient w.r.t. `gate` is proportional to `out(...)`, which is 0. The gradient
w.r.t. `out.weight` is proportional to `gate`, which is 0. Both stay exactly zero forever. Muon turns a zero gradient into a
zero update and AdamW leaves zeros at zero. Shift and scale only feed the dead branch, so they get no gradient either.

Only `cross.out` (ungated, zero-init but with nonzero input) and `patch_in`/`patch_out` learn. The trained model is a stack of
cross-attention layers with no spatial mixing and no MLP. That explains:
- the same floor (~1.05–1.10) for 50M / 116M / 253M, AdamW / Muon, batch 256 to 8192, DDP / DiLoCo, 2k to 247k steps;
- the unusually strong reliance on the text: `text_gain` 0.155 for ours vs 0.010 for Aurora;
- fine-tuning the released Aurora weights with our loop working fine (`val/loss` stayed ~0.69), because its weights aren't zero.

Verified with a 50-step local run (50m, Muon + AdamW): after training, block 3 still has max |w| = 0.0 for `attn.out.weight`,
`mlp.out_proj.weight`, the gate rows of `mod_table`, and `ada.weight`. Only `cross.out.weight` moved (8.6e-3).

**Suggested fix:** zero-initialise only one factor of each product. Standard DiT / PixArt adaLN-Zero zeroes the modulation
(`ada`, `mod_table`) and keeps normal init on `attn.out` / `mlp.out_proj`, or the other way round. Then re-run
`tests/check_model.py` and add a test asserting that every parameter receives a nonzero gradient after one step.
The text encoder is not affected (its residual branches are ungated, so its zero-init `out` layers get gradients).

## Goal

Replicate https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview (text encoder 12.6M + DiT 50.4M, rectified flow on
FLUX.2 latents, 256×256) on 8× RTX 3090 Ti (`adamo-rigga`: 6 cards on PCIe 3.0 x4, 2 on x8, modded driver with P2P, run with
`NCCL_P2P_LEVEL=SYS`; fp32-accumulate unlocked, so the bf16 peak is 153 TFLOPS). Then scale up (116m / 253m presets).

## What is verified correct

- **Forward pass / architecture:** the released weights load strict into `aurora/model.py` and give bit-identical outputs to
  the reference `pipeline.py` (`tests/check_model.py`).
- **Loss convention:** `z_t = (1-t)·x0 + t·eps`, target `eps − x0`, t ~ logit-normal. The released model scores 0.693 under our
  eval, so it was trained with the same convention.
- **Data / tokenisation / captions:** real descriptive captions (~2.7 per image). The released model reaches 0.693 through our
  loader. The per-shard eval shows no train/val contamination: Aurora has the same loss on its train and val shards.
- **Training loop:** fine-tuning from the released weights with `train.py` keeps `val/loss` ~0.69.

## Experiment history (val = EMA `val/loss` on 4 held-out shards, 4096 samples; Aurora = 0.693)

| run | setup | result |
|---|---|---|
| p10 | 116m, AdamW 2.8e-4, batch 2048, 10% data, 3 098 steps | 1.119 (old 3-shard val), noise-like samples |
| full_253m | 253m + TE 27m, Muon, batch 2048 | 1.079 at 3.2M samples |
| full_253m_b8192 | same, batch 8192 | 1.063 at 15.9M samples, flat |
| lr sweep 50m | AdamW 1e-5…1e-2, 400 steps @ 2048 | best 1.245 (1e-4); ≥1e-3 diverges |
| Muon sweep 50m | 5e-3 / 1e-2 / 2e-2 / 4e-2 | 1.158 / **1.147** / 1.155 / 1.209 |
| aurora50m_b256 | exact recipe, DDP, batch 256 | 1.152 at step 2000 (stopped, 929 img/s) |
| MuLoCo H=100 | DiLoCo, Muon 1e-2, outer 0.7/0.9 | ~1.14 then degrading |
| MuLoCo variants | from step-10000 ckpt | best H=16, plain averaging, Muon 3.5e-3: 1.135 @ 12k |
| **muloco50m_final** | 50m, H=16, avg, Muon 3.5e-3, 247 805 steps (= Aurora's budget) | **1.111 @ 12.4k → 1.0876 @ 247.8k** |

All of these share the dead-branch init, so none of the optimizer / batch conclusions can be trusted for a working model.
In particular the Muon-vs-AdamW sweep, the critical-batch reasoning and the MuLoCo settings should be re-checked after the fix.

## Infrastructure that works and is worth keeping

- `train.py`: DDP or `--diloco H` (MuLoCo: local unsharded Muon, averaged every H steps, bf16 sync, `--outer-lr/--outer-momentum`),
  constant lr + WSM (`merge.py`), EMA, wandb (auto when logged in, one run per `--out`), TFLOPS/MFU logging, `--eval-only`
  with `text_gain` and `--per-shard`, resume keeps CLI hyperparameters (not the checkpoint's).
- Throughput (50m, 8× 3090 Ti): DDP batch 256 ~930 img/s (PCIe-bound); MuLoCo H=16 ~3.6k img/s.
- Data: `prepare_data.py convert/tokenize`, `--fraction` (default 0.1), `data/aurora_full` holds the full set.

## Other known issues / to-dos
- On resume with different outer settings, the outer momentum buffer from the checkpoint is reused (stale). It should be
  reset when the outer hyperparameters change.
- The `--grad-comm powersgd` option is untested on NCCL (the local gloo test hung).
- Init details beyond the gating (std values, text encoder) were guessed. The README doesn't specify them.
