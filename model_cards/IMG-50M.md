---
license: apache-2.0
library_name: diffusers
pipeline_tag: text-to-image
language: [en]
datasets: [QLNI/FLUX-Reason-6M-flux2-latents]
tags: [text-to-image, flow-matching, rectified-flow, diffusion-transformer, tiny-model, from-scratch, muon, diloco, custom_code]
---

# IMG-50M

Experimental 256×256 text-to-image model trained from scratch, reproducing the
[AuroraIMG2-50M-Preview](https://huggingface.co/AuroraAI-Research/AuroraIMG2-50M-Preview) recipe on 8 consumer GPUs:
a **50.4M-parameter flow-matching DiT** and a **12.6M-parameter text encoder** trained jointly, decoding through the
frozen [FLUX.2 small decoder](https://huggingface.co/black-forest-labs/FLUX.2-small-decoder).
The architecture is identical to AuroraIMG2-50M (same parameter names, loads with its pipeline). Only the training differs:
distributed **MuLoCo** (DiLoCo with Muon) instead of single-GPU training, and constant lr + checkpoint averaging instead of
cosine decay.

![samples](samples/grid.png)

## Samples

Eight prompts from the AuroraIMG2 card, each with seed 0, 50 Euler steps, CFG 3.0, decoder in bfloat16. Row-major order:

1. a red fox sitting in a snowy forest at sunrise, photograph
2. a futuristic city skyline at night with neon lights reflected in the rain
3. a bowl of fresh strawberries on a wooden table, soft natural light
4. a watercolor painting of a lighthouse on a rocky coast
5. a small robot reading a book in a cozy library
6. an astronaut riding a horse on the moon, digital art
7. a street sign that says OPEN in neon letters
8. two children flying a kite on a green hill under a blue sky

## Held-out loss vs AuroraIMG2-50M

Velocity MSE (lower is better) on 4,096 held-out images: the last shard of each of the 4 dataset subsets, first caption,
fixed noise and t ~ logit-normal(0, 1). Both models were evaluated with the same code on the same samples.

| model | val/loss |
| --- | --- |
| AuroraIMG2-50M-Preview (released, EMA, cosine decay) | **0.6929** |
| **IMG-50M (this repo, WSM: mean of last 12%)** | 0.7044 |
| IMG-50M, EMA weights | 0.7030 |
| IMG-50M, raw weights at the last step | 0.7105 |

The remaining gap (~0.01) is most likely due to the schedule: Aurora decays the lr to zero with a cosine, while this run kept
the lr constant to the end. Averaging (EMA / WSM) only recovers part of what an lr decay gives. A per-shard check shows no
train/validation memorisation for either model.

## Architecture

- **Denoiser (50.4M):** DiT, 8 blocks, width 640, 10 heads (head dim 64), one token per latent pixel (16×16 grid, no
  patchify). Each block has self-attention with **2D RoPE** and QK-norm, cross-attention to the text tokens, and a SwiGLU MLP.
  Timestep conditioning is PixArt-style **adaLN-single** with zero-initialised gates.
- **Text encoder (12.6M):** 6-layer pre-norm transformer, width 384, GQA (6 query / 2 KV heads), 1D RoPE, SwiGLU, RMSNorm.
  8,192-token byte-level BPE (the AuroraIMG2 tokenizer), up to 62 tokens + BOS/EOS. Trained from scratch together with the
  denoiser.
- **Decoder (frozen):** FLUX.2 small decoder. Latents are 128×16×16 (BatchNorm-normalised, 2×2-packed FLUX.2 latents).

## Training

- **Data:** [QLNI/FLUX-Reason-6M-flux2-latents](https://huggingface.co/datasets/QLNI/FLUX-Reason-6M-flux2-latents),
  5.88M training images (all subsets: aesthetics part 1/2, imaginative, text). One random caption per image per step.
- **Budget:** 247,805 steps × batch 256 = 63.4M samples (~10.8 epochs), the same as AuroraIMG2-50M.
- **Objective:** rectified flow, `z_t = (1−t)·x + t·ε`, velocity target `ε − x`, MSE, t ~ logit-normal(0, 1).
  10% of captions are replaced by a learned null prompt for classifier-free guidance. Auxiliary masked-language-modelling
  loss on the text encoder (weight 0.05, every 4th step, 15% masking, head tied to the token embedding).
- **Hardware:** 8× RTX 3090 Ti (6 on PCIe 3.0 x4, 2 on x8), bf16 autocast, `torch.compile`. ~3,400 img/s, ~5 h.

### Optimiser: Muon inside MuLoCo

- **Muon** ([Jordan et al.](https://kellerjordan.github.io/posts/muon/)) on all hidden weight matrices: lr 3.5e-3,
  Nesterov momentum 0.95, 5 Newton–Schulz steps, weight decay 0.01. **AdamW** (β 0.9 / 0.95) on the token embedding
  (lr 1.77e-4), the input/output projections and the time MLP (1.06e-4, wd 0.01), and on norms, modulation tables and
  biases (1.06e-4, no wd). The AdamW lrs are the Aurora values (5e-4 / 3e-4 at batch 256) scaled by √(32/256) for the
  per-GPU batch.
- **MuLoCo** (DiLoCo with a Muon inner optimiser, [arXiv:2505.23725](https://arxiv.org/abs/2505.23725); DiLoCo:
  [arXiv:2311.08105](https://arxiv.org/abs/2311.08105)): every GPU trains its own replica on its own batch of 32
  (global batch 256), with no gradient communication. **Every 16 steps the 8 replicas are averaged**: outer lr 1.0,
  outer momentum 0, i.e. plain model averaging, sent as a bf16 all-reduce of the weight deltas. EMA and evaluation use
  the averaged weights.
- **Why:** with most cards on PCIe 3.0 x4, regular DDP at batch 256 was communication-bound (~930 img/s, ~6% MFU).
  MuLoCo with H=16 runs at ~3,400 img/s, ~3.7× faster for the same steps and samples.
- **Tuning notes:** in short runs, frequent syncs with plain averaging (H=16) beat H=50/100/500 with the default DiLoCo
  outer optimiser (Nesterov SGD, lr 0.7, momentum 0.9), which overshot after every sync at these sync intervals. Muon beat
  AdamW by a wide margin. These comparisons were made before an init bug was fixed (gated self-attention/MLP branches were
  dead at initialisation), so treat them as indicative rather than final.

### Schedule and released weights

- 2% linear warmup (4,956 steps), then **constant lr** to the end, with no decay.
- EMA of the averaged weights, 0.9995 per step (applied at each sync as 0.9995^16).
- **Released weights: WSM** (Warmup-Stable-Merge, [arXiv:2507.17634](https://arxiv.org/abs/2507.17634)): uniform average of
  the last 4 saved checkpoints, covering the last 12% of training (steps 218,240 / 228,160 / 238,080 / 247,805).
  This approximates a linear lr decay over that window.

Training code: https://github.com/adamo1139/trenowanie_generatora_obrazkow

## Usage

```python
import torch
from diffusers import DiffusionPipeline

REPO = "cpral/IMG-50M"  # this repository
pipe = DiffusionPipeline.from_pretrained(REPO, custom_pipeline=REPO, trust_remote_code=True).to("cuda")
image = pipe("a red fox sitting in a snowy forest at sunrise, photograph",
             num_inference_steps=50, guidance_scale=3.0,
             generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("out.png")
```

Fixed 256×256 output. English prompts, up to 62 tokens.

## License

Weights and code: Apache-2.0 (model code from AuroraIMG2-50M-Preview, Apache-2.0).
`vae/`: FLUX.2 small decoder © Black Forest Labs, Apache-2.0, unchanged. Training data has its own terms, see the dataset cards.
