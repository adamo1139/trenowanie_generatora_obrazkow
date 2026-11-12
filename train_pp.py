"""Pipeline-parallel (1F1B) training for DiTs that don't fit on one GPU, e.g. --size 1b / 1.5b / 2b.

The denoiser's blocks are split evenly over --pp stages (GPUs). Stage 0 also holds the text encoder, the time
projection and patch_in; the last stage holds the final norm, patch_out and the loss. Between stages travel the token
states, the timestep modulation t6, the text tokens and the text mask (plus their gradients on the way back), so the
text encoder and time projection on stage 0 are trained by every block. World = pp x replicas; replicas (pipelines)
are kept in sync with MuLoCo (--diloco H, averaged every H steps) or, without --diloco, by averaging gradients
every step.

  # one pipeline over 8 GPUs, 1.5B denoiser + frozen Slayer149 + REPA
  NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train_pp.py --data data/aurora_full --fraction 1 --size 1.5b \\
      --pp 8 --batch 512 --microbatches 32 --out runs/1p5b
  # two pipelines of 4 stages, MuLoCo between them
  NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train_pp.py ... --pp 4 --diloco 64 --outer-lr 1 --outer-momentum 0

Only the stages that need data load it (stage 0: inputs, last stage: targets, the REPA stage: REPA targets); the data
stream is deterministic and noise / t / caption dropout come from a generator seeded by (seed, replica, step), so all
of them see the same samples. Checkpoints are gathered to rank 0 in train.py's format (te, den, ema), so export.py,
merge.py and tests/check_export.py work unchanged; per-stage optimizer state goes to <out>/opt_stage<s>.pt.
MLM is not supported here (use --te-size slayer149, the default).

With --pp 1, forward/backward runs directly with gradient accumulation. HF captions are trimmed for the refiner
and denoiser too, to multiples of --text-bucket (default 64, 0 disables trimming). Static compilation warms up
separately for each encountered text width; allow those compilations to finish before comparing throughput.
Logged img/s includes statistics overhead (and eval/checkpoint time in the next logging window), unlike older
versions which excluded logging. Keep machine-required NCCL settings such as NCCL_P2P_LEVEL=SYS unchanged.

Growing a trained model deeper (SANA 1.5-style): --init <train.py / train_pp.py checkpoint> --depth N keeps its blocks
(EMA weights if the file has them), its stem and head, and its text encoder if it uses the same frozen LM; the
N - old_depth new blocks are appended at the end with zero output projections (attention, cross-attention, MLP), so
the grown model starts as exactly the old one. Copying trained blocks instead diverged in SANA 1.5.

  # HomeIMG 641M (6 blocks) -> 12 blocks: 4 pipelines of 2 stages, Poziomka with Polish captions
  NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train_pp.py --data data/aurora_full --fraction 1 --size 750m \\
      --depth 12 --pp 2 --te-size poziomka --te-quant int8 --pl-prob 0.5 --init runs/poziomka_stage2/latest.pt \\
      --batch 1024 --microbatches 16 --diloco 64 --outer-lr 1 --outer-momentum 0 --ema-cpu --out runs/grow12
"""
import datetime
import functools
import glob
import logging
import math
import os
import re
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from aurora.data import ConcatStore, ShardStore, TrainStream, load_val, select_shards, val_groups
from aurora.model import (CrossAttention, Denoiser, DenoiserConfig, LMEncoderConfig, TextAttention, TextEncoder,
                          build_rope_2d, default_micro_cond, make_configs, micro_cond_embedding,
                          random_rope_coords, rope_2d_from_coords, te_config_from_dict, timestep_embedding)
from aurora.muon import Muon
from aurora.profiling import TrainingProfile
from aurora.slayer import LM_PAD, trim_ids
from aurora.vae_encode import LatentEncoder
from train import (build_parser, evaluate, init_wandb, load_opt_state, lr_mult, make_muon, outer_sync, param_groups,
                   peak_tflops, resolve_defaults, wandb_logged_in, weight_sq, wrms_stats)

# GPU numbers as in nvidia-smi (--lm-gpu); CUDA reads this when it initialises, after the imports
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def get_args():
    p = build_parser()
    p.add_argument("--pp", type=int, default=8, help="pipeline stages (GPUs per pipeline); world must be a multiple")
    p.add_argument("--microbatches", type=int, default=None,
                   help="micro-batches per step and pipeline (default 4 x pp; bubble ~ (pp-1)/(microbatches+pp-1))")
    p.add_argument("--act-checkpoint", nargs="?", const="full", default=None, choices=["full", "selective"],
                   help="recompute block activations in backward: full (only block inputs are kept) or selective "
                        "(matmul and attention outputs are kept, only the cheap elementwise ops are recomputed)")
    p.add_argument("--te-cost", type=float, default=1.0,
                   help="work of the text encoder on stage 0 in denoiser blocks; stage 0 gets that many blocks fewer "
                        "so no stage straggles (0 = even split)")
    p.add_argument("--lm-stage0", action="store_true",
                   help="frozen LM text encoder only on stage 0, run inside its pipeline forward (classic PP; pair with "
                        "--te-cost ~2 so stage 0 gets fewer blocks). Default: the LM runs on every stage for 1/pp of "
                        "the micro-batches before each step and its states are sent to stage 0")
    p.add_argument("--lm-rank", type=int, nargs="?", const=1, default=0,
                   help="the LAST N ranks (default 1) only run the frozen transformers LM (Poziomka) for all the others, "
                        "which hold no LM at all: each pipeline's stage 0 sends the next step's captions while it trains "
                        "the current one and gets their LM states back; pipeline r uses LM rank r mod N "
                        "(world = N + pipelines x pp, e.g. 8 GPUs = 1 + 7 x pp 1 or 2 + 6 x pp 1)")
    p.add_argument("--res", type=int, default=256,
                   help="training resolution in pixels: latents of (res/16)^2 tokens, from <shard>.lat<res>.npy "
                        "(prepare_lowres.py) unless 256; the weights are resolution-free (per-token stem / head, RoPE "
                        "built for the grid), checkpoints record the grid")
    p.add_argument("--rope-ref", type=int, default=0,
                   help="RoPE position field (aurora.model.build_rope_2d): 0 = integer positions per grid (the grid grows "
                        "with the resolution); 16 = every resolution's tokens spread over the 256 px field, so all "
                        "curriculum stages share one spatial scale (identical to 0 at 256 px)")
    p.add_argument("--rope-random", action="store_true",
                   help="RPE-2D (arXiv:2503.18719): every training sample gets random image token positions, per axis "
                        "--res/16 distinct sorted integers of 0 .. --rope-ref - 1 (row x column grid), so every spacing "
                        "a larger sampling grid uses has been trained; sampling keeps the even --rope-ref grid. pp = 1")
    p.add_argument("--micro-cond", type=int, default=0,
                   help="RPE-2D micro-conditioning: the original size in px of the training images (256); adds the "
                        "denoiser's cond_mlp (zero output at the start) fed with original size, crop box and target "
                        "size; needed by --crop-res. 0 = off")
    p.add_argument("--crop-res", type=int, nargs="*", default=[],
                   help="RPE-2D resize-and-crop: resolutions of larger latents of the same images to cut random "
                        "--res windows from (256 = the shards' own .lat.npy; others from <--lat-dir parent>/<res>px), "
                        "e.g. 256 at --res 128: half of the samples a 128 px crop of the 256 px image (--crop-prob)")
    p.add_argument("--crop-prob", type=float, default=0.5, help="share of the training samples that are crops")
    p.add_argument("--img-dir", default="",
                   help="train from images instead of stored latents: <img-dir>/<shard>.img.bin (prepare_images.py "
                        "flux-reason), captions / tokens from --data; resized to --res in the loader, encoded by the "
                        "FLUX.2 encoder (--enc-vae) on every GPU")
    p.add_argument("--img-extra", nargs="*", default=[],
                   help="more image sources with their own tokens, e.g. data/monet_z_image (prepare_monet.py + "
                        "translate_captions.py + prepare_data.py tokenize-parquet + prepare_images.py monet); mixed "
                        "with --img-dir in their natural proportions")
    p.add_argument("--img-extra-val", nargs="*", default=[],
                   help="per --img-extra source, the file kept out of training as its validation set, e.g. 45-0/000005 "
                        "(logged as val_<folder>/...; val/ is the --img-dir held-out shards, val_all/ both together)")
    p.add_argument("--val-per-shard", action="store_true",
                   help="also log the loss of every held-out shard of --data on its own (val_<shard>/loss, e.g. "
                        "val_aesthetics-part01, val_text): the same images as val/, no extra compute")
    p.add_argument("--enc-vae", default="FLUX.2-dev-vae", help="diffusers AutoencoderKLFlux2 whose encoder makes the latents")
    p.add_argument("--enc-chunk", type=int, default=8, help="images per encoder call (memory)")
    p.add_argument("--vae-rank", type=int, default=0, choices=[0, 1],
                   help="--img-dir with --lm-rank: one extra rank (the last, on GPU --vae-gpu) that only encodes the "
                        "training images of every pipeline")
    p.add_argument("--vae-gpu", type=int, default=4, help="--vae-rank: its physical GPU (nvidia-smi order)")
    p.add_argument("--vae-remote-frac", type=float, default=1.0,
                   help="--vae-on-lm / --vae-rank: the share of each batch encoded by the server; the rest is encoded "
                        "on the training rank (when the server can't keep up with all of them)")
    p.add_argument("--vae-on-lm", action="store_true",
                   help="--img-dir with --lm-rank: the LM server rank also encodes the training images (its GPU idles "
                        "~90%% between LM requests; at 1024 px encoding took ~16%% of a training rank's step)")
    p.add_argument("--lat-dir", default="", help="folder of the --res latents if not next to the shards (e.g. "
                   "prepare_lowres.py --out-dir X -> X/128px)")
    p.add_argument("--t-shift", type=float, default=1.0,
                   help="timestep shift s, t -> s t / (1 + (s - 1) t) after t = sigmoid(randn); 1 = none. SD3's "
                        "resolution rule would be sqrt(tokens / 256): 0.25 at 64 px, 0.5 at 128, 2 at 512, 4 at 1024")
    p.add_argument("--opd", action="store_true",
                   help="on-policy distillation from FLUX.2 klein base (DiffusionOPD, arXiv:2605.15055) instead of flow "
                        "matching on the data, --pp 1 only: each step the student samples its own Euler trajectories "
                        "(conditional velocity, no CFG) from noise for the batch's captions, --opd-states states per "
                        "trajectory are sent to a teacher rank, and the student's velocity there is regressed onto "
                        "klein's CFG-guided one. The teacher always gets the ENGLISH original of the caption the student "
                        "sees (Polish or English). The student becomes guidance-distilled: sample it with CFG 1")
    p.add_argument("--teacher-ranks", type=int, default=0,
                   help="--opd: ranks holding klein (transformer + Qwen3 text encoder, ~17 GB in bf16), placed after the "
                        "training ranks and before the --lm-rank ones; every training rank splits its trajectories "
                        "over all of them")
    p.add_argument("--klein", default="black-forest-labs/FLUX.2-klein-base-4B", help="--opd: klein base repo or dir")
    p.add_argument("--opd-cfg", type=float, default=4.0, help="--opd: the teacher's CFG scale (klein base: 4)")
    p.add_argument("--klein-quant", default="", choices=["", "int8", "fp8"],
                   help="--opd: teacher linear weights in 8 bits (aurora/klein.py QLinear; ~8 GB instead of ~16, bf16 "
                        "matmuls). int8 (one scale per 128 inputs) is ~3x more accurate than fp8")
    p.add_argument("--teacher-on-lm", action="store_true",
                   help="--opd: the --lm-rank server(s) also host klein (needs --klein-quant) and take a share of the "
                        "states like the --teacher-ranks; that GPU is otherwise ~99%% idle during --opd")
    p.add_argument("--opd-rollout-steps", type=int, default=16, help="--opd: Euler steps of the student's trajectories")
    p.add_argument("--opd-states", type=int, default=4,
                   help="--opd: states per trajectory (distinct random steps) the teacher scores and the student trains on")
    p.add_argument("--dist-timeout", type=float, default=120,
                   help="minutes a collective / p2p op may wait for the other ranks (evals at 1024 px take ~10 min)")
    p.add_argument("--skip-gpus", type=int, nargs="*", default=[],
                   help="physical GPUs (nvidia-smi order) left unused, e.g. 2 for a card that overheats; with --lm-rank "
                        "the ranks then take the other GPUs (run torchrun with that many fewer processes)")
    p.add_argument("--lm-gpu", type=int, default=4,
                   help="--lm-rank: physical GPU (nvidia-smi / PCI bus order) of the (last) LM server rank, the training "
                        "ranks take the others; on adamo-rigga GPU 4 runs hottest (junction 104 C, it throttled and "
                        "held every MuLoCo sync back) and the LM GPU is only ~15-50%% busy. "
                        "-1 = plain rank order")
    p.add_argument("--lm-split", action="store_true",
                   help="frozen transformers LM (Poziomka) split by layers over the pipeline's stages: each GPU holds "
                        "1/pp of it, the hidden states go stage 0 -> 1 -> ... -> back to stage 0 (chunks of ~32 "
                        "captions, overlapped); the held-out set's LM states are computed once at the start")
    p.add_argument("--te-checkpoint", action="store_true",
                   help="activation checkpointing of the text encoder's refiner blocks (stage 0): stage 0 holds the most "
                        "micro-batches in flight under 1F1B; PP=1 trims text using --text-bucket")
    p.add_argument("--text-bucket", type=int, choices=[0, 16, 32, 64, 128, 256], default=64,
                   help="PP=1 HF text widths rounded to this bucket (default 64; 0 disables trimming). "
                        "With static compilation each new width needs its own forward/backward warmup; "
                        "benchmark only after all encountered widths have compiled.")
    p.add_argument("--mem-debug", action="store_true",
                   help="print every rank's GPU memory breakdown (weights, grads, optimizer, EMA, MuLoCo copy, frozen LM, "
                        "peak activations) after steps 1-3; the same report is always printed on an out-of-memory error")
    p.add_argument("--mem-snapshot", default="",
                   help="dir: record the CUDA allocator history and dump <dir>/rank<r>.pickle after step 2 or on OOM "
                        "(open in https://pytorch.org/memory_viz)")
    p.add_argument("--resume-opt-mismatch", action="store_true",
                   help="resume even if opt_stage*.pt is from another step than latest.pt (a save that died between "
                        "the two, e.g. on a full disk): its moments are close to the right ones, better than fresh")
    p.add_argument("--bench", action="store_true",
                   help="throughput test: no checkpoints, no merge files, no evaluation (nothing is written to --out)")
    p.add_argument("--profile-dir", default="", help="write opt-in CPU/CUDA traces here (also under --bench)")
    p.add_argument("--profile-start", type=int, default=17, help="first training step to trace, 1-based")
    p.add_argument("--profile-steps", type=int, default=4, help="number of training steps to trace")
    p.add_argument("--profile-ranks", type=int, nargs="+", default=[0], help="training ranks to trace (not LM servers)")
    p.add_argument("--masked-attention-backend", choices=["auto", "cudnn"], default="auto",
                   help="backend for DiT cross-attention and trainable text refiner; self-attention and frozen LM unchanged")
    p.set_defaults(batch=512, size="1.5b")
    return p.parse_args()


# selective activation checkpointing: keep what is expensive to recompute (matmuls, attention), recompute the rest
_SAC_SAVE = {torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default,
             torch.ops.aten._scaled_dot_product_flash_attention.default,
             torch.ops.aten._scaled_dot_product_efficient_attention.default,
             torch.ops.aten._scaled_dot_product_cudnn_attention.default}


def _sac_policy(ctx, op, *args, **kwargs):
    return CheckpointPolicy.MUST_SAVE if op in _SAC_SAVE else CheckpointPolicy.PREFER_RECOMPUTE


def checkpointed(block, mode):
    """block wrapped in activation checkpointing (mode: "full" or "selective"); compile the result, not the block."""
    if mode == "selective":  # dynamo only traces a functools.partial context_fn
        ctx = functools.partial(create_selective_checkpoint_contexts, _sac_policy)
        return lambda *args: checkpoint(block, *args, use_reentrant=False, context_fn=ctx)
    return lambda *args: checkpoint(block, *args, use_reentrant=False)


class AddAux(torch.autograd.Function):
    """Returns x unchanged; in backward, also back-propagates the scalar auxiliary loss `aux` (gradient 1).
    Lets a middle pipeline stage add its own loss term (REPA) without sending it to the last stage."""

    @staticmethod
    def forward(ctx, x, aux):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g, torch.ones((), device=g.device, dtype=torch.float32)


def microbatch_rope(seed, mb, b, side, field, d, device):
    """RPE-2D positions (cos, sin) of micro-batch `mb` of a step, drawn from (seed, mb) alone, so every pipeline stage of
    a replica draws the same ones without communicating (and pp = 1 the same as pp > 1)."""
    g = torch.Generator(device=device).manual_seed(seed * 4099 + mb)
    rows, cols = random_rope_coords(b, side, field, device, g)
    return rope_2d_from_coords(rows, cols, d)


class Stage(nn.Module):
    """One pipeline stage: optional stem (text encoder, time projection, patch_in), blocks [lo, hi), optional head."""

    def __init__(self, den, cfg, lo, hi, first, last, te=None, repa_proj=None, repa_block=-1, repa_weight=0.0,
                 act_checkpoint=None):
        super().__init__()
        self.cfg, self.first, self.last = cfg, first, last
        if first:
            self.te, self.patch_in, self.time_mlp, self.ada = te, den.patch_in, den.time_mlp, den.ada
            if cfg.micro_cond:
                self.cond_mlp = den.cond_mlp
        self.blocks = nn.ModuleList(den.blocks[lo:hi])
        if last:
            self.norm, self.patch_out = den.norm, den.patch_out
        self.repa_proj, self.repa_block, self.repa_weight = repa_proj, repa_block, repa_weight
        # callables that run each block (plain, or wrapped in activation checkpointing); main() compiles these
        self.block_fns = [checkpointed(b, act_checkpoint) if act_checkpoint else b for b in self.blocks]
        cos, sin = build_rope_2d(cfg.latent_size, cfg.dim // cfg.heads, cfg.rope_ref)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.repa_targets, self.repa_losses, self._mb = None, [], 0
        self.lm_states = None  # frozen-LM hidden states per micro-batch, computed on all stages (see main)
        self.text_cache = None  # (text, valid) to use instead of running the text encoder (--opd rollouts)
        # per micro-batch (local_train_step): (B, 8) micro-conditions (None: whole images) and (cos, sin) of random
        # RPE-2D positions (None: the even grid in the buffers)
        self.mb_cond, self.mb_rope = None, None
        # RPE-2D: random positions of 0 .. rope_field - 1 per micro-batch, from (rope_seed, _mb); set per training step
        self.rope_field, self.rope_seed = 0, None

    def forward(self, *args):
        if self.first:
            z, t, ids = args
            if self.text_cache is not None:
                text, valid = self.text_cache
            elif self.lm_states is not None and not ids.is_meta:
                text, valid = self.te(ids, lm_states=self.lm_states[self._mb])
            else:
                text, valid = self.te(ids)
            validf = valid.to(text.dtype)
            x = self.patch_in(z.flatten(2).transpose(1, 2))
            t_emb = self.time_mlp(timestep_embedding(t, 256))
            if self.cfg.micro_cond:
                cond = self.mb_cond if self.mb_cond is not None else default_micro_cond(
                    self.cfg.micro_cond, 16 * self.cfg.latent_size, z.shape[0], z.device)
                t_emb = t_emb + self.cond_mlp(micro_cond_embedding(cond).to(t_emb.dtype))
            t6 = self.ada(F.silu(t_emb)).view(z.shape[0], 6, self.cfg.dim)
        else:
            x, t6, text, validf = args
            # pipelining sends a gradient back for every stage input; the mask needs a (zero) one
            x = x + 0 * validf.sum()
        valid = validf > 0.5
        if self.mb_rope is not None:
            cos, sin = self.mb_rope
        elif self.rope_field and self.rope_seed is not None and not x.is_meta:
            cos, sin = microbatch_rope(self.rope_seed, self._mb, x.shape[0], self.cfg.latent_size, self.rope_field,
                                       self.cfg.dim // self.cfg.heads, x.device)
        else:
            cos, sin = self.rope_cos, self.rope_sin
        for i, fn in enumerate(self.block_fns):
            x = fn(x, t6, text, valid, cos, sin)
            if i == self.repa_block and self.repa_targets is not None and not x.is_meta \
                    and self._mb < len(self.repa_targets):
                target = self.repa_targets[self._mb]
                pred = self.repa_proj(x[: target.shape[0]])
                aux = -F.cosine_similarity(pred.float(), target, dim=-1).mean()
                self.repa_losses.append(aux.detach())
                x = AddAux.apply(x, self.repa_weight * aux)
        if self.last:
            n = self.cfg.latent_size
            return self.patch_out(self.norm(x)).transpose(1, 2).reshape(x.shape[0], self.cfg.latent_channels, n, n)
        return x, t6, text, validf


def ema_update_(ema, params, weight, stage=None):
    """ema <- lerp(ema, params, weight). stage: CPU buffers shaped like `ema` (allocated once, pinned) that the
    parameters are copied into when the EMA lives on the CPU, instead of a fresh CPU copy of every weight at every
    MuLoCo sync (5 GB in hundreds of tensors each time; host RAM grew ~1 GB per sync in the 384 px run)."""
    if stage is None:
        torch._foreach_lerp_(ema, [p.detach().float().to(e.device) for p, e in zip(params, ema)], weight)
        return
    for b, p in zip(stage, params):
        b.copy_(p.detach(), non_blocking=True)
    torch.cuda.synchronize()
    torch._foreach_lerp_(ema, stage, weight)


def host_rss_gb():
    """Resident memory of this process and its children (the DataLoader workers), GB: a growing host-RAM footprint
    gets the run killed by the OOM killer (it happened at a checkpoint save, the RAM peak of rank 0)."""
    def rss(pid):
        try:
            with open(f"/proc/{pid}/status") as fh:
                return next(int(l.split()[1]) for l in fh if l.startswith("VmRSS:")) / 2**20
        except (OSError, StopIteration):
            return 0.0
    me, total = os.getpid(), 0.0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as fh:
                    if int(fh.read().rsplit(")", 1)[1].split()[1]) == me:
                        total += rss(d)
            except (OSError, IndexError, ValueError):
                pass
    return rss(me) + total


def host_ram_used_gb():
    """(used, total) host RAM in GB, used = MemTotal - MemAvailable."""
    with open("/proc/meminfo") as fh:
        m = {l.split(":")[0]: int(l.split()[1]) for l in fh}
    return (m["MemTotal"] - m["MemAvailable"]) / 2**20, m["MemTotal"] / 2**20


@torch.no_grad()
def average_params(params, n, group, bucket=64 * 2**20):
    """MuLoCo with outer lr 1 and no momentum: every replica's weights become the replicas' mean (fp32 all-reduce in
    buckets of ~`bucket` elements)."""
    chunk, size = [], 0
    for i, p in enumerate(params):
        chunk.append(p)
        size += p.numel()
        if size >= bucket or i == len(params) - 1:
            flat = torch.cat([q.detach().float().flatten() for q in chunk])
            dist.all_reduce(flat, group=group)
            flat /= n
            for q, d in zip(chunk, flat.split([q.numel() for q in chunk])):
                q.copy_(d.view_as(q))
            chunk, size = [], 0


def stage_ranges(depth, pp, first_extra=0.0):
    """Contiguous block ranges per stage, balanced when stage 0 also does `first_extra` blocks' worth of other work.
    Stage 0 may end up with no blocks at all (only the text encoder and the stem), e.g. 14 blocks, 8 stages,
    first_extra 2 -> [0, 2, 2, 2, 2, 2, 2, 2]; every other stage needs at least one."""
    bounds = np.linspace(0, depth + first_extra, pp + 1) - first_extra
    bounds = np.clip(bounds.round(), 0, depth).astype(int)
    bounds[0], bounds[-1] = 0, depth
    assert (np.diff(bounds)[1:] > 0).all() and bounds[1] >= 0, f"{depth} blocks can't fill {pp} stages"
    return [(int(bounds[s]), int(bounds[s + 1])) for s in range(pp)]


def gather_to_stage(mine, M, s, pp, dst, base):
    """mine: {k: tensor} for this stage's micro-batches (k % pp == s). Point-to-point (in k order, so it can't
    deadlock) to stage `dst` of the pipeline whose stage 0 is global rank `base`; returns the M tensors there."""
    if s != dst:
        for k in sorted(mine):
            dist.send(mine[k].contiguous(), dst=base + dst)
        return None
    ref, out = mine[dst], []  # every stage owns at least micro-batch k = s (M >= pp)
    for k in range(M):
        if k % pp == s:
            out.append(mine[k])
        else:
            buf = torch.empty_like(ref)
            dist.recv(buf, src=base + k % pp)
            out.append(buf)
    return out


def lm_split_forward(lm, chunks, s, pp, base, device, hidden):
    """--lm-split: caption rows (chunks of any length) through the LM slices of a pipeline's stages in order (stage s
    holds one slice; global ranks base .. base + pp - 1), overlapped chunk by chunk; returns on stage 0 one
    (rows, T, hidden) bf16 tensor per chunk (zeros for null-prompt rows), None elsewhere. Every stage of the pipeline
    calls it with the same chunks."""
    from aurora.slayer import LM_NULL, hf_lm_part
    plans = []
    for c in chunks:
        real = c[:, 0].ne(LM_NULL)
        plans.append((c, real, trim_ids(c[real], multiple=8) if real.any() else None))
    # per pair of ranks NCCL runs the sends and receives in issue order, so both sides issue them in mirrored order:
    # stage 0 "send chunk i, receive its result", stage s "receive chunk i, send it on". Compute still overlaps:
    # stage 0 works on chunk i + 1 while the later stages work on chunk i.
    results, recvs, sends = [None] * len(plans), [], []
    for i, (c, real, sub) in enumerate(plans):
        if sub is None:
            continue
        if s == 0:
            x = sub.masked_fill(sub.eq(LM_PAD) | sub.eq(LM_NULL), 0)
        else:
            x = torch.empty(*sub.shape, hidden, dtype=torch.bfloat16, device=device)
            dist.recv(x, src=base + s - 1)
        h = hf_lm_part(lm, x, first=s == 0, last=s == pp - 1).contiguous()
        sends.append((dist.isend(h, dst=base + (s + 1) % pp), h))
        if s == 0:
            results[i] = torch.empty(*sub.shape, hidden, dtype=torch.bfloat16, device=device)
            recvs.append(dist.irecv(results[i], src=base + pp - 1))
    for req, _ in sends:
        req.wait()
    if s != 0:
        return None
    for req in recvs:
        req.wait()
    out = []
    for (c, real, sub), h in zip(plans, results):
        full = torch.zeros(c.shape[0], c.shape[1], hidden, dtype=torch.bfloat16, device=device)
        if h is not None:
            full[real, : h.shape[1]] = h
        out.append(full)
    return out


def lm_chunk_plan(ids, chunk=32):
    """CPU-only request metadata: (first row, real row indices, trimmed caption IDs).

    Both peers derive the same plan from the CPU IDs sent over the control group; no CUDA scalar reads,
    nonzero or boolean indexing are needed to allocate receives or run the LM.
    """
    from aurora.slayer import LM_NULL
    assert ids.device.type == "cpu", "plan LM requests before copying IDs to CUDA"
    out = []
    for i in range(0, len(ids), chunk):
        c = ids[i:i + chunk]
        rows = c[:, 0].ne(LM_NULL).nonzero().flatten()
        if rows.numel():
            out.append((i, rows, trim_ids(c[rows], multiple=8)))
    return out


def lm_packed_chunks(lm, plan, device, max_tokens=8192):
    """--lm-rank server: the LM states of a request's lm_chunk_plan() chunks, computed packed (aurora.slayer.
    hf_lm_packed: real tokens only, up to `max_tokens` per LM call) and laid out as the chunks' (rows, width, H)
    tensors the client receives; padding positions are zeros (masked downstream)."""
    from aurora.slayer import hf_lm_packed
    masks = [sub.ne(LM_PAD) for _, _, sub in plan]
    seqs = [row[m] for (_, _, sub), mask in zip(plan, masks) for row, m in zip(sub, mask)]
    outs, pack, n = [], [], 0
    for sq in seqs + [None]:
        if sq is None or (pack and n + len(sq) > max_tokens):
            outs.append(hf_lm_packed(lm, pack, device))
            pack, n = [], 0
        if sq is not None:
            pack.append(sq)
            n += len(sq)
    flat = torch.cat(outs)
    chunks, t0 = [], 0
    for (_, _, sub), mask in zip(plan, masks):
        k = int(mask.sum())
        out = flat.new_zeros(sub.numel(), flat.shape[-1])
        out.index_copy_(0, mask.flatten().nonzero().flatten().to(device), flat[t0:t0 + k])
        chunks.append(out.view(*sub.shape, -1))
        t0 += k
    return chunks


def lm_server(te_cfg, clients, device, log, control_group, lm=None, teacher=None, opd_cfg=4.0, teacher_clients=(),
              vae=None, serve_lm=True):
    """CPU/Gloo any-source control queue; hidden states still travel directly over NCCL.

    The optional lm argument is a small fake encoder in the distributed protocol test.
    """
    from aurora.slayer import LM_NULL, LMTextEncoder, enable_packed, hf_lm_part
    packed = serve_lm and lm is None and device.type == "cuda"  # real LM on a GPU: captions packed (varlen flash)
    if serve_lm:  # serve_lm=False: a --vae-rank server, VAE requests only
        lm = LMTextEncoder(te_cfg).to(device).lm if lm is None else lm
    if packed:
        enable_packed(lm, compile=True)
    log(f"LM server: {te_cfg.repo} for ranks {clients}{' (+ the --opd teacher)' if teacher else ''}" if serve_lm
        else f"VAE server (rank {dist.get_rank()}) for ranks {clients}", flush=True)
    # one stop per LM client, and one per teacher client when this rank also serves klein (--teacher-on-lm: every
    # training rank, not only this server's LM clients)
    stops = {c: (c in clients) + (c in teacher_clients) for c in set(clients) | set(teacher_clients)}
    null = teacher.encode([""]) if teacher else None
    active, served, t0 = set(stops), 0, time.perf_counter()
    idle = compute = transfer = vae_time = 0.0
    win_t, win_n, vae_n = t0, 0, 0
    while active:
        hdr = torch.empty(3, dtype=torch.long)
        start = time.perf_counter()
        r = dist.recv(hdr, group=control_group, tag=1)  # any ready client, not a fixed round-robin slot
        assert r in active, f"unexpected LM client {r}"
        if hdr[0].item() == 0:
            stops[r] -= 1
            if not stops[r]:
                active.remove(r)
            idle += time.perf_counter() - start
            continue
        if hdr[0].item() == 3:  # --vae-on-lm: uint8 images (B, 3, H, H) in over NCCL, latents out
            imgs = torch.empty(int(hdr[1]), 3, int(hdr[2]), int(hdr[2]), dtype=torch.uint8, device=device)
            dist.recv(imgs, src=r)
            idle += time.perf_counter() - start
            start = time.perf_counter()
            lat = vae(imgs).contiguous()
            dist.isend(lat, dst=r).wait()
            del imgs
            vae_time += time.perf_counter() - start
            vae_n += int(hdr[1])
            continue
        if hdr[0].item() == 2:  # a teacher request (aurora/klein.py protocol, header on this tag)
            from aurora.klein import serve_request
            idle += time.perf_counter() - start
            start = time.perf_counter()
            serve_request(teacher, null, r, hdr, control_group, device, opd_cfg)
            compute += time.perf_counter() - start
            continue
        ids = torch.empty(int(hdr[1]), int(hdr[2]), dtype=torch.long)
        dist.recv(ids, src=r, group=control_group, tag=2)
        idle += time.perf_counter() - start
        start = time.perf_counter()
        sends, events = [], []
        if packed:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            hs = lm_packed_chunks(lm, lm_chunk_plan(ids), device)
            end.record()
            events.append((begin, end))
            sends = [(dist.isend(h, dst=r), h) for h in hs]
        for _, _, sub in ([] if packed else lm_chunk_plan(ids)):
            sub = sub.masked_fill(sub.eq(LM_PAD) | sub.eq(LM_NULL), 0).to(device)
            if device.type == "cuda":
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
            h = hf_lm_part(lm, sub, True, True).contiguous()
            if device.type == "cuda":
                end.record()
                events.append((begin, end))
            sends.append((dist.isend(h, dst=r), h))
        for req, _ in sends:
            req.wait()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        gpu_compute = sum(a.elapsed_time(b) for a, b in events) / 1000 if events else 0.0
        compute += gpu_compute
        transfer += max(0.0, elapsed - gpu_compute)  # residual: planning, transfer and CPU dispatch, can overlap
        served += len(ids)
        win_n += len(ids)
        if time.perf_counter() - win_t > 60:
            dt = time.perf_counter() - win_t
            log(f"LM server (rank {dist.get_rank()}): LM forward (CUDA events) {100 * compute / dt:.0f}%, "
                f"transfer/dispatch residual {100 * transfer / dt:.0f}%, waiting for requests {100 * idle / dt:.0f}%, "
                f"{win_n / dt:.0f} captions/s" + (f", VAE encoding {100 * vae_time / dt:.0f}% ({vae_n / dt:.1f} images/s)"
                                                  if vae is not None else ""), flush=True)
            idle = compute = transfer = vae_time = 0.0
            win_t, win_n, vae_n = time.perf_counter(), 0, 0
    log(f"LM server: {served:,} captions in {time.perf_counter() - t0:.0f} s, done", flush=True)


class LMClient:
    """One outstanding request per client. Control tensors stay on CPU; results use the training backend."""

    def __init__(self, server, hidden, device, control_group):
        self.server, self.hidden, self.device = server, hidden, device
        self.control_group = control_group

    def request(self, ids):
        assert ids.device.type == "cpu"
        ids = ids.contiguous()
        hdr = torch.tensor([1, ids.shape[0], ids.shape[1]], dtype=torch.long)
        reqs = [dist.isend(hdr, dst=self.server, group=self.control_group, tag=1),
                dist.isend(ids, dst=self.server, group=self.control_group, tag=2)]
        plans, bufs = lm_chunk_plan(ids), []
        for _, _, sub in plans:
            bufs.append(torch.empty(*sub.shape, self.hidden, dtype=torch.bfloat16, device=self.device))
            reqs.append(dist.irecv(bufs[-1], src=self.server))
        return dict(ids=ids, hdr=hdr, plans=plans, bufs=bufs, reqs=reqs)

    def collect(self, h, out_device=None):
        for r in h["reqs"]:
            r.wait()
        ids = h["ids"]
        full = torch.zeros(*ids.shape, self.hidden, dtype=torch.bfloat16, device=out_device or self.device)
        for (i, rows, sub), buf in zip(h["plans"], h["bufs"]):
            full[:, :sub.shape[1]].index_copy_(0, (rows + i).to(full.device), buf.to(full.device))
        return full

    def request_vae(self, u8, shape):
        """--vae-on-lm: uint8 images (B, 3, H, H) on this GPU -> a handle; vae_collect() gives the (B, C, h, w) float32
        latents. Issue it before the LM request of the same batch (the server answers a client in request order, and
        NCCL matches this pair's sends / receives in issue order)."""
        hdr = torch.tensor([3, u8.shape[0], u8.shape[-1]], dtype=torch.long)
        out = torch.empty(*shape, dtype=torch.float32, device=self.device)
        reqs = [dist.isend(hdr, dst=self.server, group=self.control_group, tag=1),
                dist.isend(u8.contiguous(), dst=self.server), dist.irecv(out, src=self.server)]
        return dict(vae=out, reqs=reqs, u8=u8)

    @staticmethod
    def vae_collect(h):
        for r in h["reqs"]:
            r.wait()
        return h["vae"] if h.get("local") is None else torch.cat([h["vae"], h["local"]])

    def stop(self):
        dist.send(torch.tensor([0, 0, 0]), dst=self.server, group=self.control_group, tag=1)


def text_widths(ids_cpu, microbatch, multiple=64):
    """A small set of compile shapes for PP=1; remove only trailing LM_PAD, never valid text."""
    assert ids_cpu.device.type == "cpu"
    return [trim_ids(c, multiple=multiple).shape[1] for c in ids_cpu.split(microbatch)]


def local_train_step(stage, z, t, ids, target, microbatch, losses, widths=None, divide=True, cond=None,
                     rope_field=0, rope_seed=None):
    """PP=1: immediate backward, no pipeline output retention, variable text widths per microbatch. divide=False:
    leave the summed gradients (the caller divides once after several calls) and return the micro-batch count.
    cond: (B, 8) micro-conditions; rope_field > 0: random RPE-2D positions of 0 .. rope_field - 1 (microbatch_rope)."""
    count = z.shape[0] // microbatch
    states = stage.lm_states
    stage.rope_field, stage.rope_seed = rope_field, rope_seed
    try:
        for k in range(count):
            sl = slice(k * microbatch, (k + 1) * microbatch)
            width = widths[k] if widths is not None else ids.shape[1]
            stage._mb = k
            stage.mb_cond = cond[sl] if cond is not None else None
            if states is not None:
                # Stage indexes this list by _mb; do not retain padded views after the step.
                stage.lm_states = [None] * count
                stage.lm_states[k] = states[k][:, :width]
            pred = stage(z[sl], t[sl], ids[sl, :width])
            loss = F.mse_loss(pred.float(), target[sl])
            losses.append(loss.detach())
            # AddAux's auxiliary gradient is 1; divide ALL accumulated gradients after backward,
            # matching Schedule1F1B.perform_reduce_grad, including REPA.
            loss.backward()
        if divide:
            grads = [p.grad for p in stage.parameters() if p.grad is not None]
            torch._foreach_div_(grads, count)
    finally:
        stage.lm_states = states
        stage.mb_cond = stage.mb_rope = None
        stage.rope_field, stage.rope_seed = 0, None
    return count


@torch.no_grad()
def opd_rollout(stage, z, ids, lm_states, mb, ts, picks, widths=None):
    """The student's own Euler trajectories from noise z (B, C, h, w): conditional velocity (no CFG) for the
    captions ids (lm_states: their (B, L, H) LM states or None), time grid ts (K + 1,), in micro-batches of mb; the
    text encoder runs once per micro-batch, not once per step. -> states (B, m, C, h, w) at the step indices picks
    (B, m) (CPU)."""
    B, m = picks.shape
    chunks = []
    for j in range(0, B, mb):
        w = widths[j // mb] if widths is not None else ids.shape[1]
        if lm_states is not None:
            text, valid = stage.te(ids[j:j + mb, :w], lm_states=lm_states[j:j + mb, :w])
        else:
            text, valid = stage.te(ids[j:j + mb, :w])
        chunks.append((slice(j, j + mb), (text, valid), ids[j:j + mb, :w]))
    out = torch.empty(B, m, *z.shape[1:], device=z.device)
    try:
        for k in range(len(ts) - 1):
            b_idx, r_idx = (picks == k).nonzero(as_tuple=True)  # CPU: no GPU sync
            if len(b_idx):
                out[b_idx.to(z.device), r_idx.to(z.device)] = z[b_idx.to(z.device)]
            v = []
            for sl, cache, ids_j in chunks:
                stage.text_cache = cache
                v.append(stage(z[sl], ts[k].expand(ids_j.shape[0]), ids_j).float())
            z = z + (ts[k + 1] - ts[k]) * torch.cat(v)
    finally:
        stage.text_cache = None
    return out


def opd_step(stage, teacher, qwen_tok, texts_en, eps, ids, ids_cpu, mb, args, step, replica, losses, parts=2):
    """One --opd step for this replica's batch (see --opd). The batch is cut into `parts`: part p+1 is rolled out
    (and later trained on) while the teachers score part p, so neither side waits for the other; all parts use
    this step's weights, so it stays on-policy."""
    from aurora.klein import tokenize
    B, m, K = eps.shape[0], args.opd_states, args.opd_rollout_steps
    g = torch.Generator().manual_seed(args.seed * 1_000_003 + replica * 1009 + step + 7)
    picks = torch.stack([torch.randperm(K, generator=g)[:m] for _ in range(B)])  # distinct steps per trajectory
    # the grid's inner points shifted by a random fraction of a step every training step: the trained t cover [0, 1]
    # instead of K fixed values (the trajectories still start at pure noise and end at t = 0)
    d = torch.rand((), generator=g).item()
    u = torch.cat([torch.ones(1), 1 - (torch.arange(1, K) - d) / K, torch.zeros(1)]).clamp(0, 1).to(eps.device)
    ts = args.t_shift * u / (1 + (args.t_shift - 1) * u)
    q_ids, q_mask = tokenize(qwen_tok, texts_en)
    states = stage.lm_states
    lm_full = torch.cat(states) if states is not None else None  # (B, L, H)
    per = -(-B // parts)
    assert (per * m) % mb == 0 or parts == 1, \
        f"--opd: {per} trajectories x {m} states must split into micro-batches of {mb}"
    rep = lambda a: a.repeat_interleave(m, dim=0)
    pending = []
    for b0 in range(0, B, per):  # roll out every part and send it; the teachers start on part 0 at once
        b1 = min(B, b0 + per)
        ids_c = ids_cpu[b0:b1] if ids_cpu is not None else None
        widths = text_widths(ids_c, mb, args.text_bucket) if ids_c is not None and args.text_bucket else None
        x = opd_rollout(stage, eps[b0:b1], ids[b0:b1], lm_full[b0:b1] if lm_full is not None else None, mb, ts,
                        picks[b0:b1], widths).flatten(0, 1)
        t = ts[picks[b0:b1].to(eps.device)].flatten()
        cap = torch.arange(b1 - b0, device=eps.device).repeat_interleave(m)
        pending.append((b0, b1, x, t, teacher.request(q_ids[b0:b1], q_mask[b0:b1], x, t, cap)))
    count = 0
    try:
        for b0, b1, x, t, h in pending:  # then train on each part as its targets arrive
            if lm_full is not None:
                full = rep(lm_full[b0:b1])
                stage.lm_states = [full[i:i + mb] for i in range(0, len(full), mb)]
            ids_c = rep(ids_cpu[b0:b1]) if ids_cpu is not None else None
            w_rep = text_widths(ids_c, mb, args.text_bucket) if ids_c is not None and args.text_bucket else None
            count += local_train_step(stage, x, t, rep(ids[b0:b1]), teacher.collect(h), mb, losses, w_rep,
                                      divide=False)
    finally:
        stage.lm_states = states
    grads = [p.grad for p in stage.parameters() if p.grad is not None]
    torch._foreach_div_(grads, count)


def owned_den_names(den, s, pp, ranges):
    lo, hi = ranges[s]
    out = []
    for name, _ in den.named_parameters():
        head = name.split(".")[0]
        if head == "blocks":
            if lo <= int(name.split(".")[1]) < hi:
                out.append(name)
        elif head in ("patch_in", "time_mlp", "ada", "cond_mlp"):
            if s == 0:
                out.append(name)
        elif head in ("norm", "patch_out"):
            if s == pp - 1:
                out.append(name)
        else:
            raise ValueError(f"unassigned denoiser parameter {name}")
    return out


def load_val_cached(root, names, n, tok_suffix, null_row, lat_kind, lat_root, img_res, encode, vae, log):
    """load_val; from images (encode) the latents + token ids are cached in <lat_root>/val_cache/ (decoding and encoding
    4096 held-out images at 1024 px takes ~7 min on one GPU), keyed by everything that changes them."""
    if encode is None:
        return load_val(root, names, n, tok_suffix, null_row, lat_kind, lat_root, img_res)
    import hashlib
    key = hashlib.sha1(repr((sorted(names), n, tok_suffix, img_res, os.path.basename(os.path.normpath(vae)),
                             null_row.tolist() if null_row is not None else None)).encode()).hexdigest()[:12]
    path = os.path.join(lat_root, "val_cache", f"val_{img_res}px_{n}_{key}.pt")
    if os.path.exists(path):
        lat, ids = torch.load(path)
        log(f"validation: {len(lat)} latents from the cache {path}")
        return lat, ids
    lat, ids = load_val(root, names, n, tok_suffix, null_row, lat_kind, lat_root, img_res, encode=encode)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save((lat, ids), path + ".tmp")
    os.replace(path + ".tmp", path)
    log(f"validation: {len(lat)} latents encoded and cached in {path}")
    return lat, ids


def main():
    args = resolve_defaults(get_args())
    # long timeout: at 1024 px the held-out encoding / LM states and every eval take ~10 min on rank 0 while the
    # others wait (the default 10 min killed the first 1024 px run)
    dist_to = datetime.timedelta(minutes=args.dist_timeout)  # also given to every new_group (they don't inherit it)
    dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"), timeout=dist_to)
    rank, world_all = dist.get_rank(), dist.get_world_size()
    pp = args.pp
    # training ranks first, then --teacher-ranks (klein, --opd), then the --lm-rank LM servers, then the --vae-rank
    world = world_all - args.lm_rank - args.teacher_ranks - args.vae_rank
    lm_servers = list(range(world + args.teacher_ranks, world_all - args.vae_rank))
    vae_servers = list(range(world_all - args.vae_rank, world_all))
    teachers = list(range(world, world + args.teacher_ranks)) + (lm_servers if args.teacher_on_lm else [])
    assert world % pp == 0, f"{world} training ranks are not a multiple of --pp {pp}"
    replicas = world // pp
    replica, s = divmod(rank, pp) if rank < world else (-1, 0)
    first, last = s == 0, s == pp - 1
    n_dev = max(1, torch.cuda.device_count())
    local = int(os.environ.get("LOCAL_RANK", 0)) % n_dev
    usable = [d for d in range(n_dev) if d not in args.skip_gpus]
    if args.skip_gpus:
        assert len(usable) == world_all, f"--skip-gpus {args.skip_gpus}: {len(usable)} GPUs left for {world_all} ranks"
    tail = (([args.lm_gpu] if args.lm_rank and args.lm_gpu >= 0 else [])
            + ([args.vae_gpu] if args.vae_rank else []))
    assert len(set(tail)) == len(tail), "--lm-gpu and --vae-gpu must differ"
    if tail and len(usable) == world_all and all(d in usable for d in tail):
        # the LM server(s), then the VAE server, are the last ranks: the last LM server gets GPU --lm-gpu, the VAE
        # server --vae-gpu (nvidia-smi numbering), the training ranks the other usable GPUs in order
        order = [d for d in usable if d not in tail] + tail
        local = order[int(os.environ.get("LOCAL_RANK", 0))]
    elif args.skip_gpus:
        local = usable[int(os.environ.get("LOCAL_RANK", 0))]
    if os.environ.get("PP_CPU"):  # tests/check_pp.py: gloo on CPU
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    master = rank == 0
    log = print if master else (lambda *a, **k: None)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)

    # every rank creates every group in the same order
    pipe_groups = [dist.new_group(list(range(r * pp, (r + 1) * pp)), timeout=dist_to) for r in range(replicas)]
    stage_groups = [dist.new_group([r * pp + st for r in range(replicas)], timeout=dist_to) for st in range(pp)]
    # pp > 1 from images: stage 0 loads and encodes them and hands the latents to the last stage (which needs them only
    # for the loss target), on a communicator of their own (the hand-over runs ahead of the pipeline's own p2p)
    x0_groups = ([dist.new_group([r * pp, r * pp + pp - 1], timeout=dist_to) for r in range(replicas)]
                 if args.img_dir and pp > 1 else None)
    servers = args.lm_rank or args.teacher_ranks or args.vae_rank
    control_group = dist.new_group(backend="gloo", timeout=dist_to) if servers else None
    train_group = dist.new_group(list(range(world)), timeout=dist_to) if servers else None  # None = everyone
    # Every barrier below specifies device_ids: with --lm-gpu, rank % device_count
    # may select another process's GPU and poison this group's first NCCL communicator.
    pipe_group, stage_group = (pipe_groups[replica], stage_groups[s]) if rank < world else (None, None)

    per_pipe = args.batch // replicas
    M = args.microbatches or 4 * pp
    assert per_pipe * replicas == args.batch and per_pipe % M == 0, \
        f"batch {args.batch} must split into {replicas} pipelines x {M} micro-batches"
    mb = per_pipe // M
    # inner lrs: AdamW groups scaled to the batch each optimizer sees (the pipeline's batch with MuLoCo)
    args.lr_scale = 1.0 if args.no_lr_batch_scaling else math.sqrt((per_pipe if args.diloco else args.batch) / 256)
    args.mlm_weight = 0.0

    # ---- model: a meta skeleton everywhere, only this stage's parts are materialised -------------------------------
    den_cfg, te_cfg = make_configs(args.size, args.te_size, args.te_dim, args.te_depth, args.depth,
                                   args.te_experts, args.te_quant)
    assert args.res % 16 == 0, "--res must be a multiple of 16 (one latent token per 16 px)"
    den_cfg = replace(den_cfg, latent_size=args.res // 16, rope_ref=args.rope_ref, micro_cond=args.micro_cond)
    assert not args.rope_random or (args.rope_ref >= args.res // 16 and not args.opd), \
        "--rope-random needs --rope-ref >= the grid side (the field to sample from), no --opd"
    assert not args.crop_res or (args.micro_cond and pp == 1 and not args.opd), \
        "--crop-res needs --micro-cond, --pp 1, no --opd"
    assert all(r > args.res for r in args.crop_res), "--crop-res must be larger than --res"
    assert not (args.img_dir and args.crop_res), "--crop-res crops stored latents; not with --img-dir"
    assert args.img_dir or not args.img_extra, "--img-extra needs --img-dir"
    assert not (args.img_extra and args.lm_split), "--img-extra validation sets have no --lm-split LM states"
    assert len(args.img_extra_val) == len(args.img_extra), "--img-extra-val: one held-out file per --img-extra source"
    lat_kind = "lat" if args.res == 256 else f"lat{args.res}"
    assert not args.freeze_den, "--freeze-den is only supported by train.py"
    assert not args.ema_cpu or (args.diloco and world > pp), "--ema-cpu needs --diloco with several pipelines"
    resuming = os.path.exists(os.path.join(args.out, "latest.pt"))
    init_ck, keep_te, old_depth = None, False, 0
    if args.init:  # also on resume: the text encoder config must be derived the same way every time
        init_ck = torch.load(args.init, map_location="cpu", weights_only=False, mmap=True)
        old = DenoiserConfig(**init_ck["den_config"])
        old_depth = old.depth
        init_te = te_config_from_dict(init_ck.get("te_config", {}))
        keep_te = isinstance(init_te, LMEncoderConfig) and isinstance(te_cfg, LMEncoderConfig) \
            and init_te.repo == te_cfg.repo
        if keep_te:  # the refiner comes from the checkpoint; experts / quantisation of the frozen LM may change
            te_cfg = replace(init_te, experts=te_cfg.experts if args.te_experts else init_te.experts,
                             quant=te_cfg.quant if args.te_quant is not None else init_te.quant)
        assert asdict(replace(old, depth=den_cfg.depth, latent_size=den_cfg.latent_size,
                              rope_ref=den_cfg.rope_ref, micro_cond=den_cfg.micro_cond)) == asdict(den_cfg), \
            f"--init denoiser {old} differs from {den_cfg} in more than the depth / resolution"
        assert old_depth <= den_cfg.depth, f"--init has {old_depth} blocks, more than --depth {den_cfg.depth}"
    if args.opd:
        assert pp == 1 and teachers, "--opd needs --pp 1 and --teacher-ranks (or --teacher-on-lm)"
        assert not args.teacher_on_lm or (args.lm_rank and args.klein_quant), \
            "--teacher-on-lm needs --lm-rank and --klein-quant (klein in bf16 does not fit next to the LM)"
        args.cfg_drop = 0.0  # the target is the teacher's guided velocity for the caption; no null-prompt branch
        if rank in teachers and rank not in lm_servers:
            from aurora.klein import teacher_server
            teacher_server(args.klein, list(range(replicas)), device, print, control_group, args.opd_cfg,
                           args.klein_quant, compile=not args.no_compile)
            dist.destroy_process_group()
            return
    assert not args.vae_rank or (args.lm_rank and args.img_dir and not args.vae_on_lm), \
        "--vae-rank needs --lm-rank and --img-dir, not --vae-on-lm"
    if rank in vae_servers:
        lm_server(te_cfg, [r * pp for r in range(replicas)], device, print, control_group,
                  vae=LatentEncoder(args.enc_vae, device, args.enc_chunk, compile=not args.no_compile), serve_lm=False)
        dist.destroy_process_group()
        return
    if args.lm_rank:
        assert getattr(te_cfg, "backend", "") == "hf" and not args.lm_split and not args.lm_stage0, \
            "--lm-rank needs a transformers LM text encoder (--te-size poziomka), without --lm-split / --lm-stage0"
        if rank >= world:
            k = rank - world - args.teacher_ranks
            teacher = None
            if args.opd and args.teacher_on_lm:
                from aurora.klein import KleinTeacher
                teacher = KleinTeacher(device, args.klein, quant=args.klein_quant, compile=not args.no_compile)
            vae = (LatentEncoder(args.enc_vae, device, args.enc_chunk, compile=not args.no_compile)
                   if args.vae_on_lm else None)
            lm_server(te_cfg, [r * pp for r in range(replicas) if r % args.lm_rank == k], device, print, control_group,
                      teacher=teacher, opd_cfg=args.opd_cfg, teacher_clients=list(range(replicas)) if teacher else (),
                      vae=vae)
            dist.destroy_process_group()
            return
    assert not args.vae_on_lm or (args.lm_rank and args.img_dir), "--vae-on-lm needs --lm-rank and --img-dir"
    assert M >= pp, f"need at least {pp} micro-batches"
    ranges = stage_ranges(den_cfg.depth, pp, args.te_cost if pp > 1 else 0.0)
    lo, hi = ranges[s]
    with torch.device("meta"):
        den = Denoiser(den_cfg)
    owned = [den.blocks[i] for i in range(lo, hi)]
    if first:
        owned += [den.patch_in, den.time_mlp, den.ada] + ([den.cond_mlp] if den_cfg.micro_cond else [])
    if last:
        owned += [den.norm, den.patch_out]
    for mod in owned:
        mod.to_empty(device=device)
    torch.manual_seed(args.seed * 1000 + s)  # stage-specific init; replicas are synced below
    if first:
        den.init_stem()
    for i in range(lo, hi):
        Denoiser.init_block(den.blocks[i])
    if last:
        den.init_head()
    te = None
    if first:
        if isinstance(te_cfg, LMEncoderConfig):
            from aurora.slayer import LMTextEncoder
            te = LMTextEncoder(te_cfg, with_lm=not args.lm_rank)
        else:
            te = TextEncoder(te_cfg)
        te.init_weights()
        te.to(device)
        te.checkpoint_blocks = args.te_checkpoint
    # the frozen LM (no gradients) runs on every stage: micro-batch k on stage k % pp, hidden states sent to stage 0;
    # stage 0 keeps only the trainable refiner
    # pp = 1 with a transformers LM too: computed before the step (eager, packed: aurora.slayer.hf_states_packed),
    # outside the compiled refiner, which keeps the LM out of its graph
    lm_offload = isinstance(te_cfg, LMEncoderConfig) and not args.lm_stage0 and not args.lm_rank and (
        pp > 1 or getattr(te_cfg, "backend", "") == "hf")
    lm_enc = te
    if lm_offload and not first:
        from aurora.slayer import LMTextEncoder
        lm_enc = LMTextEncoder(replace(te_cfg, depth=0)).to(device)  # just the LM, no refiner
    if args.lm_split:
        assert lm_offload and getattr(te_cfg, "backend", "") == "hf", \
            "--lm-split needs a transformers LM text encoder (--te-size poziomka), pp > 1 and no --lm-stage0"
        from aurora.slayer import split_hf_lm
        n_lm = len(lm_enc.lm.model.layers)
        lm_lo, lm_hi = s * n_lm // pp, (s + 1) * n_lm // pp
        split_hf_lm(lm_enc.lm, lm_lo, lm_hi)
        lm_hidden = lm_enc.lm.config.hidden_size

    repa_block, repa_stage, repa_proj, repa_target = -1, -1, None, None
    if args.repa_weight > 0:
        b = args.repa_layer - 1  # block index after which the state is aligned
        repa_stage = next(st for st, (a, z) in enumerate(ranges) if a <= b < z)
        # every stage decodes 1/pp of the REPA targets (micro-batch k on stage k mod pp) and sends them to the REPA
        # stage, so the expensive latent decoder doesn't stall a single GPU
        from aurora.repa import REPATarget, make_projector
        repa_target = REPATarget(args.vae, args.repa_encoder, device, spatial_norm=args.irepa)
        if s == repa_stage:
            repa_proj = make_projector(den_cfg.dim, repa_target.dim, conv=args.irepa).to(device)
            repa_block = b - lo
    stage_mod = Stage(den, den_cfg, lo, hi, first, last, te, repa_proj, repa_block, args.repa_weight,
                      args.act_checkpoint).to(device)
    den_names = owned_den_names(den, s, pp, ranges)
    den_params = dict(den.named_parameters())
    named = [("den", n, den_params[n]) for n in den_names]
    if first:
        named = [("te", n, p) for n, p in te.named_parameters()] + named
    if repa_proj is not None:
        named += [("repa_proj", n, p) for n, p in repa_proj.named_parameters()]
    params = [p for _, _, p in named if p.requires_grad]
    model_params = [p for pre, _, p in named if pre != "repa_proj"]  # EMA / checkpoints / eval
    n_den = sum(p.numel() for p in den.parameters())
    log(f"pipeline: {replicas} x {pp} stages, blocks per stage {[b - a for a, b in ranges]}, "
        f"denoiser {n_den / 1e9:.2f}B, batch {args.batch} = {replicas} x {M} micro-batches x {mb}")
    if init_ck is not None and not resuming:
        src = "EMA" if init_ck.get("ema") else "weights"
        keys = [("te", n) for n in init_ck["te"]] + [("den", n) for n in init_ck["den"]]
        if init_ck.get("ema"):
            assert len(keys) == len(init_ck["ema"]), "checkpoint EMA does not match its state dicts"
            lookup = dict(zip(keys, init_ck["ema"]))
        else:
            lookup = {**{("te", n): v for n, v in init_ck["te"].items()},
                      **{("den", n): v for n, v in init_ck["den"].items()}}
        with torch.no_grad():
            for n in den_names:
                parts = n.split(".")
                if parts[0] == "blocks" and int(parts[1]) >= old_depth:
                    continue
                if parts[0] == "cond_mlp" and ("den", n) not in lookup:  # new micro-conditioning: no-op init
                    continue
                den_params[n].copy_(lookup[("den", n)])
            for i in range(max(lo, old_depth), hi):  # appended blocks start as the identity (their gates are not 0:
                blk = den.blocks[i]                 # adaLN-single's shared projection is trained, so zero outputs)
                for lin in (blk.attn.out, blk.cross.out, blk.mlp.out_proj):
                    nn.init.zeros_(lin.weight)
            if first and keep_te:
                for n, p_ in te.named_parameters():
                    p_.copy_(lookup[("te", n)])
        log(f"init: {old_depth} blocks + stem + head{' + text encoder' if keep_te else ''} ({src}) from {args.init} "
            f"(step {init_ck.get('step')}); blocks {old_depth}..{den_cfg.depth - 1} new, zero output projections")
    del init_ck
    if replicas > 1:  # identical start on every replica
        for p in model_params + ([] if repa_proj is None else list(repa_proj.parameters())):
            dist.broadcast(p.data, s, group=stage_group)

    opts, counts = param_groups(None, None, args, named=named)
    opts = [o if not isinstance(o, Muon) else make_muon(o.param_groups, args, named, shard=False) for o in opts]
    keep_ema = replica == 0
    ema_dev = "cpu" if args.ema_cpu else None
    ema = [p.detach().clone().float().to(ema_dev or p.device) for p in model_params] if keep_ema else None
    ema_stage = ([torch.empty_like(e, pin_memory=device.type == "cuda") for e in ema]
                 if keep_ema and ema_dev == "cpu" else None)

    # ---- resume ----------------------------------------------------------------------------------------------------
    if not args.bench:
        os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "latest.pt")
    opt_path = os.path.join(args.out, f"opt_stage{s}.pt")
    step, outer_state = 0, None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        ck_cfg = DenoiserConfig(**ck["den_config"])
        if ck_cfg.rope_ref != den_cfg.rope_ref:  # a new RoPE position field (--rope-ref) for the rest of the run
            log(f"resume: RoPE field changed from rope_ref {ck_cfg.rope_ref} to {den_cfg.rope_ref}")
        assert asdict(replace(ck_cfg, rope_ref=den_cfg.rope_ref)) == asdict(den_cfg), \
            f"checkpoint was trained with {ck['den_config']}"
        assert asdict(te_config_from_dict(ck.get("te_config", {}))) == asdict(te_cfg), "different text encoder"
        with torch.no_grad():
            for n in den_names:
                den_params[n].copy_(ck["den"][n])
            if first:
                te.load_state_dict(ck["te"])
            if keep_ema:
                full = [("te", n) for n in ck["te"]] + [("den", n) for n in ck["den"]]
                lookup = dict(zip(full, ck["ema"]))
                mine = ([("te", n) for n, _ in te.named_parameters()] if first else []) + [("den", n) for n in den_names]
                for e, key in zip(ema, mine):
                    e.copy_(lookup[key])
        # the weights in latest.pt are complete (any --pp works); optimizer state is per stage, so it is only reused
        # when the stage layout is the same as when it was saved
        # on the CPU: load_state_dict copies each tensor to its parameter's device anyway, and a GPU copy held here
        # would stay allocated for the whole run (~the optimizer's size)
        os_ck = torch.load(opt_path, map_location="cpu", weights_only=True) if os.path.exists(opt_path) else None
        same = os_ck is not None and os_ck.get("ranges", [list(r) for r in ranges]) == [list(r) for r in ranges] \
            and len(os_ck["opts"]) == len(opts) and all(
                [len(g["params"]) for g in st["param_groups"]] == [len(g["params"]) for g in o.param_groups]
                for o, st in zip(opts, os_ck["opts"]))
        if same and os_ck["step"] != ck["step"]:
            assert args.resume_opt_mismatch, (
                f"{opt_path} is from step {os_ck['step']}, latest.pt from {ck['step']} (an interrupted save); "
                f"--resume-opt-mismatch uses it anyway")
            print(f"rank {rank}: WARNING {opt_path} is from step {os_ck['step']}, latest.pt from {ck['step']}; "
                  f"using it (--resume-opt-mismatch)", flush=True)
        if same:
            for o, st in zip(opts, os_ck["opts"]):
                load_opt_state(o, st)
            if repa_proj is not None and os_ck.get("repa_proj"):
                repa_proj.load_state_dict(os_ck["repa_proj"])
            outer_state = os_ck.get("outer")
        else:
            print(f"rank {rank}: WARNING {opt_path} missing or saved with a different stage layout; weights resumed, "
                  f"optimizer state starts fresh", flush=True)
        step = ck["step"]
        log(f"resumed from {ckpt_path} at step {step}")
        del ck, os_ck

    for module in stage_mod.modules():
        if isinstance(module, (CrossAttention, TextAttention)):
            module.sdpa_backend = args.masked_attention_backend
    log(f"masked attention backend: {args.masked_attention_backend}")
    if not args.no_compile:
        # torch 2.11 inductor bug: compiling a single block's backward fails at some micro-batch sizes (e.g. 8,
        # "AssertionError: 640 v.s. 5120") unless reduction splitting is off; it only affects small reductions here
        import torch._inductor.config as inductor_config
        inductor_config.split_reductions = False
        if args.act_checkpoint:  # compile through the checkpoint so the selective policy applies inside the graph
            # silence torch's one-time "context_fn ... make sure there are no in-place ops" notice (there are none)
            logging.getLogger("torch._higher_order_ops.wrap").setLevel(logging.ERROR)
            stage_mod.block_fns = [torch.compile(f, dynamic=False) for f in stage_mod.block_fns]
        else:
            for blk in stage_mod.blocks:
                blk.compile(dynamic=False)
        if first:
            te.compile(dynamic=False)
            # one graph per text width bucket and batch size (training, --opd rollout parts, evaluation chunks):
            # more than dynamo's default 8, which would leave the rest eager
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    hf_lm = getattr(te_cfg, "backend", "") == "hf"  # remote code + MoE routing: runs eager (see aurora/slayer.py)
    lm_fn = torch.compile(lm_enc.lm_states, dynamic=False) if lm_offload and not args.no_compile and not hf_lm else \
        (lm_enc.lm_states if lm_offload else None)

    outer_opt, global_params = None, None
    # outer lr 1 without momentum is plain weight averaging: new = mean of the replicas, so no global copy is needed
    # (it only serves to send the change in bf16); average the weights themselves in fp32 instead, one fp32 model less
    # in GPU memory per rank for 2x the (every-H-steps) traffic
    plain_avg = replicas > 1 and args.diloco and args.outer_lr == 1 and args.outer_momentum == 0
    if replicas > 1 and args.diloco and not plain_avg:
        global_params = [nn.Parameter(p.detach().clone().float(), requires_grad=False) for p in params]
        outer_opt = torch.optim.SGD(global_params, lr=args.outer_lr, momentum=args.outer_momentum,
                                    nesterov=args.outer_momentum > 0)
        if outer_state is not None:
            load_opt_state(outer_opt, outer_state)

    pstage = PipelineStage(stage_mod, s, pp, device, group=pipe_group) if pp > 1 else None
    # the REPA stage needs to know which micro-batch it is computing: take it from the schedule, not a counter
    # (pipelining also runs extra forwards, e.g. shape inference on the first step)
    sched = None
    if pstage is not None:
        _fwd_chunk = pstage.forward_one_chunk

        def forward_one_chunk(fwd_chunk_id, *a, **k):
            stage_mod._mb = fwd_chunk_id
            return _fwd_chunk(fwd_chunk_id, *a, **k)
        pstage.forward_one_chunk = forward_one_chunk
        loss_fn = lambda v, target: F.mse_loss(v.float(), target)
        sched = Schedule1F1B(pstage, M, loss_fn=loss_fn)

    # ---- data: only the stages that need it ------------------------------------------------------------------------
    tok_suffix = getattr(te_cfg, "tok_suffix", "")
    ids_enc = te if first else lm_enc  # null prompt row (same on every stage that reads captions)
    null_cpu = torch.from_numpy(ids_enc.null_ids()) if ids_enc is not None else None
    null_row = null_cpu.to(device) if null_cpu is not None else None
    # pp > 1 from images: the last stage gets the latents from stage 0 instead of loading (and encoding) the images too
    x0_from_first = bool(args.img_dir) and pp > 1 and args.repa_weight == 0 and not lm_offload
    needs_data = first or (last and not x0_from_first) or args.repa_weight > 0 or lm_offload  # REPA / LM states
    alt_suffix = f"{tok_suffix}_pl" if args.pl_prob > 0 else ""
    assert not alt_suffix or tok_suffix, "--pl-prob needs a frozen-LM text encoder (--te-size poziomka)"
    train_names, val_names = select_shards(args.data, args.fraction, tok_suffix, alt_suffix, img_root=args.img_dir)
    # images: every sample is a whole image, so no micro-conditions from the loader (the model's default ones)
    has_cond = bool(args.micro_cond) and not args.img_dir
    img_res = args.res if args.img_dir else 0

    def extra_names(d, held):
        """(training files, held-out file) of an --img-extra source; `held` (--img-extra-val) is kept out of training
        as that source's own validation set."""
        names = sorted(os.path.relpath(p, d)[: -len(".img.off.npy")]
                       for p in glob.glob(os.path.join(d, "*", "*.img.off.npy")))
        names = [n for n in names if all(os.path.exists(os.path.join(d, f"{n}.{k}.npy"))
                                         for k in ("off", f"tok_{tok_suffix}") + ((f"tok_{alt_suffix}",)
                                                                                   if alt_suffix else ()))]
        assert held in names, f"--img-extra-val {held}: not a file of {d} with images and tokens"
        return [n for n in names if n != held], held
    encoder = None
    loader = store = None
    if needs_data:
        null_np = ids_enc.null_ids() if ids_enc is not None else None
        crops = [("lat", "") if r == 256 else
                 (f"lat{r}", os.path.join(os.path.dirname(args.lat_dir.rstrip("/")), f"{r}px")) for r in args.crop_res]
        store = ShardStore(args.data, train_names, tok_suffix, null_np, alt_suffix, args.pl_prob,
                           "img" if img_res else lat_kind, args.img_dir or args.lat_dir, crops, args.crop_prob,
                           args.micro_cond if has_cond else 0, img_res)
        if args.img_dir:
            sources = [f"{args.img_dir}: {len(store):,}"]
            stores = [store]
            for d, held in zip(args.img_extra, args.img_extra_val):
                names, held = extra_names(d, held)
                stores.append(ShardStore(d, names, tok_suffix, null_np, alt_suffix, args.pl_prob, "img", d,
                                         img_res=img_res))
                sources.append(f"{d}: {len(stores[-1]):,} (held out for validation: {held})")
            store = ConcatStore(stores) if len(stores) > 1 else store
            encoder = LatentEncoder(args.enc_vae, device, args.enc_chunk, compile=not args.no_compile)
            log(f"images -> latents in the loop ({args.enc_vae} encoder, {args.res} px): " + ", ".join(sources))
        if crops:
            log(f"RPE-2D resize-and-crop: {100 * args.crop_prob:.0f}% of the samples {args.res} px windows of "
                f"{', '.join(f'{r} px' for r in args.crop_res)} latents ({', '.join(k for k, _ in crops)})")
        if alt_suffix:
            log(f"captions: {100 * args.pl_prob:.0f}% Polish ({alt_suffix}), the rest English")
        stream = TrainStream(store, per_pipe, replica, replicas, args.seed, start_micro=step,
                             sort_pad=LM_PAD if hf_lm else None, rows=args.opd)
        loader = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                                  pin_memory=device.type == "cuda",
                                                  prefetch_factor=2 if args.workers else None,
                                                  persistent_workers=args.workers > 0))
    n_train = len(store) if store is not None else sum(
        len(np.load(os.path.join(args.data, f"{n}.off.npy"), mmap_mode="r")) - 1 for n in train_names)
    log(f"train: {len(train_names)} shards / {n_train:,} images; {args.steps} steps x {args.batch} = "
        f"{args.steps * args.batch / n_train:.1f} epochs")
    def lm_split_forward_(chunks):
        return lm_split_forward(lm_enc.lm, chunks, s, pp, replica * pp, device, lm_hidden)

    val, val_lm = None, None
    if master and val_names:
        # images: encoded once as they are read, as stored latents would be (only the latents kept)
        enc_val = (lambda u8: encoder(u8.to(device)).half().cpu()) if img_res else None
        val = load_val_cached(args.data, val_names, args.val_samples, tok_suffix, te.null_ids(),
                              "img" if img_res else lat_kind, args.img_dir or args.lat_dir, img_res, enc_val,
                              args.enc_vae, log)
    val_shard_groups = ()
    if master and val_names and args.val_per_shard:  # fluxdb-text-00142-of-00143 -> text
        val_shard_groups = [(re.sub(r"^fluxdb-|-\d+-of-\d+$", "", s), k)
                            for s, k in val_groups(args.data, val_names, args.val_samples)]
        log("validation per shard: " + ", ".join(f"{g} {k}" for g, k in val_shard_groups))
    # --img-extra: each source's held-out file is a second validation set, logged as val_<folder>/...
    val_extra = {}
    if master and img_res:
        for d, held in zip(args.img_extra, args.img_extra_val):
            _, held = extra_names(d, held)
            v = load_val_cached(d, [held], args.val_samples, tok_suffix, te.null_ids(), "img", d, img_res, enc_val,
                                args.enc_vae, log)
            val_extra[f"val_{os.path.basename(d.rstrip('/'))}"] = [v, None]
            log(f"validation: {len(v[0])} images of {d}/{held} as val_{os.path.basename(d.rstrip('/'))}")
    if args.lm_split and replica == 0 and val_names:  # held-out LM states once, in evaluate()'s chunks
        ids_val = (val if master else load_val(args.data, val_names, args.val_samples, tok_suffix,
                                                ids_enc.null_ids(), "img" if img_res else lat_kind,
                                                args.img_dir or args.lat_dir, img_res))[1]
        ch = 64  # evaluate()'s chunk for transformers LMs
        got = lm_split_forward_([trim_ids(ids_val[i:i + ch].to(device)) for i in range(0, len(ids_val), ch)])
        if master:
            val_lm = [h.cpu() for h in got]
            log(f"--lm-split: LM layers {lm_lo}-{lm_hi - 1} of {n_lm} on this stage; held-out LM states for "
                f"{len(ids_val)} captions precomputed")
    lm_client = (LMClient(lm_servers[replica % args.lm_rank], te_cfg.lm_dim, device, control_group) if args.lm_rank and first
                 else None)
    # images -> latents on another rank: the LM server (--vae-on-lm) or the --vae-rank server
    vae_client = (lm_client if args.vae_on_lm else
                  LMClient(vae_servers[0], 0, device, control_group) if args.vae_rank and first else None)
    teacher_client = qwen_tok = captions_en = None
    if args.opd:
        from aurora.data import CaptionText
        from aurora.klein import TeacherClient, tokenizer as klein_tokenizer
        teacher_client = TeacherClient(teachers, device, control_group, shared=lm_servers if args.teacher_on_lm else ())
        qwen_tok, captions_en = klein_tokenizer(args.klein), CaptionText(args.data, train_names)
        log(f"--opd: teacher {args.klein} on ranks {teachers} (CFG {args.opd_cfg}), {args.opd_rollout_steps}-step student "
            f"trajectories, {args.opd_states} states each, {args.batch * args.opd_states} states per step")
    if lm_client is not None and master and val_names:  # held-out LM states once (one request: see lm_server)
        ids_val = val[1].to(device)
        h = lm_client.request(val[1])
        full = lm_client.collect(h)
        ch = 64  # evaluate()'s chunk for transformers LMs
        val_lm = []
        for i in range(0, len(ids_val), ch):
            w = trim_ids(ids_val[i:i + ch]).shape[1]
            val_lm.append(full[i:i + ch, :w].cpu())
        del full, h
        log(f"--lm-rank: LM on ranks {lm_servers}; held-out LM states for {len(ids_val)} captions precomputed")
        for label, entry in val_extra.items():  # the same for the extra validation sets
            ids_x = entry[0][1].to(device)
            full = lm_client.collect(lm_client.request(entry[0][1]))
            entry[1] = [full[i:i + ch, :trim_ids(ids_x[i:i + ch]).shape[1]].cpu() for i in range(0, len(ids_x), ch)]
            del full

    # FLOPs per sample (fwd + bwd), analytic: matmuls 6 N T, attention 12 L d T (T + T_text), frozen LM forward 2 N T
    T, Tt = den_cfg.latent_size ** 2, getattr(te_cfg, "max_tokens", 64)
    flops_sample = 6 * n_den * T + 12 * den_cfg.depth * den_cfg.dim * T * (T + Tt)
    if first:
        flops_sample += 6 * sum(p.numel() for p in te.parameters()) * Tt
        if getattr(te, "lm", None) is not None:
            flops_sample += 2 * sum(p.numel() for p in te.lm.parameters()) * Tt
    peak = args.peak_tflops or (peak_tflops(torch.cuda.get_device_name(device)) if device.type == "cuda" else 0)

    wb = None
    if master and args.wandb and not args.no_wandb and not args.bench:
        if wandb_logged_in():
            wb = init_wandb(args)
        else:
            log("wandb: off (not logged in; run `wandb login` to enable, or pass --no-wandb to silence this)")

    # ---- gathering to rank 0 (replica 0) for eval / checkpoints -----------------------------------------------------
    te_names = None
    if master:
        te_names = [n for n, _ in te.named_parameters()]

    def gather(tensors):
        """tensors: this rank's model_params-ordered tensors. Rank 0 gets {('te'|'den', name): cpu tensor}."""
        if replica != 0:
            return None
        if not master:
            for t_ in tensors[len(tensors) - len(den_names):]:
                dist.send(t_.detach().float().to(device).contiguous(), dst=0)  # the EMA may live on the CPU
            return None
        out = {}
        n_te = len(te_names)
        for n, t_ in zip(te_names, tensors[:n_te]):
            out[("te", n)] = t_.detach().float().cpu()
        for n, t_ in zip(den_names, tensors[n_te:]):
            out[("den", n)] = t_.detach().float().cpu()
        for src in range(1, pp):
            for n in owned_den_names(den, src, pp, ranges):
                buf = torch.empty(den_params[n].shape, dtype=torch.float32, device=device)
                dist.recv(buf, src=src)
                out[("den", n)] = buf.cpu()
        return out

    def save(merge_only=False):
        raw = gather([p for p in model_params])
        emas = None if merge_only else gather(ema if keep_ema else model_params)
        if master:
            te_sd = {n: raw[("te", n)] for n in te_names}
            den_sd = {n: raw[("den", n)] for n, _ in den.named_parameters()}
            meta = dict(step=step, den_config=asdict(den_cfg), te_config=asdict(te_cfg), te=te_sd, den=den_sd)
            if merge_only:
                d = os.path.join(args.out, "merge")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"step_{step:07d}.pt")
                torch.save(dict(meta, total_steps=args.steps), path + ".tmp")
                os.replace(path + ".tmp", path)
                return
        if merge_only:
            return
        # two phases: every file is written to .tmp first, and only when all of them are complete are they renamed,
        # so a failed write (e.g. a full disk) leaves the previous optimizer states and latest.pt together
        if replica == 0:  # per-stage optimizer state
            torch.save(dict(step=step, ranges=[list(r) for r in ranges], opts=[o.state_dict() for o in opts],
                            outer=outer_opt.state_dict() if outer_opt is not None else None,
                            repa_proj={k: v.cpu() for k, v in repa_proj.state_dict().items()} if repa_proj else None),
                       opt_path + ".tmp")
        if master:  # latest.pt (the step marker)
            ema_list = [emas[("te", n)] for n in te_names] + [emas[("den", n)] for n, _ in den.named_parameters()]
            torch.save(dict(meta, ema=ema_list, args=vars(args), opts=None), ckpt_path + ".tmp")
        dist.barrier(group=train_group, device_ids=[device.index] if device.type == "cuda" else None)
        if replica == 0:
            os.replace(opt_path + ".tmp", opt_path)
        if master:
            os.replace(ckpt_path + ".tmp", ckpt_path)

    @torch.no_grad()
    def ema_eval():
        emas = gather(ema)
        if not master:
            return None
        with torch.device("meta"):
            den_eval = Denoiser(den_cfg)
        den_eval.to_empty(device=device)
        den_eval.load_state_dict({n: emas[("den", n)] for n, _ in den_eval.named_parameters()})
        cos, sin = build_rope_2d(den_cfg.latent_size, den_cfg.dim // den_cfg.heads, den_cfg.rope_ref)
        den_eval.rope_cos, den_eval.rope_sin = cos.to(device), sin.to(device)
        te_live = [p.detach().clone() for p in te.parameters()]
        torch._foreach_copy_(list(te.parameters()), [emas[("te", n)].to(device) for n in te_names])
        dc = max(1, 16384 // den_cfg.latent_size ** 2)  # denoiser images per call: 4 at 1024 px, 64 at 256 px
        res = evaluate(te, den_eval, val, device, lm_states=val_lm, groups=val_shard_groups, den_chunk=dc)
        if val_extra:  # every source on its own, and val_all/: all held-out images together (size-weighted)
            # only the val/ keys: val_<shard>/ (--val-per-shard) has no counterpart in the other sources
            n_all, sums = len(val[0]), {k: x * len(val[0]) for k, x in res.items() if k.startswith("val/")}
            for label, (v, v_lm) in val_extra.items():
                got = evaluate(te, den_eval, v, device, lm_states=v_lm, den_chunk=dc)
                res.update({k.replace("val/", f"{label}/", 1): x for k, x in got.items()})
                n_all += len(v[0])
                for k, x in got.items():
                    sums[k] += x * len(v[0])
            res.update({k.replace("val/", "val_all/", 1): s / n_all for k, s in sums.items()})
        torch._foreach_copy_(list(te.parameters()), te_live)
        del den_eval
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    warmup = max(1, int(args.warmup_frac * args.steps))
    repa_steps = int(args.repa_until * args.steps) if args.repa_weight > 0 else 0
    if args.repa_weight > 0:
        log(f"REPA{' (iREPA)' if args.irepa else ''}: {args.repa_encoder} <- DiT block {args.repa_layer} "
            f"(stage {repa_stage}), {args.repa_frac:.0%} of each micro-batch, every {args.repa_every} step(s), until "
            f"step {repa_steps}; decoding spread over all {pp} stages")
    log(f"resolution {args.res} px ({den_cfg.latent_size}x{den_cfg.latent_size} latents, {lat_kind}), t shift {args.t_shift:.2f}")
    if args.rope_random or args.micro_cond:
        whole = default_micro_cond(args.micro_cond, args.res, 1)[0].int().tolist() if args.micro_cond else None
        mc = ("off" if not args.micro_cond else
              f"from the loader (crops of a {args.micro_cond} px original, else {whole})" if has_cond else
              f"every sample a whole image: {whole} [orig h, w, crop t, l, b, r, target h, w], as at sampling")
        log(f"RPE-2D: {'random positions of 0..' + str(args.rope_ref - 1) if args.rope_random else 'even positions'}"
            f", micro-conditioning {mc}")
    log(f"lr: {args.schedule} after {warmup} warmup steps, Muon {args.lr_muon}, AdamW x{args.lr_scale:.2f}, "
        f"EMA {args.ema:.5f}; sync: {'MuLoCo every %d steps%s' % (args.diloco, ' (plain averaging, no global copy)' if plain_avg else '') if replicas > 1 and args.diloco else 'gradients every step' if replicas > 1 else 'single pipeline'}")
    acc = torch.zeros(4, device=device)  # sum loss, n, sum repa, n repa
    t0, t_last = time.time(), step
    GB = 2 ** 30
    nbytes = lambda ts: sum(t.numel() * t.element_size() for t in ts if t is not None and t.is_cuda)

    def mem_report(tag):
        """This rank's memory: what is resident between steps, and the peak that the step's activations add."""
        if device.type != "cuda":
            return
        ms = torch.cuda.memory_stats(device)
        trainable = [p for _, _, p in named if p.requires_grad]
        parts = {
            "weights": nbytes(p for _, _, p in named),
            "grads (in backward)": nbytes(trainable),
            "optimizer": nbytes(v for o in opts for st in o.state.values() for v in st.values()
                                if torch.is_tensor(v)),
            "EMA": nbytes(ema or []),
            "MuLoCo global copy": nbytes(global_params or []),
            "frozen LM": nbytes([*lm_mod.parameters(), *lm_mod.buffers()]) if lm_mod is not None else 0,
        }
        static = sum(parts.values())
        peak = ms.get("allocated_bytes.all.peak", 0)
        line = " | ".join(f"{k} {v / GB:.2f}" for k, v in parts.items() if v)
        print(f"[mem {tag}] rank {rank} stage {s} blocks {lo}-{hi}: now {ms.get('allocated_bytes.all.current', 0) / GB:.2f}"
              f" GB, peak {peak / GB:.2f} GB, reserved {ms.get('reserved_bytes.all.current', 0) / GB:.2f} GB, "
              f"alloc retries {ms.get('num_alloc_retries', 0)} || {line} || peak - these = activations + temporaries "
              f"{(peak - static) / GB:.2f} GB", flush=True)

    def mem_report_all(tag):
        for r in range(world):  # one rank after another, so the lines don't interleave
            if r == rank:
                mem_report(tag)
            dist.barrier(group=train_group, device_ids=[device.index] if device.type == "cuda" else None)

    lm_owner = te if first else lm_enc
    lm_mod = getattr(lm_owner, "lm", None) if lm_owner is not None else None
    if args.mem_snapshot and device.type == "cuda":
        os.makedirs(args.mem_snapshot, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    def dump_snapshot(tag):
        if args.mem_snapshot and device.type == "cuda":
            path = os.path.join(args.mem_snapshot, f"rank{rank}.pickle")
            torch.cuda.memory._dump_snapshot(path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"[mem {tag}] rank {rank}: allocator history -> {path}", flush=True)

    if device.type == "cuda":  # the peak should be the training step's, not the one from loading (e.g. LM -> int8)
        torch.cuda.reset_peak_memory_stats(device)
    def next_batch(at):
        """The pipeline's batch of step `at` (latents, t, noise, captions with CFG drop); None on stages without data."""
        if not needs_data:
            if not (x0_from_first and last):
                return None, None, None, None, None, None, None, None
            # x0 itself comes from stage 0 right before this step's pipeline (see the main loop)
            shape = (per_pipe, den_cfg.latent_channels, den_cfg.latent_size, den_cfg.latent_size)
            g = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + replica * 1009 + at)
            t = torch.sigmoid(torch.randn(per_pipe, device=device, generator=g))  # the same t and noise as stage 0
            if args.t_shift != 1.0:
                t = args.t_shift * t / (1 + (args.t_shift - 1) * t)
            eps = torch.randn(shape, device=device, generator=g)
            return torch.empty(shape, device=device), None, t, eps, None, None, None, None
        with profile.phase("data_loader"):
            got = next(loader)  # with a transformers LM: sorted by caption length in the loader (TrainStream)
            mcond = got[-1].to(device, non_blocking=True) if has_cond else None
            got = got[:-1] if has_cond else got
            lat, ids, rows = got if args.opd else (*got, None)
        ids_cpu = ids
        shape = (per_pipe, den_cfg.latent_channels, den_cfg.latent_size, den_cfg.latent_size)
        with profile.phase("vae_encode"):
            if vae_client is not None and encoder is not None and round(args.vae_remote_frac * per_pipe) > 0:
                # a handle (the first k images on the server, the rest here); the main loop collects it
                u8 = lat.to(device, non_blocking=True)
                k = round(args.vae_remote_frac * per_pipe)
                x0 = vae_client.request_vae(u8[:k], (k, *shape[1:]))
                x0["local"] = encoder(u8[k:]) if k < per_pipe else None
            elif encoder is not None:
                x0 = encoder(lat.to(device, non_blocking=True))
            else:
                x0 = lat.to(device, non_blocking=True).float()
        ids = ids.to(device, non_blocking=True)
        g = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + replica * 1009 + at)
        t = torch.sigmoid(torch.randn(per_pipe, device=device, generator=g))
        if args.t_shift != 1.0:
            t = args.t_shift * t / (1 + (args.t_shift - 1) * t)
        eps = torch.randn(shape if isinstance(x0, dict) else x0.shape, device=device, generator=g)
        # caption dropout drawn on the CPU (same seed), so the CPU copy of the captions needs no GPU -> CPU sync
        g_cpu = torch.Generator().manual_seed(args.seed * 1_000_003 + replica * 1009 + at)
        drop_cpu = torch.rand(per_pipe, generator=g_cpu) < args.cfg_drop
        drop = drop_cpu.to(device, non_blocking=True)
        cond = torch.where(drop[:, None], null_row, ids) if null_row is not None else None
        cond_cpu = (torch.where(drop_cpu[:, None], null_cpu, ids_cpu)
                    if hf_lm and null_row is not None else None)
        return x0, ids, t, eps, cond, cond_cpu, rows, mcond

    profile = TrainingProfile(args.profile_dir, rank, args.profile_ranks, args.profile_start, args.profile_steps,
                              device.type == "cuda", metadata={"args": vars(args), "den_config": asdict(den_cfg),
                                                               "te_config": asdict(te_cfg), "rank": rank})
    opt_window = []  # keep every step in the logging window, including cheap periodic-Muon steps
    pending = None  # --lm-rank: the next step's batch and its LM request, sent while the current step trains
    # all training ranks enter step 1 together: stages without data / validation work would otherwise start the
    # pipeline's first p2p (NCCL communicator setup) long before stage 0 is ready
    dist.barrier(group=train_group, device_ids=[device.index] if device.type == "cuda" else None)
    while step < args.steps:
        profile.begin_step(step + 1)
        if lm_client is not None:
            if pending is None:
                nb = next_batch(step)
                with profile.phase("lm_request"):
                    pending = (nb, lm_client.request(nb[5]))
            (x0, ids, t, eps, cond_ids, cond_cpu, rows, mcond), req = pending
            if isinstance(x0, dict):  # --vae-on-lm: this step's latents from the LM server
                with profile.phase("vae_wait"):
                    x0 = LMClient.vae_collect(x0)
            with profile.phase("lm_wait_collect"):
                lm_full = lm_client.collect(req)
            del req
            pending = None
            if step + 1 < args.steps:
                nb = next_batch(step + 1)
                with profile.phase("lm_request"):
                    pending = (nb, lm_client.request(nb[5]))
        else:
            x0, ids, t, eps, cond_ids, cond_cpu, rows, mcond = next_batch(step)
        stage_mod.lm_states = None
        if lm_client is not None:
            stage_mod.lm_states = [lm_full[k * mb:(k + 1) * mb] for k in range(M)]
            del lm_full
        if lm_offload and args.lm_split:  # whole step's captions through the stages' LM slices, ~32 per chunk
            per = max(1, 32 // mb)
            groups = [list(range(i, min(M, i + per))) for i in range(0, M, per)]
            got = lm_split_forward_([torch.cat([cond_ids[k * mb:(k + 1) * mb] for k in g]) for g in groups])
            if got is not None:
                stage_mod.lm_states = [h[j * mb:(j + 1) * mb] for g, h in zip(groups, got) for j in range(len(g))]
        elif lm_offload:
            mine = {}
            ks = [k for k in range(M) if k % pp == s]
            if hf_lm and pp == 1:  # the whole batch in one packed LM call (hf_states_packed cuts it to ~8k tokens)
                with profile.phase("lm_local"):
                    h = lm_fn(cond_ids)
                mine = {k: h[k * mb:(k + 1) * mb] for k in range(M)}
            elif hf_lm:  # ~32 captions per LM call (the MoE's sweet spot; micro-batches of similar length go together)
                per = max(1, 32 // mb)
                for i in range(0, len(ks), per):
                    group = ks[i:i + per]
                    h = lm_fn(torch.cat([cond_ids[k * mb:(k + 1) * mb] for k in group]))
                    for j, k in enumerate(group):
                        mine[k] = h[j * mb:(j + 1) * mb]
            else:
                for k in ks:
                    with amp:
                        mine[k] = lm_fn(cond_ids[k * mb:(k + 1) * mb])
            stage_mod.lm_states = gather_to_stage(mine, M, s, pp, 0, replica * pp)
        stage_mod.repa_targets = None
        if step < repa_steps and step % args.repa_every == 0:
            n_repa = max(1, round(args.repa_frac * mb))
            mine = {k: repa_target(x0[k * mb: k * mb + n_repa]).bfloat16() for k in range(M) if k % pp == s}
            got = gather_to_stage(mine, M, s, pp, repa_stage, replica * pp)
            stage_mod.repa_targets = [g_.float() for g_ in got] if got is not None else None
        stage_mod.repa_losses = []
        losses = []
        with profile.phase("forward_backward"):
            # RPE-2D positions: the same seed on every stage of this replica (pipelined stages draw them per micro-batch)
            rope_seed = args.seed * 1_000_003 + replica * 1009 + step + 13 if args.rope_random else None
            if pp > 1 and args.rope_random:
                stage_mod.rope_field, stage_mod.rope_seed = args.rope_ref, rope_seed
            if x0_from_first and (first or last):  # this step's latents, stage 0 -> last stage, right before the
                # pipeline: blocking and in the same order on both (an async send of the prefetched step + 1 next to
                # the pipeline's own p2p deadlocked NCCL)
                if first:
                    dist.send(x0, dst=replica * pp + pp - 1, group=x0_groups[replica])
                else:
                    dist.recv(x0, src=replica * pp, group=x0_groups[replica])
            try:
                with amp:
                    if first:
                        tt = t.view(-1, 1, 1, 1)
                        z_t = (1 - tt) * x0 + tt * eps
                        if args.opd:
                            opd_step(stage_mod, teacher_client, qwen_tok, captions_en(rows.tolist()), eps, cond_ids,
                                     cond_cpu, mb, args, step, replica, losses)
                        elif pp == 1:
                            widths = (text_widths(cond_cpu, mb, args.text_bucket)
                                      if cond_cpu is not None and args.text_bucket else None)
                            local_train_step(stage_mod, z_t, t, cond_ids, eps - x0, mb, losses, widths,
                                             cond=mcond, rope_field=args.rope_ref if args.rope_random else 0,
                                             rope_seed=rope_seed)
                        elif last:
                            sched.step(z_t, t, cond_ids, target=eps - x0, losses=losses, return_outputs=False)
                        else:
                            sched.step(z_t, t, cond_ids, return_outputs=False)
                    elif last:
                        sched.step(target=eps - x0, losses=losses, return_outputs=False)
                    else:
                        sched.step(return_outputs=False)
            except torch.OutOfMemoryError:
                mem_report(f"OOM at step {step + 1}")  # no barrier: the other ranks may be stuck in communication
                dump_snapshot("OOM")
                raise
            finally:
                stage_mod.rope_field, stage_mod.rope_seed = 0, None
        stage_mod.lm_states = None  # backward has finished; release current LM states before optimizer temporaries
        if losses:
            acc[0] += torch.stack([l.detach().float() for l in losses]).mean()
            acc[1] += 1
        if stage_mod.repa_losses:
            acc[2] += torch.stack(stage_mod.repa_losses).mean()
            acc[3] += 1

        with profile.phase("gradient_sync_clip"):
            if replicas > 1 and not args.diloco:  # plain data parallel across pipelines
                grads = [p.grad for p in params if p.grad is not None]
                flat = torch.cat([gr.flatten() for gr in grads]).bfloat16()
                dist.all_reduce(flat, group=stage_group)
                flat = flat.float() / replicas
                for gr, d in zip(grads, flat.split([gr.numel() for gr in grads])):
                    gr.copy_(d.view_as(gr))
            grads = [p.grad for p in params if p.grad is not None]
            sq = torch.stack(torch._foreach_norm(grads)).float().square().sum()
            if pp > 1:
                dist.all_reduce(sq, group=pipe_group)
            gnorm = sq.sqrt()
            coef = (args.clip / (gnorm + 1e-6)).clamp(max=1.0)
            torch._foreach_mul_(grads, coef)
            del grads
        mult = lr_mult(step, args.steps, warmup, args.schedule)
        if device.type == "cuda":
            opt_ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            opt_window.append(opt_ev)
            opt_ev[0].record()
        with profile.phase("optimizer"):
            try:
                for o in opts:
                    for gr in o.param_groups:
                        gr["lr"] = gr["base_lr"] * mult
                    o.step()
                    o.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    opt_ev[1].record()
            except torch.OutOfMemoryError:
                mem_report(f"OOM in the optimizer step {step + 1}")
                dump_snapshot("OOM")
                raise
        step += 1
        if args.mem_debug and step <= 3:
            mem_report_all(f"step {step}")
        if step == 2:
            dump_snapshot("step 2")
        with profile.phase("outer_sync_ema"):
            if outer_opt is not None or plain_avg:
                if step % args.diloco == 0 or step == args.steps:
                    if plain_avg:
                        average_params(params, replicas, stage_group)
                    else:
                        outer_sync(global_params, params, outer_opt, replicas, args.outer_comm, stage_group)
                    if keep_ema:
                        n_sync = step // args.diloco
                        d = min(args.ema ** args.diloco, (1 + n_sync) / (10 + n_sync))
                        ema_update_(ema, model_params, 1 - d, ema_stage)
            elif keep_ema:
                d = min(args.ema, step / (9 + step))
                torch._foreach_lerp_(ema, [p.detach().float() for p in model_params], 1 - d)

        if step % args.log_every == 0 or step == args.steps:
            dist.all_reduce(acc, group=train_group)
            n = step - t_last
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # else the CPU runs ahead and short windows look too fast
            opt_times = [a.elapsed_time(b) for a, b in opt_window]
            opt_ms = sum(opt_times) / len(opt_times) if opt_times else 0.0
            opt_max_ms = max(opt_times, default=0.0)
            opt_window.clear()
            stats = {"opt_ms": opt_ms, "opt_max_ms": opt_max_ms, "train/samples": step * args.batch, "train/loss_fm": (acc[0] / acc[1].clamp(min=1)).item(),
                     "train/loss_repa": (acc[2] / acc[3].clamp(min=1)).item(), "train/grad_norm": gnorm.item(),
                     "lr_mult": mult}
            mfu = ""
            repa_s = f"repa {stats['train/loss_repa']:.3f} | " if acc[3] > 0 else ""
            if replica == 0:
                wsq = weight_sq(named, device)
                if pp > 1:
                    dist.all_reduce(wsq, group=pipe_group)
                if master:
                    stats.update(wrms_stats(wsq))
            # peak memory and allocator retries (cudaFree + retry = memory pressure, stalls the whole pipeline)
            mem_s = ""
            if device.type == "cuda":
                ms = torch.cuda.memory_stats(device)
                mine_mem = torch.tensor([ms.get("allocated_bytes.all.peak", 0) / 2**30, ms.get("num_alloc_retries", 0), opt_ms, opt_max_ms,
                                         host_rss_gb()], device=device)
                all_mem = [torch.zeros_like(mine_mem) for _ in range(world)]
                dist.all_gather(all_mem, mine_mem, group=train_group)
                all_mem = torch.stack(all_mem).cpu()
                worst = int(all_mem[:, 0].argmax())
                retries = int(all_mem[:, 1].sum())
                stats["opt_slowest_rank_ms"] = all_mem[:, 2].max().item()
                stats["opt_global_max_ms"] = all_mem[:, 3].max().item()
                stats.update({"mem_peak_gb": all_mem[worst, 0].item(), "alloc_retries": retries})
                mem_s = f" | mem {all_mem[worst, 0]:.1f} GB (GPU {worst}){f', ALLOC RETRIES {retries}' if retries else ''}"
                used, total = host_ram_used_gb()
                big = int(all_mem[:, 4].argmax())
                stats.update({"host_ram_used_gb": used, "host_rss_max_rank_gb": all_mem[big, 4].item(),
                              "host_rss_ranks_gb": all_mem[:, 4].sum().item()})
                mem_s += (f" | RAM {used:.0f}/{total:.0f} GB, ranks+workers {all_mem[:, 4].sum():.0f} GB "
                          f"(max {all_mem[big, 4]:.1f} GB, rank {big})")
            # Include statistics / communication in the window. The timestamp is taken before print;
            # printing, W&B and any following eval/checkpoint are included in the next window.
            stamp = time.time()
            dt = stamp - t0
            stats.update(steps_per_s=n / dt, imgs_per_s=n * args.batch / dt,
                         tflops_per_gpu=flops_sample * args.batch * n / dt / world / 1e12)
            if peak:
                stats["mfu"] = stats["tflops_per_gpu"] / peak
                mfu = f" MFU {stats['mfu']:.0%}"
            log(f"step {step:>7} | fm {stats['train/loss_fm']:.4f} | {repa_s}grad_norm {stats['train/grad_norm']:.3f} | "
                f"{f'lr x{mult:.3f} | ' if mult != 1.0 else ''}{stats['imgs_per_s']:.0f} img/s | "
                f"~{stats['tflops_per_gpu']:.1f} TFLOPS/GPU{mfu} | opt avg {opt_ms:.0f} ms | eta {(args.steps - step) / stats['steps_per_s'] / 3600:.1f} h{mem_s}",
                flush=True)
            if wb:
                wb.log(stats, step=step)
            acc.zero_()
            t0, t_last = stamp, step

        profile.end_step(step)
        if args.bench:
            continue
        if step % args.ckpt_every == 0 or step == args.steps:  # before the eval: an eval that fails loses no steps
            save()
        if step % args.eval_every == 0 or step == args.steps:
            res = ema_eval()
            if master and res is not None:
                log(f"step {step:>7} | EMA " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
                if wb:
                    wb.log({**res, "train/samples": step * args.batch}, step=step)
            dist.barrier(group=train_group, device_ids=[device.index] if device.type == "cuda" else None)
        if args.merge_every and step >= args.merge_start * args.steps and (
                step % args.merge_every == 0 or step == args.steps):
            save(merge_only=True)
            dist.barrier(group=train_group, device_ids=[device.index] if device.type == "cuda" else None)

    profile.close()
    if lm_client is not None:
        lm_client.stop()
    if vae_client is not None and vae_client is not lm_client:
        vae_client.stop()
    if teacher_client is not None:
        teacher_client.stop()
    log("done")
    if wb:
        wb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
