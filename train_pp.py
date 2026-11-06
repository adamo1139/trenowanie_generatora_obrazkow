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
import functools
import logging
import math
import os
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from aurora.data import ShardStore, TrainStream, load_val, select_shards
from aurora.model import (Denoiser, DenoiserConfig, LMEncoderConfig, TextEncoder, build_rope_2d, make_configs,
                          te_config_from_dict, timestep_embedding)
from aurora.muon import Muon
from aurora.slayer import LM_PAD, trim_ids
from train import (build_parser, evaluate, init_wandb, load_opt_state, lr_mult, make_muon, outer_sync, param_groups,
                   peak_tflops, resolve_defaults, wandb_logged_in, weight_sq, wrms_stats)


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
    p.add_argument("--lm-split", action="store_true",
                   help="frozen transformers LM (Poziomka) split by layers over the pipeline's stages: each GPU holds "
                        "1/pp of it, the hidden states go stage 0 -> 1 -> ... -> back to stage 0 (chunks of ~32 "
                        "captions, overlapped); the held-out set's LM states are computed once at the start")
    p.add_argument("--te-checkpoint", action="store_true",
                   help="activation checkpointing of the text encoder's refiner blocks (stage 0): stage 0 holds the most "
                        "micro-batches in flight under 1F1B; PP=1 trims text using --text-bucket")
    p.add_argument("--text-bucket", type=int, choices=[0, 64, 128, 256], default=64,
                   help="PP=1 HF text widths rounded to this bucket (default 64; 0 disables trimming). "
                        "With static compilation each new width needs its own forward/backward warmup; "
                        "benchmark only after all encountered widths have compiled.")
    p.add_argument("--mem-debug", action="store_true",
                   help="print every rank's GPU memory breakdown (weights, grads, optimizer, EMA, MuLoCo copy, frozen LM, "
                        "peak activations) after steps 1-3; the same report is always printed on an out-of-memory error")
    p.add_argument("--mem-snapshot", default="",
                   help="dir: record the CUDA allocator history and dump <dir>/rank<r>.pickle after step 2 or on OOM "
                        "(open in https://pytorch.org/memory_viz)")
    p.add_argument("--bench", action="store_true",
                   help="throughput test: no checkpoints, no merge files, no evaluation (nothing is written to --out)")
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


class Stage(nn.Module):
    """One pipeline stage: optional stem (text encoder, time projection, patch_in), blocks [lo, hi), optional head."""

    def __init__(self, den, cfg, lo, hi, first, last, te=None, repa_proj=None, repa_block=-1, repa_weight=0.0,
                 act_checkpoint=None):
        super().__init__()
        self.cfg, self.first, self.last = cfg, first, last
        if first:
            self.te, self.patch_in, self.time_mlp, self.ada = te, den.patch_in, den.time_mlp, den.ada
        self.blocks = nn.ModuleList(den.blocks[lo:hi])
        if last:
            self.norm, self.patch_out = den.norm, den.patch_out
        self.repa_proj, self.repa_block, self.repa_weight = repa_proj, repa_block, repa_weight
        # callables that run each block (plain, or wrapped in activation checkpointing); main() compiles these
        self.block_fns = [checkpointed(b, act_checkpoint) if act_checkpoint else b for b in self.blocks]
        cos, sin = build_rope_2d(cfg.latent_size, cfg.dim // cfg.heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.repa_targets, self.repa_losses, self._mb = None, [], 0
        self.lm_states = None  # frozen-LM hidden states per micro-batch, computed on all stages (see main)

    def forward(self, *args):
        if self.first:
            z, t, ids = args
            if self.lm_states is not None and not ids.is_meta:
                text, valid = self.te(ids, lm_states=self.lm_states[self._mb])
            else:
                text, valid = self.te(ids)
            validf = valid.to(text.dtype)
            x = self.patch_in(z.flatten(2).transpose(1, 2))
            t6 = self.ada(F.silu(self.time_mlp(timestep_embedding(t, 256)))).view(z.shape[0], 6, self.cfg.dim)
        else:
            x, t6, text, validf = args
            # pipelining sends a gradient back for every stage input; the mask needs a (zero) one
            x = x + 0 * validf.sum()
        valid = validf > 0.5
        for i, fn in enumerate(self.block_fns):
            x = fn(x, t6, text, valid, self.rope_cos, self.rope_sin)
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


def lm_server(te_cfg, clients, device, log, control_group, lm=None):
    """CPU/Gloo any-source control queue; hidden states still travel directly over NCCL.

    The optional lm argument is a small fake encoder in the distributed protocol test.
    """
    from aurora.slayer import LM_NULL, LMTextEncoder, hf_lm_part
    lm = LMTextEncoder(te_cfg).to(device).lm if lm is None else lm
    log(f"LM server: {te_cfg.repo} for ranks {clients}", flush=True)
    active, served, t0 = set(clients), 0, time.perf_counter()
    idle = compute = transfer = 0.0
    win_t, win_n = t0, 0
    while active:
        hdr = torch.empty(3, dtype=torch.long)
        start = time.perf_counter()
        r = dist.recv(hdr, group=control_group, tag=1)  # any ready client, not a fixed round-robin slot
        assert r in active, f"unexpected LM client {r}"
        if hdr[0].item() == 0:
            active.remove(r)
            idle += time.perf_counter() - start
            continue
        ids = torch.empty(int(hdr[1]), int(hdr[2]), dtype=torch.long)
        dist.recv(ids, src=r, group=control_group, tag=2)
        idle += time.perf_counter() - start
        start = time.perf_counter()
        sends, events = [], []
        for _, _, sub in lm_chunk_plan(ids):
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
                f"{win_n / dt:.0f} captions/s", flush=True)
            idle = compute = transfer = 0.0
            win_t, win_n = time.perf_counter(), 0
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

    def stop(self):
        dist.send(torch.tensor([0, 0, 0]), dst=self.server, group=self.control_group, tag=1)


def text_widths(ids_cpu, microbatch, multiple=64):
    """A small set of compile shapes for PP=1; remove only trailing LM_PAD, never valid text."""
    assert ids_cpu.device.type == "cpu"
    return [trim_ids(c, multiple=multiple).shape[1] for c in ids_cpu.split(microbatch)]


def local_train_step(stage, z, t, ids, target, microbatch, losses, widths=None):
    """PP=1: immediate backward, no pipeline output retention, variable text widths per microbatch."""
    count = z.shape[0] // microbatch
    states = stage.lm_states
    try:
        for k in range(count):
            sl = slice(k * microbatch, (k + 1) * microbatch)
            width = widths[k] if widths is not None else ids.shape[1]
            stage._mb = k
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
        grads = [p.grad for p in stage.parameters() if p.grad is not None]
        torch._foreach_div_(grads, count)
    finally:
        stage.lm_states = states


def owned_den_names(den, s, pp, ranges):
    lo, hi = ranges[s]
    out = []
    for name, _ in den.named_parameters():
        head = name.split(".")[0]
        if head == "blocks":
            if lo <= int(name.split(".")[1]) < hi:
                out.append(name)
        elif head in ("patch_in", "time_mlp", "ada"):
            if s == 0:
                out.append(name)
        elif head in ("norm", "patch_out"):
            if s == pp - 1:
                out.append(name)
        else:
            raise ValueError(f"unassigned denoiser parameter {name}")
    return out


def main():
    args = resolve_defaults(get_args())
    dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"))
    rank, world_all = dist.get_rank(), dist.get_world_size()
    pp = args.pp
    world = world_all - args.lm_rank  # training ranks; with --lm-rank N the last N serve the LM
    lm_servers = list(range(world, world_all))
    assert world % pp == 0, f"{world} training ranks are not a multiple of --pp {pp}"
    replicas = world // pp
    replica, s = divmod(rank, pp) if rank < world else (-1, 0)
    first, last = s == 0, s == pp - 1
    local = int(os.environ.get("LOCAL_RANK", 0)) % max(1, torch.cuda.device_count())
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
    pipe_groups = [dist.new_group(list(range(r * pp, (r + 1) * pp))) for r in range(replicas)]
    stage_groups = [dist.new_group([r * pp + st for r in range(replicas)]) for st in range(pp)]
    control_group = dist.new_group(backend="gloo") if args.lm_rank else None
    train_group = dist.new_group(list(range(world))) if args.lm_rank else None  # None = everyone
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
        assert asdict(replace(old, depth=den_cfg.depth)) == asdict(den_cfg), \
            f"--init denoiser {old} differs from {den_cfg} in more than the depth"
        assert old_depth <= den_cfg.depth, f"--init has {old_depth} blocks, more than --depth {den_cfg.depth}"
    if args.lm_rank:
        assert getattr(te_cfg, "backend", "") == "hf" and not args.lm_split and not args.lm_stage0, \
            "--lm-rank needs a transformers LM text encoder (--te-size poziomka), without --lm-split / --lm-stage0"
        if rank >= world:
            k = rank - world
            lm_server(te_cfg, [r * pp for r in range(replicas) if r % args.lm_rank == k], device, print, control_group)
            dist.destroy_process_group()
            return
    assert M >= pp, f"need at least {pp} micro-batches"
    ranges = stage_ranges(den_cfg.depth, pp, args.te_cost if pp > 1 else 0.0)
    lo, hi = ranges[s]
    with torch.device("meta"):
        den = Denoiser(den_cfg)
    owned = [den.blocks[i] for i in range(lo, hi)]
    if first:
        owned += [den.patch_in, den.time_mlp, den.ada]
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
    lm_offload = isinstance(te_cfg, LMEncoderConfig) and pp > 1 and not args.lm_stage0 and not args.lm_rank
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

    # ---- resume ----------------------------------------------------------------------------------------------------
    if not args.bench:
        os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "latest.pt")
    opt_path = os.path.join(args.out, f"opt_stage{s}.pt")
    step, outer_state = 0, None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        assert ck["den_config"] == asdict(den_cfg), f"checkpoint was trained with {ck['den_config']}"
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
        os_ck = torch.load(opt_path, map_location=device, weights_only=True) if os.path.exists(opt_path) else None
        same = os_ck is not None and os_ck.get("ranges", [list(r) for r in ranges]) == [list(r) for r in ranges] \
            and len(os_ck["opts"]) == len(opts) and all(
                [len(g["params"]) for g in st["param_groups"]] == [len(g["params"]) for g in o.param_groups]
                for o, st in zip(opts, os_ck["opts"]))
        if same:
            assert os_ck["step"] == ck["step"], f"{opt_path} is from step {os_ck['step']}, latest.pt from {ck['step']}"
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
        del ck

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
    needs_data = first or last or args.repa_weight > 0 or lm_offload  # REPA targets / LM states: every stage
    alt_suffix = f"{tok_suffix}_pl" if args.pl_prob > 0 else ""
    assert not alt_suffix or tok_suffix, "--pl-prob needs a frozen-LM text encoder (--te-size poziomka)"
    train_names, val_names = select_shards(args.data, args.fraction, tok_suffix, alt_suffix)
    loader = store = None
    if needs_data:
        store = ShardStore(args.data, train_names, tok_suffix, ids_enc.null_ids() if ids_enc is not None else None,
                           alt_suffix, args.pl_prob)
        if alt_suffix:
            log(f"captions: {100 * args.pl_prob:.0f}% Polish ({alt_suffix}), the rest English")
        stream = TrainStream(store, per_pipe, replica, replicas, args.seed, start_micro=step,
                             sort_pad=LM_PAD if hf_lm else None)
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
        val = load_val(args.data, val_names, args.val_samples, tok_suffix, te.null_ids())
    if args.lm_split and replica == 0 and val_names:  # held-out LM states once, in evaluate()'s chunks
        ids_val = (val if master else load_val(args.data, val_names, args.val_samples, tok_suffix,
                                                ids_enc.null_ids()))[1]
        ch = 64  # evaluate()'s chunk for transformers LMs
        got = lm_split_forward_([trim_ids(ids_val[i:i + ch].to(device)) for i in range(0, len(ids_val), ch)])
        if master:
            val_lm = [h.cpu() for h in got]
            log(f"--lm-split: LM layers {lm_lo}-{lm_hi - 1} of {n_lm} on this stage; held-out LM states for "
                f"{len(ids_val)} captions precomputed")
    lm_client = (LMClient(lm_servers[replica % args.lm_rank], te_cfg.lm_dim, device, control_group) if args.lm_rank and first
                 else None)
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
        if replica == 0:  # per-stage optimizer state first, then latest.pt (the step marker) on rank 0
            torch.save(dict(step=step, ranges=[list(r) for r in ranges], opts=[o.state_dict() for o in opts],
                            outer=outer_opt.state_dict() if outer_opt is not None else None,
                            repa_proj={k: v.cpu() for k, v in repa_proj.state_dict().items()} if repa_proj else None),
                       opt_path + ".tmp")
            os.replace(opt_path + ".tmp", opt_path)
        dist.barrier(group=train_group)
        if master:
            ema_list = [emas[("te", n)] for n in te_names] + [emas[("den", n)] for n, _ in den.named_parameters()]
            torch.save(dict(meta, ema=ema_list, args=vars(args), opts=None), ckpt_path + ".tmp")
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
        cos, sin = build_rope_2d(den_cfg.latent_size, den_cfg.dim // den_cfg.heads)
        den_eval.rope_cos, den_eval.rope_sin = cos.to(device), sin.to(device)
        te_live = [p.detach().clone() for p in te.parameters()]
        torch._foreach_copy_(list(te.parameters()), [emas[("te", n)].to(device) for n in te_names])
        res = evaluate(te, den_eval, val, device, lm_states=val_lm)
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
            dist.barrier(group=train_group)

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
            return None, None, None, None, None, None
        lat, ids = next(loader)  # with a transformers LM: sorted by caption length in the loader (TrainStream)
        ids_cpu = ids
        x0 = lat.to(device, non_blocking=True).float()
        ids = ids.to(device, non_blocking=True)
        g = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + replica * 1009 + at)
        t = torch.sigmoid(torch.randn(per_pipe, device=device, generator=g))
        eps = torch.randn(x0.shape, device=device, generator=g)
        # caption dropout drawn on the CPU (same seed), so the CPU copy of the captions needs no GPU -> CPU sync
        g_cpu = torch.Generator().manual_seed(args.seed * 1_000_003 + replica * 1009 + at)
        drop_cpu = torch.rand(per_pipe, generator=g_cpu) < args.cfg_drop
        drop = drop_cpu.to(device, non_blocking=True)
        cond = torch.where(drop[:, None], null_row, ids) if null_row is not None else None
        cond_cpu = (torch.where(drop_cpu[:, None], null_cpu, ids_cpu)
                    if hf_lm and null_row is not None else None)
        return x0, ids, t, eps, cond, cond_cpu

    pending = None  # --lm-rank: the next step's batch and its LM request, sent while the current step trains
    while step < args.steps:
        if lm_client is not None:
            if pending is None:
                nb = next_batch(step)
                pending = (nb, lm_client.request(nb[5]))
            (x0, ids, t, eps, cond_ids, cond_cpu), req = pending
            lm_full = lm_client.collect(req)
            del req
            pending = None
            if step + 1 < args.steps:
                nb = next_batch(step + 1)
                pending = (nb, lm_client.request(nb[5]))
        else:
            x0, ids, t, eps, cond_ids, cond_cpu = next_batch(step)
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
            if hf_lm:  # ~32 captions per LM call (the MoE's sweet spot; micro-batches of similar length go together)
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
        try:
            with amp:
                if first:
                    tt = t.view(-1, 1, 1, 1)
                    z_t = (1 - tt) * x0 + tt * eps
                    if pp == 1:
                        widths = (text_widths(cond_cpu, mb, args.text_bucket)
                                  if cond_cpu is not None and args.text_bucket else None)
                        local_train_step(stage_mod, z_t, t, cond_ids, eps - x0, mb, losses, widths)
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
        stage_mod.lm_states = None  # backward has finished; release current LM states before optimizer temporaries
        if losses:
            acc[0] += torch.stack([l.detach().float() for l in losses]).mean()
            acc[1] += 1
        if stage_mod.repa_losses:
            acc[2] += torch.stack(stage_mod.repa_losses).mean()
            acc[3] += 1

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
        try:
            for o in opts:
                for gr in o.param_groups:
                    gr["lr"] = gr["base_lr"] * mult
                o.step()
                o.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError:
            mem_report(f"OOM in the optimizer step {step + 1}")
            dump_snapshot("OOM")
            raise
        step += 1
        if args.mem_debug and step <= 3:
            mem_report_all(f"step {step}")
        if step == 2:
            dump_snapshot("step 2")
        if outer_opt is not None or plain_avg:
            if step % args.diloco == 0 or step == args.steps:
                if plain_avg:
                    average_params(params, replicas, stage_group)
                else:
                    outer_sync(global_params, params, outer_opt, replicas, args.outer_comm, stage_group)
                if keep_ema:
                    n_sync = step // args.diloco
                    d = min(args.ema ** args.diloco, (1 + n_sync) / (10 + n_sync))
                    torch._foreach_lerp_(ema, [p.detach().float().to(ema_dev or p.device) for p in model_params], 1 - d)
        elif keep_ema:
            d = min(args.ema, step / (9 + step))
            torch._foreach_lerp_(ema, [p.detach().float() for p in model_params], 1 - d)

        if step % args.log_every == 0 or step == args.steps:
            dist.all_reduce(acc, group=train_group)
            n = step - t_last
            if device.type == "cuda":
                torch.cuda.synchronize(device)  # else the CPU runs ahead and short windows look too fast
            stats = {"train/samples": step * args.batch, "train/loss_fm": (acc[0] / acc[1].clamp(min=1)).item(),
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
                mine_mem = torch.tensor([ms.get("allocated_bytes.all.peak", 0) / 2**30, ms.get("num_alloc_retries", 0)],
                                        device=device)
                all_mem = [torch.zeros_like(mine_mem) for _ in range(world)]
                dist.all_gather(all_mem, mine_mem, group=train_group)
                all_mem = torch.stack(all_mem).cpu()
                worst = int(all_mem[:, 0].argmax())
                retries = int(all_mem[:, 1].sum())
                stats.update({"mem_peak_gb": all_mem[worst, 0].item(), "alloc_retries": retries})
                mem_s = f" | mem {all_mem[worst, 0]:.1f} GB (GPU {worst}){f', ALLOC RETRIES {retries}' if retries else ''}"
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
                f"~{stats['tflops_per_gpu']:.1f} TFLOPS/GPU{mfu} | eta {(args.steps - step) / stats['steps_per_s'] / 3600:.1f} h{mem_s}",
                flush=True)
            if wb:
                wb.log(stats, step=step)
            acc.zero_()
            t0, t_last = stamp, step

        if args.bench:
            continue
        if step % args.eval_every == 0 or step == args.steps:
            res = ema_eval()
            if master and res is not None:
                log(f"step {step:>7} | EMA " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
                if wb:
                    wb.log({**res, "train/samples": step * args.batch}, step=step)
            dist.barrier(group=train_group)
        if step % args.ckpt_every == 0 or step == args.steps:
            save()
        if args.merge_every and step >= args.merge_start * args.steps and (
                step % args.merge_every == 0 or step == args.steps):
            save(merge_only=True)
            dist.barrier(group=train_group)

    if lm_client is not None:
        lm_client.stop()
    log("done")
    if wb:
        wb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
