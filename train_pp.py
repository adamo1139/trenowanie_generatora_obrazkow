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
"""
import functools
import logging
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts
from torch.distributed.pipelining import PipelineStage, Schedule1F1B

from aurora.data import ShardStore, TrainStream, load_val, select_shards
from aurora.model import (Denoiser, LMEncoderConfig, TextEncoder, build_rope_2d, make_configs,
                          te_config_from_dict, timestep_embedding)
from aurora.muon import Muon
from train import (build_parser, evaluate, init_wandb, load_opt_state, lr_mult, param_groups, peak_tflops,
                   resolve_defaults, wandb_logged_in)


def get_args():
    p = build_parser()
    p.add_argument("--pp", type=int, default=8, help="pipeline stages (GPUs per pipeline); world must be a multiple")
    p.add_argument("--microbatches", type=int, default=None,
                   help="micro-batches per step and pipeline (default 4 x pp; bubble ~ (pp-1)/(microbatches+pp-1))")
    p.add_argument("--act-checkpoint", nargs="?", const="full", default=None, choices=["full", "selective"],
                   help="recompute block activations in backward: full (only block inputs are kept) or selective "
                        "(matmul and attention outputs are kept, only the cheap elementwise ops are recomputed)")
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

    def forward(self, *args):
        if self.first:
            z, t, ids = args
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


def stage_ranges(depth, pp):
    bounds = np.linspace(0, depth, pp + 1).round().astype(int)
    return [(int(bounds[s]), int(bounds[s + 1])) for s in range(pp)]


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
    rank, world = dist.get_rank(), dist.get_world_size()
    pp = args.pp
    assert world % pp == 0, f"world size {world} is not a multiple of --pp {pp}"
    replicas = world // pp
    replica, s = divmod(rank, pp)
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
    pipe_group, stage_group = pipe_groups[replica], stage_groups[s]

    per_pipe = args.batch // replicas
    M = args.microbatches or 4 * pp
    assert per_pipe * replicas == args.batch and per_pipe % M == 0, \
        f"batch {args.batch} must split into {replicas} pipelines x {M} micro-batches"
    mb = per_pipe // M
    # inner lrs: AdamW groups scaled to the batch each optimizer sees (the pipeline's batch with MuLoCo)
    args.lr_scale = 1.0 if args.no_lr_batch_scaling else math.sqrt((per_pipe if args.diloco else args.batch) / 256)
    args.mlm_weight = 0.0

    # ---- model: a meta skeleton everywhere, only this stage's parts are materialised -------------------------------
    den_cfg, te_cfg = make_configs(args.size, args.te_size)
    assert den_cfg.depth >= pp, f"{den_cfg.depth} blocks can't fill {pp} stages"
    ranges = stage_ranges(den_cfg.depth, pp)
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
            te = LMTextEncoder(te_cfg)
        else:
            te = TextEncoder(te_cfg)
        te.init_weights()
        te.to(device)

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
    if replicas > 1:  # identical start on every replica
        for p in model_params + ([] if repa_proj is None else list(repa_proj.parameters())):
            dist.broadcast(p.data, s, group=stage_group)

    opts, counts = param_groups(None, None, args, named=named)
    opts = [o if not isinstance(o, Muon) else Muon(o.param_groups, lr=args.lr_muon, momentum=args.momentum,
                                                   ns_steps=args.ns_steps, weight_decay=args.wd,
                                                   compile_ns=not args.no_compile, shard=False) for o in opts]
    keep_ema = replica == 0
    ema = [p.detach().clone().float() for p in model_params] if keep_ema else None

    # ---- resume ----------------------------------------------------------------------------------------------------
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

    outer_opt, global_params = None, None
    if replicas > 1 and args.diloco:
        global_params = [nn.Parameter(p.detach().clone().float(), requires_grad=False) for p in params]
        outer_opt = torch.optim.SGD(global_params, lr=args.outer_lr, momentum=args.outer_momentum,
                                    nesterov=args.outer_momentum > 0)
        if outer_state is not None:
            load_opt_state(outer_opt, outer_state)

    pstage = PipelineStage(stage_mod, s, pp, device, group=pipe_group)
    # the REPA stage needs to know which micro-batch it is computing: take it from the schedule, not a counter
    # (pipelining also runs extra forwards, e.g. shape inference on the first step)
    _fwd_chunk = pstage.forward_one_chunk

    def forward_one_chunk(fwd_chunk_id, *a, **k):
        stage_mod._mb = fwd_chunk_id
        return _fwd_chunk(fwd_chunk_id, *a, **k)
    pstage.forward_one_chunk = forward_one_chunk
    loss_fn = lambda v, target: F.mse_loss(v.float(), target)
    # every stage needs loss_fn: the schedule only enables backward (and gradient exchange) when it is set
    sched = Schedule1F1B(pstage, M, loss_fn=loss_fn)

    # ---- data: only the stages that need it ------------------------------------------------------------------------
    tok_suffix = getattr(te_cfg, "tok_suffix", "")
    null_row = None
    if first:
        null_row = torch.from_numpy(te.null_ids()).to(device)
    needs_data = first or last or args.repa_weight > 0  # REPA targets are decoded on every stage
    train_names, val_names = select_shards(args.data, args.fraction, tok_suffix)
    loader = store = None
    if needs_data:
        store = ShardStore(args.data, train_names, tok_suffix, te.null_ids() if first else None)
        stream = TrainStream(store, per_pipe, replica, replicas, args.seed, start_micro=step)
        loader = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                                  pin_memory=device.type == "cuda",
                                                  prefetch_factor=2 if args.workers else None,
                                                  persistent_workers=args.workers > 0))
    n_train = len(store) if store is not None else sum(
        len(np.load(os.path.join(args.data, f"{n}.off.npy"), mmap_mode="r")) - 1 for n in train_names)
    log(f"train: {len(train_names)} shards / {n_train:,} images; {args.steps} steps x {args.batch} = "
        f"{args.steps * args.batch / n_train:.1f} epochs")
    val = None
    if master and val_names:
        val = load_val(args.data, val_names, args.val_samples, tok_suffix, te.null_ids())

    # FLOPs per sample (fwd + bwd), analytic: matmuls 6 N T, attention 12 L d T (T + T_text), frozen LM forward 2 N T
    T, Tt = den_cfg.latent_size ** 2, getattr(te_cfg, "max_tokens", 64)
    flops_sample = 6 * n_den * T + 12 * den_cfg.depth * den_cfg.dim * T * (T + Tt)
    if first:
        flops_sample += 6 * sum(p.numel() for p in te.parameters()) * Tt
        if hasattr(te, "lm"):
            flops_sample += 2 * sum(p.numel() for p in te.lm.parameters()) * Tt
    peak = args.peak_tflops or (peak_tflops(torch.cuda.get_device_name(device)) if device.type == "cuda" else 0)

    wb = None
    if master and args.wandb and not args.no_wandb:
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
                dist.send(t_.detach().float().contiguous(), dst=0)
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
        dist.barrier()
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
        res = evaluate(te, den_eval, val, device)
        torch._foreach_copy_(list(te.parameters()), te_live)
        del den_eval
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return res

    @torch.no_grad()
    def outer_sync():
        deltas = [g - p.detach().float() for g, p in zip(global_params, params)]
        flat = torch.cat([d.flatten() for d in deltas])
        if args.outer_comm == "bf16":
            flat_c = flat.bfloat16()
            dist.all_reduce(flat_c, group=stage_group)
            flat = flat_c.float()
        else:
            dist.all_reduce(flat, group=stage_group)
        flat /= replicas
        for g, d in zip(global_params, flat.split([d.numel() for d in deltas])):
            g.grad = d.view_as(g)
        outer_opt.step()
        outer_opt.zero_grad(set_to_none=True)
        torch._foreach_copy_(params, global_params)

    warmup = max(1, int(args.warmup_frac * args.steps))
    repa_steps = int(args.repa_until * args.steps) if args.repa_weight > 0 else 0
    if args.repa_weight > 0:
        log(f"REPA{' (iREPA)' if args.irepa else ''}: {args.repa_encoder} <- DiT block {args.repa_layer} "
            f"(stage {repa_stage}), {args.repa_frac:.0%} of each micro-batch, every {args.repa_every} step(s), until "
            f"step {repa_steps}; decoding spread over all {pp} stages")
    log(f"lr: {args.schedule} after {warmup} warmup steps, Muon {args.lr_muon}, AdamW x{args.lr_scale:.2f}, "
        f"EMA {args.ema:.5f}; sync: {'MuLoCo every %d steps' % args.diloco if replicas > 1 and args.diloco else 'gradients every step' if replicas > 1 else 'single pipeline'}")
    acc = torch.zeros(4, device=device)  # sum loss, n, sum repa, n repa
    t0, t_last = time.time(), step
    while step < args.steps:
        x0 = ids = None
        if needs_data:
            lat, ids = next(loader)
            x0 = lat.to(device, non_blocking=True).float()
            ids = ids.to(device, non_blocking=True)
            g = torch.Generator(device=device).manual_seed(args.seed * 1_000_003 + replica * 1009 + step)
            t = torch.sigmoid(torch.randn(per_pipe, device=device, generator=g))
            eps = torch.randn(x0.shape, device=device, generator=g)
            drop = torch.rand(per_pipe, device=device, generator=g) < args.cfg_drop
        stage_mod.repa_targets = None
        if step < repa_steps and step % args.repa_every == 0:
            n_repa = max(1, round(args.repa_frac * mb))
            mine = {k: repa_target(x0[k * mb: k * mb + n_repa]) for k in range(M) if k % pp == s}
            repa_rank = replica * pp + repa_stage
            if s == repa_stage:
                targets = []
                for k in range(M):
                    if k % pp == s:
                        targets.append(mine[k].bfloat16().float())  # same precision as the received ones
                    else:
                        buf = torch.empty(n_repa, den_cfg.latent_size ** 2, repa_target.dim, dtype=torch.bfloat16,
                                          device=device)
                        dist.recv(buf, src=replica * pp + k % pp)
                        targets.append(buf.float())
                stage_mod.repa_targets = targets
            else:
                for k in sorted(mine):
                    dist.send(mine[k].bfloat16().contiguous(), dst=repa_rank)
        stage_mod.repa_losses = []
        losses = []
        with amp:
            if first:
                tt = t.view(-1, 1, 1, 1)
                z_t = (1 - tt) * x0 + tt * eps
                cond_ids = torch.where(drop[:, None], null_row, ids)
                if last:
                    sched.step(z_t, t, cond_ids, target=eps - x0, losses=losses)
                else:
                    sched.step(z_t, t, cond_ids)
            elif last:
                sched.step(target=eps - x0, losses=losses)
            else:
                sched.step()
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
        sq = torch.zeros((), device=device)
        for p in params:
            if p.grad is not None:
                sq += p.grad.float().pow(2).sum()
        dist.all_reduce(sq, group=pipe_group)
        gnorm = sq.sqrt()
        coef = (args.clip / (gnorm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_([p.grad for p in params if p.grad is not None], coef)
        mult = lr_mult(step, args.steps, warmup, args.schedule)
        for o in opts:
            for gr in o.param_groups:
                gr["lr"] = gr["base_lr"] * mult
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1
        if outer_opt is not None:
            if step % args.diloco == 0 or step == args.steps:
                outer_sync()
                if keep_ema:
                    n_sync = step // args.diloco
                    d = min(args.ema ** args.diloco, (1 + n_sync) / (10 + n_sync))
                    torch._foreach_lerp_(ema, [p.detach().float() for p in model_params], 1 - d)
        elif keep_ema:
            d = min(args.ema, step / (9 + step))
            torch._foreach_lerp_(ema, [p.detach().float() for p in model_params], 1 - d)

        if step % args.log_every == 0 or step == args.steps:
            dist.all_reduce(acc)
            n = step - t_last
            dt = time.time() - t0
            stats = {"train/samples": step * args.batch, "train/loss_fm": (acc[0] / acc[1].clamp(min=1)).item(),
                     "train/loss_repa": (acc[2] / acc[3].clamp(min=1)).item(), "train/grad_norm": gnorm.item(),
                     "lr_mult": mult, "steps_per_s": n / dt, "imgs_per_s": n * args.batch / dt,
                     "tflops_per_gpu": flops_sample * args.batch * n / dt / world / 1e12}
            mfu = ""
            if peak:
                stats["mfu"] = stats["tflops_per_gpu"] / peak
                mfu = f" MFU {stats['mfu']:.0%}"
            repa_s = f"repa {stats['train/loss_repa']:.3f} | " if acc[3] > 0 else ""
            log(f"step {step:>7} | fm {stats['train/loss_fm']:.4f} | {repa_s}grad_norm {stats['train/grad_norm']:.3f} | "
                f"lr x{mult:.3f} | {stats['steps_per_s']:.2f} it/s {stats['imgs_per_s']:.0f} img/s | "
                f"~{stats['tflops_per_gpu']:.1f} TFLOPS/GPU{mfu} | eta {(args.steps - step) / stats['steps_per_s'] / 3600:.1f} h",
                flush=True)
            if wb:
                wb.log(stats, step=step)
            acc.zero_()
            t0, t_last = time.time(), step

        if step % args.eval_every == 0 or step == args.steps:
            res = ema_eval()
            if master and res is not None:
                log(f"step {step:>7} | EMA " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
                if wb:
                    wb.log({**res, "train/samples": step * args.batch}, step=step)
            dist.barrier()
        if step % args.ckpt_every == 0 or step == args.steps:
            save()
        if args.merge_every and step >= args.merge_start * args.steps and (
                step % args.merge_every == 0 or step == args.steps):
            save(merge_only=True)
            dist.barrier()

    log("done")
    if wb:
        wb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
