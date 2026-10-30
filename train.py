"""AuroraIMG2-50M replication: joint text-encoder + flow-matching DiT training on FLUX.2 latents.

  # 8 GPUs, defaults: 253m, global batch 2048 = 8 GPU x 64 x accum 4; 10% of the data, 3 098 steps (~11 epochs)
  torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/aurora
  # out of memory: --micro-batch 128 (accumulation 2 is derived automatically, global batch stays 2048)
  # full dataset, 30 976 steps
  torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/aurora_full --fraction 1
  # the released recipe's batch (256 = 8 x 32, 24 780 steps at 10%); lr / steps / EMA rescale automatically
  torchrun --nproc_per_node 8 train.py --data data/aurora --out runs/b256 --batch 256 --micro-batch 32

  # 1 GPU smoke test (gradient accumulation keeps the global batch)
  python train.py --data data/aurora --out runs/test --steps 200

  # reference: val loss of the released EMA weights on our held-out shards
  python train.py --data data/aurora --out runs/ref --eval-only --load-hf AuroraIMG2-50M-Preview
"""
import argparse
import contextlib
import json
import math
import os
import time
import uuid
from dataclasses import asdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.flop_counter import FlopCounterMode

from aurora.data import (MASK_ID, ShardStore, TrainStream, full_shard_list, list_shards, load_val, null_ids,
                         select_shards, split_shards)
from aurora.model import SIZES, TE_SIZES, DenoiserConfig, TextEncoderConfig, build_models, make_configs
from aurora.muon import Muon


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/aurora")
    p.add_argument("--out", required=True)
    p.add_argument("--size", choices=sorted(SIZES), default="253m",
                   help="denoiser size preset (50m = released model; 116m / 253m = 768x12 / 1024x16)")
    p.add_argument("--te-size", choices=sorted(TE_SIZES), default="27m",
                   help="text encoder size preset (13m = released 384x6, 27m = 512x8)")
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                   help="optimizer for the hidden matrices; everything else is always AdamW. Muon (lr 1e-2, as in the "
                        "released recipe) won the 50m sweep: val/loss 1.147 vs 1.245 for the best AdamW at 400 steps")
    p.add_argument("--lr-hidden", type=float, default=None,  # all AdamW lrs are given at batch 256
                   help="AdamW peak lr for hidden matrices (--optimizer adamw). Default 1e-4 x 768 / width "
                        "(muP-style 1/width from the 768-wide 116m, see PLAN.md)")
    p.add_argument("--beta2", type=float, default=None, help="AdamW beta2 (default: 0.99 with adamw, 0.95 with muon)")
    p.add_argument("--steps", type=int, default=None,
                   help="default: 247805 x fraction, i.e. the released run's ~11 epochs over the chosen data")
    p.add_argument("--batch", type=int, default=2048,
                   help="global batch. 2048 (8 GPU x 256) keeps PCIe x4 all-reduce small relative to compute. "
                        "Steps, AdamW lrs and EMA decay are rescaled from the released recipe's batch 256")
    p.add_argument("--fraction", type=float, default=0.1,
                   help="fraction of the FULL dataset's training shards to train on (same pick as prepare_data "
                        "--fraction); 1 = full dataset")
    p.add_argument("--micro-batch", type=int, default=64,  # 128 and 256 OOM on 24 GB for 253m + 27m TE
                   help="per-GPU micro batch; accumulation = batch / (micro x GPUs)")
    p.add_argument("--warmup-frac", type=float, default=0.02)
    p.add_argument("--lr-muon", type=float, default=1e-2)
    p.add_argument("--lr-emb", type=float, default=5e-4)
    p.add_argument("--lr-adam", type=float, default=3e-4)
    p.add_argument("--no-lr-batch-scaling", action="store_true",
                   help="AdamW lrs are given at batch 256 and multiplied by sqrt(batch / 256) (Malladi et al. 2022); "
                        "this flag uses them as-is")
    p.add_argument("--wd", type=float, default=None,
                   help="decoupled weight decay on matrices (default: 0.1 with adamw, 0.01 with muon)")
    p.add_argument("--schedule", choices=["constant", "cosine"], default="constant",
                   help="after warmup: constant lr, or cosine to 0 (released recipe)")
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--ns-steps", type=int, default=5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=None,
                   help="EMA decay per step; default 0.9995 ** (batch / 256), i.e. the released recipe's window in samples")
    p.add_argument("--cfg-drop", type=float, default=0.1)
    p.add_argument("--mlm-weight", type=float, default=0.05)
    p.add_argument("--mlm-every", type=int, default=4)
    p.add_argument("--mlm-prob", type=float, default=0.15)
    p.add_argument("--grad-comm", choices=["fp32", "bf16", "powersgd"], default="bf16",
                   help="bf16 halves DDP all-reduce traffic (PCIe-only 3090 Ti boxes); powersgd compresses matrix "
                        "gradients to low rank with error feedback (Vogels et al. 2019), ~50-100x less traffic. "
                        "Meant for --optimizer adamw (Muon orthogonalises the low-rank gradient)")
    p.add_argument("--powersgd-rank", type=int, default=8, help="PowerSGD approximation rank")
    p.add_argument("--diloco", type=int, default=0,
                   help="DiLoCo / MuLoCo: every GPU trains its own copy for H inner steps (no DDP, local Muon), then the "
                        "averaged weight change is applied by an outer Nesterov-SGD step. H = this value, 0 = off (DDP)")
    p.add_argument("--outer-lr", type=float, default=0.7, help="DiLoCo outer learning rate")
    p.add_argument("--outer-momentum", type=float, default=0.9, help="DiLoCo outer Nesterov momentum")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers per GPU")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=None, help="default: steps / 20")
    p.add_argument("--ckpt-every", type=int, default=None,
                   help="overwrite <out>/latest.pt (full state, for resuming); default: steps / 50")
    p.add_argument("--merge-every", type=int, default=None,
                   help="WSM: save raw weights to <out>/merge/ every N steps (default steps/25, i.e. 4%%); 0 = off. "
                        "Combine them afterwards with merge.py")
    p.add_argument("--merge-start", type=float, default=0.8,
                   help="WSM: start saving merge checkpoints at this fraction of training (0.8 -> last 20%%)")
    p.add_argument("--keep-every", type=int, default=None,
                   help="also keep <out>/snap_<step>.pt (weights + EMA only, ~1 GB for 116m); default steps/10, 0 = off")
    p.add_argument("--val-samples", type=int, default=4096)
    p.add_argument("--peak-tflops", type=float, default=0,
                   help="GPU peak bf16 TFLOPS for MFU (default: looked up from the GPU name, 0 = unknown)")
    p.add_argument("--wandb", default="trenowanie_generatora_obrazkow",
                   help="wandb project name. Logging is on automatically when wandb is logged in (`wandb login`), "
                        "otherwise off; a resumed run continues the same wandb run (id in <out>/wandb_id.txt)")
    p.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--load-hf", default="", help="init from a released AuroraIMG2 repo dir")
    p.add_argument("--per-shard", type=int, default=0,
                   help="--eval-only: also report loss per shard (N samples each) for the val shards and for the first "
                        "and a middle training shard of every subset, to tell generalisation from memorisation")
    return p.parse_args()


class FlowModel(nn.Module):
    """Both trainable nets in one module so DDP buckets and syncs them together."""

    def __init__(self, te, den, mlm_weight):
        super().__init__()
        self.te, self.den, self.mlm_weight = te, den, mlm_weight

    def forward(self, x0, ids, t, eps, mlm_ids=None, mlm_labels=None):
        text, valid = self.te(ids)
        tt = t.view(-1, 1, 1, 1)
        z_t = (1 - tt) * x0 + tt * eps
        v = self.den(z_t, t, text, valid)
        loss_fm = F.mse_loss(v.float(), eps - x0)
        if mlm_ids is None:
            return loss_fm, loss_fm.detach(), None
        h, _ = self.te(mlm_ids)
        logits = h @ self.te.token_embedding.weight.to(h.dtype).T  # MLM head tied to the token embedding
        loss_mlm = F.cross_entropy(logits.float().flatten(0, 1), mlm_labels.flatten(), ignore_index=-100)
        return loss_fm + self.mlm_weight * loss_mlm, loss_fm.detach(), loss_mlm.detach()


def make_mlm(ids, prob):
    maskable = ids > MASK_ID  # never mask PAD/UNK/BOS/EOS/NULL/MASK
    m = (torch.rand(ids.shape, device=ids.device) < prob) & maskable
    return ids.masked_fill(m, MASK_ID), ids.masked_fill(~m, -100)


def param_groups(te, den, args):
    muon, emb, proj, other = [], [], [], []
    for prefix, mod in (("te", te), ("den", den)):
        for name, p in mod.named_parameters():
            full = f"{prefix}.{name}"
            if full == "te.token_embedding.weight":
                emb.append(p)
            elif ".blocks." in f".{name}" and p.ndim == 2 and not name.endswith("mod_table"):
                muon.append(p)  # hidden matrices: attention, cross-attention, MLP
            elif p.ndim == 2 and not name.endswith("mod_table"):
                proj.append(p)  # patch_in / patch_out / time_mlp / ada
            else:
                other.append(p)  # norms, mod tables, biases
    adam_groups = [
        dict(params=emb, base_lr=args.lr_emb * args.lr_scale, weight_decay=0.0),
        dict(params=proj, base_lr=args.lr_adam * args.lr_scale, weight_decay=args.wd),
        dict(params=other, base_lr=args.lr_adam * args.lr_scale, weight_decay=0.0),
    ]
    opts = []
    if args.optimizer == "muon":
        opts.append(Muon([dict(params=muon, base_lr=args.lr_muon)], lr=args.lr_muon, momentum=args.momentum,
                         ns_steps=args.ns_steps, weight_decay=args.wd, compile_ns=not args.no_compile,
                         shard=not args.diloco))
    else:
        adam_groups.append(dict(params=muon, base_lr=args.lr_hidden * args.lr_scale, weight_decay=args.wd))
    for g in adam_groups:
        g["lr"] = g["base_lr"]
    beta2 = args.beta2 if args.beta2 is not None else (0.99 if args.optimizer == "adamw" else 0.95)
    opts.append(torch.optim.AdamW(adam_groups, betas=(0.9, beta2), eps=1e-8, fused=True))
    counts = {k: sum(p.numel() for p in v) for k, v in
              dict(hidden=muon, emb=emb, proj=proj, other=other).items()}
    return opts, counts


# dense bf16 tensor-core peak with fp32 accumulation (what autocast matmuls use). Stock GeForce runs this at half the
# fp16-acc rate; the 3090 Ti entry is for the unlocked cards on adamo-rigga (full rate, 153; stock would be 80).
PEAK_TFLOPS = {"3090 Ti": 153.0, "3090": 71.0, "4090": 165.2, "5080": 112.6, "5090": 209.5, "A100": 312.0, "H100": 989.0}


def peak_tflops(name: str) -> float:
    for key in sorted(PEAK_TFLOPS, key=len, reverse=True):
        if key in name:
            return PEAK_TFLOPS[key]
    return 0.0


def measure_flops(model, mb, device, mlm_every, mlm_prob, probe=4):
    """Exact training FLOPs (fwd + bwd) of one micro-batch, averaged over MLM / non-MLM steps. Counted eagerly on a
    `probe`-sample batch (eager needs far more memory than the compiled step) and scaled: FLOPs are linear in batch."""
    x0 = torch.randn(probe, 128, 16, 16, device=device)
    ids = torch.randint(MASK_ID + 1, 8192, (probe, 64), device=device)
    t = torch.rand(probe, device=device)
    flops = []
    for mlm in (False, True):
        with FlopCounterMode(display=False) as fc:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = model(x0, ids, t, torch.randn_like(x0), *(make_mlm(ids, mlm_prob) if mlm else (None, None)))
            loss.backward()
        flops.append(fc.get_total_flops())
    model.zero_grad(set_to_none=True)
    return ((mlm_every - 1) * flops[0] + flops[1]) / mlm_every * mb / probe


def lr_mult(step, total, warmup, schedule):
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


@torch.no_grad()
def evaluate(te, den, val, device, seed=1234, chunk=256):
    """Velocity MSE on the held-out set with fixed t and noise, so numbers are comparable across runs."""
    lat, ids = val
    g = torch.Generator(device="cpu").manual_seed(seed)
    t_all = torch.sigmoid(torch.randn(len(lat), generator=g))
    eps_all = torch.randn(lat.shape, generator=g)
    te.eval(), den.eval()
    tot, n = 0.0, 0
    buckets = torch.zeros(4), torch.zeros(4)
    for i in range(0, len(lat), chunk):
        x0 = lat[i:i + chunk].to(device).float()
        t, eps = t_all[i:i + chunk].to(device), eps_all[i:i + chunk].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text, valid = te(ids[i:i + chunk].to(device))
            tt = t.view(-1, 1, 1, 1)
            v = den((1 - tt) * x0 + tt * eps, t, text, valid)
        per = (v.float() - (eps - x0)).pow(2).mean(dim=(1, 2, 3)).cpu()
        tot += per.sum().item()
        n += len(per)
        b = (t.cpu() * 4).long().clamp(max=3)
        buckets[0].index_add_(0, b, per)
        buckets[1].index_add_(0, b, torch.ones_like(per))
    te.train(), den.train()
    out = {"val/loss": tot / n}
    for k in range(4):
        out[f"val/loss_t{k / 4:.2f}-{(k + 1) / 4:.2f}"] = (buckets[0][k] / buckets[1][k].clamp(min=1)).item()
    return out


def hf_configs(repo):
    def read(sub, cls):
        with open(os.path.join(repo, sub, "config.json")) as f:
            cfg = json.load(f)
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})
    return read("transformer", DenoiserConfig), read("text_encoder", TextEncoderConfig)


def eval_per_shard(te, den, args, device, log):
    full = full_shard_list(args.data)
    present = set(list_shards(args.data))
    train, val = split_shards(full)
    groups = {}
    for n in train:
        groups.setdefault(n.rsplit("-", 3)[0], []).append(n)
    picks = [(n, "val") for n in val] + [(g[i], "train") for g in groups.values() for i in (0, len(g) // 2)]
    log(f"{'shard':45s} {'split':5s} {'loss':>7s} {'no text':>8s} {'gain':>6s}")
    for name, split in picks:
        if name not in present:
            log(f"{name:45s} {split:5s}  (not prepared)")
            continue
        lat, ids = load_val(args.data, [name], args.per_shard)
        a = evaluate(te, den, (lat, ids), device)["val/loss"]
        b = evaluate(te, den, (lat, torch.from_numpy(null_ids()).expand_as(ids).clone()), device)["val/loss"]
        log(f"{name:45s} {split:5s} {a:7.4f} {b:8.4f} {b - a:6.3f}")


def load_hf(te, den, repo):
    from safetensors.torch import load_file
    te.load_state_dict(load_file(os.path.join(repo, "text_encoder", "diffusion_pytorch_model.safetensors")))
    den.load_state_dict(load_file(os.path.join(repo, "transformer", "diffusion_pytorch_model.safetensors")))


def wandb_logged_in() -> bool:
    """True if wandb is installed and has credentials: WANDB_API_KEY, offline mode, or the ~/.netrc entry that
    `wandb login` writes."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") in ("offline", "dryrun"):
        return True
    import netrc
    from urllib.parse import urlparse
    host = urlparse(os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")).hostname
    try:
        return netrc.netrc(os.environ.get("NETRC")).authenticators(host) is not None
    except (FileNotFoundError, netrc.NetrcParseError):
        return False


def init_wandb(args):
    """One wandb run per output dir: the id is stored in <out>/wandb_id.txt, so resuming continues the same run."""
    import wandb
    id_path = os.path.join(args.out, "wandb_id.txt")
    run_id = open(id_path).read().strip() if os.path.exists(id_path) else uuid.uuid4().hex[:12]
    with open(id_path, "w") as f:
        f.write(run_id)
    wb = wandb.init(project=args.wandb, id=run_id, resume="allow", dir=args.out,
                    name=os.path.basename(os.path.normpath(args.out)))
    wb.config.update(vars(args), allow_val_change=True)
    return wb


def load_opt_state(opt, state):
    """Restore optimizer state (momenta etc.) but keep the hyperparameters from the current command line:
    torch's load_state_dict would silently bring back the checkpoint's lr / momentum / weight decay."""
    fresh = [{k: v for k, v in g.items() if k != "params"} for g in opt.param_groups]
    opt.load_state_dict(state)
    for g, f in zip(opt.param_groups, fresh):
        g.update(f)


def resolve_defaults(args):
    """Defaults that depend on other arguments (size, batch, fraction)."""
    if args.lr_hidden is None:
        args.lr_hidden = 1e-4 * 768 / (SIZES[args.size].dim if not args.load_hf else 768)
    ref_ratio = args.batch / 256  # released recipe: batch 256, 247 805 steps (~11 epochs over the full data)
    if args.steps is None:
        args.steps = round(247_805 * args.fraction / ref_ratio)
    # DiLoCo: each GPU's inner optimizer only sees its own share of the batch, so lrs are scaled to that
    lr_ratio = args.batch / int(os.environ.get("WORLD_SIZE", 1)) / 256 if args.diloco else ref_ratio
    args.lr_scale = 1.0 if args.no_lr_batch_scaling else math.sqrt(lr_ratio)
    if args.ema is None:
        args.ema = 0.9995 ** ref_ratio
    if args.eval_every is None:
        args.eval_every = max(1, args.steps // 20)
    if args.ckpt_every is None:
        args.ckpt_every = max(1, args.steps // 50)
    if args.merge_every is None:
        args.merge_every = max(1, args.steps // 25)
    if args.keep_every is None:
        args.keep_every = max(1, args.steps // 10)
    if args.wd is None:
        args.wd = 0.1 if args.optimizer == "adamw" else 0.01
    if args.diloco:  # evals, checkpoints and WSM saves happen right after an outer sync
        up = lambda n: n and -(-n // args.diloco) * args.diloco
        args.eval_every, args.ckpt_every = up(args.eval_every), up(args.ckpt_every)
        args.merge_every, args.keep_every = up(args.merge_every), up(args.keep_every)
    return args


def main():
    args = resolve_defaults(get_args())
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group(os.environ.get("DIST_BACKEND", "nccl"))  # gloo only for single-GPU tests
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    local %= torch.cuda.device_count()
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    master = rank == 0
    torch.manual_seed(args.seed * 1000 + rank)  # different noise / t / dropout per rank
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log = print if master else (lambda *a, **k: None)

    mb = args.micro_batch or args.batch // world
    accum = args.batch // (mb * world)
    assert mb * world * accum == args.batch, "batch must be divisible by micro_batch * world"

    # ---- data ----------------------------------------------------------------------------------
    train_names, val_names = select_shards(args.data, args.fraction)
    store = ShardStore(args.data, train_names)
    log(f"train: {len(train_names)} shards / {len(store):,} images, val: {len(val_names)} shards {val_names}")
    log(f"{args.steps} steps x {args.batch} = {args.steps * args.batch / len(store):.1f} epochs over the training set")
    val = load_val(args.data, val_names, args.val_samples) if (master and val_names) else None

    # ---- model ---------------------------------------------------------------------------------
    den_cfg, te_cfg = hf_configs(args.load_hf) if args.load_hf else make_configs(args.size, args.te_size)
    te, den = build_models(den_cfg, te_cfg)
    if args.load_hf:
        load_hf(te, den, args.load_hf)
    te.to(device), den.to(device)
    log(f"params: text encoder {sum(p.numel() for p in te.parameters()) / 1e6:.2f}M, "
        f"denoiser {sum(p.numel() for p in den.parameters()) / 1e6:.2f}M")

    if args.eval_only:
        if val is not None:
            res = evaluate(te, den, val, device)
            # same images, empty prompt: the gap shows how much the model actually uses the text
            no_text = evaluate(te, den, (val[0], torch.from_numpy(null_ids()).expand_as(val[1]).clone()), device)
            res.update({k.replace("val/", "val_no_text/"): v for k, v in no_text.items()})
            res["text_gain"] = no_text["val/loss"] - res["val/loss"]
            log(json.dumps(res, indent=1))
            if args.per_shard:
                eval_per_shard(te, den, args, device, log)
        if distributed:
            dist.destroy_process_group()
        return

    opts, counts = param_groups(te, den, args)
    log(f"optimizer: {args.optimizer} for hidden matrices; split (params): {counts}")
    ema = [p.detach().clone().float() for p in list(te.parameters()) + list(den.parameters())] if master else None

    # ---- resume --------------------------------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "latest.pt")
    step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        assert ck["den_config"] == asdict(den_cfg), f"checkpoint was trained with {ck['den_config']}"
        ck_te = ck.get("te_config", asdict(TextEncoderConfig()))
        assert ck_te == asdict(te_cfg), f"checkpoint was trained with text encoder {ck_te} (see --te-size)"
        te.load_state_dict(ck["te"]), den.load_state_dict(ck["den"])
        for o, s in zip(opts, ck["opts"]):
            load_opt_state(o, s)
        if master:
            for e, v in zip(ema, ck["ema"]):
                e.copy_(v)
        step = ck["step"]
        outer_state = ck.get("outer")
        log(f"resumed from {ckpt_path} at step {step}")
        del ck
    else:
        outer_state = None

    model = FlowModel(te, den, args.mlm_weight)
    flops_micro = measure_flops(model, mb, device, args.mlm_every, args.mlm_prob)  # before compile: eager is countable
    if not args.no_compile:
        te.compile()
        den.compile()
    peak = args.peak_tflops or peak_tflops(torch.cuda.get_device_name(device))
    log(f"{flops_micro / mb / 1e9:.1f} GFLOP per training sample; peak {peak or '?'} TFLOPS/GPU "
        f"({torch.cuda.get_device_name(device)})")
    ddp = distributed and not args.diloco
    if args.diloco:
        live = list(te.parameters()) + list(den.parameters())
        if distributed:  # same starting point on every GPU (DDP would broadcast it)
            for p in live:
                dist.broadcast(p.data, 0)
        global_params = [nn.Parameter(p.detach().clone().float(), requires_grad=False) for p in live]
        outer_opt = torch.optim.SGD(global_params, lr=args.outer_lr, momentum=args.outer_momentum, nesterov=True)
        if outer_state is not None:
            load_opt_state(outer_opt, outer_state)
        log(f"DiLoCo/MuLoCo: {world} workers, outer sync every {args.diloco} steps "
            f"(outer Nesterov SGD lr {args.outer_lr}, momentum {args.outer_momentum}), inner {args.optimizer} unsharded")
    if ddp:
        model = DDP(model, device_ids=[local], bucket_cap_mb=64, gradient_as_bucket_view=True)
        if args.grad_comm == "bf16":
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(None, default_hooks.bf16_compress_hook)
        elif args.grad_comm == "powersgd":
            from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as psgd
            # plain all-reduce until the lr warmup is over (recommended by the authors); the hook counts optimizer
            # steps, and its error-feedback state is not checkpointed, so a resumed run restarts that count
            start = max(2, int(args.warmup_frac * args.steps) - step)
            state = psgd.PowerSGDState(process_group=None, matrix_approximation_rank=args.powersgd_rank,
                                       start_powerSGD_iter=start, use_error_feedback=True, warm_start=True,
                                       random_seed=args.seed)
            model.register_comm_hook(state, psgd.powerSGD_hook)
            log(f"PowerSGD rank {args.powersgd_rank}, full all-reduce for the first {start} steps")
            if args.optimizer == "muon":
                log("warning: PowerSGD with Muon - Muon orthogonalises the low-rank gradient, prefer --optimizer adamw")
    params = [p for p in model.parameters()]

    stream = TrainStream(store, mb, rank, world, args.seed, start_micro=step * accum)
    loader = iter(torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                              pin_memory=True, prefetch_factor=2 if args.workers else None,
                                              persistent_workers=args.workers > 0))
    null_row = torch.from_numpy(null_ids()).to(device)

    wb = None
    if master and args.wandb and not args.no_wandb:
        if wandb_logged_in():
            wb = init_wandb(args)
        else:
            log("wandb: off (not logged in; run `wandb login` to enable, or pass --no-wandb to silence this)")

    def save():
        if not master:
            return
        sd = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        tmp = ckpt_path + ".tmp"
        torch.save(dict(step=step, den_config=asdict(den_cfg), te_config=asdict(te_cfg), te=sd(te), den=sd(den), opts=[o.state_dict() for o in opts],
                        ema=[e.cpu() for e in ema], args=vars(args),
                        outer=outer_opt.state_dict() if args.diloco else None), tmp)
        os.replace(tmp, ckpt_path)
        if args.keep_every and step % args.keep_every == 0:
            torch.save(dict(step=step, den_config=asdict(den_cfg), te_config=asdict(te_cfg), te=sd(te), den=sd(den), ema=[e.cpu() for e in ema]),
                       os.path.join(args.out, f"snap_{step:07d}.pt"))

    def save_merge():
        """WSM: raw (non-EMA) fp32 weights only, ~0.46 GB for 116m; merge.py averages them."""
        d = os.path.join(args.out, "merge")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"step_{step:07d}.pt")
        sd = lambda m: {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save(dict(step=step, total_steps=args.steps, den_config=asdict(den_cfg), te_config=asdict(te_cfg), te=sd(te), den=sd(den)),
                   path + ".tmp")
        os.replace(path + ".tmp", path)

    @torch.no_grad()
    def outer_sync():
        """DiLoCo outer step: pseudo-gradient = global - local weights, averaged over workers, applied with Nesterov
        SGD to the global weights, which then replace every worker's local weights."""
        deltas = [g - p.detach().float() for g, p in zip(global_params, live)]
        flat = torch.cat([d.flatten() for d in deltas])
        if distributed:
            dist.all_reduce(flat)
            flat /= world
        for g, d in zip(global_params, flat.split([d.numel() for d in deltas])):
            g.grad = d.view_as(g)
        outer_opt.step()
        outer_opt.zero_grad(set_to_none=True)
        torch._foreach_copy_(live, global_params)

    @torch.no_grad()
    def ema_eval():
        """Copy EMA weights into the live modules, evaluate, restore."""
        live = [p.detach().clone() for p in list(te.parameters()) + list(den.parameters())]
        torch._foreach_copy_(list(te.parameters()) + list(den.parameters()), ema)
        res = evaluate(te, den, val, device)
        torch._foreach_copy_(list(te.parameters()) + list(den.parameters()), live)
        return res

    warmup = max(1, int(args.warmup_frac * args.steps))
    lrs = ", ".join(f"{g['base_lr']:.2e}" for o in opts for g in o.param_groups)
    log(f"lr: {args.schedule} after {warmup} warmup steps, peak lrs [{lrs}] (x{args.lr_scale:.2f} for batch "
        f"{args.batch}), weight decay {args.wd}, EMA {args.ema:.5f}, eval every {args.eval_every}, "
        f"latest.pt every {args.ckpt_every}")
    if args.merge_every:
        log(f"WSM: raw weights to {args.out}/merge/ every {args.merge_every} steps from step "
            f"{math.ceil(args.merge_start * args.steps)} (~{int((1 - args.merge_start) * args.steps / args.merge_every)} files)")
    log(f"global batch {args.batch} = {world} GPU x {mb} x accum {accum}; {args.steps} steps, warmup {warmup}")
    acc = torch.zeros(3, device=device)  # sum fm, sum mlm, n mlm
    t0, t_last = time.time(), step
    while step < args.steps:
        mlm_step = args.mlm_weight > 0 and step % args.mlm_every == 0
        for micro in range(accum):
            lat, ids = next(loader)
            x0 = lat.to(device, non_blocking=True).float()
            ids = ids.to(device, non_blocking=True)
            b = x0.shape[0]
            t = torch.sigmoid(torch.randn(b, device=device))  # logit-normal(0, 1)
            eps = torch.randn_like(x0)
            drop = torch.rand(b, device=device) < args.cfg_drop
            cond_ids = torch.where(drop[:, None], null_row, ids)
            mlm_ids, mlm_labels = make_mlm(ids, args.mlm_prob) if mlm_step else (None, None)
            sync = contextlib.nullcontext() if (not ddp or micro == accum - 1) else model.no_sync()
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, l_fm, l_mlm = model(x0, cond_ids, t, eps, mlm_ids, mlm_labels)
                (loss / accum).backward()
            acc[0] += l_fm / accum
            if l_mlm is not None:
                acc[1] += l_mlm / accum
                acc[2] += 1 / accum

        gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        mult = lr_mult(step, args.steps, warmup, args.schedule)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * mult
            o.step()
            o.zero_grad(set_to_none=True)
        step += 1
        if args.diloco:
            if step % args.diloco == 0 or step == args.steps:
                outer_sync()
                if master:  # EMA of the global weights, same window in samples as the per-step EMA
                    n = step // args.diloco
                    d = min(args.ema ** args.diloco, (1 + n) / (10 + n))
                    torch._foreach_lerp_(ema, [p.detach() for p in live], 1 - d)
        elif master:
            d = min(args.ema, (step) / (9 + step))
            torch._foreach_lerp_(ema, [p.detach() for p in list(te.parameters()) + list(den.parameters())], 1 - d)

        if step % args.log_every == 0 or step == args.steps:
            if distributed:
                dist.all_reduce(acc)
                acc /= world
            n = step - t_last
            dt = time.time() - t0
            stats = {"train/samples": step * args.batch, "train/epoch": step * args.batch / len(store),
                     "train/loss_fm": acc[0].item() / n, "train/loss_mlm": (acc[1] / acc[2].clamp(min=1e-9)).item(),
                     "train/grad_norm": gnorm.item(), "lr_mult": mult, "steps_per_s": n / dt,
                     "imgs_per_s": n * args.batch / dt, "tflops_per_gpu": flops_micro * accum * n / dt / 1e12}
            mfu = ""
            if peak:
                stats["mfu"] = stats["tflops_per_gpu"] / peak
                mfu = f" MFU {stats['mfu']:.0%}"
            eta_h = (args.steps - step) / stats["steps_per_s"] / 3600
            log(f"step {step:>7} | fm {stats['train/loss_fm']:.4f} | mlm {stats['train/loss_mlm']:.3f} | "
                f"gn {stats['train/grad_norm']:.3f} | lr x{mult:.3f} | {stats['steps_per_s']:.2f} it/s "
                f"{stats['imgs_per_s']:.0f} img/s | {stats['tflops_per_gpu']:.1f} TFLOPS/GPU{mfu} | eta {eta_h:.1f} h", flush=True)
            if wb:
                wb.log(stats, step=step)
            acc.zero_()
            t0, t_last = time.time(), step

        if val is not None and (step % args.eval_every == 0 or step == args.steps):
            res = ema_eval()
            log(f"step {step:>7} | EMA " + " ".join(f"{k}={v:.4f}" for k, v in res.items()), flush=True)
            if wb:
                wb.log({**res, "train/samples": step * args.batch}, step=step)
        if step % args.ckpt_every == 0 or step == args.steps or (args.keep_every and step % args.keep_every == 0):
            save()
        if master and args.merge_every and step >= args.merge_start * args.steps and (
                step % args.merge_every == 0 or step == args.steps):
            save_merge()
        if distributed and (step % args.eval_every == 0 or step % args.ckpt_every == 0
                            or (args.keep_every and step % args.keep_every == 0)):
            dist.barrier()

    log("done")
    if wb:
        wb.finish()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
