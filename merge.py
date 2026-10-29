"""WSM (Warmup-Stable-Merge, Tian et al. 2025, arXiv:2507.17634): instead of decaying the lr, train at constant lr and
average the last checkpoints with weights that emulate a decay schedule.

train.py saves raw weights to <run>/merge/step_*.pt over the last 30% of training (--merge-start / --merge-every).
This script tries every (scheme, window) combination on the held-out val set, compares with the EMA from latest.pt,
and writes the best one as a checkpoint that export.py takes:

  python merge.py runs/p10                                   # sweep defaults, write runs/p10/merged.pt
  python merge.py runs/p10 --schemes 1-sqrt --windows 0.2    # one fixed choice
  python export.py runs/p10/merged.pt exports/p10_wsm

Theorem 3.1 of the paper: merging checkpoints theta_0..theta_k (theta_0 = start of the window) with
c_0 = 1 - w_1, c_j = w_j - w_{j+1}, c_k = w_k is equivalent to having scaled the gradient updates between them by the
non-increasing decay coefficients w_1..w_k. The paper finds 1-sqrt best, mean (= linear decay) close, EMA clearly worse,
and the window length (merge duration) the most important knob.
"""
import argparse
import glob
import math
import os

import torch

from aurora.data import full_shard_list, load_val, split_shards
from aurora.model import build_models, configs_from_ckpt
from train import evaluate

DECAYS = {  # lr multiplier as a function of progress p in (0, 1) through the window
    "mean": lambda p: 1 - p,  # linear decay <-> uniform average
    "1-sqrt": lambda p: 1 - math.sqrt(p),
    "cosine": lambda p: 0.5 * (1 + math.cos(math.pi * p)),
}


def merge_weights(scheme: str, n_ckpts: int):
    """Weights c_0..c_k (k = n_ckpts - 1) for checkpoints ordered old -> new."""
    k = n_ckpts - 1
    if k == 0:
        return [1.0]
    w = [DECAYS[scheme](i / (k + 1)) for i in range(1, k + 1)]  # decay coefficient of interval i
    c = [1 - w[0]] + [w[j] - w[j + 1] for j in range(k - 1)] + [w[-1]]
    assert all(x >= -1e-9 for x in c) and abs(sum(c) - 1) < 1e-6
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", help="training output dir (contains merge/ and latest.pt)")
    p.add_argument("--schemes", default="1-sqrt,mean,cosine")
    p.add_argument("--windows", default="0.05,0.1,0.2,0.3",
                   help="merge durations as fractions of the whole training run (counted back from the last step)")
    p.add_argument("--data", default="data/aurora")
    p.add_argument("--val-samples", type=int, default=4096)
    p.add_argument("--out", default=None, help="default: <run>/merged.pt")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.run, "merge", "step_*.pt")))
    assert files, f"no merge checkpoints in {args.run}/merge (train with --merge-every > 0)"
    meta = [torch.load(f, map_location="cpu", weights_only=False, mmap=True) for f in files]
    steps = [m["step"] for m in meta]
    total, last = meta[-1]["total_steps"], steps[-1]
    print(f"{len(files)} merge checkpoints, steps {steps[0]}..{last} (run of {total} steps)")

    combos = []
    for window in (float(x) for x in args.windows.split(",")):
        idx = [i for i, s in enumerate(steps) if s >= last - window * total]
        for scheme in args.schemes.split(","):
            combos.append(dict(scheme=scheme, window=window, idx=idx, c=merge_weights(scheme, len(idx))))

    # stream checkpoints once, accumulating every combination (fp32, on GPU)
    dev = "cuda"
    keys = [("te", k) for k in meta[0]["te"]] + [("den", k) for k in meta[0]["den"]]
    for cb in combos:
        cb["acc"] = {key: torch.zeros_like(meta[0][key[0]][key[1]], dtype=torch.float32, device=dev) for key in keys}
    for i, m in enumerate(meta):
        tensors = {key: m[key[0]][key[1]].to(dev, torch.float32) for key in keys}
        for cb in combos:
            if i in cb["idx"]:
                cj = cb["c"][cb["idx"].index(i)]
                for key in keys:
                    cb["acc"][key].add_(tensors[key], alpha=cj)

    val = load_val(args.data, split_shards(full_shard_list(args.data))[1], args.val_samples)
    te, den = build_models(*configs_from_ckpt(meta[0]))
    te.to(dev), den.to(dev)

    def score(te_sd, den_sd):
        te.load_state_dict(te_sd), den.load_state_dict(den_sd)
        return evaluate(te, den, val, dev)["val/loss"]

    results = []
    raw = score(*({k: v.to(dev) for k, v in meta[-1][part].items()} for part in ("te", "den")))
    print(f"{'raw weights @ last step':34s} val/loss {raw:.5f}")
    latest = os.path.join(args.run, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        n_te = len(ck["te"])
        ema_te = dict(zip(ck["te"].keys(), ck["ema"][:n_te]))
        ema_den = dict(zip(ck["den"].keys(), ck["ema"][n_te:]))
        print(f"{'EMA @ step ' + str(ck['step']):34s} val/loss {score(ema_te, ema_den):.5f}")
    for cb in combos:
        te_sd = {k: v for (part, k), v in cb["acc"].items() if part == "te"}
        den_sd = {k: v for (part, k), v in cb["acc"].items() if part == "den"}
        loss = score(te_sd, den_sd)
        results.append((loss, cb, te_sd, den_sd))
        label = f"WSM {cb['scheme']:7s} window {cb['window']:.0%} ({len(cb['idx'])} ckpt)"
        print(f"{label:34s} val/loss {loss:.5f}")

    loss, cb, te_sd, den_sd = min(results, key=lambda r: r[0])
    out = args.out or os.path.join(args.run, "merged.pt")
    te_sd = {k: v.cpu() for k, v in te_sd.items()}
    den_sd = {k: v.cpu() for k, v in den_sd.items()}
    # same layout as a training checkpoint; "ema" holds the merged weights too, so export.py picks them by default
    torch.save(dict(step=last, den_config=meta[0]["den_config"], te_config=meta[0].get("te_config", {}), te=te_sd, den=den_sd,
                    ema=list(te_sd.values()) + list(den_sd.values()),
                    wsm=dict(scheme=cb["scheme"], window=cb["window"], steps=[steps[i] for i in cb["idx"]],
                             weights=cb["c"], val_loss=loss)), out)
    print(f"best: {cb['scheme']} window {cb['window']:.0%} val/loss {loss:.5f} -> {out}")


if __name__ == "__main__":
    main()
