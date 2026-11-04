"""Cost of a frozen LM text encoder inside training: forward time and GPU memory for a micro-batch of captions, per
number of kept layers and active MoE experts (weights are what stays resident next to the training state).

  python tests/bench_te.py cpral/Poziomka-Baza-2026-09-05 --layers 8 12 16 --experts 16 32 --batch 32 --tokens 256

Reference for the 2560x6 denoiser: one micro-batch of 32 takes ~0.3 s fwd+bwd on a 3090 Ti, so a text encoder at
~0.03 s per micro-batch costs ~10%.
"""
import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, ".")
from aurora.hf_compat import load_causal_lm

p = argparse.ArgumentParser()
p.add_argument("repo")
p.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16])
p.add_argument("--experts", type=int, nargs="+", default=[16, 32])
p.add_argument("--batch", type=int, default=32)
p.add_argument("--tokens", type=int, default=256)
p.add_argument("--compile", action="store_true", help="also try torch.compile on the decoder")
a = p.parse_args()
dev = "cuda"
print(f"{torch.cuda.get_device_name()}, micro-batch {a.batch} x {a.tokens} tokens\n")
print(f"{'layers':>6} {'experts':>7} {'weights GB':>10} {'peak GB':>8} {'fwd ms':>8}")
for L in a.layers:
    for E in a.experts:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            model, tok = load_causal_lm(a.repo, dev, num_layers=L, num_experts_per_tok=E, fused_moe=a.fused_moe)
        except Exception as e:  # noqa: BLE001
            print(f"{L:>6} {E:>7}  load failed: {e!r}")
            continue
        w = (torch.cuda.memory_allocated() - base) / 2**30
        body = model.model
        if a.compile:
            body = torch.compile(body)
        ids = torch.randint(1000, 30000, (a.batch, a.tokens), device=dev)
        with torch.no_grad():
            for _ in range(2):
                body(input_ids=ids)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            n = 5
            for _ in range(n):
                body(input_ids=ids)
            torch.cuda.synchronize()
        ms = (time.time() - t0) / n * 1000
        peak = (torch.cuda.max_memory_allocated() - base) / 2**30
        print(f"{L:>6} {E:>7} {w:10.2f} {peak:8.2f} {ms:8.1f}", flush=True)
        del model, body, tok
        gc.collect()
