"""train_pp.lm_split_forward (--lm-split: the frozen LM's layers spread over the pipeline stages) gives the same LM
states as the whole LM on one rank: a small stand-in LM with the BailingMoeV2 layout, on CPU (gloo).

  DIST_BACKEND=gloo torchrun --nproc_per_node 2 tests/check_lm_split.py
  DIST_BACKEND=gloo torchrun --nproc_per_node 3 tests/check_lm_split.py
"""
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, ".")
from aurora.slayer import LM_NULL, LM_PAD, hf_lm_part, split_hf_lm, trim_ids
from train_pp import lm_split_forward

H, V, L = 32, 200, 6


class Layer(nn.Module):  # causal toy layer: mixes each position with the running mean of the previous ones
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(H, H)

    def forward(self, x, position_ids=None, position_embeddings=None, use_cache=False):
        prev = x.cumsum(1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
        return (x + torch.tanh(self.lin(prev + position_embeddings)),)


def make_lm():
    torch.manual_seed(0)
    m = nn.Module()
    m.word_embeddings, m.layers, m.norm = nn.Embedding(V, H), nn.ModuleList(Layer() for _ in range(L)), nn.LayerNorm(H)
    m.rotary_emb = lambda x, pos: pos[..., None].to(x.dtype) * 0.01
    lm = nn.Module()
    lm.model = m
    return lm.to(torch.bfloat16)


dist.init_process_group("gloo")
rank, pp = dist.get_rank(), dist.get_world_size()
torch.manual_seed(1)
chunks = []
for n, w in ((5, 40), (7, 64), (3, 16)):
    c = torch.full((n, 64), LM_PAD, dtype=torch.long)
    for r in range(n):
        k = torch.randint(2, w, ()).item()
        c[r, :k] = torch.randint(1, V, (k,))
    c[0, :] = LM_PAD
    c[0, 0] = LM_NULL  # a CFG null prompt in every chunk
    chunks.append(c)
chunks.append(torch.full((2, 64), LM_PAD, dtype=torch.long).index_fill_(1, torch.tensor([0]), LM_NULL))  # all null

lm = make_lm()
ref = []
for c in chunks:  # whole LM, one rank
    real = c[:, 0].ne(LM_NULL)
    out = torch.zeros(c.shape[0], c.shape[1], H, dtype=torch.bfloat16)
    if real.any():
        sub = trim_ids(c[real], multiple=8)
        out[real, : sub.shape[1]] = hf_lm_part(lm, sub.masked_fill(sub.eq(LM_PAD), 0), True, True)
    ref.append(out)
split_hf_lm(lm, rank * L // pp, (rank + 1) * L // pp)
got = lm_split_forward(lm, chunks, rank, pp, 0, torch.device("cpu"), H)
if rank == 0:
    diff = max((a.float() - b.float()).abs().max().item() for a, b in zip(got, ref))
    print(f"{pp} stages, layers per stage {[(r + 1) * L // pp - r * L // pp for r in range(pp)]}: max diff {diff}")
    print("OK" if diff == 0 else "MISMATCH")
dist.destroy_process_group()
