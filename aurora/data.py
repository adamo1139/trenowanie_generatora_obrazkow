"""Prepared-data layout (written by prepare_data.py), one set of files per source shard <name>:

  <name>.lat.npy   float16 [N, 128, 16, 16]   latents, already BatchNorm-normalised + 2x2 packed by the dataset
  <name>.tok.npy   uint16  [C, 64]            tokenised captions: [BOS] ids[:62] [EOS] [PAD]...
  <name>.off.npy   int64   [N + 1]            captions of image i are tok[off[i]:off[i+1]]

Everything is opened with mmap, so the page cache is shared by all ranks and loader workers on a node.
"""
import glob
import os
import re

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

MAX_TOKENS = 64
PAD_ID, BOS_ID, EOS_ID, NULL_ID, MASK_ID = 0, 2, 3, 4, 5


def null_ids() -> np.ndarray:
    row = np.full(MAX_TOKENS, PAD_ID, dtype=np.int64)
    row[:3] = (BOS_ID, NULL_ID, EOS_ID)
    return row


def list_shards(root: str):
    names = sorted(os.path.basename(p)[: -len(".off.npy")] for p in glob.glob(os.path.join(root, "*.off.npy")))
    return [n for n in names if os.path.exists(os.path.join(root, n + ".lat.npy"))]


def split_shards(names, val_names=None):
    """Default hold-out: the last shard of every subset (aesthetics-part01, -part02, imaginative, text)."""
    if val_names is None:
        groups = {}
        for n in names:
            groups.setdefault(re.sub(r"-\d+-of-\d+$", "", n), []).append(n)
        val_names = [g[-1] for g in groups.values() if len(g) > 1]
    val = [n for n in names if n in set(val_names)]
    return [n for n in names if n not in set(val)], val


class ShardStore:
    def __init__(self, root: str, names):
        self.root, self.names = root, list(names)
        self.offs = [np.load(self._p(n, "off")) for n in self.names]
        self.sizes = np.array([len(o) - 1 for o in self.offs], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)])
        self._lat, self._tok = None, None

    def _p(self, name, kind):
        return os.path.join(self.root, f"{name}.{kind}.npy")

    def __len__(self):
        return int(self.starts[-1])

    def _open(self):  # lazily, so every DataLoader worker gets its own handles
        if self._lat is None:
            self._lat = [np.load(self._p(n, "lat"), mmap_mode="r") for n in self.names]
            self._tok = [np.load(self._p(n, "tok"), mmap_mode="r") for n in self.names]

    def fetch(self, indices: np.ndarray, rng: np.random.Generator | None):
        """Latents + one caption per image (random if rng is given, else the first one)."""
        self._open()
        lat = np.empty((len(indices), 128, 16, 16), dtype=np.float16)
        ids = np.empty((len(indices), MAX_TOKENS), dtype=np.int64)
        shard = np.searchsorted(self.starts, indices, side="right") - 1
        for j, (s, gi) in enumerate(zip(shard, indices)):
            i = gi - self.starts[s]
            lat[j] = self._lat[s][i]
            lo, hi = self.offs[s][i], self.offs[s][i + 1]
            if hi == lo:
                ids[j] = null_ids()
            else:
                ids[j] = self._tok[s][lo if rng is None else rng.integers(lo, hi)]
        return lat, ids


class TrainStream(IterableDataset):
    """Deterministic, resumable stream of per-rank micro-batches.

    Every epoch is a seeded global permutation; global batch k of the epoch is perm[k*G:(k+1)*G], and rank r takes its
    contiguous slice of it. DataLoader workers take micro-batches round-robin, which the loader yields back in order,
    so resuming from `start_micro` reproduces exactly the same sequence.
    """

    def __init__(self, store: ShardStore, micro_batch: int, rank: int, world: int, seed: int, start_micro: int = 0):
        self.store, self.mb, self.rank, self.world = store, micro_batch, rank, world
        self.seed, self.start = seed, start_micro
        self.per_epoch = len(store) // (micro_batch * world)  # micro-batches per rank per epoch

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        perm_epoch, perm = -1, None
        m = self.start + wid
        while True:
            epoch, k = divmod(m, self.per_epoch)
            if epoch != perm_epoch:
                perm = np.random.default_rng([self.seed, epoch]).permutation(len(self.store))
                perm_epoch = epoch
            g0 = k * self.mb * self.world + self.rank * self.mb
            idx = np.sort(perm[g0:g0 + self.mb])
            lat, ids = self.store.fetch(idx, np.random.default_rng([self.seed, m, self.rank, 1]))
            yield torch.from_numpy(lat), torch.from_numpy(ids)
            m += nw


def load_val(root: str, names, n: int):
    """First n images of the held-out shards (round-robin across shards), first caption each."""
    store = ShardStore(root, names)
    per = -(-n // len(names))
    idx = np.concatenate([store.starts[s] + np.arange(min(per, store.sizes[s])) for s in range(len(names))])[:n]
    lat, ids = store.fetch(idx, None)
    return torch.from_numpy(lat), torch.from_numpy(ids)
