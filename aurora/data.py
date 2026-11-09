"""Prepared-data layout (written by prepare_data.py), one set of files per source shard <name>:

  <name>.lat.npy   float16 [N, 128, 16, 16]   latents, already BatchNorm-normalised + 2x2 packed by the dataset
  <name>.tok.npy   uint16  [C, 64]            tokenised captions: [BOS] ids[:62] [EOS] [PAD]...
  <name>.off.npy   int64   [N + 1]            captions of image i are tok[off[i]:off[i+1]]
  <name>.tok_<enc>.npy  uint16 [C, L]          the same captions for another text encoder (e.g. tok_slayer149,
                                             L = 96, PAD = 65535); same order, so .off.npy is shared
  <name>.tok_<enc>_<lang>.npy                 the translated captions (e.g. tok_poziomka_pl), row for row the same

Everything is opened with mmap, so the page cache is shared by all ranks and loader workers on a node.
"""
import glob
import json
import os
import re
import resource

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

MAX_TOKENS = 64
PAD_ID, BOS_ID, EOS_ID, NULL_ID, MASK_ID = 0, 2, 3, 4, 5


def null_ids() -> np.ndarray:
    row = np.full(MAX_TOKENS, PAD_ID, dtype=np.int64)
    row[:3] = (BOS_ID, NULL_ID, EOS_ID)
    return row


def list_shards(root: str, img_root: str = ""):
    """Shards with captions (.off.npy) and latents (.lat.npy), or with images (<img_root>/<name>.img.off.npy)."""
    names = sorted(os.path.basename(p)[: -len(".off.npy")] for p in glob.glob(os.path.join(root, "*.off.npy"))
                   if not p.endswith(".img.off.npy"))
    if img_root:
        return [n for n in names if os.path.exists(os.path.join(img_root, n + ".img.off.npy"))]
    return [n for n in names if os.path.exists(os.path.join(root, n + ".lat.npy"))]


def _by_subset(names):
    groups = {}
    for n in sorted(names):
        groups.setdefault(re.sub(r"-\d+-of-\d+$", "", n), []).append(n)
    return groups


def split_shards(names, val_names=None):
    """Default hold-out: the last shard of every subset (aesthetics-part01, -part02, imaginative, text)."""
    if val_names is None:
        val_names = [g[-1] for g in _by_subset(names).values() if len(g) > 1]
    val = [n for n in names if n in set(val_names)]
    return [n for n in names if n not in set(val)], val


def select_fraction(train_names, fraction: float):
    """Deterministic subset of training shards: evenly spaced shards from every subset, so the mix of
    aesthetics / imaginative / text stays the same as in the full dataset. Used by both prepare_data.py and train.py,
    so `convert --fraction f` downloads exactly what `train --train-fraction f` would pick from the full data."""
    if fraction >= 1:
        return list(train_names)
    picked = []
    for g in _by_subset(train_names).values():
        k = max(1, round(fraction * len(g)))
        picked += [g[i] for i in sorted(set(np.linspace(0, len(g) - 1, k).round().astype(int)))]
    return sorted(picked)


def full_shard_list(root: str):
    """All shards of the source dataset (manifest.json from prepare_data.py), else whatever is prepared."""
    manifest = os.path.join(root, "manifest.json")
    return json.load(open(manifest))["shards"] if os.path.exists(manifest) else list_shards(root)


def tok_kind(suffix: str = "") -> str:
    return f"tok_{suffix}" if suffix else "tok"


def select_shards(root: str, fraction: float, tok_suffix: str = "", alt_suffix: str = "", img_root: str = ""):
    """(train, val) shard names for a fraction of the FULL dataset; val shards do not depend on the fraction.
    img_root: shards must have images there (prepare_images.py) instead of latents."""
    kinds = [tok_kind(tok_suffix)] + ([tok_kind(alt_suffix)] if alt_suffix else [])
    present = [n for n in list_shards(root, img_root)
               if all(os.path.exists(os.path.join(root, f"{n}.{k}.npy")) for k in kinds)]
    train, val = split_shards(full_shard_list(root))
    train = select_fraction(train, fraction)
    missing = sorted(set(train + val) - set(present))
    if missing:
        raise SystemExit(f"{len(missing)} shards selected by --fraction {fraction} are not prepared in {root} "
                         f"(e.g. {missing[0]}); run: prepare_data.py convert ... --fraction {fraction} + tokenize"
                         + (f" --encoder {tok_suffix}" if tok_suffix else "")
                         + (f" (and --lang {alt_suffix.split('_')[-1]})" if alt_suffix else ""))
    assert train, f"no prepared shards in {root}"
    return train, val


def ensure_open_files(needed: int):
    """Every mmap holds a file descriptor and each DataLoader worker maps 2 files per shard (~2300 for the full set),
    above the usual soft limit of 1024. Raise the soft limit to the hard one (inherited by forked workers)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY or soft >= needed:
        return
    target = needed if hard == resource.RLIM_INFINITY else min(max(needed, 65536), hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    if target < needed:
        raise SystemExit(f"need ~{needed} open files per process for {needed // 2} shards, hard limit is {hard}; "
                         f"raise it (e.g. `ulimit -Hn 65536` as root / limits.conf)")


class ShardStore:
    def __init__(self, root: str, names, tok_suffix: str = "", null_row=None, alt_suffix: str = "", alt_prob=0.0,
                 lat_kind: str = "lat", lat_root: str = "", crops=(), crop_prob: float = 0.0, cond_base: int = 0,
                 img_res: int = 0):
        """alt_suffix / alt_prob: with probability alt_prob a sample's caption comes from <name>.tok_<alt_suffix>.npy
        (the same caption translated, e.g. poziomka_pl) instead. lat_kind: which latents, e.g. "lat128" for the
        <name>.lat128.npy files of prepare_lowres.py (any grid; the shape is read from the files).
        RPE-2D's resize-and-crop augmentation (arXiv:2503.18719): with probability crop_prob a training sample is a
        random window, as large as the lat_kind grid, of the latents of one of `crops` [(lat_kind, lat_root)] (larger
        grids of the same images, e.g. the 256 px "lat") instead of the whole resized image. cond_base > 0: fetch also
        returns the (B, 8) micro-conditions in pixels of a cond_base-px original (aurora.model.default_micro_cond).
        img_res > 0: images instead of latents, from <lat_root>/<name>.img.bin + .img.off.npy (prepare_images.py),
        decoded and resized to img_res x img_res here (in the DataLoader workers) -> uint8 (B, 3, img_res, img_res);
        the latents are made on the GPU (aurora.vae_encode.LatentEncoder)."""
        self.root, self.names, self.lat_kind = root, list(names), lat_kind
        self.crops, self.crop_prob, self.cond_base, self.img_res = list(crops), crop_prob, cond_base, img_res
        assert not self.crops or cond_base, "crops need micro-conditions (cond_base)"
        assert not (img_res and self.crops), "latent crops and images do not mix"
        self.lat_root = lat_root or root  # latents may live in another folder (prepare_lowres.py --out-dir)
        self.tok = tok_kind(tok_suffix)
        self.alt = tok_kind(alt_suffix) if alt_suffix and alt_prob > 0 else None
        self.alt_prob = alt_prob
        self.null_row = null_ids() if null_row is None else np.asarray(null_row, dtype=np.int64)
        ensure_open_files(((3 if self.alt else 2) + len(self.crops)) * len(self.names) + 512)
        self.offs = [np.load(self._p(n, "off")) for n in self.names]
        self.sizes = np.array([len(o) - 1 for o in self.offs], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.sizes)])
        self._lat, self._tok, self._alt, self._crop = None, None, None, None

    def _p(self, name, kind):
        return os.path.join(self.root, f"{name}.{kind}.npy")

    def __len__(self):
        return int(self.starts[-1])

    def _open(self):  # lazily, so every DataLoader worker gets its own handles
        if self._lat is None:
            if self.img_res:
                self._lat = [(np.memmap(os.path.join(self.lat_root, f"{n}.img.bin"), dtype=np.uint8, mode="r"),
                              np.load(os.path.join(self.lat_root, f"{n}.img.off.npy"))) for n in self.names]
            else:
                self._lat = [np.load(os.path.join(self.lat_root, f"{n}.{self.lat_kind}.npy"), mmap_mode="r")
                             for n in self.names]
            self._tok = [np.load(self._p(n, self.tok), mmap_mode="r") for n in self.names]
            if self.alt:
                self._alt = [np.load(self._p(n, self.alt), mmap_mode="r") for n in self.names]
            self._crop = [[np.load(os.path.join(r or self.root, f"{n}.{k}.npy"), mmap_mode="r") for n in self.names]
                          for k, r in self.crops]

    def fetch(self, indices: np.ndarray, rng: np.random.Generator | None, rows: bool = False):
        """Latents + one caption per image (random if rng is given, else the first one). rows: also (B, 2) [shard,
        caption row] of the caption picked (-1 for images without one); the row is the same in the translated files,
        so it also finds the English original of a Polish caption (CaptionText)."""
        self._open()
        picked = np.full((len(indices), 2), -1, dtype=np.int64)
        r = self.img_res
        lat = (np.empty((len(indices), 3, r, r), dtype=np.uint8) if r
               else np.empty((len(indices), *self._lat[0].shape[1:]), dtype=np.float16))
        ids = np.empty((len(indices), self._tok[0].shape[1]), dtype=np.int64)
        n = lat.shape[-1]  # target grid side
        cond = np.tile(np.array([self.cond_base] * 2 + [0, 0] + [self.cond_base] * 2 + [16 * n] * 2, np.float32),
                       (len(indices), 1))
        shard = np.searchsorted(self.starts, indices, side="right") - 1
        for j, (s, gi) in enumerate(zip(shard, indices)):
            i = gi - self.starts[s]
            if self._crop and rng is not None and rng.random() < self.crop_prob:
                src = self._crop[rng.integers(len(self._crop))][s]
                side = src.shape[-1]
                r, c = rng.integers(0, side - n + 1, size=2)
                lat[j] = src[i, :, r:r + n, c:c + n]
                px = self.cond_base / side  # original pixels per source token
                cond[j, 2:6] = r * px, c * px, (r + n) * px, (c + n) * px
            elif r:
                blob, off = self._lat[s]
                lat[j] = decode_image(blob[off[i]:off[i + 1]], r)
            else:
                lat[j] = self._lat[s][i]
            lo, hi = self.offs[s][i], self.offs[s][i + 1]
            if hi == lo:
                ids[j] = self.null_row
                continue
            c = lo if rng is None else rng.integers(lo, hi)
            if rng is None:
                ids[j] = self._tok[s][c]
            else:
                ids[j] = (self._alt if self.alt and rng.random() < self.alt_prob else self._tok)[s][c]
            picked[j] = s, c
        out = (lat, ids, picked) if rows else (lat, ids)
        return out + (cond,) if self.cond_base else out


def decode_image(data, res: int) -> np.ndarray:
    """Encoded image bytes -> (3, res, res) uint8 RGB: a JPEG is first decoded at the smallest DCT scale still >= res
    (cheap), then resized with PIL's antialiased bicubic (as prepare_lowres.py's downscale). Square sources only."""
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(bytes(data)))
    im.draft("RGB", (res, res))
    im = im.convert("RGB")
    if im.size != (res, res):
        im = im.resize((res, res), Image.BICUBIC, reducing_gap=None)
    return np.asarray(im).transpose(2, 0, 1)


class ConcatStore:
    """Several ShardStores as one index space (e.g. FLUX-Reason + MONET images), in their natural proportions. The
    stores must give the same layout (images of one size, captions of one width, no micro-conditions)."""

    def __init__(self, stores):
        self.stores = list(stores)
        self.starts = np.concatenate([[0], np.cumsum([len(s) for s in self.stores])])

    def __len__(self):
        return int(self.starts[-1])

    def fetch(self, indices: np.ndarray, rng, rows: bool = False):
        indices = np.asarray(indices)
        which = np.searchsorted(self.starts, indices, side="right") - 1
        lat = ids = None
        for k, store in enumerate(self.stores):
            pos = np.nonzero(which == k)[0]
            if not len(pos):
                continue
            got = store.fetch(indices[pos] - self.starts[k], rng)
            assert len(got) == 2, "ConcatStore: stores with micro-conditions are not supported"
            if lat is None:
                lat = np.empty((len(indices), *got[0].shape[1:]), dtype=got[0].dtype)
                ids = np.empty((len(indices), got[1].shape[1]), dtype=got[1].dtype)
            assert got[1].shape[1] == ids.shape[1], "ConcatStore: caption widths differ between stores"
            lat[pos], ids[pos] = got
        return (lat, ids, np.full((len(indices), 2), -1, dtype=np.int64)) if rows else (lat, ids)


class CaptionText:
    """Caption strings by (shard, caption row) from <shard>.caps[_<lang>].json, the files the token rows were made
    from (prepare_data.py flattens them in the same order); a few shards kept parsed."""

    def __init__(self, root: str, names, lang: str = "", keep: int = 8):
        self.root, self.names, self.lang, self.keep = root, list(names), lang, keep
        self._cache = {}

    def _flat(self, s: int):
        if s not in self._cache:
            if len(self._cache) >= self.keep:
                self._cache.pop(next(iter(self._cache)))
            suffix = f"caps_{self.lang}" if self.lang else "caps"
            with open(os.path.join(self.root, f"{self.names[s]}.{suffix}.json")) as f:
                self._cache[s] = [c for caps in json.load(f) for c in caps]
        return self._cache[s]

    def __call__(self, rows) -> list:
        return ["" if s < 0 else self._flat(int(s))[int(c)] for s, c in rows]


class TrainStream(IterableDataset):
    """Deterministic, resumable stream of per-rank micro-batches.

    Every epoch is a seeded global permutation; global batch k of the epoch is perm[k*G:(k+1)*G], and rank r takes its
    contiguous slice of it. DataLoader workers take micro-batches round-robin, which the loader yields back in order,
    so resuming from `start_micro` reproduces exactly the same sequence.
    """

    def __init__(self, store: ShardStore, micro_batch: int, rank: int, world: int, seed: int, start_micro: int = 0,
                 sort_pad=None, rows: bool = False):
        """sort_pad: caption pad id -> rows ordered by caption length (stable), here in the worker, so the loader's
        pinned batch reaches the GPU without another copy (train_pp.py's similar-length micro-batches). rows: also
        yield ShardStore.fetch's (B, 2) caption rows."""
        self.store, self.mb, self.rank, self.world = store, micro_batch, rank, world
        self.seed, self.start, self.sort_pad, self.rows = seed, start_micro, sort_pad, rows
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
            got = self.store.fetch(idx, np.random.default_rng([self.seed, m, self.rank, 1]), rows=True)
            lat, ids, picked, *cond = (torch.from_numpy(x) for x in got)
            if self.sort_pad is not None:
                order = ids.ne(self.sort_pad).sum(1).argsort(stable=True)
                lat, ids, picked, cond = lat[order], ids[order], picked[order], [c[order] for c in cond]
            yield ((lat, ids, picked) if self.rows else (lat, ids)) + tuple(cond)
            m += nw


def load_val(root: str, names, n: int, tok_suffix: str = "", null_row=None, lat_kind: str = "lat", lat_root: str = "",
             img_res: int = 0):
    """First n images of the held-out shards (round-robin across shards), first caption each. img_res: uint8 images
    (ShardStore img_res) instead of latents."""
    store = ShardStore(root, names, tok_suffix, null_row, lat_kind=lat_kind, lat_root=lat_root, img_res=img_res)
    per = -(-n // len(names))
    idx = np.concatenate([store.starts[s] + np.arange(min(per, store.sizes[s])) for s in range(len(names))])[:n]
    lat, ids = store.fetch(idx, None)
    return torch.from_numpy(lat), torch.from_numpy(ids)
