"""Images for VAE-in-the-loader training (no stored latents), packed for random access by row:
<name>.img.bin (the encoded images back to back, bytes unchanged) + <name>.img.off.npy (int64 [N + 1] byte offsets),
read with mmap like the latents.

  # LucasFang/FLUX-Reason-6M, 1024 px JPEGs, shard by shard (each parquet is downloaded, packed and deleted, so the
  # disk peak is the packed size + --workers parquets). Shard names and row order are those of the latent shards in
  # --data (QLNI/FLUX-Reason-6M-flux2-latents), so their captions / tokens (.off.npy, .tok_poziomka*.npy) apply as is;
  # with --data the image count of every shard is checked against its .off.npy.
  python prepare_images.py flux-reason --out data/flux_reason_1024 --data data/aurora_full --workers 16
  python prepare_images.py flux-reason --out data/flux_reason_1024 --fraction 0.5   # half of the training shards

  # MONET parquets of prepare_monet.py: their 384 px WEBP thumbnails, next to each parquet
  python prepare_images.py monet --parquet-dir data/monet_z_image
"""
import argparse
import glob
import os
import shutil
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq

REPO = "LucasFang/FLUX-Reason-6M"


def pack(blobs, out_base):  # whole list in memory: small sources only (MONET thumbnails)
    """list of bytes -> <out_base>.img.bin + .img.off.npy, atomically (offsets written last = shard complete)."""
    off = np.zeros(len(blobs) + 1, dtype=np.int64)
    np.cumsum([len(b) for b in blobs], out=off[1:])
    with open(out_base + ".img.bin.tmp", "wb") as f:
        for b in blobs:
            f.write(b)
    os.replace(out_base + ".img.bin.tmp", out_base + ".img.bin")
    with open(out_base + ".img.off.npy.tmp", "wb") as f:
        np.save(f, off)
    os.replace(out_base + ".img.off.npy.tmp", out_base + ".img.off.npy")
    return int(off[-1])


def image_bytes(col):
    return [x["bytes"] if isinstance(x, dict) else x for x in col]


class PackWriter:
    """Streaming pack(): images appended one row group at a time, so a shard is never held in memory whole."""

    def __init__(self, out_base):
        self.base, self.sizes = out_base, []
        self.f = open(out_base + ".img.bin.tmp", "wb")

    def add(self, blobs):
        for b in blobs:
            self.f.write(b)
            self.sizes.append(len(b))

    def close(self):
        self.f.close()
        off = np.zeros(len(self.sizes) + 1, dtype=np.int64)
        np.cumsum(self.sizes, out=off[1:])
        os.replace(self.base + ".img.bin.tmp", self.base + ".img.bin")
        with open(self.base + ".img.off.npy.tmp", "wb") as f:
            np.save(f, off)
        os.replace(self.base + ".img.off.npy.tmp", self.base + ".img.off.npy")  # offsets last = shard complete
        return len(self.sizes), int(off[-1])


def flux_reason_one(job):
    """One shard, in its own short-lived process (Pool maxtasksperchild=1): whatever pyarrow / the downloader keep
    allocated goes back to the OS when it exits (in threads it piled up until the OOM killer stopped the run)."""
    path, out, data, tmp_root = job
    from huggingface_hub import hf_hub_download
    name = os.path.basename(path)[: -len(".parquet")]
    base = os.path.join(out, name)
    if os.path.exists(base + ".img.off.npy"):
        return name, 0, 0, True
    tmp = tempfile.mkdtemp(dir=tmp_root)
    try:
        local = hf_hub_download(REPO, path, repo_type="dataset", local_dir=tmp)
        pf = pq.ParquetFile(local, memory_map=True)
        w = PackWriter(base)
        for g in range(pf.metadata.num_row_groups):
            w.add(image_bytes(pf.read_row_group(g, columns=["image"])["image"].to_pylist()))
        if data:
            n_lat = len(np.load(os.path.join(data, f"{name}.off.npy"))) - 1
            assert n_lat == len(w.sizes), f"{name}: {len(w.sizes)} images, {n_lat} in {data}"
        n, size = w.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return name, n, size, False


def cmd_flux_reason(a):
    from huggingface_hub import HfFileSystem
    from aurora.data import select_fraction, split_shards
    files = sorted(p.split(f"datasets/{REPO}/", 1)[1] for p in HfFileSystem().glob(f"datasets/{REPO}/*/*.parquet"))
    by_name = {os.path.basename(p)[: -len(".parquet")]: p for p in files}
    train, val = split_shards(sorted(by_name))  # the same held-out shards as the latent training
    names = select_fraction(train, a.fraction) + val
    if a.max_shards:
        names = names[: a.max_shards]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(by_name)} parquets, {len(names)} selected ({len(val)} held-out) -> {a.out}", flush=True)
    t0, done, imgs, gb = time.time(), 0, 0, 0.0
    jobs = [(by_name[n], a.out, a.data, a.tmp or a.out) for n in names]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for name, n, size, skipped in pool.imap_unordered(flux_reason_one, jobs):
            done += 1
            imgs += n
            gb += size / 1e9
            el = time.time() - t0
            print(f"[{done}/{len(names)}] {name}: {'skip' if skipped else f'{n} images, {size / 1e9:.2f} GB'} | "
                  f"{imgs:,} images, {gb:.1f} GB, eta {(len(names) - done) * el / done / 3600:.1f} h", flush=True)


def cmd_monet(a):
    files = sorted(f for f in glob.glob(os.path.join(a.parquet_dir, "*", "*.parquet"))
                   if os.path.basename(f).count(".") == 1)
    for i, f in enumerate(files):
        base = f[: -len(".parquet")]
        if os.path.exists(base + ".img.off.npy"):
            continue
        blobs = image_bytes(pq.read_table(f, columns=["thumbnail"])["thumbnail"].to_pylist())
        n, size = len(blobs), pack(blobs, base)
        print(f"[{i + 1}/{len(files)}] {os.path.relpath(f, a.parquet_dir)}: {n} images, {size / 1e6:.0f} MB", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("flux-reason")
    s.add_argument("--out", required=True)
    s.add_argument("--data", default="", help="latent shards to check the image counts against (their .off.npy)")
    s.add_argument("--fraction", type=float, default=1.0, help="of the training shards (held-out ones always)")
    s.add_argument("--workers", type=int, default=16, help="parquets downloaded and packed at once (one process each)")
    s.add_argument("--tmp", default="", help="where the parquets are downloaded to (default --out)")
    s.add_argument("--max-shards", type=int, default=0, help="only the first N selected shards (a test)")
    s = sub.add_parser("monet")
    s.add_argument("--parquet-dir", required=True)
    a = p.parse_args()
    {"flux-reason": cmd_flux_reason, "monet": cmd_monet}[a.cmd](a)


if __name__ == "__main__":
    main()
