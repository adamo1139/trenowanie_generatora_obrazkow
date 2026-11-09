"""MONET synthetic thumbnails + text (jasperai/monet, v1.2.0/synthetic/<source>): from every parquet only the 384 px
WEBP thumbnail and the small columns (prompt, VLM captions, scores, hashes, size) are read, i.e. not the embeddings or
the SANA latents (~2/3 of the bytes). Each parquet is written slimmed to <out>/<shard dir>/<file>.parquet with an extra
column `caption_short`: the shortest non-empty VLM caption of the image (mostly Florence-2's), the one to translate.
Files already done are skipped, so an interrupted run resumes; files are fetched in parallel.

  python prepare_monet.py --source z-image --out data/monet_z_image --workers 8
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = "datasets/jasperai/monet/v1.2.0/synthetic"
CAPTIONS = ["caption_florence-2-large", "caption_internvl-3-8b", "caption_sharegpt4v-7b", "caption_gemini-2.5-flash-lite"]


def shortest(table):
    cols = [table[c].to_pylist() for c in CAPTIONS if c in table.column_names]
    out = []
    for caps in zip(*cols):
        caps = [" ".join(c.split()) for c in caps if c and c.strip()]
        out.append(min(caps, key=lambda c: len(c.split())) if caps else None)
    return pa.array(out, pa.string())


def one(fs, path, out_dir):
    rel = path.split("/synthetic/", 1)[1].split("/", 1)[1]  # <shard dir>/<file>.parquet
    dst = os.path.join(out_dir, rel)
    if os.path.exists(dst):
        return dst, 0, True
    pf = pq.ParquetFile(fs.open(path, block_size=2 ** 23))
    cols = [n for n in pf.schema_arrow.names if not n.startswith("embedding_")]
    t = pf.read(columns=cols)
    t = t.append_column("caption_short", shortest(t))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    pq.write_table(t, dst + ".tmp", compression="zstd")
    os.replace(dst + ".tmp", dst)
    return dst, t.num_rows, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="z-image", choices=["z-image", "flux-klein", "flux-schnell"])
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-files", type=int, default=0, help="only the first N parquets (a test)")
    a = p.parse_args()
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{ROOT}/{a.source}/*/*.parquet"))
    if a.max_files:
        files = files[: a.max_files]
    print(f"{a.source}: {len(files)} parquet files -> {a.out}", flush=True)
    t0, rows, done = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, fs, f, a.out) for f in files]
        for fut in as_completed(futs):
            dst, n, skipped = fut.result()
            done += 1
            rows += n
            el = time.time() - t0
            print(f"[{done}/{len(files)}] {'skip' if skipped else f'{n} rows'} {dst} | {rows:,} rows, "
                  f"{rows / max(el, 1):.0f} rows/s, eta {(len(files) - done) * el / done / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
