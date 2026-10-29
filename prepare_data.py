"""Convert QLNI/FLUX-Reason-6M-flux2-latents (.npz shards) into the mmap-friendly layout read by aurora/data.py.

  # 0) look at one raw shard first (caption format check)
  python prepare_data.py inspect data/raw/fluxdb-imaginative-00127-of-00168.npz

  # 1) download + convert shard by shard (raw .npz deleted after conversion, so peak disk ~ converted size + a few GB).
  #    Default --fraction 0.1: 10% of the training shards (~39 GB) + the held-out val shards.
  python prepare_data.py convert --out data/aurora --download --delete-raw --workers 6
  #    or from already downloaded files:
  python prepare_data.py convert --out data/aurora --src /data/raw

  #    the full dataset (~385 GB); already converted shards are skipped, so this tops up a 10% download:
  python prepare_data.py convert --out data/aurora --download --delete-raw --workers 6 --fraction 1

  # 2) tokenize captions (uses the released BPE tokenizer by default)
  python prepare_data.py tokenize --data data/aurora --tokenizer AuroraIMG2-50M-Preview/tokenizer/tokenizer.json

Only `lat` and `caps` are kept; `dino` (first 60 shards) and `clip` are dropped.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from aurora.data import select_fraction, split_shards

REPO = "QLNI/FLUX-Reason-6M-flux2-latents"
MAX_TOKENS = 64
PAD_ID, BOS_ID, EOS_ID = 0, 2, 3


def replace_file(path, write):
    """Write via a temp file + rename: atomic, and never modifies a file hard-linked from another data dir
    (`cp -al data/aurora data/aurora_full` shares inodes)."""
    tmp = path + ".tmp"
    write(tmp)
    os.replace(tmp, path)


def save_npy(path, arr):
    replace_file(path, lambda tmp: np.save(open(tmp, "wb"), arr))


def save_json(path, obj, **kw):
    def write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, **kw)
    replace_file(path, write)


def normalize_caps(c):
    """Flatten whatever one `caps` entry is (list / array / dict / JSON string / str) into a list of strings."""
    if isinstance(c, np.ndarray):
        c = c.tolist()
    if isinstance(c, bytes):
        c = c.decode("utf-8", "replace")
    if c is None:
        return []
    if isinstance(c, str):
        s = c.strip()
        if s[:1] in "[{":
            try:
                return normalize_caps(json.loads(s))
            except json.JSONDecodeError:
                pass
        return [s] if s else []
    if isinstance(c, dict):
        c = list(c.values())
    if isinstance(c, (list, tuple)):
        return [s for x in c for s in normalize_caps(x)]
    return []


def cmd_inspect(args):
    import zipfile
    print([(i.filename, i.file_size, "deflated" if i.compress_type else "stored")
           for i in zipfile.ZipFile(args.file).infolist()])
    d = np.load(args.file, allow_pickle=True)
    for k in d.files:
        a = d[k]
        print(f"{k}: shape={a.shape} dtype={a.dtype}")
    lat = d["lat"][:256].astype(np.float32)
    print(f"lat[:256] mean={lat.mean():.4f} std={lat.std():.4f} (normalised latents should be ~0 / ~1)")
    for i in range(2):
        print(f"--- caps[{i}] raw type {type(d['caps'][i]).__name__}:\n{repr(d['caps'][i])[:2000]}")
        print(f"--- normalised -> {normalize_caps(d['caps'][i])}")


def convert_one(src: str, out: str, delete_raw: bool):
    name = os.path.basename(src)[: -len(".npz")]
    d = np.load(src, allow_pickle=True)
    lat = np.ascontiguousarray(d["lat"], dtype=np.float16)
    assert lat.shape[1:] == (128, 16, 16), lat.shape
    caps = [normalize_caps(c) for c in d["caps"]]
    assert len(caps) == len(lat)
    tmp = os.path.join(out, name + ".lat.tmp.npy")
    np.save(tmp, lat)
    save_json(os.path.join(out, name + ".caps.json"), caps, ensure_ascii=False)
    os.replace(tmp, os.path.join(out, name + ".lat.npy"))  # .lat.npy appears last = shard done
    if delete_raw:
        os.remove(src)
    return name, len(lat), sum(map(len, caps))


def download_and_convert(fname: str, raw: str, out: str, delete_raw: bool):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(REPO, fname, repo_type="dataset", local_dir=raw)
    return convert_one(path, out, delete_raw)


def write_manifest(out, files):
    """Full shard list of the source, so train.py --fraction picks the same shards whatever has been downloaded."""
    save_json(os.path.join(out, "manifest.json"),
              {"shards": sorted(os.path.basename(x)[: -len(".npz")] for x in files)}, indent=0)


def pick_fraction(files, fraction):
    """Training-shard fraction plus all held-out val shards, so val/loss stays comparable with full-data runs."""
    if fraction >= 1:
        return files
    by_name = {os.path.basename(f)[: -len(".npz")]: f for f in files}
    train, val = split_shards(sorted(by_name))
    keep = select_fraction(train, fraction) + val
    print(f"--fraction {fraction}: {len(keep) - len(val)}/{len(train)} training shards + {len(val)} val shards")
    return sorted(by_name[n] for n in keep)


def cmd_convert(args):
    os.makedirs(args.out, exist_ok=True)
    done = lambda n: os.path.exists(os.path.join(args.out, n + ".lat.npy"))
    if args.download:
        from huggingface_hub import list_repo_files
        files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".npz"))
        if args.include:
            files = [f for f in files if args.include in f]
        files = files[: args.limit] if args.limit else files
        write_manifest(args.out, files)
        files = pick_fraction(files, args.fraction)
        todo = [f for f in files if not done(os.path.basename(f)[:-4])]
        raw = args.src or os.path.join(args.out, "_raw")
        jobs = [(download_and_convert, f, raw) for f in todo]
    else:
        files = sorted(glob.glob(os.path.join(args.src, "**", "*.npz"), recursive=True))
        write_manifest(args.out, files)
        files = pick_fraction(files, args.fraction)
        todo = [f for f in files if not done(os.path.basename(f)[:-4])]
        jobs = [(convert_one, f) for f in todo]
    print(f"{len(files)} shards, {len(files) - len(todo)} already converted, {len(todo)} to do")
    total = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(*job, args.out, args.delete_raw) for job in jobs]
        for i, fut in enumerate(as_completed(futs)):
            name, n, c = fut.result()
            total += n
            print(f"[{i + 1}/{len(futs)}] {name}: {n} images, {c} captions", flush=True)
    print(f"done, {total} images converted in this run")


def tokenize_one(name: str, data: str, tokenizer_path: str):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    with open(os.path.join(data, name + ".caps.json")) as f:
        caps = json.load(f)
    flat, off = [], np.zeros(len(caps) + 1, dtype=np.int64)
    for i, cs in enumerate(caps):
        # identical to AuroraIMG2Pipeline._tokenize: collapse whitespace, explicit leading space
        flat += [" " + " ".join(c.split()) for c in cs if c.split()]
        off[i + 1] = len(flat)
    rows = np.full((len(flat), MAX_TOKENS), PAD_ID, dtype=np.uint16)
    for j, enc in enumerate(tok.encode_batch(flat, add_special_tokens=False)):
        ids = enc.ids[: MAX_TOKENS - 2]
        rows[j, 0] = BOS_ID
        rows[j, 1:1 + len(ids)] = ids
        rows[j, 1 + len(ids)] = EOS_ID
    save_npy(os.path.join(data, name + ".tok.npy"), rows)
    save_npy(os.path.join(data, name + ".off.npy"), off)  # .off.npy written last = shard tokenised
    return name, len(caps), len(flat)


def cmd_tokenize(args):
    names = sorted(os.path.basename(p)[: -len(".caps.json")] for p in glob.glob(os.path.join(args.data, "*.caps.json")))
    todo = [n for n in names if args.force or not os.path.exists(os.path.join(args.data, n + ".off.npy"))]
    print(f"{len(names)} shards, {len(todo)} to tokenise")
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(tokenize_one, n, args.data, args.tokenizer) for n in todo]
        for i, fut in enumerate(as_completed(futs)):
            name, n, c = fut.result()
            print(f"[{i + 1}/{len(futs)}] {name}: {n} images, {c} captions", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inspect")
    s.add_argument("file")
    s = sub.add_parser("convert")
    s.add_argument("--out", default="data/aurora")
    s.add_argument("--src", help="dir with raw .npz (or download dir with --download)")
    s.add_argument("--download", action="store_true")
    s.add_argument("--include", help="only shards whose path contains this string")
    s.add_argument("--limit", type=int, default=0, help="only the first N shards (for tests)")
    s.add_argument("--fraction", type=float, default=0.1,
                   help="fraction of the training shards to download/convert (+ val shards); 1 = full dataset")
    s.add_argument("--delete-raw", action="store_true")
    s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("tokenize")
    s.add_argument("--data", default="data/aurora")
    s.add_argument("--tokenizer", default="AuroraIMG2-50M-Preview/tokenizer/tokenizer.json")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--force", action="store_true")
    args = p.parse_args()
    {"inspect": cmd_inspect, "convert": cmd_convert, "tokenize": cmd_tokenize}[args.cmd](args)


if __name__ == "__main__":
    main()
