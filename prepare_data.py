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
  #    for a frozen-LM text encoder, e.g. Poziomka, English and then the Polish translations (translate_captions.py):
  python prepare_data.py tokenize --data data/aurora --encoder poziomka
  python prepare_data.py tokenize --data data/aurora --encoder poziomka --lang pl
  #    MONET parquets (prepare_monet.py), English caption_short and its translation (translate_captions.py --parquet-dir):
  python prepare_data.py tokenize-parquet --parquet-dir data/monet_z_image
  python prepare_data.py tokenize-parquet --parquet-dir data/monet_z_image --lang pl

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


# text encoders with their own caption tokenisation (stored next to the default .tok.npy)
ENCODERS = {
    # frozen LM: [<|endoftext|>] ids[:95], right-padded with 65535 (see aurora/slayer.py)
    "slayer149": dict(repo="SlayerLab/Slayer149", width=96, bos=12285, pad=65535, leading_space=False),
    # [EOS = document separator] ids[:255]; 256 covers p95 of the English captions (243 tokens), Polish ones are shorter
    "poziomka": dict(repo="cpral/Poziomka-Baza-2026-09-05", width=256, bos=1, pad=65535, leading_space=False),
}


def tokenize_one(name: str, data: str, tokenizer_path: str, encoder: str = "", lang: str = ""):
    """lang (e.g. "pl"): the translated captions from <shard>.caps_<lang>.json (translate_captions.py), row for row
    aligned with the English ones (same .off.npy); a translation flagged as broken falls back to the English caption."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    enc_cfg = ENCODERS.get(encoder)
    lead = " " if enc_cfg is None else ("" if not enc_cfg["leading_space"] else " ")
    with open(os.path.join(data, name + ".caps.json")) as f:
        caps = json.load(f)
    trans, fallback = None, 0
    if lang:
        from translate_captions import suspicious
        with open(os.path.join(data, f"{name}.caps_{lang}.json")) as f:
            trans = json.load(f)
        assert len(trans) == len(caps), name
    flat, off = [], np.zeros(len(caps) + 1, dtype=np.int64)
    for i, cs in enumerate(caps):
        # Aurora tokenizer: identical to AuroraIMG2Pipeline._tokenize (collapsed whitespace, explicit leading space)
        if trans is None:
            flat += [lead + " ".join(c.split()) for c in cs if c.split()]
        else:
            for c, t in zip(cs, trans[i]):
                if c.split():  # same captions kept as for English, so the rows line up
                    bad = suspicious(c, t)
                    fallback += bad
                    flat.append(lead + " ".join((c if bad else t).split()))
        off[i + 1] = len(flat)
    encoded = tok.encode_batch(flat, add_special_tokens=False)
    if enc_cfg is None:
        rows = np.full((len(flat), MAX_TOKENS), PAD_ID, dtype=np.uint16)
        for j, e in enumerate(encoded):
            ids = e.ids[: MAX_TOKENS - 2]
            rows[j, 0] = BOS_ID
            rows[j, 1:1 + len(ids)] = ids
            rows[j, 1 + len(ids)] = EOS_ID
        out = name + ".tok.npy"
    else:
        w = enc_cfg["width"]
        rows = np.full((len(flat), w), enc_cfg["pad"], dtype=np.uint16)
        for j, e in enumerate(encoded):
            ids = e.ids[: w - 1]
            rows[j, 0] = enc_cfg["bos"]
            rows[j, 1:1 + len(ids)] = ids
        out = f"{name}.tok_{encoder}{'_' + lang if lang else ''}.npy"
    save_npy(os.path.join(data, out), rows)
    if lang:  # .off.npy belongs to the English captions; check instead of rewriting it
        assert np.array_equal(np.load(os.path.join(data, name + ".off.npy")), off), f"{name}: captions do not line up"
    else:
        save_npy(os.path.join(data, name + ".off.npy"), off)  # .off.npy written last = shard tokenised
    return name, len(caps), len(flat), fallback


def tokenize_parquet_one(path: str, tokenizer_path: str, encoder: str, column: str, lang: str = ""):
    """prepare_monet.py parquet (one caption per image in `column`, or its translation <file>.<lang>.parquet column
    <column>_<lang>) -> <file>.tok_<encoder>[_<lang>].npy, one row per image in the parquet's order. "The image shows"
    boilerplate is dropped from the English (translate_captions.strip_prefix); the translations are taken as they are,
    since translate_captions.py --strip-prefix already dropped it before translating (stripping "Obraz przedstawia"
    afterwards would also cut "A painting of ..." -> "Obraz przedstawia ..."). A missing caption, or a translation
    flagged as broken, becomes the English caption (Polish) or an empty prompt (English: BOS only, the CFG null prompt)."""
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    from translate_captions import strip_prefix, suspicious
    tok, cfg = Tokenizer.from_file(tokenizer_path), ENCODERS[encoder]
    base = path[: -len(".parquet")]
    en = [strip_prefix(c) for c in pq.read_table(path, columns=[column])[column].to_pylist()]
    texts, fallback = en, 0
    if lang:
        pl = pq.read_table(f"{base}.{lang}.parquet")
        assert pl["__key__"].to_pylist() == pq.read_table(path, columns=["__key__"])["__key__"].to_pylist(), path
        texts = []
        for e, p in zip(en, pl[f"{column}_{lang}"].to_pylist()):
            bad = bool(e) and (not p or suspicious(e, p))
            fallback += bad
            texts.append(e if bad else p)
    flat = [" ".join((c or "").split()) for c in texts]
    w = cfg["width"]
    rows = np.full((len(flat), w), cfg["pad"], dtype=np.uint16)
    for j, e in enumerate(tok.encode_batch(flat, add_special_tokens=False)):
        ids = e.ids[: w - 1]
        rows[j, 0] = cfg["bos"]
        rows[j, 1:1 + len(ids)] = ids
    save_npy(f"{base}.tok_{encoder}{'_' + lang if lang else ''}.npy", rows)
    if not lang:  # one caption per image: the aurora.data.ShardStore layout (<name>.off.npy) for training
        save_npy(f"{base}.off.npy", np.arange(len(flat) + 1, dtype=np.int64))
    return path, len(flat), sum(not c for c in flat), fallback


def cmd_tokenize_parquet(args):
    from huggingface_hub import hf_hub_download
    tokenizer = hf_hub_download(ENCODERS[args.encoder]["repo"], "tokenizer.json")
    suffix = f"tok_{args.encoder}" + (f"_{args.lang}" if args.lang else "")
    files = sorted(f for f in glob.glob(os.path.join(args.parquet_dir, "*", "*.parquet"))
                   if os.path.basename(f).count(".") == 1)  # not the <file>.<lang>.parquet translations
    todo = [f for f in files if args.force or not os.path.exists(f"{f[: -len('.parquet')]}.{suffix}.npy")]
    if args.lang:
        todo = [f for f in todo if os.path.exists(f"{f[: -len('.parquet')]}.{args.lang}.parquet")]
    print(f"{len(files)} parquets, {len(todo)} to tokenise for {suffix}")
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(tokenize_parquet_one, f, tokenizer, args.encoder, args.column, args.lang) for f in todo]
        for i, fut in enumerate(as_completed(futs)):
            path, n, empty, fb = fut.result()
            print(f"[{i + 1}/{len(futs)}] {os.path.relpath(path, args.parquet_dir)}: {n} images, {empty} empty"
                  + (f", {fb} flagged -> English" if args.lang else ""), flush=True)


def cmd_tokenize(args):
    names = sorted(os.path.basename(p)[: -len(".caps.json")] for p in glob.glob(os.path.join(args.data, "*.caps.json")))
    tokenizer = args.tokenizer
    if args.encoder:
        from huggingface_hub import hf_hub_download
        tokenizer = hf_hub_download(ENCODERS[args.encoder]["repo"], "tokenizer.json")
    assert not args.lang or args.encoder, "--lang needs --encoder (and the English captions tokenised first)"
    suffix = f"tok_{args.encoder}" + (f"_{args.lang}" if args.lang else "")
    done = lambda n: os.path.exists(os.path.join(args.data, f"{n}.{suffix}.npy" if args.encoder else n + ".off.npy"))
    todo = [n for n in names if args.force or not done(n)]
    if args.lang:  # shards without a translation are skipped (train.py then refuses --pl-prob)
        todo = [n for n in todo if os.path.exists(os.path.join(args.data, f"{n}.caps_{args.lang}.json"))]
    print(f"{len(names)} shards, {len(todo)} to tokenise" + (f" for {suffix}" if args.encoder else ""))
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(tokenize_one, n, args.data, tokenizer, args.encoder, args.lang) for n in todo]
        for i, fut in enumerate(as_completed(futs)):
            name, n, c, fb = fut.result()
            print(f"[{i + 1}/{len(futs)}] {name}: {n} images, {c} captions"
                  + (f", {fb} flagged -> English" if args.lang else ""), flush=True)


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
    s.add_argument("--encoder", choices=[""] + sorted(ENCODERS), default="",
                   help="tokenise for another text encoder (writes <shard>.tok_<encoder>.npy, keeps .tok.npy)")
    s.add_argument("--lang", default="", help="tokenise the translated captions <shard>.caps_<lang>.json instead "
                   "(writes <shard>.tok_<encoder>_<lang>.npy; e.g. --encoder poziomka --lang pl)")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("tokenize-parquet", help="prepare_monet.py parquets: one caption per image")
    s.add_argument("--parquet-dir", required=True)
    s.add_argument("--encoder", choices=sorted(ENCODERS), default="poziomka")
    s.add_argument("--column", default="caption_short")
    s.add_argument("--lang", default="", help="the translation <file>.<lang>.parquet (translate_captions.py)")
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--force", action="store_true")
    args = p.parse_args()
    {"inspect": cmd_inspect, "convert": cmd_convert, "tokenize": cmd_tokenize,
     "tokenize-parquet": cmd_tokenize_parquet}[args.cmd](args)


if __name__ == "__main__":
    main()
