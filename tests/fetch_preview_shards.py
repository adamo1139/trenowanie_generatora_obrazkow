"""One shard of each candidate image source, for a look before choosing data (tests/shard_viewer.py):
  * jasperai/monet synthetic (flux-klein, z-image, flux-schnell): the parquet's 384 px WEBP thumbnails, prompts,
    VLM captions and scores, plus the full-resolution JPEGs of the matching tar, extracted while streaming (the tar
    itself is never stored)
  * LucasFang/FLUX-Reason-6M: the 1024 px JPEGs, a 384 px version made the way training data would be (antialiased
    bicubic, WEBP q90) and its captions, with our Polish translations (cpral/FLUX-Reason-6M-captions-pl)

Writes <out>/<source>/{orig,thumb}/<key>.<ext> and <out>/<source>/meta.jsonl.

  python tests/fetch_preview_shards.py --out /path/with/space/data_preview
"""
import argparse
import io
import json
import os
import tarfile

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

MONET = "datasets/jasperai/monet/v1.2.0/synthetic"
MONET_COLS = ["__key__", "thumbnail", "caption_original", "caption_florence-2-large", "caption_gemini-2.5-flash-lite",
              "caption_internvl-3-8b", "caption_sharegpt4v-7b", "aesthetic_jasperai", "aesthetic_laion-improved",
              "wk_jasperai", "width", "height"]
FR_SHARD = "fluxdb-aesthetics-part01-00000-of-00415"
FR_FILE = f"datasets/LucasFang/FLUX-Reason-6M/Aesthetics-Part01/{FR_SHARD}.parquet"


def monet(fs, sub, out):
    d = os.path.join(out, f"monet-{sub}")
    for k in ("orig", "thumb"):
        os.makedirs(os.path.join(d, k), exist_ok=True)
    t = pq.ParquetFile(fs.open(f"{MONET}/{sub}/0-0/000000.parquet")).read(columns=MONET_COLS).to_pylist()
    with open(os.path.join(d, "meta.jsonl"), "w") as f:
        for r in t:
            b = r.pop("thumbnail")
            b = b["bytes"] if isinstance(b, dict) else b
            key = r.pop("__key__")
            with open(os.path.join(d, "thumb", f"{key}.webp"), "wb") as g:
                g.write(b)
            f.write(json.dumps(dict(key=key, prompt=r.pop("caption_original"), **r), ensure_ascii=False) + "\n")
    print(f"monet-{sub}: {len(t)} thumbnails", flush=True)
    n = 0
    with fs.open(f"{MONET}/{sub}/0-0/000000.tar", block_size=2 ** 24) as raw, tarfile.open(fileobj=raw, mode="r|") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".jpg"):
                key = m.name[: -len(".jpg")].split("/")[-1]
                with open(os.path.join(d, "orig", f"{key}.jpg"), "wb") as g:
                    g.write(tar.extractfile(m).read())
                n += 1
                if n % 1000 == 0:
                    print(f"monet-{sub}: {n} originals", flush=True)
    print(f"monet-{sub}: {n} originals done", flush=True)


def flux_reason(fs, out):
    d = os.path.join(out, "flux-reason")
    for k in ("orig", "thumb"):
        os.makedirs(os.path.join(d, k), exist_ok=True)
    pl = pq.read_table(hf_hub_download("cpral/FLUX-Reason-6M-captions-pl", "data/fluxdb-aesthetics-part01-000.parquet",
                                       repo_type="dataset"))
    pl = {x["row"]: x for x in pl.filter(pc.equal(pl["shard"], FR_SHARD)).to_pylist()}
    pf = pq.ParquetFile(fs.open(FR_FILE, block_size=2 ** 24))
    caps = [c for c in pf.schema_arrow.names if c.startswith("caption_") and not c.endswith("_cn")]
    scores = [c for c in pf.schema_arrow.names if c.startswith("score_")]
    row = 0
    with open(os.path.join(d, "meta.jsonl"), "w") as f:
        for g in range(pf.metadata.num_row_groups):
            for r in pf.read_row_group(g, columns=["id", "image"] + caps + scores).to_pylist():
                b = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
                key = f"{row:06d}"
                with open(os.path.join(d, "orig", f"{key}.jpg"), "wb") as h:
                    h.write(b)
                im = Image.open(io.BytesIO(b)).convert("RGB")
                im.resize((384, 384), Image.BICUBIC, reducing_gap=None).save(  # PIL's resize antialiases on downscale
                    os.path.join(d, "thumb", f"{key}.webp"), quality=90)
                x = pl.get(row, {})
                f.write(json.dumps(dict(key=key, id=r["id"], prompt=r.get("caption_composition"),
                                        **{c: r[c] for c in caps + scores},
                                        captions_pl=x.get("captions_pl"), captions_en=x.get("captions_en")),
                                   ensure_ascii=False) + "\n")
                row += 1
            print(f"flux-reason: {row} images", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sources", nargs="+", default=["flux-reason", "flux-klein", "z-image", "flux-schnell"])
    a = p.parse_args()
    fs = HfFileSystem()
    for s in a.sources:
        flux_reason(fs, a.out) if s == "flux-reason" else monet(fs, s, a.out)


if __name__ == "__main__":
    main()
