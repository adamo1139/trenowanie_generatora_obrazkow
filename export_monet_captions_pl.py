"""Export the Polish translations of the MONET synthetic captions (prepare_monet.py + translate_captions.py
--parquet-dir --strip-prefix) as a Hugging Face dataset: one parquet per MONET shard folder with, per image, the join
keys into jasperai/monet (source parquet path + row, __key__, id), the English caption that was translated and its
Polish translation, plus README.md.

  python export_monet_captions_pl.py --parquet-dir data/monet_z_image --out exports/MONET-z-image-captions-pl
  hf upload cpral/MONET-z-image-captions-pl exports/MONET-z-image-captions-pl . --repo-type dataset
"""
import argparse
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

from translate_captions import strip_prefix, suspicious

CAPTIONS = ["caption_florence-2-large", "caption_internvl-3-8b", "caption_sharegpt4v-7b", "caption_gemini-2.5-flash-lite"]
SCHEMA = pa.schema([("source_file", pa.string()), ("row", pa.int32()), ("key", pa.string()), ("id", pa.string()),
                    ("caption_source", pa.string()), ("caption_en", pa.string()), ("caption_pl", pa.string()),
                    ("pl_flagged", pa.bool_())])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--source", default="z-image", help="the MONET synthetic source the parquets came from")
    p.add_argument("--suffix", default="pl")
    p.add_argument("--repo", default="cpral/MONET-z-image-captions-pl", help="for the README only")
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    dirs = sorted({os.path.dirname(f) for f in glob.glob(os.path.join(a.parquet_dir, "*", "*.parquet"))},
                  key=lambda d: [int(x) for x in os.path.basename(d).split("-")])
    tot = dict(images=0, translated=0, flagged=0, empty=0, quoted=0, retried=0, too_long=0)
    sources = {}
    for st_file in glob.glob(os.path.join(a.parquet_dir, "*", f"*.{a.suffix}.stats.json")):
        st = json.load(open(st_file))
        for k in ("quoted", "retried", "too_long"):
            tot[k] += st.get(k, 0)
    for d in dirs:
        cols = {c: [] for c in SCHEMA.names}
        for f in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            if os.path.basename(f).count(".") != 1:
                continue
            src = pq.read_table(f, columns=["__key__", "id", "caption_short"] + CAPTIONS).to_pylist()
            pl = pq.read_table(f[: -len(".parquet")] + f".{a.suffix}.parquet")
            assert pl["__key__"].to_pylist() == [r["__key__"] for r in src], f
            rel = f"v1.2.0/synthetic/{a.source}/{os.path.relpath(f, a.parquet_dir)}"
            for i, (r, q) in enumerate(zip(src, pl[f"caption_short_{a.suffix}"].to_pylist())):
                c = r["caption_short"]
                which = next((k for k in CAPTIONS if r[k] and " ".join(r[k].split()) == c), None) if c else None
                en = strip_prefix(c) if c else None
                bad = bool(en) and (not q or suspicious(en, q))
                for k, v in zip(SCHEMA.names, (rel, i, r["__key__"], r["id"], which and which[len("caption_"):],
                                                en, q, bad)):
                    cols[k].append(v)
                tot["images"] += 1
                tot["empty"] += not en
                tot["translated"] += bool(q)
                tot["flagged"] += bad
                if which:
                    sources[which] = sources.get(which, 0) + 1
        name = f"data/{a.source}-{os.path.basename(d)}.parquet"
        pq.write_table(pa.table(cols, schema=SCHEMA), os.path.join(a.out, name), compression="zstd")
        print(f"{name}: {len(cols['row']):,} images", flush=True)
    src_lines = "\n".join(f"| `{k[len('caption_'):]}` | {v:,} ({100 * v / max(1, tot['images']):.1f}%) |"
                          for k, v in sorted(sources.items(), key=lambda kv: -kv[1]))
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(README.format(repo=a.repo, source=a.source, n_files=len(dirs), src_lines=src_lines, **tot,
                              flag_pct=100 * tot["flagged"] / max(1, tot["translated"]),
                              quoted_pct=100 * tot["quoted"] / max(1, tot["translated"])))
    print(json.dumps(dict(tot, sources=sources)))


README = """---
license: apache-2.0
language: [pl, en]
task_categories: [text-to-image, translation]
pretty_name: MONET {source} captions in Polish
size_categories: [1M<n<10M]
configs:
  - config_name: default
    data_files: "data/*.parquet"
---

# MONET {source} captions in Polish

Polish machine translations of one short caption per image of the **synthetic {source} subset** of
[jasperai/monet](https://huggingface.co/datasets/jasperai/monet) (`v1.2.0/synthetic/{source}`): **{images:,} images,
{translated:,} translated captions**, each with the English text it was translated from and the keys to join it with
MONET. Made for training a bilingual (English / Polish) text-to-image model.

## Which caption

MONET gives every image the generation prompt and up to four VLM captions. This dataset translates the **shortest
non-empty VLM caption** of each image, which is close to what a user would type:

| caption source | images |
| --- | --- |
{src_lines}

The VLM boilerplate at the start ("The image shows ...", "The image depicts ...", "In this image we can see ...") was
removed **before** translation, so `caption_en` / `caption_pl` start with the subject itself (and the Polish subject is
in the nominative case, not the accusative object of "Obraz przedstawia ...").

## Format

{n_files} parquet files in `data/`, one per MONET shard folder, one row per image:

| column | type | |
| --- | --- | --- |
| `source_file` | string | the MONET parquet, e.g. `v1.2.0/synthetic/{source}/0-0/000000.parquet` |
| `row` | int32 | row index within that parquet |
| `key` | string | MONET `__key__` |
| `id` | string | MONET `id` |
| `caption_source` | string | which MONET caption column was the shortest (e.g. `florence-2-large`) |
| `caption_en` | string | that caption without the leading boilerplate: the text that was translated |
| `caption_pl` | string | its Polish translation (null when the image has no caption, or the caption was too long to translate) |
| `pl_flagged` | bool | the translation looks broken (empty, contains CJK characters, or length ratio outside 0.4–3.0) |

```python
from datasets import load_dataset
ds = load_dataset("{repo}", split="train")
r = ds[0]
print(r["source_file"], r["row"], r["caption_pl"])   # join with jasperai/monet on (source_file, row) or key
```

## How it was made

- Model: [IndexTeam/Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B) (Apache-2.0) in vLLM,
  greedy decoding, the prompt template from its model card (English → Polish), one request per unique caption.
- **Text in double quotes was kept verbatim** (usually text rendered in the image): quoted spans were replaced by
  placeholders before translation and restored afterwards. {quoted:,} captions ({quoted_pct:.1f}%) contained quotes; in
  {retried:,} of them the model broke a placeholder and the caption was translated again without protection.
- {too_long:,} captions did not fit the model's context (degenerate VLM output) and were not translated; {empty:,}
  images have no caption at all.
- {flagged:,} translations ({flag_pct:.2f}%) are flagged in `pl_flagged`; filter them or fall back to `caption_en`.
- Code: https://github.com/adamo1139/trenowanie_generatora_obrazkow (`prepare_monet.py`, `translate_captions.py`,
  `export_monet_captions_pl.py`).

## Quality

A 2B translation model: the Polish is fluent in most cases, but expect occasional mistranslated words and grammatical
errors. Good enough as conditioning text for image generation; not a reference-quality translation corpus.

## License

Apache-2.0, as MONET and the translation model. The images and English captions come from jasperai/monet; please also
cite that work if you use this dataset.
"""


if __name__ == "__main__":
    main()
