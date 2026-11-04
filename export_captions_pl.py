"""Export the Polish captions (translate_captions.py output) as a Hugging Face dataset: parquet files with, per image,
the source shard and row (join key into QLNI/FLUX-Reason-6M-flux2-latents, whose .npz shards keep the source parquet
names and row order), the English captions, their Polish translations and a per-caption quality flag; plus README.md.

  python export_captions_pl.py --data data/aurora_full --out exports/FLUX-Reason-6M-captions-pl
  hf upload cpral/FLUX-Reason-6M-captions-pl exports/FLUX-Reason-6M-captions-pl . --repo-type dataset

Needs pyarrow (`uv pip install pyarrow`; it does not touch torch).
"""
import argparse
import glob
import json
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq

from translate_captions import suspicious

SCHEMA = pa.schema([("shard", pa.string()), ("row", pa.int32()), ("captions_en", pa.list_(pa.string())),
                    ("captions_pl", pa.list_(pa.string())), ("pl_flagged", pa.list_(pa.bool_()))])


def subset(name):  # fluxdb-aesthetics-part01-00414-of-00415 -> fluxdb-aesthetics-part01
    return re.sub(r"-\d+-of-\d+$", "", name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suffix", default="pl")
    p.add_argument("--shards-per-file", type=int, default=50)
    p.add_argument("--repo", default="cpral/FLUX-Reason-6M-captions-pl", help="for the README only")
    a = p.parse_args()

    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(a.data, f"*.caps_{a.suffix}.json")))
    names = [os.path.basename(f)[: -len(f".caps_{a.suffix}.json")] for f in srcs]
    groups = {}
    for n in names:
        groups.setdefault(subset(n), []).append(n)
    tot = dict(images=0, captions=0, flagged=0, quoted=0, retried=0)
    for st_file in glob.glob(os.path.join(a.data, f"*.caps_{a.suffix}.stats.json")):
        st = json.load(open(st_file))
        tot["quoted"] += st["quoted"]
        tot["retried"] += st["retried"]
    files = []
    for sub, members in sorted(groups.items()):
        for k in range(0, len(members), a.shards_per_file):
            part = members[k:k + a.shards_per_file]
            cols = {c: [] for c in SCHEMA.names}
            for n in part:
                en = json.load(open(os.path.join(a.data, f"{n}.caps.json")))
                pl = json.load(open(os.path.join(a.data, f"{n}.caps_{a.suffix}.json")))
                assert len(en) == len(pl), n
                for i, (e, q) in enumerate(zip(en, pl)):
                    assert len(e) == len(q), (n, i)
                    flags = [suspicious(x, y) for x, y in zip(e, q)]
                    for c, v in zip(SCHEMA.names, (n, i, e, q, flags)):
                        cols[c].append(v)
                    tot["images"] += 1
                    tot["captions"] += len(e)
                    tot["flagged"] += sum(flags)
            fname = f"data/{sub}-{k // a.shards_per_file:03d}.parquet"
            pq.write_table(pa.table(cols, schema=SCHEMA), os.path.join(a.out, fname), compression="zstd",
                           row_group_size=50_000)
            files.append(fname)
            print(f"{fname}: {len(part)} shards, {len(cols['row']):,} images", flush=True)

    readme = README.format(repo=a.repo, n_files=len(files), n_shards=len(names), **tot,
                           flag_pct=100 * tot["flagged"] / max(1, tot["captions"]),
                           quoted_pct=100 * tot["quoted"] / max(1, tot["captions"]),
                           retried_pct=100 * tot["retried"] / max(1, tot["quoted"]))
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(readme)
    print(json.dumps(tot))


README = """---
license: apache-2.0
language: [pl, en]
task_categories: [text-to-image, translation]
pretty_name: FLUX-Reason-6M captions in Polish
size_categories: [1M<n<10M]
configs:
  - config_name: default
    data_files: "data/*.parquet"
---

# FLUX-Reason-6M captions in Polish

Polish machine translations of all captions of
[QLNI/FLUX-Reason-6M-flux2-latents](https://huggingface.co/datasets/QLNI/FLUX-Reason-6M-flux2-latents)
(itself derived from [LucasFang/FLUX-Reason-6M](https://huggingface.co/datasets/LucasFang/FLUX-Reason-6M)):
**{images:,} images, {captions:,} captions**, each with its English original, aligned row by row with the source
shards so the Polish captions can be used directly with the existing image latents. Made for training a bilingual
(English / Polish) text-to-image model.

## Format

{n_files} parquet files in `data/`, one row per image:

| column | type | |
| --- | --- | --- |
| `shard` | string | source shard name, e.g. `fluxdb-imaginative-00127-of-00168` (= the `.npz` file name in QLNI/FLUX-Reason-6M-flux2-latents) |
| `row` | int32 | row index within that shard |
| `captions_en` | list[string] | the original English captions of the image (usually 3) |
| `captions_pl` | list[string] | their Polish translations, same order |
| `pl_flagged` | list[bool] | per caption: the translation looks broken (empty, contains CJK characters, or length ratio outside 0.4–3.0) |

```python
from datasets import load_dataset
import numpy as np
ds = load_dataset("{repo}", split="train")
r = ds[0]
lat = np.load(f"{{r['shard']}}.npz", allow_pickle=True)["lat"][r["row"]]   # from QLNI/FLUX-Reason-6M-flux2-latents
print(r["captions_pl"][0])
```

## How it was made

- Model: [IndexTeam/Index-Translate-2B](https://huggingface.co/IndexTeam/Index-Translate-2B) (Apache-2.0) in vLLM,
  greedy decoding, the prompt template from its model card (English → Polish), one request per unique caption.
- **Text in double quotes was kept verbatim** (it is usually text rendered in the image, e.g. a sign that says
  "OPEN"): quoted spans were replaced by placeholders before translation and restored afterwards.
  {quoted:,} captions ({quoted_pct:.1f}%) contained quotes; in {retried:,} of them ({retried_pct:.1f}%) the model
  broke a placeholder and the caption was translated again without protection (the quote may then be translated).
- {flagged:,} captions ({flag_pct:.2f}%) are flagged in `pl_flagged`; filter them or fall back to `captions_en`.
- Code: https://github.com/adamo1139/trenowanie_generatora_obrazkow (`translate_captions.py`, `export_captions_pl.py`).

## Quality

A 2B translation model: the Polish is fluent in most cases, but expect occasional mistranslated words and grammatical
errors (e.g. wrong gender agreement, "maple syrup" → "dżem z klonu"). Good enough as conditioning text for image
generation; not a reference-quality translation corpus.

## License

Apache-2.0, as the source datasets and the translation model. The captions originate from
LucasFang/FLUX-Reason-6M; please also cite that work if you use this dataset.
"""


if __name__ == "__main__":
    main()
