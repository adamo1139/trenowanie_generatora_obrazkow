"""Translate the training captions to Polish with IndexTeam/Index-Translate-2B (Apache-2.0, Qwen3.5-based MT model) in
vLLM: <shard>.caps.json (per image a list of English captions) -> <shard>.caps_pl.json, same structure and order, so the
latents and .off.npy stay valid and `prepare_data.py tokenize` can read the Polish file like the English one.

vLLM pins its own torch, so install it in a separate venv, not the training one:
  python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm
  .venv-vllm/bin/python translate_captions.py --data data/aurora_full --pilot 2000     # speed + 15 sample pairs
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i .venv-vllm/bin/python translate_captions.py \\
      --data data/aurora_full --rank $i --world 8 > translate_$i.log 2>&1 & done

--parquet-dir: parquets of prepare_monet.py instead (one caption per image in --column, default caption_short): every
<shard dir>/<file>.parquet gets a <file>.<suffix>.parquet next to it with `__key__` and `<column>_<suffix>`, in the same
row order; empty captions stay empty.
  .venv-vllm/bin/python translate_captions.py --parquet-dir data/monet_z_image --pilot 2000
  for i in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$i .venv-vllm/bin/python translate_captions.py \\
      --parquet-dir data/monet_z_image --rank $i --world 8 > translate_$i.log 2>&1 & done

Text in double quotes is what is written in the image (a sign that says "OPEN"), so it is swapped for placeholders before
translation and put back afterwards: the caption must keep matching the image. Captions whose placeholders do not survive
are translated again without them. Each shard is written atomically with per-shard counters in <shard>.caps_pl.stats.json;
a rerun skips finished shards.
"""
import argparse
import glob
import json
import os
import random
import re
import time

PROMPT = "请将以下{src}文本翻译为{tgt}，直接输出翻译结果，不要进行任何解释。\n\n{text}"  # the model card's template
QUOTE = re.compile(r'"[^"\n]{1,80}"|“[^”\n]{1,80}”|„[^”"\n]{1,80}[”"]')
PH = "[[{}]]"
CJK = re.compile(r"[぀-ヿ㐀-鿿]")
# VLM boilerplate at the start of a caption ("The image shows ...", Florence-2 nearly always): users do not write it
PREFIX_EN = re.compile(r"^(?:(?:the|this) (?:image|picture|photo(?:graph)?|illustration) (?:shows(?: shows| features| "
                       r"depicts)?|depicts|features|captures|presents|displays|showcases|is(?: of)?)|in (?:this|the) "
                       r"(?:image|picture)(?:,)? (?:we can see|there (?:is|are)|we see)|it is)\s+", re.I)
# the same in Polish translations; only verbs that take what follows as their object (not "jest" + instrumental)
PREFIX_PL = re.compile(r"^(?:(?:ten |to )?(?:obraz(?:ek)?|zdjęcie|ilustracja|fotografia)(?: ten)? (?:przedstawia|pokazuje|"
                       r"ukazuje|prezentuje)|na (?:tym )?(?:obrazie|obrazku|zdjęciu|ilustracji|fotografii),? (?:widać|"
                       r"widzimy|znajduje się|znajdują się))\s*:?\s+", re.I)


def strip_prefix(text, lang="en"):
    """Drop the leading "The image shows" / "Obraz przedstawia" boilerplate and capitalise the rest."""
    if not text:
        return text
    out = (PREFIX_PL if lang == "pl" else PREFIX_EN).sub("", text.strip(), count=1)
    return (out[:1].upper() + out[1:]) if out else text


def protect(text):
    """Quoted spans -> [[0]], [[1]]...; returns (text, spans)."""
    spans = []

    def sub(m):
        spans.append(m.group(0))
        return PH.format(len(spans) - 1)
    return QUOTE.sub(sub, text), spans


def restore(text, spans):
    """Put the quoted spans back; None if any placeholder is missing or duplicated."""
    for i, s in enumerate(spans):
        if text.count(PH.format(i)) != 1:
            return None
        text = text.replace(PH.format(i), s)
    return None if re.search(r"\[\[\d+\]\]", text) else text


def suspicious(src, out):
    return (not out or CJK.search(out) is not None or not 0.4 <= len(out) / max(1, len(src)) <= 3.0)


def pilot(flat, translate):
    """Speed and 15 sample pairs."""
    t0 = time.time()
    res, st = translate(flat)
    dt = time.time() - t0
    print(json.dumps(st), f"| {len(flat) / dt:.1f} captions/s, {st['out_tokens'] / dt:.0f} output tok/s on this GPU")
    random.seed(0)
    for i in random.sample(range(len(flat)), min(15, len(flat))):
        print(f"\nEN: {flat[i]}\nPL: {res[i]}")


def parquet_mode(a, translate):
    import pyarrow as pa  # imported before the model loads too (main), so a missing package fails fast
    import pyarrow.parquet as pq
    suffix = f".{a.suffix}.parquet"
    files = sorted(f for f in glob.glob(os.path.join(a.parquet_dir, "*", "*.parquet")) if not f.endswith(suffix))
    if a.pilot:
        texts = [strip_prefix(c) if a.strip_prefix else c
                 for c in pq.read_table(files[0], columns=[a.column])[a.column].to_pylist() if c]
        pilot(list(dict.fromkeys(texts))[: a.pilot], translate)
        return
    mine = files[a.rank::a.world]
    for n, path in enumerate(mine):
        out = path[: -len(".parquet")] + suffix
        if os.path.exists(out):
            continue
        t0 = time.time()
        src = pq.read_table(path, columns=["__key__", a.column])
        texts = [strip_prefix(c) for c in src[a.column].to_pylist()] if a.strip_prefix else src[a.column].to_pylist()
        uniq = list(dict.fromkeys(c for c in texts if c))
        res, st = translate(uniq)
        table = dict(zip(uniq, res))
        pq.write_table(pa.table({"__key__": src["__key__"],
                                 f"{a.column}_{a.suffix}": pa.array([table.get(c) if c else None for c in texts],
                                                                    pa.string())}), out + ".tmp", compression="zstd")
        os.replace(out + ".tmp", out)
        with open(out[: -len(".parquet")] + ".stats.json", "w") as fh:
            json.dump(st, fh)
        dt = time.time() - t0
        print(f"[{n + 1}/{len(mine)}] {os.path.relpath(path, a.parquet_dir)}: {st['captions']} captions, "
              f"{st['quoted']} quoted, {st['retried']} retried, {st['flagged']} flagged, {len(uniq) / dt:.1f} captions/s",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="", help="folder of <shard>.caps.json (FLUX-Reason training data)")
    p.add_argument("--parquet-dir", default="", help="folder of prepare_monet.py parquets (<shard dir>/<file>.parquet)")
    p.add_argument("--column", default="caption_short", help="--parquet-dir: the caption column to translate")
    p.add_argument("--strip-prefix", action="store_true",
                   help="--parquet-dir: drop \"The image shows\" etc. before translating (the Polish then starts with the "
                        "subject in the nominative instead of the accusative object of \"Obraz przedstawia\")")
    p.add_argument("--model", default="IndexTeam/Index-Translate-2B")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--suffix", default="pl")
    p.add_argument("--src", default="英语")      # English
    p.add_argument("--tgt", default="波兰语")    # Polish
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-num-seqs", type=int, default=1024, help="sequences decoded at once (vLLM default 256)")
    p.add_argument("--max-batched-tokens", type=int, default=16384, help="tokens per scheduler step (prefill chunk)")
    p.add_argument("--pilot", type=int, default=0, help="translate this many captions of the first shard, print samples")
    a = p.parse_args()

    if a.parquet_dir:
        import pyarrow.parquet  # noqa: F401  (the vLLM venv does not ship it: pip install pyarrow)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    # the model is a multimodal Qwen3.5: text only, so vLLM does not profile (and OOM on) the video encoder
    llm = LLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=4096, enable_prefix_caching=True,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=a.max_num_seqs,
              max_num_batched_tokens=a.max_batched_tokens)
    params = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    def prompt(text):
        msg = [{"role": "user", "content": PROMPT.format(src=a.src, tgt=a.tgt, text=text)}]
        try:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def translate(texts):
        """Unique English captions -> Polish, with quoted-span protection, a retry without it, and counters."""
        prot = [protect(t) for t in texts]
        outs = [o.outputs[0].text.strip() for o in llm.generate([prompt(pt) for pt, _ in prot], params)]
        res, retry, st = [], [], dict(captions=len(texts), quoted=sum(bool(s) for _, s in prot), retried=0, flagged=0,
                                      out_tokens=0)
        for i, ((pt, spans), out) in enumerate(zip(prot, outs)):
            r = restore(out, spans) if spans else out
            if r is None:
                retry.append(i)
            res.append(r)
        if retry:  # placeholders mangled: translate the original text instead
            st["retried"] = len(retry)
            outs2 = [o.outputs[0].text.strip() for o in llm.generate([prompt(texts[i]) for i in retry], params)]
            for i, out in zip(retry, outs2):
                res[i] = out
        for t, r in zip(texts, res):
            st["flagged"] += suspicious(t, r)
            st["out_tokens"] += len(tok(r, add_special_tokens=False).input_ids) if r else 0
        return res, st

    assert bool(a.data) != bool(a.parquet_dir), "give --data or --parquet-dir"
    if a.parquet_dir:
        return parquet_mode(a, translate)
    shards = sorted(glob.glob(os.path.join(a.data, "*.caps.json")))
    if a.pilot:
        caps = json.load(open(shards[0]))
        flat = list(dict.fromkeys(c for img in caps for c in img))[: a.pilot]
        pilot(flat, translate)
        return

    mine = shards[a.rank::a.world]
    for n, path in enumerate(mine):
        out = path.replace(".caps.json", f".caps_{a.suffix}.json")
        if os.path.exists(out):
            continue
        t0 = time.time()
        caps = json.load(open(path))
        uniq = list(dict.fromkeys(c for img in caps for c in img))
        res, st = translate(uniq)
        table = dict(zip(uniq, res))
        for f, obj in ((out, [[table[c] for c in img] for img in caps]),
                       (out.replace(".json", ".stats.json"), st)):
            with open(f + ".tmp", "w") as fh:
                json.dump(obj, fh, ensure_ascii=False)
            os.replace(f + ".tmp", f)
        dt = time.time() - t0
        print(f"[{n + 1}/{len(mine)}] {os.path.basename(path)}: {st['captions']} captions, {st['quoted']} quoted, "
              f"{st['retried']} retried, {st['flagged']} flagged, {len(uniq) / dt:.1f} captions/s", flush=True)


if __name__ == "__main__":
    main()
    # vLLM's engine-core process sometimes keeps the interpreter alive after the work is done: exit hard (all output
    # files are already written and renamed, stdout is flushed)
    import sys
    sys.stdout.flush()
    os._exit(0)
