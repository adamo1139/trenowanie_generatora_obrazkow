"""BLiMP (Warstadt et al. 2020, arXiv:1912.00582): 67 English minimal-pair paradigms x 1,000 pairs; a model passes a
pair when it gives the grammatical sentence the higher total log-probability. Used here to check how much English a
candidate text encoder knows (a text-to-image model conditioned on it can only separate what the LM separates).

  python blimp_eval.py cpral/Poziomka-Baza-2026-09-05 SlayerLab/GoLLeM-v6-250M SlayerLab/Slayer149 --out blimp.json
  python blimp_eval.py SlayerLab/GoLLeM-v6-250M --limit 100      # quick: 100 pairs per paradigm

Backends: GoLLeM v6 repos (their own modeling_gollem_v6.py), GoLLeM/Slayer GPT repos (aurora/slayer.py), anything
else through transformers AutoModelForCausalLM (trust_remote_code). Every sentence gets a BOS / document-start token
in front, so its first word is scored too. Chance level = 50%.
"""
import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_blimp(limit):
    files = sorted(f for f in list_repo_files("nyu-mll/blimp", repo_type="dataset") if f.endswith(".parquet"))
    out = []
    for f in files:
        rows = pq.read_table(hf_hub_download("nyu-mll/blimp", f, repo_type="dataset")).to_pylist()
        out += rows[:limit] if limit else rows
    return out


def load_model(repo, device):
    """-> (logits_fn(ids LongTensor B x T) -> B x T x V, encode(str) -> list[int] incl. BOS, label)."""
    files = list_repo_files(repo)
    if "modeling_gollem_v6.py" in files:
        path = snapshot_download(repo)
        sys.path.insert(0, path)
        from modeling_gollem_v6 import load_gollem_v6
        model, tok = load_gollem_v6(path, device)
        bos = tok.token_to_id("<|endoftext|>")
        return (lambda ids: model(ids)), (lambda s: [bos] + tok.encode(s).ids), "gollem-v6"
    cfg = json.load(open(hf_hub_download(repo, "config.json")))
    if "n_layer" in cfg:  # GoLLeM-style GPT (Slayer149)
        from tokenizers import Tokenizer

        from aurora.slayer import load_lm
        lm, tok_path = load_lm(repo)
        lm.to(device)
        tok = Tokenizer.from_file(tok_path)
        bos = tok.token_to_id("<|endoftext|>")
        if bos is None:
            bos = 12285  # Slayer149's document start (prepare_data.py ENCODERS)
        logits = lambda ids: lm.head(lm.lnf(lm.hidden(ids, len(lm.blocks))))
        return logits, (lambda s: [bos] + tok.encode(s).ids), "gollem-gpt"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    enc = lambda s: [bos] + tok(s, add_special_tokens=False).input_ids
    return (lambda ids: model(input_ids=ids).logits), enc, "transformers"


@torch.no_grad()
def score(logits_fn, seqs, device, batch=128):
    """Total log-probability of tokens 1.. of every sequence (right-padded batches; causal, so padding is harmless)."""
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        T = max(map(len, chunk))
        ids = torch.zeros(len(chunk), T, dtype=torch.long)
        mask = torch.zeros(len(chunk), T - 1)
        for j, s in enumerate(chunk):
            ids[j, :len(s)] = torch.tensor(s)
            mask[j, :len(s) - 1] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = logits_fn(ids[:, :-1])
        lp = F.log_softmax(lg.float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
        out += (lp * mask).sum(1).tolist()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="+")
    p.add_argument("--limit", type=int, default=0, help="pairs per paradigm (0 = all 1,000)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    data = load_blimp(a.limit)
    print(f"BLiMP: {len(data):,} pairs, {len({r['UID'] for r in data})} paradigms", flush=True)
    results = {}
    for repo in a.models:
        t0 = time.time()
        logits_fn, enc, kind = load_model(repo, a.device)
        good = score(logits_fn, [enc(r["sentence_good"]) for r in data], a.device)
        bad = score(logits_fn, [enc(r["sentence_bad"]) for r in data], a.device)
        ok = [g > b for g, b in zip(good, bad)]
        by = lambda key: {k: 100 * sum(o for o, r in zip(ok, data) if r[key] == k) / sum(r[key] == k for r in data)
                          for k in sorted({r[key] for r in data})}
        res = dict(backend=kind, overall=100 * sum(ok) / len(ok), phenomena=by("linguistics_term"), paradigms=by("UID"))
        results[repo] = res
        print(f"\n{repo} ({kind}): overall {res['overall']:.1f}%  ({time.time() - t0:.0f} s)", flush=True)
        for k, v in res["phenomena"].items():
            print(f"  {k:28s} {v:5.1f}", flush=True)
        del logits_fn
        torch.cuda.empty_cache()
    if len(results) > 1:
        names = list(results)
        print("\n" + " " * 30 + "".join(f"{n.split('/')[-1][:18]:>20s}" for n in names))
        for k in results[names[0]]["phenomena"]:
            print(f"{k:30s}" + "".join(f"{results[n]['phenomena'][k]:20.1f}" for n in names))
        print(f"{'OVERALL':30s}" + "".join(f"{results[n]['overall']:20.1f}" for n in names))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
