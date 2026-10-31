"""Print a train.py command that continues a checkpoint with its original settings, in a separate output directory
(the original run is left untouched), e.g. to measure the gradient noise scale late in training.

  python tests/resume_cmd.py runs/253m/latest.pt --out runs/gns_253m --extra-steps 200 -- --gns-every 32
  # then: mkdir -p runs/gns_253m && cp runs/253m/latest.pt runs/gns_253m/ and run the printed command

Everything after `--` is appended (and overrides the stored value of the same flag).
"""
import argparse
import sys

import torch

sys.path.insert(0, ".")
from train import build_parser

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("--out", required=True)
p.add_argument("--extra-steps", type=int, default=200)
args, extra = p.parse_known_args()
extra = [a for a in extra if a != "--"]

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
stored, step = ck.get("args") or {}, ck["step"]
if not stored:
    sys.exit("checkpoint has no stored args")
override = {a.lstrip("-").replace("-", "_") for a in extra if a.startswith("--")}
skip = {"out", "steps", "eval_every", "ckpt_every", "merge_every", "keep_every", "log_every", "load_hf"} | override
flags = []
for action in build_parser()._actions:
    k = action.dest
    if k in skip or k == "help" or k not in stored or stored[k] is None or not action.option_strings:
        continue
    opt, v = action.option_strings[-1], stored[k]
    if action.nargs == 0:  # store_true / store_false
        if v != action.default:
            flags.append(opt)
    elif v != action.default:
        flags += [opt, " ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)]
end = step + args.extra_steps
print(f"# checkpoint at step {step} (of {stored.get('steps')}), batch {stored.get('batch')}, size {stored.get('size')}, "
      f"te {stored.get('te_size')}, diloco {stored.get('diloco')}")
print("NCCL_P2P_LEVEL=SYS torchrun --nproc_per_node 8 train.py " + " ".join(flags) +
      f" --steps {end} --eval-every {end} --ckpt-every {end} --merge-every 0 --keep-every 0 --out {args.out} " +
      " ".join(extra))
