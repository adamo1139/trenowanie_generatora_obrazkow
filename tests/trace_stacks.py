"""Where in Python an operator is called from, from a torch.profiler chrome trace recorded with stacks
(PROFILE_STACK=1 train_pp.py --profile-dir ...): every CPU event named --op is matched to the Python frames
(python_function events) that enclose it on the same thread; the most common call paths are printed with counts.

  python tests/trace_stacks.py profiles/64px_stack/rank0_step25.trace.json --op aten::fill_
"""
import argparse
import json
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--op", default="aten::fill_")
    p.add_argument("--depth", type=int, default=10, help="innermost Python frames to show")
    p.add_argument("--top", type=int, default=8)
    a = p.parse_args()
    print(f"reading {a.trace} ...", flush=True)
    events = json.load(open(a.trace))["traceEvents"]
    per_tid = {}  # tid -> [(start, -end, kind, name)]: frames before the ops they enclose at the same start
    n_ops = 0
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") == "python_function":
            per_tid.setdefault(e["tid"], []).append((e["ts"], -(e["ts"] + e["dur"]), 0, e["name"]))
        elif e.get("name") == a.op:
            per_tid.setdefault(e["tid"], []).append((e["ts"], -(e["ts"] + e["dur"]), 1, e["name"]))
            n_ops += 1
    del events
    paths = Counter()
    for items in per_tid.values():  # one sweep per thread: a stack of the Python frames still open
        items.sort()
        stack = []
        for start, neg_end, kind, name in items:
            while stack and stack[-1][0] < start:
                stack.pop()
            if kind == 0:
                stack.append((-neg_end, name))
            else:
                paths[tuple(f[1] for f in reversed(stack[-a.depth:]))] += 1
    ops = range(n_ops)
    print(f"{len(ops)} x {a.op}")
    for path, n in paths.most_common(a.top):
        print(f"\n{n} calls:")
        for frame in path:
            print(f"    {frame}")


if __name__ == "__main__":
    main()
