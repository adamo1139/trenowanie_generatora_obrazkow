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
    events = json.load(open(a.trace))["traceEvents"]
    py, ops = {}, []
    for e in events:
        if e.get("ph") != "X":
            continue
        if e.get("cat") == "python_function":
            py.setdefault(e["tid"], []).append((e["ts"], e["ts"] + e["dur"], e["name"]))
        elif e.get("name") == a.op:
            ops.append(e)
    for frames in py.values():
        frames.sort()
    paths = Counter()
    for op in ops:
        t = op["ts"]
        enclosing = [f for f in py.get(op["tid"], []) if f[0] <= t <= f[1]]
        enclosing.sort(key=lambda f: f[1] - f[0])  # innermost first
        paths[tuple(f[2] for f in enclosing[: a.depth])] += 1
    print(f"{len(ops)} x {a.op}")
    for path, n in paths.most_common(a.top):
        print(f"\n{n} calls:")
        for frame in path:
            print(f"    {frame}")


if __name__ == "__main__":
    main()
