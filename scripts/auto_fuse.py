#!/usr/bin/env python3
"""auto_fuse: imatrix-voted donor maps for layer fusion. No manual --map.

The manual era (eyeball importance, hand-write flags) ends here: this tool
loads one imatrix per donor, compares them, and emits the donor map —
then builds via fuse_layers.py. Manual patches still allowed on top.

Rules:
  rank-vote  — per block, donor whose imatrix ranks it highest *relatively*
               (rank inside its own model, scale-free). Ties and Ox-wins
               stay on the base. Neo never wins anything on Qwen3.5
               (near-noop proven) — this rule shows it, not assumes it.
  diverge    — top-N blocks by cross-donor weight divergence (needs BF16s,
               slow); kept for reference, rank-vote is the fast compass.

Usage:
  auto_fuse.py --a OX.gguf --b ORN.gguf --ia OX.imatrix --ib ORN.imatrix \\
      --rule rank-vote --out M7.gguf
  auto_fuse.py ... --dry                       # print map only
  auto_fuse.py ... --patch "31:c" --c NEO.gguf --ic NEO.imatrix
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def layer_of(name):
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def load_imatrix(path):
    from imatrix_reader import read_imatrix
    return read_imatrix(path)["tensors"]


def rank_vote(ia, ib, ic=None, base="a"):
    """Best relative rank wins. Returns {block: donor} for non-base wins."""
    donors = {"a": ia, "b": ib}
    if ic is not None:
        donors["c"] = ic
    common = set(ia) & set(ib)
    if ic is not None:
        common &= set(ic)
    per = {k: defaultdict(list) for k in donors}
    for n in common:
        b = layer_of(n)
        if b < 0:
            continue
        for k, t in donors.items():
            per[k][b].append(t[n]["importance_mean"])
    mean = {k: {b: sum(v) / len(v) for b, v in d.items()}
            for k, d in per.items()}
    rank = {}
    for k, m in mean.items():
        order = sorted(m, key=lambda b: -m[b])
        rank[k] = {b: i for i, b in enumerate(order)}
    amap = {}
    for b in sorted(set().union(*[set(m) for m in mean.values()])):
        r = {k: rank[k][b] for k in rank if b in rank[k]}
        if not r:
            continue
        best = min(r.values())
        winners = sorted(k for k, v in r.items() if v == best)
        # base wins ties (stability); a win must be strict AND non-base
        if len(winners) == 1 and winners[0] != base:
            amap[b] = winners[0]
    return amap


def parse_patch(spec):
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        blk, _, donor = part.partition(":")
        out[int(blk.strip())] = donor.strip().lower()
    return out


def main():
    ap = argparse.ArgumentParser(description="imatrix-voted auto fusion")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None)
    ap.add_argument("--ia", required=True, help="imatrix for A")
    ap.add_argument("--ib", required=True, help="imatrix for B")
    ap.add_argument("--ic", default=None, help="imatrix for C")
    ap.add_argument("--rule", choices=["rank-vote"], default="rank-vote")
    ap.add_argument("--patch", default=None,
                    help="manual overrides, e.g. '31:c'")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ia, ib = load_imatrix(args.ia), load_imatrix(args.ib)
    ic = load_imatrix(args.ic) if args.ic and args.c else None
    amap = rank_vote(ia, ib, ic)
    amap.update(parse_patch(args.patch))
    letters = {"a": "A", "b": "B", "c": "C"}
    desc = ", ".join(f"{b}:{letters[d]}" for b, d in sorted(amap.items()))
    print("auto map (%s): %s" % (args.rule, desc or "(all base)"))
    print(json.dumps({"rule": args.rule, "map": amap}))
    if args.dry or not args.out:
        return
    cmd = [sys.executable,
           os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fuse_layers.py"),
           "--a", args.a, "--b", args.b, "--out", args.out,
           "--map", ",".join(f"{b}:{d}" for b, d in sorted(amap.items()))]
    if args.c:
        cmd += ["--c", args.c]
    print("exec:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
