#!/usr/bin/env python3
"""Audit a quant file: which tier sits on which layer.

Usage:
  python scripts/audit_tiers.py --model FILE.gguf [--per-layer]
  python scripts/audit_tiers.py --model A.gguf --compare B.gguf

Prints: tier histogram (count + MiB), a class x tier table (what the
allocator spent on attention vs FFN vs embeddings), then optionally a
per-layer map. --compare B does the same table side by side plus the
per-tensor tier movement between the two files — the "whose distribution
is shaped how" view (e.g. stock Q4_K_M vs a MERNIK build).
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gguf import GGUFReader
import gguf

from constants import get_tensor_class, get_tensor_type, TIER_ORDER

# File tiers we may see (F32 is written by the binary for 1D tensors and
# never appears in TIER_ORDER, so it joins at the bottom).
TIER_RANK = {t: i for i, t in enumerate(TIER_ORDER)}
TIER_RANK["F32"] = -1

GMAP = {str(v): k for k, v in vars(gguf.GGMLQuantizationType).items()
        if not k.startswith("_")}
BPW = {"F32": 32.0, "F16": 16.0, "Q8_0": 8.5, "Q6_K": 6.5625, "Q5_K": 5.5,
       "Q4_K": 4.5, "Q4_0": 4.5, "Q5_0": 5.5, "Q8_1": 8.0, "Q2_K": 2.5625,
       "Q3_K": 3.4375, "IQ4_XS": 4.25, "IQ4_NL": 4.5, "IQ3_XXS": 3.0625,
       "IQ3_S": 3.44, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125, "IQ2_S": 2.5,
       "IQ1_S": 1.5625}


def layer_of(name):
    parts = name.replace(".weight", "").replace(".bias", "").split(".")
    if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
        try:
            return int(parts[1])
        except ValueError:
            pass
    if "token_embd" in name or "embed" in name:
        return "embd"
    if name.startswith("output"):
        return "output"
    return "global"


def read_tiers(path):
    """-> (name -> tier, name -> MiB) straight from the file."""
    r = GGUFReader(path)
    tiers, mibs = {}, {}
    for t in r.tensors:
        tt = GMAP.get(str(t.tensor_type), "?")
        n = 1
        for d in t.shape:
            n *= int(d)
        tiers[t.name] = tt
        mibs[t.name] = n * BPW.get(tt, 0.0) / (8 * 1024 * 1024)
    return tiers, mibs


def class_table(tiers):
    """class -> Counter(tier) — what the file spends per functional class."""
    agg = defaultdict(Counter)
    for name, tt in tiers.items():
        agg[get_tensor_class(get_tensor_type(name))][tt] += 1
    return agg


def print_class_table(agg, title):
    print(title)
    for cls in sorted(agg):
        parts = " ".join("%sx%d" % (tt, n)
                         for tt, n in sorted(agg[cls].items()))
        print("  %-14s %s" % (cls, parts))


def compare(a_tiers, b_tiers, a_mibs, b_mibs, top=15):
    """Movement report: --model A vs --compare B, per tensor by name."""
    common = sorted(set(a_tiers) & set(b_tiers))
    only_a = sorted(set(a_tiers) - set(b_tiers))
    only_b = sorted(set(b_tiers) - set(a_tiers))
    up = down = same = 0
    moved = []
    for n in common:
        ra = TIER_RANK.get(a_tiers[n], -99)
        rb = TIER_RANK.get(b_tiers[n], -99)
        if rb > ra:
            up += 1
        elif rb < ra:
            down += 1
        else:
            same += 1
        if rb != ra:
            moved.append((abs(rb - ra), n, a_tiers[n], b_tiers[n],
                          b_mibs.get(n, 0.0) - a_mibs.get(n, 0.0)))
    print("tensors: common %d | only in A %d | only in B %d"
          % (len(common), len(only_a), len(only_b)))
    if only_a:
        print("  only A: %s%s" % (", ".join(only_a[:8]),
                                  " ..." if len(only_a) > 8 else ""))
    if only_b:
        print("  only B: %s%s" % (", ".join(only_b[:8]),
                                  " ..." if len(only_b) > 8 else ""))
    print("tier movement (B vs A): B higher on %d, lower on %d, same on %d"
          % (up, down, same))
    if moved:
        moved.sort(reverse=True)
        print("biggest movers (rank delta, then MiB):")
        for _, n, ta, tb, dmi in moved[:top]:
            print("  %-44s %s -> %s  (%+.1f MiB)" % (n, ta, tb, dmi))
    else:
        print("no tier differences on common tensors.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-layer", action="store_true",
                    help="print every layer's tier map")
    ap.add_argument("--layer", default=None,
                    help="show only this layer (number, embd, output, global)")
    ap.add_argument("--big", type=int, default=0,
                    help="list top-N biggest tensors with tiers")
    ap.add_argument("--compare", default=None, metavar="B.gguf",
                    help="side-by-side class table + per-tensor tier "
                         "movement against this second file")
    ap.add_argument("--top", type=int, default=15,
                    help="how many movers to list in --compare")
    ap.add_argument("--no-class", action="store_true",
                    help="skip the class x tier table")
    args = ap.parse_args()

    tiers, mibs = read_tiers(args.model)
    tot_c, tot_m = Counter(), Counter()
    layers = defaultdict(list)
    biggest = []
    for name, tt in tiers.items():
        tot_c[tt] += 1
        tot_m[tt] += mibs[name]
        layers[layer_of(name)].append((name, tt))
        biggest.append((mibs[name], name, tt))
    biggest.sort(reverse=True)

    print("%s" % args.model)
    print("tensors: %d  est: %.0f MiB" % (sum(tot_c.values()), sum(tot_m.values())))
    for tt in sorted(tot_c):
        print("  %-8s %4d  %8.1f MiB" % (tt, tot_c[tt], tot_m[tt]))

    agg_a = class_table(tiers)
    if not args.no_class:
        print_class_table(agg_a, "class x tier:")

    if args.compare:
        b_tiers, b_mibs = read_tiers(args.compare)
        agg_b = class_table(b_tiers)
        print()
        print("A = %s" % args.model)
        print("B = %s" % args.compare)
        # side by side: one line per class, both files
        print("class x tier (A | B):")
        for cls in sorted(set(agg_a) | set(agg_b)):
            fa = " ".join("%sx%d" % (tt, n)
                          for tt, n in sorted(agg_a[cls].items())) or "-"
            fb = " ".join("%sx%d" % (tt, n)
                          for tt, n in sorted(agg_b[cls].items())) or "-"
            print("  %-14s %-32s | %s" % (cls, fa, fb))
        print()
        compare(tiers, b_tiers, mibs, b_mibs, top=args.top)
        return

    if args.big > 0:
        print("biggest %d:" % args.big)
        for mib, name, tt in biggest[:args.big]:
            print("  %8.1f MiB  %-44s %s" % (mib, name, tt))
    want = None
    if args.layer is not None:
        want = args.layer if args.layer in ("embd", "output", "global") \
            else int(args.layer)
    if args.per_layer or want is not None:
        keys = [want] if want is not None else sorted(
            layers, key=lambda x: (isinstance(x, str), x))
        for layer in keys:
            if layer not in layers:
                print("layer %s: (none)" % (layer,))
                continue
            ts = sorted(layers[layer])
            print("layer %-6s: %s" % (layer, " ".join(
                "%s=%s" % (n.split(".")[2] if len(n.split(".")) > 2 else n, tt)
                for n, tt in ts)))


if __name__ == "__main__":
    main()
