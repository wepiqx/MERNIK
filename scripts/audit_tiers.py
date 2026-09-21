#!/usr/bin/env python3
"""Audit a quant file: which tier sits on which layer.

Usage:
  python scripts/audit_tiers.py --model FILE.gguf [--per-layer]

Prints: tier histogram (count + MiB), then optionally a per-layer map
showing every tensor's tier. Answers "how many layers, where, in what tier"
natively from the file — no config needed.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gguf import GGUFReader
import gguf

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-layer", action="store_true",
                    help="print every layer's tier map")
    ap.add_argument("--layer", default=None,
                    help="show only this layer (number, embd, output, global)")
    ap.add_argument("--big", type=int, default=0,
                    help="list top-N biggest tensors with tiers")
    args = ap.parse_args()

    r = GGUFReader(args.model)
    tot_c, tot_m = Counter(), Counter()
    layers = defaultdict(list)
    biggest = []
    for t in r.tensors:
        tt = GMAP.get(str(t.tensor_type), "?")
        n = 1
        for d in t.shape:
            n *= int(d)
        mib = n * BPW.get(tt, 0.0) / (8 * 1024 * 1024)
        tot_c[tt] += 1
        tot_m[tt] += mib
        layers[layer_of(t.name)].append((t.name, tt))
        biggest.append((mib, t.name, tt))
    biggest.sort(reverse=True)

    print("tensors: %d  est: %.0f MiB" % (sum(tot_c.values()), sum(tot_m.values())))
    for tt in sorted(tot_c):
        print("  %-8s %4d  %8.1f MiB" % (tt, tot_c[tt], tot_m[tt]))
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
