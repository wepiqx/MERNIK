#!/usr/bin/env python3
"""SMSE vs Gnom allocation comparison at equal --size (CPU-only).

Runs the exact classifier paths main.py uses twice — utility=smse
(bottom-up) and utility=gnom (top-down by default) with a netpred table —
then diffs the per-tensor tier maps: tier histograms, tissue x tier moves,
byte flow, and the top upgraded/downgraded tensors.

    compare_alloc.py --model M.gguf --imatrix I.gguf --size 5310
                     --netpred models/netpred_m.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from llama_inspect import read_model
from imatrix_reader import (read_imatrix, build_importance_table,
                            detect_tied_groups, combine_imatrix)
from classifier import optimal_classify, optimal_classify_topdown


def run(model, imp_table, tied_groups, target, mode, netpred,
        free_pins=False, allow_q3=True, top_down=False):
    raw = mode
    mode = {"netpure": "gnom", "netfd": "gnom"}.get(mode, mode)
    if raw == "netfd":
        top_down = True
    uopt = {"mode": mode, "top_down": top_down, "netpred": netpred}
    classify = optimal_classify_topdown if top_down else optimal_classify
    kwargs = {} if top_down else {
        "relief": None, "relief_thr": -0.5,
        "allowed_tiers": None, "pin_small": 0.0}
    assignments, _ = classify(
        imp_table, tied_groups, model, target_size_mib=target,
        allow_q3=allow_q3, free_pins=free_pins, uopt=uopt,
        legacy_1d=False, **kwargs)
    return assignments


def tissue(n):
    p = n.split(".")
    if p[0] == "blk" and len(p) > 2:
        return p[2]
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--size", type=float, required=True)
    ap.add_argument("--netpred", default=None)
    ap.add_argument("--a", default="smse", help="baseline utility")
    ap.add_argument("--b", default="gnom", help="challenger utility")
    ap.add_argument("--a-bottom-up", action="store_true",
                    help="run baseline bottom-up (default: its native direction)")
    ap.add_argument("--b-bottom-up", action="store_true",
                    help="run challenger bottom-up (default: Gnom top-down)")
    ap.add_argument("--free-pins", action="store_true",
                    help="let output/token_embd/routers fight for budget")
    ap.add_argument("--allow-q3-or-lower", dest="allow_q3",
                    action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--dont-allow-q3", dest="allow_q3",
                    action="store_false",
                    help="keep the Q4_K floor even if --size is smaller")
    ap.set_defaults(allow_q3=True)
    args = ap.parse_args()

    model = read_model(args.model)
    im = combine_imatrix([read_imatrix(args.imatrix)])
    tied_groups = detect_tied_groups(im)
    imp_table = build_importance_table(im, model)
    ne_map = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}

    needs_table = ("gnom", "netpure", "netfd")
    if (args.a in needs_table or args.b in needs_table) and not args.netpred:
        print("Gnom side needs --netpred", flush=True)
        sys.exit(2)
    a = run(model, imp_table, tied_groups, args.size, args.a,
            args.netpred if args.a in needs_table else None,
            args.free_pins, args.allow_q3,
            top_down=args.a in ("gnom", "netfd") and not args.a_bottom_up)
    b = run(model, imp_table, tied_groups, args.size, args.b,
            args.netpred if args.b in needs_table else None,
            args.free_pins, args.allow_q3,
            top_down=args.b in ("gnom", "netfd") and not args.b_bottom_up)
    print(f"baseline={args.a} challenger={args.b}")

    tiers = sorted(set(a.values()) | set(b.values()))
    print(f"{'tier':8s} {'a_n':>7s} {'b_n':>6s} "
          f"{'a_MiB':>9s} {'b_MiB':>8s}")
    for t in tiers:
        na = sum(1 for v in a.values() if v == t)
        nb = sum(1 for v in b.values() if v == t)
        ma = sum(ne_map.get(k, 0) for k, v in a.items() if v == t)
        mb = sum(ne_map.get(k, 0) for k, v in b.items() if v == t)
        print(f"{t:8s} {na:7d} {nb:6d} {ma / 2**20:9.1f} {mb / 2**20:8.1f}")

    # tissue x tier moves: which tissues gnom upgrades/downgrades
    from collections import Counter
    order = ["F32", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K_M", "Q3_K_S",
             "Q3_K_L", "Q2_K"]
    rank = {t: i for i, t in enumerate(order)}
    moves = Counter()
    moved_tensors = []
    for k in a:
        if a[k] != b.get(k):
            d = rank.get(b.get(k, "?"), 99) - rank.get(a[k], 99)
            moves[(tissue(k), "UP" if d < 0 else "DOWN")] += 1
            moved_tensors.append((k, a[k], b.get(k)))
    print(f"\nmoved tensors: {len(moved_tensors)}/{len(a)}")
    for (tis, direction), c in sorted(moves.items()):
        print(f"  {tis:14s} {direction:4s} x{c}")
    print("\nsample UP (gnom spends more here):")
    shown = 0
    for k, ta, tb in moved_tensors:
        if rank.get(tb, 99) < rank.get(ta, 99):
            print(f"  {k:42s} {ta} -> {tb}")
            shown += 1
            if shown >= 15:
                break
    print("sample DOWN (gnom saves here):")
    shown = 0
    for k, ta, tb in moved_tensors:
        if rank.get(tb, 99) > rank.get(ta, 99):
            print(f"  {k:42s} {ta} -> {tb}")
            shown += 1
            if shown >= 15:
                break


if __name__ == "__main__":
    main()
