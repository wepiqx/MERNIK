#!/usr/bin/env python3
"""SMSE vs netdmg allocation comparison at equal --size (CPU-only).

Runs the exact classifier path main.py uses (optimal_classify, bottom-up)
twice — utility=smse and utility=netdmg with a netpred table — then diffs
the per-tensor tier maps: tier histograms, tissue x tier moves, byte flow,
and the top upgraded/downgraded tensors.

    compare_alloc.py --model M.gguf --imatrix I.gguf --size 5310
                     --netpred models/netpred_m.json [--netdmg-w 0.5]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from model_reader import read_model
from imatrix_reader import (read_imatrix, build_importance_table,
                            detect_tied_groups, combine_imatrix)
from classifier import optimal_classify


def run(model, imp_table, tied_groups, target, mode, netpred, w,
        free_pins=False, allow_q3=False):
    uopt = {"mode": mode, "netpred": netpred, "netdmg_w": w}
    assignments, _ = optimal_classify(
        imp_table, tied_groups, model, target_size_mib=target,
        allow_q3=allow_q3, free_pins=free_pins, uopt=uopt, legacy_1d=False,
        relief=None, relief_thr=-0.5, allowed_tiers=None, pin_small=0.0)
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
    ap.add_argument("--netdmg-w", type=float, default=0.5)
    ap.add_argument("--a", default="smse", help="baseline utility")
    ap.add_argument("--b", default="netdmg", help="challenger utility")
    ap.add_argument("--free-pins", action="store_true",
                    help="let output/token_embd/routers fight for budget")
    ap.add_argument("--allow-q3-or-lower", action="store_true",
                    help="low-importance tensors may go to Q3_K or lower")
    args = ap.parse_args()

    model = read_model(args.model)
    im = combine_imatrix([read_imatrix(args.imatrix)])
    tied_groups = detect_tied_groups(im)
    imp_table = build_importance_table(im, model)
    ne_map = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}

    if (args.a in ("netdmg", "netpure") or
            args.b in ("netdmg", "netpure")) and not args.netpred:
        print("netdmg side needs --netpred", flush=True)
        sys.exit(2)
    _NEEDS_TABLE = ("netdmg", "netpure")
    a = run(model, imp_table, tied_groups, args.size, args.a,
            args.netpred if args.a in _NEEDS_TABLE else None,
            args.netdmg_w, args.free_pins, args.allow_q3_or_lower)
    b = run(model, imp_table, tied_groups, args.size, args.b,
            args.netpred if args.b in _NEEDS_TABLE else None,
            args.netdmg_w, args.free_pins, args.allow_q3_or_lower)
    print(f"baseline={args.a} challenger={args.b}")

    tiers = sorted(set(a.values()) | set(b.values()))
    print(f"{'tier':8s} {'smse_n':>7s} {'net_n':>6s} "
          f"{'smse_MiB':>9s} {'net_MiB':>8s}")
    for t in tiers:
        na = sum(1 for v in a.values() if v == t)
        nb = sum(1 for v in b.values() if v == t)
        ma = sum(ne_map.get(k, 0) for k, v in a.items() if v == t)
        mb = sum(ne_map.get(k, 0) for k, v in b.items() if v == t)
        print(f"{t:8s} {na:7d} {nb:6d} {ma / 2**20:9.1f} {mb / 2**20:8.1f}")

    # tissue x tier moves: which tissues netdmg upgrades/downgrades
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
    print("\nsample UP (netdmg spends more here):")
    shown = 0
    for k, ta, tb in moved_tensors:
        if rank.get(tb, 99) < rank.get(ta, 99):
            print(f"  {k:42s} {ta} -> {tb}")
            shown += 1
            if shown >= 15:
                break
    print("sample DOWN (netdmg saves here):")
    shown = 0
    for k, ta, tb in moved_tensors:
        if rank.get(tb, 99) > rank.get(ta, 99):
            print(f"  {k:42s} {ta} -> {tb}")
            shown += 1
            if shown >= 15:
                break


if __name__ == "__main__":
    main()
