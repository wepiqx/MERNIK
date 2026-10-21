#!/usr/bin/env python3
"""Consolidate MERNIK damage labels into a clean pool for Gnom training.

Filters the mixed label archive (PPL-era + KLD-era, old + new) down to the
KLD-based files the new Gnom consumes, tags each with model/tier/corpus/columns,
and writes a clean pool.json + a unified directory.

Usage:
    python scripts/consolidate_labels.py [--forge PATH] [--out assets/labels/pool]
"""
import argparse
import json
import os
import shutil
import sys

# The KLD-based label files we keep, with measured metadata.
# tier = drop tier (base is Q5_K everywhere). corpus = what the KLD pass ran on.
# n_cols = how many of the 5 KLD columns are present (5 = multi-target, 1 = mean-only).
SOURCES = [
    # The clean rlmini pool: the new KLD-only 5-column files, Q4 and Q3 legs
    # (multitier: same units at two drop tiers -> the tier curve, and the
    # model axis for a leave-one-MODEL-out exam).
    # The old damage_17_rlmini / damage_4b_rlmini are duplicates (same units, ~0.002
    # KLD drift from different measurement settings) — retired in favour of these.
    {"name": "rl17_q4",  "model": "spark17", "tier": "Q4_K", "corpus": "rlmini",
     "n_cols": 5, "path": "/mnt/Vsio/assets/labels/damage_17b_rlmini_kldonly.jsonl"},
    {"name": "rl17_q3",  "model": "spark17", "tier": "Q3_K", "corpus": "rlmini",
     "n_cols": 5, "path": "/mnt/Vsio/assets/labels/damage_17b_rlmini_q3.jsonl"},
    {"name": "rl4b_q4",  "model": "spark4b", "tier": "Q4_K", "corpus": "rlmini",
     "n_cols": 5, "path": "/mnt/Vsio/assets/labels/damage_4b_rlmini_kldonly.jsonl"},
    {"name": "rl4b_q3",  "model": "spark4b", "tier": "Q3_K", "corpus": "rlmini",
     "n_cols": 5, "path": "/mnt/Vsio/assets/labels/damage_4b_rlmini_q3.jsonl"},
    # 9B — new clean 5-col (Q4, accumulating on the forge).
    {"name": "rl9b_q4",  "model": "ornith", "tier": "Q4_K", "corpus": "rlmini",
     "n_cols": 5, "path": None},  # --forge supplies this
]

KLD_COLS = ["damage_kld", "damage_kld_max", "damage_kld_median",
            "damage_dp_rms", "damage_same_top"]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forge", default=None,
                    help="path to the 9B forge labels (damage_ornith_rlmini.jsonl)")
    ap.add_argument("--out", default="assets/labels/pool")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pool = []
    for src in SOURCES:
        path = src["path"] if src["path"] else args.forge
        if not path or not os.path.exists(path):
            print(f"  SKIP {src['name']}: no file ({path})")
            continue
        rows = load(path)
        cols = set()
        for r in rows:
            cols |= set(r.keys())
        have = [c for c in KLD_COLS if c in cols]
        n_units = sum(1 for r in rows if r.get("unit") != "BASELINE")
        n_term = sum(1 for r in rows if r.get("terminated"))
        # copy into the clean pool with a uniform name
        dst = os.path.join(args.out, f"{src['name']}.jsonl")
        shutil.copyfile(path, dst)
        pool.append({**src, "file": os.path.basename(dst),
                     "rows": len(rows), "units": n_units,
                     "terminated": n_term, "kld_cols": have})
        print(f"  OK {src['name']:8s} rows={len(rows):4d} units={n_units:4d} "
              f"term={n_term} cols={len(have)}")

    with open(os.path.join(args.out, "pool.json"), "w") as f:
        json.dump({"base_tier": "Q5_K", "files": pool}, f, indent=2)
    print(f"\npool.json written: {len(pool)} files -> {args.out}")


if __name__ == "__main__":
    main()
