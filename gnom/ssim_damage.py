#!/usr/bin/env python3
"""SSIM-damage sweep (Gnom third metric): per-group structural damage.

Fake-quantizes each tied group at --drop-tier and measures mean (1-SSIM)
over members — a CPU-only label (no GPU, no inference). Same jsonl schema
as teacher labels but with "ssim_damage" (+ "damage" mirror for loader
compat). Rank-correct per the zoo lesson (emulation lies 4x on scale,
never on order) — and Spearman only needs order.

    ssim_damage.py --model M-F16.gguf --imatrix I.gguf
                   --out gnom/labels/ssim_17_q4.jsonl [--drop-tier Q4_K]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_dtypes  # noqa: F401
from gguf import GGUFReader
from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups
from ssim_probe import fake_quant, block_ssim, TIERS, _perceptual_row


def unit_tag(unit):
    return "+".join(n.replace(".weight", "").split(".")[-1] + "@" +
                    n.split(".")[1] for n in unit
                    if n.startswith("blk.")) or "global"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-tier", default="Q4_K")
    ap.add_argument("--perceptual", action="store_true",
                    help="weight column damage by imatrix input energy "
                         "(damage in high-traffic columns counts more)")
    ap.add_argument("--colblock", type=int, default=64)
    ap.add_argument("--ms", action="store_true",
                    help="multi-scale SSIM: geometric blend over block "
                         "sizes 64/256/1024 (coarse=drift, fine=texture)")
    ap.add_argument("--gmsd", action="store_true",
                    help="gradient-magnitude deviation pooling: std (not mean) "
                         "of per-block gradient similarity — the weakest "
                         "link dominates (outlier channels, sub-4)")
    args = ap.parse_args()

    bits, blk = TIERS[args.drop_tier]
    model = read_model(args.model)
    im = read_imatrix(args.imatrix)
    energy = {}
    if args.perceptual:
        for name, d in im["tensors"].items():
            v = d.get("in_sum2_raw")
            if v is not None:
                energy[name] = np.asarray(v, dtype=np.float64)
    groups = detect_tied_groups(im)
    reader = GGUFReader(args.model)
    wmap = {t.name: t for t in reader.tensors}

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["unit"])
    n_new = 0
    with open(args.out, "a") as f:
        for gi, group in enumerate(groups):
            tag = unit_tag(group)
            if tag in done:
                continue
            ds = []
            for n in group:
                if n not in wmap:
                    continue
                try:
                    w = np.array(wmap[n].data, dtype=np.float32).ravel()
                except Exception:
                    continue
                q = fake_quant(w, bits, blk)
                if args.gmsd:
                    # 1D gradient magnitude per 256-block, similarity map,
                    # pooled by STD: uneven damage scores worse than uniform.
                    n = (w.size // 256) * 256
                    if n < 512:
                        ds.append(0.0)
                        continue
                    A = w[:n].reshape(-1, 256)
                    B = q[:n].reshape(-1, 256)
                    ga = np.abs(np.diff(A, axis=1))
                    gb = np.abs(np.diff(B, axis=1))
                    c = (0.03 * max(w.max() - w.min(), 1e-12)) ** 2
                    gms = (2 * ga * gb + c) / (ga * ga + gb * gb + c)
                    ds.append(float(gms.std()))
                    continue
                if args.ms and not (args.perceptual and n in energy):
                    import math
                    m = 1.0
                    for bsize in (64, 256, 1024):
                        s, _, _, _ = block_ssim(w, q, block=bsize)
                        m *= max(s, 1e-12) ** (1.0 / 3.0)
                    ds.append(1.0 - m)
                elif args.perceptual and n in energy:
                    t = wmap[n]
                    fq = {args.drop_tier: q}
                    prow = _perceptual_row(t, w, fq, energy[n], args.colblock)
                    ds.append(prow[args.drop_tier])
                else:
                    s, _, _, _ = block_ssim(w, q)
                    ds.append(1.0 - s)
            if not ds:
                continue
            dmg = float(np.mean(ds))
            rec = {"unit": tag, "tensors": group,
                   "ssim_damage": dmg, "damage": dmg}
            f.write(json.dumps(rec) + "\n")
            n_new += 1
            if n_new % 20 == 0:
                print(f"  [{gi + 1}/{len(groups)}] {tag} ssim_dmg={dmg:.5f}",
                      flush=True)
    print(f"SSIM labels: +{n_new} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
