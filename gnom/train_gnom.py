#!/usr/bin/env python3
"""Gnom-0.2 trainer: pooled multi-model damage regression.

    train_gnom.py --pool pool.json --out gnom/models/gnom02.npz
                  [--variant trunk|modelbias|moeshared|all] [--seed 7]

pool.json: [{"features": path.npz, "labels": path.jsonl,
             "model": tag, "base_bpw": 5.5, "drop_bpw": 4.5}, ...]
Label damage is divided by file BASELINE (relative). Tier bpws appended
(curve learning). Per-model standardization of scale-sensitive features
(timp_mean/max z-scored, logN vs model median) happens here, not in npz.
--metric picks the label key: damage (PPL) | damage_kld (KLD) | ssim.

Reports random-split AND leave-one-model-out Spearman + timp baseline.
Duel bar: LOMO >= 0.45.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinynet import (init_params, train, forward, standardize, fit_mixed,
                     mixed_predict, moe_train_shared, moe_predict_shared)

DIM_MAP = {"trunk": [16, 8, 4, 1], "modelbias": [18, 8, 4, 1]}

# raw feature columns (tinynet.featurize order): 0-5 onehot, 6 layer,
# 7 logN, 8 timp_mean, 9 timp_max, 10 frag, 11 econc, 12-14 w-stats
SCALE_COLS = [7, 8, 9]


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float))).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def load_pool(path, metric="damage"):
    specs = json.load(open(path))
    Xs, ys, models, gtags, tiers = [], [], [], [], []
    for sp in specs:
        m = sp.get("metric", metric)  # per-file override: pool PPL+KLD rows
        z = np.load(sp["features"])
        X0 = np.asarray(z["X"], dtype=np.float64)
        names = [str(t) for t in z["names"]]
        tmu = float(z["timp_mean"][0]) if "timp_mean" in z else 0.0
        tsd = float(z["timp_std"][0]) if "timp_std" in z else 1.0
        lmed = float(z["logN_med"][0]) if "logN_med" in z else 0.0
        row = {n: i for i, n in enumerate(names)}
        lab, base = {}, None
        with open(sp["labels"]) as f:
            for line in f:
                d = json.loads(line)
                if d["unit"] == "BASELINE":
                    base = d["ppl"]
                elif m in d:
                    lab[d["unit"]] = (d[m],
                                      [t for t in d.get("tensors", [])])
        kf = 1.0 if m == "damage_kld" else (1.0 / base if base else 1.0)
        for unit, (dmg, tensors) in lab.items():
            idx = [row[t] for t in tensors if t in row]
            if not idx:
                continue
            A = X0[idx]
            g = A.mean(axis=0)
            # scale-invariant: timp z-scored in-model, logN vs model median
            g[8] = (A[:, 8].mean() - tmu) / (tsd + 1e-12)
            g[9] = (A[:, 9].max() - tmu) / (tsd + 1e-12)
            g[7] = A[:, 7].mean() - lmed
            Xs.append(np.concatenate([g, [sp["base_bpw"], sp["drop_bpw"]]]))
            ys.append(dmg * kf)
            models.append(sp["model"])
            gtags.append(f'{sp["model"]}:{m}:{unit}')
    return (np.array(Xs), np.array(ys), np.array(models), gtags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default="gnom/models/gnom02.npz")
    ap.add_argument("--variant",
                    choices=["trunk", "modelbias", "moeshared", "all"],
                    default="all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--metric", default="damage",
                    help="label key: damage (PPL) | damage_kld (KLD) | "
                         "ssim_damage (structural)")
    args = ap.parse_args()

    X, y, models, gtags = load_pool(args.pool, metric=args.metric)
    print(f"pooled: n={len(y)}, models={sorted(set(models))}", flush=True)
    uniq_m = sorted(set(models))
    midx = np.array([uniq_m.index(m) for m in models])
    DIMS = [X.shape[1], 8, 4, 1]

    def run_variant(variant, tr_idx, va_idx, seed):
        if variant == "moeshared":
            exp = moe_train_shared(X[tr_idx][:, :16], y[tr_idx], seed=seed,
                                   epochs=args.epochs)
            p = moe_predict_shared(exp, X[:, :16])
            return p
        if variant == "modelbias":
            # honest subset fit: trunk never sees va labels; unseen models
            # get bias 0 (pure trunk generalization).
            Xtr, ytr = X[tr_idx], y[tr_idx]
            mtr = midx[tr_idx]
            ug = sorted(set(mtr.tolist()))
            r = {g: i for i, g in enumerate(ug)}
            gs = np.array([r[g] for g in mtr])
            mp = fit_mixed(Xtr, ytr, gs, len(ug), DIMS, seed=seed,
                           epochs=args.epochs, log_every=args.epochs)
            Xs = (X - mp["xmu"]) / mp["xsd"]
            base = forward(mp["trunk"], Xs, "mish")[0][-1].ravel() \
                * mp["tsd"] + mp["tmu"]
            b = np.zeros(len(X))
            known = np.array([r[m] if m in r else -1 for m in midx])
            kk = known >= 0
            b[kk] = mp["bias"][known[kk]]
            return (base + b) * mp["ysd"] + mp["ymu"]
        ps = init_params(DIMS, seed=seed)
        Xs, _, _ = standardize(X)
        w = np.abs(y[tr_idx]) + 0.05 * np.abs(y[tr_idx]).max()
        train(ps, Xs[tr_idx], y[tr_idx], epochs=args.epochs,
              log_every=args.epochs, noise_std=0.05, sample_w=w,
              activation="mish", dropout=0.0, seed=seed)
        mu, sd = float(y.mean()), float(y.std() + 1e-12)
        return forward(ps, Xs, "mish")[0][-1].ravel() * sd + mu

    variants = (["trunk", "modelbias", "moeshared"]
                if args.variant == "all" else [args.variant])
    rng = np.random.default_rng(args.seed)
    # random split (group-disjoint)
    uniq_g = sorted(set(gtags))
    gperm = rng.permutation(len(uniq_g))
    gtr = set(gperm[:max(2, len(uniq_g) * 4 // 5)])
    gidx = np.array([uniq_g.index(t) for t in gtags])
    tr = np.array([i for i in range(len(X)) if gidx[i] in gtr])
    va = np.array([i for i in range(len(X)) if gidx[i] not in gtr])
    print(f"split: train {len(tr)} / heldout {len(va)}", flush=True)
    results = {}
    for v in variants:
        p = run_variant(v, tr, va, args.seed)
        sp_r = spearman(p[va], y[va])
        # LOMO needs >=2 models; single-metric pools report random only.
        sp_lomo = []
        if len(uniq_m) > 1:
            for m in uniq_m:
                trm = np.where(models != m)[0]
                vam = np.where(models == m)[0]
                if len(vam) < 5 or len(trm) < 5:
                    continue
                pm = run_variant(v, trm, vam, args.seed)
                sp_lomo.append(spearman(pm[vam], y[vam]))
        lomo = float(np.mean(sp_lomo)) if sp_lomo else 0.0
        base = spearman(X[va, 8], y[va])
        print(f"{v}: random={sp_r:.3f} LOMO={lomo:.3f} "
              f"(per-model {[f'{s:.2f}' for s in sp_lomo]}) "
              f"timp_base={base:.3f}", flush=True)
        results[v] = {"random": sp_r, "lomo": lomo}
    best = max(results, key=lambda v: results[v]["lomo"])
    print(f"BEST: {best} LOMO={results[best]['lomo']:.3f} "
          f"(bar 0.45: {'PASS' if results[best]['lomo'] >= 0.45 else 'FAIL'})",
          flush=True)
    # Export: trunk trained on FULL data (moeshared/modelbias export TODO).
    # Whole gnom: ~220 floats ≈ 2KB in one .npz (precompute per model into
    # netpred.json; no live inference needed in the queue).
    full = np.arange(len(X))
    ps = init_params(DIMS, seed=args.seed)
    Xs, xmu, xsd = standardize(X)
    ymu, ysd = float(y.mean()), float(y.std() + 1e-12)
    w = np.abs(y) + 0.05 * np.abs(y).max()
    train(ps, Xs, (y - ymu) / ysd, epochs=args.epochs,
          log_every=args.epochs, noise_std=0.05, sample_w=w,
          activation="mish", dropout=0.0, seed=args.seed)
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(args.out)) or ".",
                 exist_ok=True)
    np.savez(args.out, xmu=xmu, xsd=xsd, ymu=np.array(ymu), ysd=np.array(ysd),
             dims=np.array(DIMS),
             **{f"p{i}{k}": v for i, p in enumerate(ps) for k, v in p.items()})
    print(f"saved trunk {args.out}", flush=True)


if __name__ == "__main__":
    main()
