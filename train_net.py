"""Train the tiny damage net (battlefield zoo).

Reads group features (models/features_17.npz, per-tensor) + teacher labels
(models/damage.jsonl, per-group) and trains a SMALL regressor:

    12 -> 8 -> 4 -> 1   (~150 params, honest size for ~100 labels)

Group features: timp summed over members (damage scales with total weight),
everything else averaged. Standardized. Huber-ish robustness via sample
weights ~ |damage| + feature jitter (anti-parrot).

Outputs: models/netdmg.npz (params + norm stats).
Ablation: --no-timp drops timp features (the real exam).

With --synthetic N: shakedown mode — fake labels damage = log(timp_sum) +
noise, to validate the training machinery before teacher labels land.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tinynet import init_params, train, forward, standardize
ACT = 'mish'

DIMS_SMALL = [15, 8, 4, 1]


def load_features(path):
    z = np.load(path)
    return {str(k): np.asarray(z[k]) for k in z.files}


def group_matrix(feat, groups, no_timp=False):
    """groups: {tag: [tensor_names]} -> X matrix, tags list."""
    X, tags = [], []
    for tag, members in groups.items():
        rows = [feat[m] for m in members if m in feat]
        if not rows:
            continue
        A = np.array(rows)
        g = A.mean(axis=0)
        g[8] = A[:, 8].sum()  # timp_mean -> timp_sum over members
        if no_timp:
            g[8] = 0.0
            g[9] = 0.0
        X.append(g)
        tags.append(tag)
    return np.array(X), tags


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="models/features_17.npz")
    ap.add_argument("--group-features", default=None,
                    help="prebuilt group matrix npz (X, tags); skips group_matrix")
    ap.add_argument("--labels", default="models/damage.jsonl")
    ap.add_argument("--groups-from", default=None,
                    help="json {tag: [tensors]}; default: singletons from features")
    ap.add_argument("--out", default="models/netdmg.npz")
    ap.add_argument("--no-timp", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0,
                    help="shakedown: fake labels, N unused (uses all tensors)")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mixed", action="store_true",
                    help="mixed-effects Gnom: shared trunk + per-group bias")
    ap.add_argument("--lam", type=float, default=1.0,
                    help="James-Stein shrinkage for group biases")
    ap.add_argument("--label-spec", default=None,
                    help="'path:base_bpw:drop_bpw,...' pool of label files; "
                         "tier bpws become features (curve learning)")
    args = ap.parse_args()

    feat = load_features(args.features) if not args.group_features else None
    if args.group_features:
        z = np.load(args.group_features, allow_pickle=True)
        X0 = np.asarray(z["X"], dtype=np.float64)
        tags0 = [str(t) for t in z["tags"]]
    elif args.groups_from:
        groups = json.load(open(args.groups_from))
    else:
        # singleton groups keyed by tensor name
        inv = {}
        for k in feat:
            inv[k] = [n for n in [k] if True]
        groups = {k: [k] for k in feat}

    if not args.group_features:
        X0, tags0 = group_matrix(feat, groups, args.no_timp)
    tag2row = {t: X0[i] for i, t in enumerate(tags0)}
    if args.synthetic:
        rng = np.random.default_rng(11)
        X, tags = X0, tags0
        y = np.log10(X[:, 8] + 1) + rng.normal(0, 0.15, len(X))
        print(f"synthetic labels: damage=log(timp_sum)+noise, n={len(X)}")
    else:
        # pool of label files; tier bpws appended as features
        specs = []
        if args.label_spec:
            for part in args.label_spec.split(","):
                p, b, d = part.split(":")
                specs.append((p, float(b), float(d)))
        else:
            specs = [(args.labels, 5.5, 4.5)]
        Xs_list, ys_list, gtags = [], [], []
        for path, base_bpw, drop_bpw in specs:
            lab, base = {}, None
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    if d["unit"] == "BASELINE":
                        base = d["ppl"]
                    else:
                        lab[d["unit"]] = d["damage"]
            if base:
                kf = 1.0 / base
                print(f"{path}: relative labels (base={base:.2f}, "
                      f"tiers {base_bpw}->{drop_bpw})")
            else:
                kf = 1.0
            for t, dmg in lab.items():
                if t not in tag2row:
                    continue
                row = np.concatenate([tag2row[t], [base_bpw, drop_bpw]])
                Xs_list.append(row)
                ys_list.append(dmg * kf)
                gtags.append(t)
        X = np.array(Xs_list)
        y = np.array(ys_list)
        print(f"real labels pooled: n={len(y)}")

    from tinynet import fit_mixed, mixed_predict
    DIMS = [17, 8, 4, 1] if X.shape[1] == 17 else DIMS_SMALL
    if args.synthetic:
        tags = tags0 if 'tags0' in dir() else list(range(len(X)))
        gtags = list(tags)

    # split by GROUP (same tag never leaks across train/heldout)
    uniq = sorted(set(gtags))
    gidx_of = {t: i for i, t in enumerate(uniq)}
    gidx = np.array([gidx_of[t] for t in gtags])
    rng = np.random.default_rng(args.seed)
    gperm = rng.permutation(len(uniq))
    gcut = max(2, len(uniq) * 4 // 5)
    gtr = set(gperm[:gcut])
    tr = np.array([i for i in range(len(X)) if gidx[i] in gtr])
    va = np.array([i for i in range(len(X)) if gidx[i] not in gtr])

    if args.mixed:
        mp = fit_mixed(X, y, gidx, len(uniq), DIMS, lam=args.lam,
                       seed=args.seed, epochs=args.epochs,
                       log_every=args.epochs // 3, noise_std=args.noise)
        import copy as _copy
        # honest heldout: zero biases of heldout groups (no label leakage)
        va_groups = set(gidx[va])
        mp_h = _copy.deepcopy(mp)
        for g in va_groups:
            mp_h["bias"][g] = 0.0
        for name, ii, mpp in (("train", tr, mp), ("heldout", va, mp_h)):
            p = mixed_predict(mpp, X[ii], gidx[ii], len(uniq), ACT)
            rmse = float(np.sqrt(np.mean((p - y[ii]) ** 2)))
            sp = spearman(p, y[ii])
            print(f"{name}: RMSE={rmse:.4f} Spearman={sp:.3f} (n={len(ii)})")
        print(f"timp_sum Spearman={spearman(X[:, 8], y):.3f} <- beat this")
        np.savez(args.out, **{"bias": mp["bias"]},
                 xmu=mp["xmu"], xsd=mp["xsd"],
                 ymu=np.array(mp["ymu"]), ysd=np.array(mp["ysd"]),
                 tmu=np.array(mp["tmu"]), tsd=np.array(mp["tsd"]),
                 dims=np.array(DIMS),
                 **{f"p{i}{k}": v for i, p in enumerate(mp["trunk"])
                    for k, v in p.items()})
        print("saved mixed", args.out)
        return

    Xs, mu, sd = standardize(X)

    w = np.abs(ys[tr]) + 0.05 * np.abs(ys[tr]).max()
    ps = init_params(DIMS, seed=args.seed)
    train(ps, Xs[tr], ys[tr], epochs=args.epochs, log_every=args.epochs // 3,
          noise_std=args.noise, sample_w=w, activation=ACT, dropout=args.dropout)

    for name, ii in (("train", tr), ("heldout", va)):
        p = forward(ps, Xs[ii], ACT)[0][-1].ravel() * ysd + ymu
        rmse = float(np.sqrt(np.mean((p - y[ii]) ** 2)))
        sp = spearman(p, y[ii])
        print(f"{name}: RMSE={rmse:.4f} Spearman={sp:.3f} (n={len(ii)})")

    # baseline: raw timp_sum rank correlation (what we must beat)
    print(f"timp_sum Spearman={spearman(X[:, 8], y):.3f} <- beat this")
    np.savez(args.out, **{f"p{i}{k}": v
                           for i, p in enumerate(ps) for k, v in p.items()},
             mu=mu, sd=sd, dims=np.array(DIMS))
    print("saved", args.out)


if __name__ == "__main__":
    main()
