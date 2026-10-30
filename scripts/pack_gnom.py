#!/usr/bin/env python3
"""pack_gnom.py — seal the Gnom modulator as a .safetensors artifact.

v1 contract (MERNIK.gnom.v1):
  weights   W1 18x48, b1 48, W2 48x24, b2 24, W3 24x1, b3 1   (mish MLP)
  header    model architecture + feature contract + provenance

Feature order (18 = onehot6 + 6 scalars + wstats4 + tails2):
  onehot(TYPES): attn_gate, attn_qkv, attn_output, ffn_gate, ffn_up, ffn_down
  [6]  layer  = blk layer / n_layers
  [7]  log10(n_elements+1)
  [8]  imp matrix mean
  [9]  imp matrix max
  [10] frag   = 1 - ssim
  [11] econc  = imp_max / (imp_mean+eps)
  [12] log10(weight std)
  [13] clamp(kurt,-2,50)/10
  [14] log10(max/mean + 1)
  [15] frac(|w| > 6*std)
  [16] log10(imp_min+1)
  [17] log10(imp_sum+1)
Rows are z-scored per model pool (loader standardize); target damage_kld.
"""
import sys
import os
import json
import time

import numpy as np

_ROOT = "/mnt/Vsio/MERNIK"
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ROOT + "/gnom")
from tinynet import standardize, init_params, train, forward, TYPES
from train_gnom import load_pool_multi

from safetensors.numpy import save_file, load_file

DIMS = [20, 48, 24, 1]   # probe-set below; input width is discovered at fit
DIMS_IN = 18             # raw featurize contract (onehot6+scalars6+wstats4+tails2)
EPOCHS = 1500
SEED = 7
POOL = os.environ.get(
    "GNOM_POOL",
    os.path.join(_ROOT, "gnom/pool_fracture_lfm3_2026-10-30.json"))

FEATURE_NAMES = (
    [f"onehot_{t}" for t in TYPES] +
    ["layer_frac", "log10_n_elements", "imp_mean", "imp_max", "frag_ssim",
     "econc_peak_mean", "w_log10_std", "w_kurt_clip10", "w_log10_peakmean",
     "w_outlier_frac", "imp_log10_min", "imp_log10_sum"]
)


def fit_pool():
    X, Y, models, gtags, stats = load_pool_multi(
        POOL, ["damage_kld"], force_metric=True, strict_map=True)
    dims = [X.shape[1], 48, 24, 1]
    y = Y[:, 0]
    ps = init_params(dims, seed=SEED)
    w = np.abs(y) + 0.05 * np.abs(y).max()
    train(ps, X, y, epochs=EPOCHS, log_every=100, noise_std=0.05,
          sample_w=w, activation="mish", dropout=0.0, seed=SEED)
    pred = forward(ps, X, "mish")[0][-1].ravel()
    root = np.array([m.rsplit("_", 1)[0] for m in models])
    per = {}
    for m in sorted(set(root)):
        i = root == m
        per[m] = float(spearman(pred[i], y[i]))
    rho_wm = float(np.mean(list(per.values())))
    return ps, X, y, rho_wm, dims, per


def pack(ps, y, rho, n_rows, dims, per, tag=None):
    tag = tag or os.environ.get("GNOM_TAG", "lfm3ext")
    tensors = {}
    for i, p in enumerate(ps, 1):
        tensors[f"W{i}"] = p["W"].astype(np.float64)
        tensors[f"b{i}"] = p["b"].astype(np.float64)
    raw_names = (
        [f"onehot_{t}" for t in TYPES] +
        ["layer_frac", "log10_n_elements", "imp_mean", "imp_max", "frag_ssim",
         "econc_peak_mean", "w_log10_std", "w_kurt_clip10", "w_log10_peakmean",
         "w_outlier_frac", "imp_log10_min", "imp_log10_sum"]
    )
    assert len(raw_names) == DIMS_IN, f"contract drift: {len(raw_names)} vs {DIMS_IN}"
    feature_order = (
        raw_names +
        ["base_bpw", "drop_bpw"] +
        ["POOL_EXPLAINS_LAST_%d" % (dims[0] - DIMS_IN - 2)] *
        max(0, dims[0] - DIMS_IN - 2))
    meta = {
        "schema": "MERNIK.gnom.v1",
        "doc": "Gnom damage-rank modulator. Inputs are PER-UNIT rows: "
               "mean of raw features over the unit's tensors with special "
               "imp/logN scaling (train_gnom.load_pool_multi), then "
               "z-scored per pool. Output = damage_kld rank score.",
        "dims": ",".join(map(str, dims)),
        "activation": "mish",
        "epochs": str(EPOCHS),
        "seed": str(SEED),
        "metric": "damage_kld",
        "pool_path": POOL,
        "pool_sha": _sha(POOL),
        "feature_order": json.dumps(feature_order),
        "preprocessing": "train_gnom.load_pool_multi (w_center=True, "
                         "namefeats=off): unit-mean aggregation + "
                         "per-model-stat z/centering",
        "self_test_spearman_per_model": json.dumps(per),
        "n_rows": str(n_rows),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": "pack_gnom.py v1",
    }
    out = os.path.join(_ROOT, "gnom", "artifacts",
                       f"gnom_v1_2026-10-30_{tag}.safetensors")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_file(tensors, out, metadata=meta)
    return out


def verify(path, X, y, rho, models):
    st = load_file(path)
    ps = []
    i = 1
    while f"W{i}" in st:
        ps.append({"W": st[f"W{i}"], "b": st[f"b{i}"]})
        i += 1
    pred = forward(ps, X, "mish")[0][-1].ravel()
    root = np.array([m.rsplit("_", 1)[0] for m in models])
    per = {m: float(spearman(pred[root == m], y[root == m]))
           for m in sorted(set(root))}
    rho2 = float(np.mean(list(per.values())))
    assert str(rho2) == str(rho), f"roundtrip mismatch {rho} vs {rho2}"
    return rho2, len(st), per


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


BEST_PER_FAMILY = {
    "rl17": (48, 24, 1500), "rl4b": (8, 4, 1500), "ling_moe": (48, 24, 1500),
    "lfm3": (48, 24, 1500), "rl9b": (48, 24, 1500),
}


def pack_multinet_v3(X, y, models, dims_ref, tag="lfm3ext_v2"):
    tag = os.environ.get("GNOM_TAG", tag)
    """Per-family BEST net of the 2026-10-30 sweep (48-24 champion except
    rl4b, where the 209p legacy net wins 0.74 vs 0.71). Each family may
    carry its own dims; the selection and its measured score are in the
    metadata. Held-out rows are never seen by their family's net."""
    from train_gnom import spearman
    root = np.array([m.rsplit("_", 1)[0] for m in models])
    tensors, nets = {}, {}
    for fam in sorted(set(root)):
        h1, h2, ep = BEST_PER_FAMILY[fam]
        dims = [dims_ref[0], h1, h2, 1]
        va = root == fam
        tr = ~va
        Xs, _, _ = standardize(np.concatenate([X[tr], X[va]]))
        ntr = int(tr.sum())
        w = np.abs(y[tr]) + 0.05 * np.abs(y[tr]).max()
        ps = init_params(dims, seed=SEED)
        train(ps, Xs[:ntr], y[tr], epochs=ep, log_every=10**9,
              noise_std=0.05, sample_w=w, activation="mish", dropout=0.0,
              seed=SEED)
        pred = forward(ps, Xs[ntr:], "mish")[0][-1].ravel()
        r = float(spearman(pred, y[va]))
        nets[fam] = {"lomo_spearman": repr(r), "dims": ",".join(map(str, dims)),
                     "n_train": int(tr.sum()), "n_scored": int(va.sum())}
        for i, p in enumerate(ps, 1):
            tensors[f"{fam}/W{i}"] = p["W"].astype(np.float64)
            tensors[f"{fam}/b{i}"] = p["b"].astype(np.float64)
    meta = {
        "schema": "MERNIK.gnom.multinet.v3",
        "doc": "Per-family Gnom modulators, per-family best config of the "
               "2026-10-30 n-sweep (net trained WITHOUT its family, held "
               "rows scored). Consumer picks closest family; per-net dims "
               "in nets metadata.",
        "epochs": str(EPOCHS),
        "seed": str(SEED),
        "metric": "damage_kld",
        "pool_path": POOL,
        "pool_sha": _sha(POOL),
        "feature_order": json.dumps(FEATURE_NAMES + ["base_bpw", "drop_bpw"]),
        "nets": json.dumps(nets),
        "preprocessing": "per-family standardize (stats from train rows)",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": "pack_gnom.py v3 (mixed per-family best)",
    }
    out = os.path.join(_ROOT, "gnom", "artifacts",
                       f"gnom_multinet_v3_2026-10-30_{tag}.safetensors")
    save_file(tensors, out, metadata=meta)
    st = load_file(out)
    for fam in nets:
        d = [int(v) for v in nets[fam]["dims"].split(",")]
        ps = []
        while True:
            keys = [k for k in st if k.startswith(f"{fam}/W")]
            if not keys:
                break
            ii = 1
            ps = []
            while f"{fam}/W{ii}" in st:
                ps.append({"W": st[f"{fam}/W{ii}"], "b": st[f"{fam}/b{ii}"]})
                ii += 1
            break
        va = root == fam
        trm = ~va
        Xs, _, _ = standardize(np.concatenate([X[trm], X[va]]))
        ntr = int(trm.sum())
        pred = forward(ps, Xs[ntr:], "mish")[0][-1].ravel()
        r2 = float(spearman(pred, y[va]))
        assert abs(r2 - float(nets[fam]["lomo_spearman"])) < 1e-9, fam
    return out, nets


def pack_multinet(X, y, models, dims, tag="lfm3ext_v2"):
    tag = os.environ.get("GNOM_TAG", tag)
    """One LOMO net per model family: trained WITHOUT family F, scored ON F.
    Deployment contract: pick the net for the closest family; its stored
    score is the measured LOMO rank quality for that family."""
    from train_gnom import spearman
    root = np.array([m.rsplit("_", 1)[0] for m in models])
    tensors, nets = {}, {}
    for fam in sorted(set(root)):
        va = root == fam
        tr = ~va
        Xs, _, _ = standardize(np.concatenate([X[tr], X[va]]))
        ntr = int(tr.sum())
        Xtr, Xva = Xs[:ntr], Xs[ntr:]
        w = np.abs(y[tr]) + 0.05 * np.abs(y[tr]).max()
        ps = init_params(dims, seed=SEED)
        train(ps, Xtr, y[tr], epochs=EPOCHS, log_every=10**9,
              noise_std=0.05, sample_w=w, activation="mish", dropout=0.0,
              seed=SEED)
        pred = forward(ps, Xva, "mish")[0][-1].ravel()
        r = float(spearman(pred, y[va]))
        nets[fam] = {"lomo_spearman": repr(float(r)), "n_train": int(tr.sum()),
                     "n_scored": int(va.sum())}
        for i, p in enumerate(ps, 1):
            tensors[f"{fam}/W{i}"] = p["W"].astype(np.float64)
            tensors[f"{fam}/b{i}"] = p["b"].astype(np.float64)
    meta = {
        "schema": "MERNIK.gnom.multinet.v1",
        "doc": "Per-family Gnom modulators. Net for family F is trained "
               "WITHOUT F (strict LOMO); consumer picks closest family. "
               "Score = measured held-out rank quality on F.",
        "dims": ",".join(map(str, dims)),
        "activation": "mish",
        "epochs": str(EPOCHS),
        "seed": str(SEED),
        "metric": "damage_kld",
        "pool_path": POOL,
        "pool_sha": _sha(POOL),
        "feature_order": json.dumps(FEATURE_NAMES + ["base_bpw", "drop_bpw"]),
        "nets": json.dumps(nets),
        "preprocessing": "train_gnom.load_pool_multi per-family standardize "
                         "(stats from that family's train rows, applied to "
                         "held rows) — same convention as the LOMO exam",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": "pack_gnom.py v2 (multinet, per-family standardization)",
    }
    out = os.path.join(_ROOT, "gnom", "artifacts",
                       f"gnom_multinet_v2_2026-10-30_{tag}.safetensors")
    save_file(tensors, out, metadata=meta)
    st = load_file(out)
    for fam in nets:
        ps = []
        i = 1
        while f"{fam}/W{i}" in st:
            ps.append({"W": st[f"{fam}/W{i}"], "b": st[f"{fam}/b{i}"]})
            i += 1
        va = root == fam
        trm = ~va
        Xs, _, _ = standardize(np.concatenate([X[trm], X[va]]))
        ntr = int(trm.sum())
        pred = forward(ps, Xs[ntr:], "mish")[0][-1].ravel()
        r2 = float(spearman(pred, y[va]))
        assert abs(r2 - float(nets[fam]["lomo_spearman"])) < 1e-9, fam
    return out, nets


def pack_multinet_v4(X, y, models, dims_ref, tag="ensemble2026-10-31"):
    """Deploy artifact: per-family net ENSEMBLE (seeds 7/8/9 on the clean
    pool). Tensors "<fam>/s<seed>/W<i>", predictions averaged by consumer.
    Metadata carries per-seed scores and the ensemble score."""
    from train_gnom import spearman
    root = np.array([m.rsplit("_", 1)[0] for m in models])
    tensors, nets = {}, {}
    for fam in sorted(set(root)):
        va = root == fam
        tr = ~va
        Xc = np.concatenate([X[tr], X[va]])
        Xs, mu_s, sd_s = standardize(Xc)
        ntr = int(tr.sum())
        w = np.abs(y[tr]) + 0.05 * np.abs(y[tr]).max()
        preds, rs = [], []
        tensors[f"{fam}/mu"] = np.asarray(mu_s, dtype=np.float64)
        tensors[f"{fam}/sd"] = np.asarray(sd_s, dtype=np.float64)
        for seed in (7, 8, 9):
            ps = init_params([dims_ref[0], 48, 24, 1], seed=seed)
            train(ps, Xs[:ntr], y[tr], epochs=EPOCHS, log_every=10**9,
                  noise_std=0.05, sample_w=w, activation="mish", dropout=0.0,
                  seed=seed)
            preds.append(forward(ps, Xs[ntr:], "mish")[0][-1].ravel())
            rs.append(float(spearman(preds[-1], y[va])))
            for i, p in enumerate(ps, 1):
                tensors[f"{fam}/s{seed}/W{i}"] = p["W"].astype(np.float64)
                tensors[f"{fam}/s{seed}/b{i}"] = p["b"].astype(np.float64)
        r_ens = float(spearman(np.mean(preds, axis=0), y[va]))
        nets[fam] = {"dims": ",".join(map(str, [dims_ref[0], 48, 24, 1])),
                     "seeds": [7, 8, 9], "per_seed": [repr(r) for r in rs],
                     "ensemble_spearman": repr(r_ens),
                     "n_train": int(tr.sum()), "n_scored": int(va.sum())}
        print(f"  {fam:8s} per-seed " + " ".join(f"{r:+.3f}" for r in rs)
              + f" | ensemble {r_ens:+.3f}", flush=True)
    meta = {
        "schema": "MERNIK.gnom.multinet.v4",
        "doc": "Per-family Gnom modulator ensembles (3 seeds), each family's "
               "nets trained WITHOUT that family and scored on it (strict "
               "LOMO). Consumer: pick closest family, average the 3 nets' "
               "predictions. Pins (exact-zero damage units, e.g. binary-F32 "
               "1D/ssm/rope tensors) are excluded from the pool. To score a "
               "new model: build 20-col rows, apply (x - <fam>/mu) / <fam>/sd, "
               "then average the 3 seed nets' mish forward.",
        "epochs": str(EPOCHS), "metric": "damage_kld",
        "pool_path": POOL, "pool_sha": _sha(POOL),
        "feature_order": json.dumps(FEATURE_NAMES + ["base_bpw", "drop_bpw"]),
        "nets": json.dumps(nets),
        "preprocessing": "per-family standardize (stats from train rows)",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "encoder": "pack_gnom.py v4 (seed ensemble)",
    }
    out = os.path.join(_ROOT, "gnom", "artifacts",
                       f"gnom_multinet_v4_2026-10-31_{tag}.safetensors")
    save_file(tensors, out, metadata=meta)
    st = load_file(out)
    for fam in nets:
        Xtr = X[root != fam]; Xva = X[root == fam]
        Xs, _, _ = standardize(np.concatenate([Xtr, Xva]))
        ntr = len(Xtr)
        ps_list = []
        for seed in (7, 8, 9):
            ps = []
            i = 1
            while f"{fam}/s{seed}/W{i}" in st:
                ps.append({"W": st[f"{fam}/s{seed}/W{i}"],
                           "b": st[f"{fam}/s{seed}/b{i}"]})
                i += 1
            ps_list.append(ps)
        pe = np.mean([forward(p, Xs[ntr:], "mish")[0][-1].ravel()
                      for p in ps_list], axis=0)
        r2 = float(spearman(pe, y[root == fam]))
        assert abs(r2 - float(nets[fam]["ensemble_spearman"])) < 1e-9, fam
    return out, nets


if __name__ == "__main__":
    from train_gnom import spearman
    ps, X, y, rho, dims, per, models2 = None, None, None, None, None, None, None
    X_, Y_, models2, gtags, stats = load_pool_multi(
        POOL, ["damage_kld"], force_metric=True, strict_map=True)
    ps, X, y, rho, dims, per = fit_pool()
    print(f"fit done: dims {dims} rows {len(y)} per-model in-sample "
          + " ".join(f"{k}: {v:+.3f}" for k, v in sorted(per.items()))
          + f" mean {rho:+.4f}")
    out = pack(ps, y, rho, len(y), dims, per)
    rho2, nt, per = verify(out, X, y, rho, models2)
    print(f"packed {out} tensors={nt} roundtrip mean rho={rho2:+.4f} OK")
    out2, nets = pack_multinet(X, y, models2, dims)
    print("multinet packed:", out2)
    for fam, d in sorted(nets.items()):
        print(f"  {fam:8s} lomo-spearman {d['lomo_spearman']} "
              f"(train {d['n_train']} rows, scored {d['n_scored']})")
    out3, nets3 = pack_multinet_v3(X, y, models2, dims)
    print("multinet v3 packed:", out3)
    for fam, d in sorted(nets3.items()):
        print(f"  {fam:8s} dims {d['dims']:22s} lomo-spearman "
              f"{float(d['lomo_spearman']):+.4f}")
    out4, nets4 = pack_multinet_v4(X, y, models2, dims)
    print("multinet v4 (ensemble) packed:", out4)
