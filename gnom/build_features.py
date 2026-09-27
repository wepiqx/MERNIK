#!/usr/bin/env python3
"""Build per-tensor feature matrix for Gnom training (deterministic).

    build_features.py --model M.gguf --imatrix I.gguf --out gnom/features/<m>.npz
                      [--ssim-table models/ssim_table.npz] [--n-layers N]

Output .npz: X (n_tensors x 15, RAW, see gnom/README), names (tensor names),
stats (timp_mean/std, logN_median for per-model standardization at train).
Weight stats read streaming (one tensor at a time, ~GB peak transient).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ml_dtypes  # noqa: F401  (bfloat16 decode)
from gguf import GGUFReader
from model_reader import read_model
from imatrix_reader import read_imatrix, build_importance_table
from tinynet import featurize


def load_ssim(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        z = np.load(path, allow_pickle=False)
        return {n: float(np.ravel(z[n])[0]) for n in z.files}  # Q4_K first
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ssim-table", default=None)
    ap.add_argument("--n-layers", type=int, default=0,
                    help="0 = estimate from tensor names")
    args = ap.parse_args()

    model = read_model(args.model)
    im = read_imatrix(args.imatrix)
    table = build_importance_table(im, model)
    ssim = load_ssim(args.ssim_table)

    reader = GGUFReader(args.model)
    wmap = {t.name: t for t in reader.tensors}

    names = sorted(table.keys())
    if args.n_layers <= 0:
        layers = [int(n.split(".")[1]) for n in names
                  if n.startswith("blk.") and n.split(".")[1].isdigit()]
        n_layers = max(layers) + 1 if layers else 28
    else:
        n_layers = args.n_layers

    X, kept = [], []
    for n in names:
        info = table[n]
        frag = 1.0 - ssim.get(n, 1.0)
        tm0, tx0 = info.get("importance_mean", 0.0), info.get("importance_max", 0.0)
        econc = tx0 / (tm0 + 1e-12)  # peak/mean concentration ratio
        w = None
        if n in wmap:
            try:
                w = np.array(wmap[n].data, dtype=np.float32).ravel()
            except Exception:
                w = None
        row = featurize(n, info.get("n_elements", 0), tm0, tx0,
                        frag=frag, econc=econc, n_layers=n_layers, w=w)
        X.append(row)
        kept.append(n)
        del w
    X = np.array(X, dtype=np.float64)
    tm = np.array([table[n].get("importance_mean", 0.0) for n in kept])
    logn = np.log10(np.array([table[n].get("n_elements", 1)
                              for n in kept]) + 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, X=X, names=np.array(kept),
             timp_mean=np.array([tm.mean()]), timp_std=np.array([tm.std()]),
             logN_med=np.array([np.median(logn)]))
    print(f"{args.out}: {X.shape[0]} tensors x {X.shape[1]}, "
          f"n_layers={n_layers}", flush=True)


if __name__ == "__main__":
    main()
