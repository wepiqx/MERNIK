#!/usr/bin/env python3
"""smooth_probe: SmoothQuant-style per-channel smoothing from in_sum2.

Outlier channels eat the Q4 range; smoothing migrates magnitude from
activations into weights (W' = W*s, activations /s folded into the
preceding norm at quant time). in_sum2 from the imatrix IS the per-channel
activation energy — no extra calibration runs.

v1 probe (CPU only, no GPU, no quant binary): fake-Q4 SSIM damage of W vs
smoothed W'. Necessary (not sufficient) condition — real duel decides.
Sweep --alpha 0.25/0.5/0.75.

Usage:
  python smooth_probe.py --model Spark-1.7B-F16.gguf \\
      --imatrix Spark-1.7B-imatrix.gguf [--alpha 0.5]
"""
import argparse
import sys
import os
import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ssim_probe import fake_quant, block_ssim

CANDIDATES = ("ffn_down", "ffn_gate", "ffn_up", "attn_output", "attn_qkv",
              "attn_q", "attn_k", "attn_v")


def channel_scales(W: np.ndarray, energy: np.ndarray, alpha: float):
    """Per-input-channel smoothing scales, geometric-mean normalized."""
    in_dim = W.shape[-1]
    rms_act = np.sqrt(np.maximum(energy[:in_dim], 1e-12))
    rms_w = np.sqrt(np.maximum((W.astype(np.float64) ** 2).mean(axis=0),
                               1e-24))
    s = (rms_act ** alpha) / (rms_w ** (1.0 - alpha) + 1e-24)
    s = s / (np.exp(np.log(np.maximum(s, 1e-24)).mean()) + 1e-24)
    return s.astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--alpha", type=float, nargs="+",
                    default=[0.25, 0.5, 0.75])
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from imatrix_reader import read_imatrix
    im = read_imatrix(args.imatrix)
    im_raw = {n: np.asarray(d["in_sum2_raw"], dtype=np.float64)
              for n, d in im["tensors"].items()
              if d.get("in_sum2_raw") is not None}

    r = gguf.GGUFReader(args.model)
    for alpha in args.alpha:
        print(f"=== alpha={alpha} ===")
        tot0 = tot1 = 0.0
        n = 0
        per_layer = {}
        for t in r.tensors:
            name = t.name
            if t.tensor_type != gguf.GGMLQuantizationType.F16:
                continue
            if not any(c in name for c in CANDIDATES):
                continue
            if name not in im_raw:
                continue
            W = np.asarray(t.data, dtype=np.float32)
            e = im_raw[name]
            if e.shape[0] < W.shape[-1]:
                continue
            w = W.astype(np.float64).ravel()
            d0 = 1.0 - block_ssim(w, fake_quant(w, 4, 256))[0]
            s = channel_scales(W.astype(np.float64), e, alpha)
            ws = (W.astype(np.float64) * s).ravel()
            d1 = 1.0 - block_ssim(ws, fake_quant(ws, 4, 256))[0]
            tot0 += d0
            tot1 += d1
            n += 1
            if name.startswith("blk."):
                L = int(name.split(".")[1])
                a, b = per_layer.get(L, (0.0, 0.0))
                per_layer[L] = (a + d0, b + d1)
        print(f"  tensors: {n}  Q4-damage raw={tot0 / max(n, 1):.4f} "
              f"smoothed={tot1 / max(n, 1):.4f} "
              f"delta={100 * (tot1 - tot0) / max(tot0, 1e-12):+.1f}%")
        for L in sorted(per_layer):
            a, b = per_layer[L]
            print(f"    blk.{L:2d} raw={a:.3f} sm={b:.3f}")


if __name__ == "__main__":
    main()
