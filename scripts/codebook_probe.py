#!/usr/bin/env python3
"""Codebook probe (operator's new-algorithm idea, stage 1): can a
2-bit outlier-aware codebook beat IQ2_XXS on weight fidelity at the
SAME bitrate? numpy-only, CPU minutes. If yes -> output-MSE stage,
then engine work is justified. If no -> the idea dies cheap.
"""
import subprocess
import numpy as np


def kmeans_w(x, w, k=4, iters=50, seed=7):
    rng = np.random.default_rng(seed)
    qs = np.quantile(x, np.linspace(0.02, 0.98, k))
    c = qs + rng.normal(0, 1e-6, k)
    for _ in range(iters):
        d = np.abs(x[:, None] - c[None, :]) * (1.0 + w[:, None])
        a = d.argmin(axis=1)
        nc = np.array([np.average(x[a == j], weights=w[a == j] + 1e-12)
                       if (a == j).any() else c[j] for j in range(k)])
        if np.allclose(nc, c):
            break
        c = nc
    a = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
    return c[a], c


def run(src, dst, typ):
    r = subprocess.run(
        ["/home/wepiqx/llama.cpp/build/bin/llama-quantize",
         "--tensor-type", ".*=%s" % typ, src, dst, "Q8_0"],
        capture_output=True, text=True)
    return r.returncode == 0


if __name__ == "__main__":
    import sys
    import gguf
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/mnt/Vsio/Downloads/Spark-X2.5-4B.gguf"  # 7.6GB: fits 16GB RAM
    picks = sys.argv[2:] or ["blk.12.ffn_down.weight", "blk.12.attn_output.weight",
                             "blk.0.ffn_down.weight", "blk.20.attn_qkv.weight"]
    r = gguf.GGUFReader(path)
    T = {t.name: t for t in r.tensors}
    for name in picks:
        t = T[name]
        w = np.ascontiguousarray(t.data).astype(np.float64).ravel()
        # candidate: Fisher-|x|-weighted 4-centroid codebook (2 bits) +
        # fp16 scales (negligible) -> ~2.0bpw + grid overhead, vs IQ2_XXS ~2.56
        xw = np.abs(w) / (np.abs(w).mean() + 1e-12)
        q, _ = kmeans_w(w, xw, k=4)
        mse_km = float(np.mean((w - q) ** 2))
        var = float(np.var(w))
        print("%s: var=%.4g kmeans-relMSE=%.5f" % (name, var, mse_km / var),
              flush=True)
    print("NOTE: IQ2_XXS side needs same-tensor extraction; "
          "llama-quantize whole-file compare next.", flush=True)
