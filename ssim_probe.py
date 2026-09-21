"""SSIM structural probe (battlefield zoo).

Fake-quantizes F16 weights per tier emulation and measures block-SSIM
vs the original — a *measured* per-(tensor, tier) structural signal,
decomposed into luminance / contrast / structure.

Emulation (affine, per-block): Q8_0: 8bit/32, Q6_K: 6bit/256,
Q5_K: 5bit/256, Q4_K: 4bit/256. Close enough to rank structural damage.

Usage:
    python ssim_probe.py --model <F16.gguf> --out models/ssim_table.npz
"""
import argparse
import numpy as np
import gguf


def fake_quant(w: np.ndarray, bits: int, block: int) -> np.ndarray:
    f = w.astype(np.float32).ravel()
    n = (f.size // block) * block
    f, tail = f[:n].reshape(-1, block), f[n:]
    lo, hi = f.min(axis=1, keepdims=True), f.max(axis=1, keepdims=True)
    scale = (hi - lo) / (2 ** bits - 1)
    scale[scale == 0] = 1.0
    q = np.round((f - lo) / scale) * scale + lo
    return np.concatenate([q.ravel(), tail])


def block_ssim(a: np.ndarray, b: np.ndarray, block: int = 256):
    """Mean SSIM over blocks + global luminance/contrast/structure."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    n = (a.size // block) * block
    if n == 0:
        return 1.0, 1.0, 1.0, 1.0
    A, B = a[:n].reshape(-1, block), b[:n].reshape(-1, block)
    dr = max(a.max() - a.min(), b.max() - b.min(), 1e-12)
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mu_a, mu_b = A.mean(axis=1), B.mean(axis=1)
    va, vb = A.var(axis=1), B.var(axis=1)
    cab = ((A - mu_a[:, None]) * (B - mu_b[:, None])).mean(axis=1)
    lum = (2 * mu_a * mu_b + C1) / (mu_a**2 + mu_b**2 + C1)
    con = (2 * np.sqrt(va * vb) + C2) / (va + vb + C2)
    str_ = (cab + C2 / 2) / (np.sqrt(va * vb) + C2 / 2)
    ssim = lum * con * str_
    return float(ssim.mean()), float(lum.mean()), float(con.mean()), float(str_.mean())


TIERS = {"Q4_K": (4, 256), "Q5_K": (5, 256), "Q6_K": (6, 256), "Q8_0": (8, 32)}


def _perceptual_row(t, w: np.ndarray, fq: dict, energy: np.ndarray,
                      colblock: int) -> dict:
    """Activation-weighted structural damage per tier.

    Columns are grouped in blocks; each block's SSIM damage (1-SSIM vs the
    fake-quantized tier) is weighted by the block's mean input energy
    (in_sum2). Damage in high-traffic columns counts more — the network's
    own notion of perceptual importance.
    """
    W = w.reshape(t.data.shape)
    in_dim = W.shape[-1]
    if energy.shape[0] != in_dim:  # shape mismatch: fall back to unweighted
        return {tier: 1.0 - block_ssim(w, q)[0] for tier, q in fq.items()}
    nb = max(1, in_dim // colblock)
    cols = np.array_split(np.arange(in_dim), nb)
    eblk = np.array([energy[c].mean() for c in cols])
    wsum = eblk.sum()
    if wsum <= 0:
        return {tier: 1.0 - block_ssim(w, q)[0] for tier, q in fq.items()}
    out = {}
    for tier, q in fq.items():
        Q = q.reshape(t.data.shape)
        dmg = 0.0
        for c, e in zip(cols, eblk):
            s, _, _, _ = block_ssim(W[..., c].ravel(), Q[..., c].ravel())
            dmg += e * (1.0 - s)
        out[tier] = dmg / wsum
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imatrix", default=None,
                    help="optional: also emit perceptual (activation-weighted) damage table")
    ap.add_argument("--pout", default=None)
    ap.add_argument("--colblock", type=int, default=64)
    args = ap.parse_args()

    im_raw = {}
    if args.imatrix:
        import sys
        sys.path.insert(0, ".")
        from imatrix_reader import read_imatrix
        im = read_imatrix(args.imatrix)
        for name, d in im["tensors"].items():
            v = d.get("in_sum2_raw")
            if v is not None:
                im_raw[name] = np.asarray(v, dtype=np.float64)

    r = gguf.GGUFReader(args.model)
    table, layers, prow = {}, {}, {}
    for t in r.tensors:
        name = t.name
        if t.tensor_type != gguf.GGMLQuantizationType.F16:
            continue
        if "norm" in name or name in ("output.weight", "token_embd.weight"):
            continue  # pinned anyway / no imatrix
        w = np.asarray(t.data, dtype=np.float32)
        row = {}
        fq = {}
        for tier, (bits, blk) in TIERS.items():
            q = fake_quant(w, bits, blk)
            fq[tier] = q
            s, l, c, s_ = block_ssim(w, q)
            row[tier] = (s, l, c, s_)
        table[name] = row
        if name in im_raw and args.pout:
            prow[name] = _perceptual_row(t, w, fq, im_raw[name], args.colblock)
        parts = name.split(".")
        if parts[0] == "blk":
            layers.setdefault(int(parts[1]), []).append((name, row["Q4_K"][0]))

    np.savez(args.out, **{k: np.array([v[t] for t in TIERS], dtype=np.float64)
                          for k, v in table.items()})
    print(f"tensors: {len(table)} -> {args.out}")
    if args.pout and prow:
        np.savez(args.pout, **{k: np.array([v[t] for t in TIERS], dtype=np.float64)
                               for k, v in prow.items()})
        print(f"perceptual tensors: {len(prow)} -> {args.pout}")

    print("\nLayer structural fragility @Q4 (1-SSIM, mean over tensors):")
    for L in sorted(layers):
        frag = np.mean([1.0 - s for _, s in layers[L]])
        print(f"  blk.{L:2d}  frag={frag:.4f}")


if __name__ == "__main__":
    main()
