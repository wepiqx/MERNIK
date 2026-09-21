"""Real-quant SSIM probe (battlefield resuscitation).

SSIM between source F16 and F16-roundtripped real quants (F16->T->F16 via
llama-quantize) — no fake-quant emulation. Plus perceptual damage weighted
by in_sum2 energy from a matching-domain imatrix (wiki, not Qwen).

Usage:
    python real_probe.py --src <F16.gguf> --roundtrips REAL-Q8_0:REAL-Q8_0-F16.gguf,... \\
        --imatrix <wiki.imatrix.gguf> --out models/real_ssim_table.npz \\
        --pout models/real_ssim_ptable.npz
"""
import argparse
import numpy as np
import gguf
from ssim_probe import block_ssim, _perceptual_row, TIERS


def load_f16(path: str) -> dict:
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            out[t.name] = (t.data.shape, np.asarray(t.data, dtype=np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--roundtrips", required=True,
                    help="TIER:path,TIER:path... (F16-roundtripped real quants)")
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pout", default=None)
    ap.add_argument("--colblock", type=int, default=64)
    args = ap.parse_args()

    rt = dict(p.split(":") for p in args.roundtrips.split(","))
    # Streaming: one tensor in RAM at a time (full float32 copies OOM).
    readers = {"SRC": gguf.GGUFReader(args.src)}
    for tier, p in rt.items():
        readers[tier] = gguf.GGUFReader(p)
    by_name = {}
    for key, r in readers.items():
        for t in r.tensors:
            if t.tensor_type == gguf.GGMLQuantizationType.F16:
                by_name.setdefault(t.name, {})[key] = t

    im_raw = {}
    if args.imatrix:
        from imatrix_reader import read_imatrix
        im = read_imatrix(args.imatrix)
        for name, d in im["tensors"].items():
            v = d.get("in_sum2_raw")
            if v is not None:
                im_raw[name] = np.asarray(v, dtype=np.float64)

    table, prow, layers = {}, {}, {}
    for name, variants in by_name.items():
        if "norm" in name or name in ("output.weight", "token_embd.weight"):
            continue
        if "SRC" not in variants:
            continue
        t0 = variants["SRC"]
        shape = t0.data.shape
        w = np.asarray(t0.data, dtype=np.float32).ravel()
        row, fq, prow_row = {}, {}, {}
        for tier in TIERS:
            if tier not in variants:
                continue
            q = np.asarray(variants[tier].data, dtype=np.float32).ravel()
            fq[tier] = q
            s, l, c, s_ = block_ssim(w, q)
            row[tier] = (s, l, c, s_)
        del w
        if not row:
            continue
        table[name] = row
        if name in im_raw and args.pout:
            class _T:  # minimal shim for _perceptual_row
                data = type("D", (), {"shape": shape})()
            prow[name] = _perceptual_row(_T(), np.asarray(
                t0.data, dtype=np.float32).ravel(), fq, im_raw[name],
                args.colblock)
        if name.startswith("blk."):
            layers.setdefault(int(name.split(".")[1]), []).append(
                (name, row.get("Q4_K", (1.0,))[0]))

    np.savez(args.out, **{k: np.array([v[t] for t in TIERS if t in v],
                                      dtype=np.float64)
                          for k, v in table.items()})
    print(f"real tensors: {len(table)} -> {args.out}")
    if args.pout and prow:
        np.savez(args.pout, **{k: np.array([v[t] for t in TIERS if t in v],
                                           dtype=np.float64)
                               for k, v in prow.items()})
        print(f"real perceptual: {len(prow)} -> {args.pout}")
    print("\nReal fragility @Q4:")
    for L in sorted(layers):
        print(f"  blk.{L:2d}  frag={np.mean([1.0 - s for _, s in layers[L]]):.4f}")


if __name__ == "__main__":
    main()
