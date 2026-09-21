#!/usr/bin/env python3
"""Build top-down with norms pinned at F16 (the norms-F16 probe).

Usage: build_normstd.py --utility MODE --size MIB --output FILE
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from classifier import optimal_classify_topdown, compute_stats
from config_generator import generate_flags, format_flags
from quantizer import run_dry_run, run_quantization

M = "/mnt/Vsio/Downloads/NeoHorse-1-9B-BF16.gguf"
I = "/mnt/Vsio/Downloads/NeoHorse-1-9B.imatrix.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utility", default="mse")
    ap.add_argument("--size", type=float, default=6500)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = read_model(M)
    im = read_imatrix(I)
    tg = detect_tied_groups(im)
    imp = build_importance_table(im, model)
    a, pad = optimal_classify_topdown(imp, tg, model, args.size,
                                      uopt={"mode": args.utility})
    ne = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}
    n = 0
    for t in a:
        if "norm" in t or ne.get(t, 10 ** 9) < 100000:
            if a[t] != "F16":
                n += 1
            a[t] = "F16"
    st = compute_stats(a, ne, pad)
    print("norms rescued: %d total=%.1f (target %.0f)" % (n, st["total_mib"], args.size))
    flags = generate_flags(a, model, "Q5_K_M", args.size)
    flags["imatrix"] = I
    print(format_flags(flags))
    dry = run_dry_run(flags, M)
    print("dry-run: %.0f" % (dry or -1))
    ok = run_quantization(flags, M, args.output)
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
