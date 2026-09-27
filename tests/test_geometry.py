#!/usr/bin/env python3
"""Golden geometry — regression net for the allocator.

Two agents edit this tree (see AGENTS.md). Any change to classifier.py /
constants.py / config_generator.py moves the tier map, and until today
nothing noticed: the 1D/F32 fix silently reshaped every distribution in the
lab, and the only reason it was caught at all was the pre-flight audit I
happened to add. A golden fixture is cheaper than that argument.

Golden values were recorded from the artefacts on 2026-09-28, AFTER the
1D/F32 law landed, and the ZOO row is the stand the README quotes. If a row
fails, either the allocator changed (say so in the commit and re-record
deliberately) or something broke (that is the bug).

Models are gitignored and move around, so a missing model is a skip, never a
failure. No GPU, no network, no writes outside stdout.

    python tests/test_geometry.py
"""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.simplefilter("ignore")

from model_reader import read_model
from imatrix_reader import (read_imatrix, combine_imatrix, detect_tied_groups,
                            build_importance_table)
from classifier import optimal_classify, optimal_classify_topdown, compute_stats

DL = "/mnt/Vsio/Downloads"
PROG = "/mnt/Vsio/shq-program/models"

# (label, model, imatrix, size_mib, utility, top_down, pin_norms, allow_q3, golden)
CASES = [
    ("zoo-stand 1.7B @1000 mse  (README-ZOO baseline)",
     f"{DL}/Spark-X2.5-1.7B-BF16.gguf", f"{PROG}/Spark-X2.5-1.7B-imatrix.gguf",
     1000, "mse", False, False, False,
     {"F32": 57, "Q4_K": 105, "Q5_K": 8, "Q6_K": 56}),

    ("MiniCPM5-2B @1700 mse    (bottom-up default)",
     f"{DL}/MiniCPM5-2B-BF16.gguf", f"{DL}/MiniCPM5-2B.imatrix.gguf",
     1700, "mse", False, False, False,
     {"F32": 85, "IQ4_XS": 40, "Q4_K": 19, "Q5_K": 71, "Q6_K": 45, "Q8_0": 121}),

    ("MiniCPM5-2B @1700 smse   (blend lens)",
     f"{DL}/MiniCPM5-2B-BF16.gguf", f"{DL}/MiniCPM5-2B.imatrix.gguf",
     1700, "smse", False, False, False,
     {"F32": 85, "IQ4_XS": 40, "Q4_K": 19, "Q5_K": 71, "Q6_K": 45, "Q8_0": 121}),

    ("MiniCPM5-2B @1400 top-down",
     f"{DL}/MiniCPM5-2B-BF16.gguf", f"{DL}/MiniCPM5-2B.imatrix.gguf",
     1400, "mse", True, False, False,
     {"F32": 85, "IQ4_XS": 39, "Q4_K": 255, "Q5_K": 2}),

    ("MiniCPM5-2B @1200 smse allow-q3",
     f"{DL}/MiniCPM5-2B-BF16.gguf", f"{DL}/MiniCPM5-2B.imatrix.gguf",
     1200, "smse", False, False, True, None),   # None = recorded on first run

    ("RINIQ-M2 @5100 mse       (qwen35 hybrid + MTP)",
     f"{DL}/RINIQ-M2-BF16.gguf", f"{DL}/RINIQ-M2-qwen38.imatrix.gguf",
     5100, "mse", False, False, False,
     {"F32": 177, "Q4_K": 216, "Q5_K": 34}),
]

GOLDEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "geometry_golden.json")


def load_golden():
    if os.path.exists(GOLDEN_FILE):
        import json
        return json.load(open(GOLDEN_FILE))
    return {}


def save_golden(g):
    import json
    with open(GOLDEN_FILE, "w") as f:
        json.dump(g, f, indent=2, sort_keys=True)


def geometry(case):
    (_label, model, imat, size, util, td, pin, q3, _gold) = case
    m = read_model(model)
    im = combine_imatrix([read_imatrix(imat)])
    tg = detect_tied_groups(im)
    it = build_importance_table(im, m)
    uopt = {"mode": util}
    if td:
        a, _ = optimal_classify_topdown(it, tg, m, target_size_mib=size,
                                        uopt=uopt, pin_norms=pin)
    else:
        a, _ = optimal_classify(it, tg, m, target_size_mib=size, allow_q3=q3,
                                uopt=uopt)
    ne = {k: v["n_elements"] for k, v in m["tensors"].items()}
    from classifier import _f32_map
    st = compute_stats(a, ne, None, _f32_map(m["tensors"]))
    return {k: v for k, v in st["by_tier_count"].items() if v}, st["total_mib"]


def main():
    golden = load_golden()
    failures = skipped = 0
    for case in CASES:
        label = case[0]
        if not (os.path.exists(case[1]) and os.path.exists(case[2])):
            print(f"SKIP  {label}  (model or imatrix not on this box)")
            skipped += 1
            continue
        got, total = geometry(case)
        key = label
        if key not in golden:
            golden[key] = {"tiers": got, "total_mib": round(total, 1)}
            save_golden(golden)
            print(f"REC   {label}  -> {got} ({total:.1f} MiB)")
            continue
        want = golden[key]["tiers"]
        if got == want:
            print(f"ok    {label}  ({total:.1f} MiB)")
        else:
            failures += 1
            print(f"FAIL  {label}")
            print(f"        want {want}")
            print(f"        got  {got}")
            for t in sorted(set(want) | set(got)):
                if want.get(t) != got.get(t):
                    print(f"        {t}: {want.get(t, 0)} -> {got.get(t, 0)}")
    print(f"\n{len(CASES) - skipped - failures} ok, {failures} failed, "
          f"{skipped} skipped")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
