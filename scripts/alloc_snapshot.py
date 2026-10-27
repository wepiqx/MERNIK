#!/usr/bin/env python3
"""Record greedy allocation distributions for the water-filling baseline.

One dry-run per (model, size, utility). Each run writes a JSON dump with the
final per-tensor map, planned tier histogram, and the real dry-run geometry
(llama-quantize --dry-run), plus the full log next to it.

    python scripts/alloc_snapshot.py                 # default model matrix
    python scripts/alloc_snapshot.py --only ling --force
    python scripts/alloc_snapshot.py --out DIR --matrix FILE
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = "/mnt/Vsio/assets/models"

DEFAULT_MATRIX = [
    {
        "tag": "ling",
        "model": f"{ASSETS}/Ling-3.0-tiny-bf16.gguf",
        "imatrix": "/mnt/Vsio/shq-program/models/Ling-3.0-tiny-imatrix.gguf",
        "netpred": "models/netpred_ling_groups.json",
        "sizes": [4500, 5100, 6500, 9000],
        "utilities": ["smse", "gnom"],
    },
    {
        "tag": "k2",
        "model": f"{ASSETS}/K2-Horizon-7B-bf16.gguf",
        "imatrix": f"{ASSETS}/K2-Horizon-7B-imatrix.gguf",
        "sizes": [5200, 5500, 6500],
        "utilities": ["smse"],
    },
    {
        "tag": "frognano",
        "model": f"{ASSETS}/FrogNano-4B-2609-bf16.gguf",
        "imatrix": f"{ASSETS}/FrogNano-4B-2609-imatrix.gguf",
        "sizes": [2600, 3000, 3500],
        "utilities": ["smse"],
    },
    {
        "tag": "mellum",
        "model": f"{ASSETS}/Mellum2.1-12B-A2.5B-Thinking-BF16.gguf",
        "imatrix": f"{ASSETS}/Mellum2.1-12B-A2.5B-Thinking-imatrix.gguf",
        "sizes": [7600, 8500, 10000],
        "utilities": ["smse"],
    },
    {
        "tag": "mimo",
        "model": f"{ASSETS}/MiMo-V2.6-Distill-Qwen-9B-bf16.gguf",
        "imatrix": f"{ASSETS}/MiMo-V2.6-Distill-Qwen-9B-imatrix.gguf",
        "sizes": [5200, 6500, 8350],
        "utilities": ["smse"],
    },
    {
        "tag": "minicpm",
        "model": f"{ASSETS}/MiniCPM5-2B-BF16.gguf",
        "imatrix": f"{ASSETS}/MiniCPM5-2B.imatrix.gguf",
        "sizes": [1500, 1700, 2300],
        "utilities": ["smse"],
    },
    {
        "tag": "ornith",
        "model": f"{ASSETS}/Ornith-1.5-9B-BF16.gguf",
        "imatrix": f"{ASSETS}/Ornith-1.5-Paretrix-imatrix.gguf",
        "netpred": "models/netpred_ornith.json",
        "sizes": [5300, 6500, 8350],
        "utilities": ["smse", "gnom"],
    },
    {
        "tag": "oxcoder",
        "model": f"{ASSETS}/OxCoder-9B.BF16.gguf",
        "imatrix": f"{ASSETS}/OxCoder-9B.imatrix.gguf",
        "sizes": [5200, 6500, 8350],
        "utilities": ["smse"],
    },
    {
        "tag": "spark4b",
        "model": f"{ASSETS}/Spark-X2.5-4B.gguf",
        "imatrix": f"{ASSETS}/Spark-X2.5-4B-imatrix.gguf",
        "sizes": [2300, 3000, 4050],
        "utilities": ["smse"],
    },
    {
        "tag": "spark17b",
        "model": f"{ASSETS}/Spark-X2.5-1.7B-BF16.gguf",
        "imatrix": f"{ASSETS}/Spark-X2.5-1.7B-imatrix.gguf",
        "sizes": [1000, 1400, 1700],
        "utilities": ["smse"],
    },
]


def _load_matrix(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("matrix must be a JSON list")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "provenance", "alloc_baselines"))
    ap.add_argument("--matrix", default=None, help="JSON matrix override")
    ap.add_argument("--only", action="append", default=[], metavar="TAG")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    matrix = _load_matrix(args.matrix) if args.matrix else DEFAULT_MATRIX
    if args.only:
        matrix = [m for m in matrix if m["tag"] in set(args.only)]
    os.makedirs(args.out, exist_ok=True)

    index_path = os.path.join(args.out, "index.json")
    index = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "entries": []}
    if os.path.exists(index_path) and not args.force:
        try:
            with open(index_path) as f:
                index = json.load(f)
        except Exception:
            pass
    done = {(e.get("tag"), e.get("utility"), e.get("size")) for e in index["entries"]}

    for entry in matrix:
        tag = entry["tag"]
        model = entry["model"]
        imatrix = entry["imatrix"]
        netpred = entry.get("netpred")
        if not os.path.exists(model):
            print(f"SKIP {tag}: model missing {model}")
            continue
        if not os.path.exists(imatrix):
            print(f"SKIP {tag}: imatrix missing {imatrix}")
            continue
        for utility in entry["utilities"]:
            if utility == "gnom" and not netpred:
                print(f"SKIP {tag}/{utility}: no netpred")
                continue
            for size in entry["sizes"]:
                key = (tag, utility, size)
                stem = f"{tag}_{utility}_td_{size}"
                out_json = os.path.join(args.out, stem + ".json")
                out_log = os.path.join(args.out, stem + ".log")
                if key in done and os.path.exists(out_json) and not args.force:
                    print(f"SKIP {stem}: exists")
                    continue
                cmd = [sys.executable, os.path.join(ROOT, "main.py"),
                       "--model", model, "--imatrix", imatrix,
                       "--size", str(size), "--utility", utility,
                       "--top-down", "--dump-alloc", out_json]
                if netpred and utility == "gnom":
                    cmd += ["--netpred", os.path.join(ROOT, netpred)]
                if args.dry_run:
                    print("RUN", " ".join(cmd))
                    continue
                print(f"RUN {stem}")
                t0 = time.time()
                with open(out_log, "w") as lf:
                    rc = subprocess.run(cmd, cwd=ROOT, stdout=lf,
                                        stderr=subprocess.STDOUT).returncode
                elapsed = time.time() - t0
                rec = {"tag": tag, "utility": utility, "size": size,
                       "top_down": True, "json": out_json, "log": out_log,
                       "returncode": rc, "elapsed_s": round(elapsed, 2)}
                index["entries"] = [e for e in index["entries"]
                                    if (e.get("tag"), e.get("utility"), e.get("size")) != key]
                index["entries"].append(rec)
                with open(index_path, "w") as f:
                    json.dump(index, f, indent=2, sort_keys=True)
                print(f"  rc={rc} {elapsed:.1f}s")

    if not args.dry_run:
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
        bad = sum(1 for e in index["entries"] if e.get("returncode") != 0)
        print(f"\nindex: {index_path}  entries={len(index['entries'])} failed={bad}")


if __name__ == "__main__":
    main()
