#!/usr/bin/env python3
"""twin.py — stolen from Paretrix (R19 stock-twin detection), MERNIK-flavored.

Before spending GPU on a battery: is this exact file (by sha256) already
measured under some tag? Twins are byte-identical files, not same names.

    python scripts/twin.py --model /path/to/MODEL.gguf [--json]

Exit 0 + prints TWIN tags (with their HE counts) or "no twin".
"""
import argparse
import glob
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sha256(path, _bytes=1 << 26):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    want = sha256(args.model)
    hits = []
    for mp in glob.glob(os.path.join(REPO, "eval_results", "*.meta.json")):
        try:
            meta = json.load(open(mp))
        except Exception:
            continue
        if meta.get("serve_sha256") == want:
            tag = os.path.basename(mp)[:-len(".meta.json")]
            he = None
            rp = os.path.join(REPO, "eval_results", tag + ".jsonl_results.jsonl")
            try:
                rows = [json.loads(l) for l in open(rp)]
                he = sum(1 for r in rows if r.get("passed"))
                he = "%d/%d" % (he, len(rows))
            except Exception:
                pass
            hits.append({"tag": tag, "HE": he,
                         "finished": meta.get("finished"),
                         "temp": (meta.get("sample") or {}).get("temperature")})
    if args.json:
        print(json.dumps({"sha256": want, "twins": hits}, indent=1))
    elif hits:
        print("TWIN of %s:" % args.model)
        for h in hits:
            print("  %s  HE %s  temp %s  (%s)" % (
                h["tag"], h["HE"], h["temp"], h["finished"]))
    else:
        print("no twin: safe to measure")
    return 0


if __name__ == "__main__":
    main()
