#!/usr/bin/env python3
"""lab_index.py — persistent full-text index of the lab (operator's idea:
ask anything, get it). One build (~30-60 min CPU), then lab_ask reads it
in seconds. GGUFs/binaries excluded — text only.
"""
import glob
import json
import os
import urllib.request

import numpy as np

EMB_URL = "http://127.0.0.1:28083/embedding"
DIM_TRUNC = 256
OUT = "/mnt/Vsio/MERNIK/provenance/lab_index.npz"

SKIP_EXT = (".gguf", ".dat", ".npz", ".bin", ".so", ".o", ".a", ".png",
            ".jpg", ".jpeg", ".ttf", ".whl", ".xz", ".zst", ".pyc")
SKIP_DIR = ("node_modules", "__pycache__", ".git/", "build/", ".cache")


def want(path):
    if path.endswith(SKIP_EXT):
        return False
    if any(s in path for s in SKIP_DIR):
        return False
    try:
        if os.path.getsize(path) > 8_000_000:
            return False
    except OSError:
        return False
    return True


def collect():
    roots = ["/mnt/Vsio/chain-*.log", "/mnt/Vsio/*.log",
             "/mnt/Vsio/MERNIK/*.md", "/mnt/Vsio/MERNIK/gnom/*.md",
             "/mnt/Vsio/MERNIK/scripts/*.py", "/mnt/Vsio/MERNIK/*.py",
             "/mnt/Vsio/hfwork/*.md",
             "/mnt/Vsio/MERNIK/eval_results/*.meta.json",
             "/mnt/Vsio/opencode-backup/*.sh"]
    files = []
    for pat in roots:
        files += [f for f in glob.glob(pat) if want(f)]
    return sorted(set(files))


def chunk(text, name):
    out = []
    for i in range(0, len(text), 1200):
        c = text[i:i + 1200].strip()
        if len(c) > 100:
            out.append((name, c))
    return out


def embed_batch(texts):
    vecs = []
    for t in texts:
        req = json.dumps({"content": t[:450]}).encode()
        try:
            r = urllib.request.urlopen(
                urllib.request.Request(EMB_URL, data=req, headers={
                    "Content-Type": "application/json"}), timeout=120)
            raw = json.loads(r.read())
            if isinstance(raw, list):
                raw = raw[0].get("embedding", raw[0])
            v = np.array(raw, dtype=np.float64).ravel()[:DIM_TRUNC]
            vecs.append(v / (np.linalg.norm(v) + 1e-12))
        except Exception as e:
            print("embed fail:", str(e)[:60], flush=True)
            vecs.append(np.zeros(DIM_TRUNC))
    return np.array(vecs)


def main():
    files = collect()
    print("files:", len(files), flush=True)
    chunks = []
    for f in files:
        try:
            with open(f, errors="replace") as fh:
                chunks += chunk(fh.read(), f)
        except Exception:
            pass
    print("chunks:", len(chunks), flush=True)
    B = 8
    E = []
    for i in range(0, len(chunks), B):
        E.append(embed_batch(["title: none | text: " + c
                              for _, c in chunks[i:i + B]]))
        if i % 80 == 0:
            print("  %d/%d" % (i, len(chunks)), flush=True)
    E = np.vstack(E)
    names = np.array([n for n, _ in chunks])
    texts = np.array([c for _, c in chunks])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, E=E, names=names, texts=texts)
    print("saved %s (%d vectors)" % (OUT, len(E)), flush=True)


if __name__ == "__main__":
    main()
