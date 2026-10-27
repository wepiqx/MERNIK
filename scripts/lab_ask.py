#!/usr/bin/env python3
"""lab_ask.py — RAG over the lab (operator's idea).

Corpus: chain-log tails + AGENTS.md (STATE + mailbox) + ledger standings
+ one LIVE status snapshot (GPU/procs/disk). Embeddings via local
llama-server (embeddinggemma-2 Q8, CPU, :28083). Cosine in numpy.
Reader: the agent (me) — this tool prints the top chunks + live status.

    python scripts/lab_ask.py "what is on the GPU now" [--k 6]
"""
import glob
import json
import os
import subprocess
import sys
import urllib.request

import numpy as np

EMB_URL = "http://127.0.0.1:28083/embedding"
DIM_TRUNC = 256  # Matryoshka: plenty for lab notes
EMB_MODEL = "/mnt/Vsio/Downloads/embed/embeddinggemma-2-Q8_0.gguf"
EMB_BIN = "/home/wepiqx/llama.cpp/build/bin/llama-server"


def ensure_server():
    """On-demand embeddings: server rises per question, idles otherwise.
    Survives reboots with zero manual steps (and zero idle RAM)."""
    import time
    try:
        urllib.request.urlopen("http://127.0.0.1:28083/health",
                               timeout=5).read()
        return
    except Exception:
        pass
    subprocess.Popen(
        [EMB_BIN, "-m", EMB_MODEL, "--port", "28083", "-ngl", "0",
         "--embeddings"],
        stdout=open("/tmp/opencode/embed.log", "a"),
        stderr=subprocess.STDOUT, start_new_session=True,
        stdin=subprocess.DEVNULL)
    for _ in range(24):
        time.sleep(5)
        try:
            urllib.request.urlopen("http://127.0.0.1:28083/health",
                                   timeout=5).read()
            return
        except Exception:
            pass
    raise RuntimeError("embedding server did not rise")


def embed(texts, prefix=""):
    out = []
    for t in texts:
        # server batch caps input tokens; Cyrillic is token-dense (~3x),
        # so chunks are small. Task prefixes per the manual (retrieval).
        chunks = [t[i:i + 500] for i in range(0, max(1, len(t)), 500)] or [t]
        vecs = []
        for ch in chunks[:8]:
            req = json.dumps({"content": prefix + ch}).encode()
            r = urllib.request.urlopen(
                urllib.request.Request(EMB_URL, data=req,
                                       headers={"Content-Type": "application/json"}),
                timeout=120)
            raw = json.loads(r.read())
            if isinstance(raw, list):
                raw = raw[0].get("embedding", raw[0])
            elif isinstance(raw, dict):
                raw = raw.get("embedding", raw.get("data", [{}])[0].get("embedding"))
            v = np.array(raw, dtype=np.float64).ravel()
            v = v[:DIM_TRUNC]
            vecs.append(v / (np.linalg.norm(v) + 1e-12))
        m = np.mean(vecs, axis=0)
        out.append(m / (np.linalg.norm(m) + 1e-12))
    return np.array(out)


def live_status():
    parts = []
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        parts.append("GPU now: " + smi.stdout.strip())
    except Exception as e:
        parts.append("GPU now: nvidia-smi failed (%s)" % e)
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=20).stdout
        keys = [l.split()[10:13] for l in ps.splitlines()
                if any(k in l for k in ("llama-", "run_humaneval", "group_damage",
                                        "train_", "chain-", "snapshot_download"))
                and "grep" not in l]
        parts.append("Lab processes now (%d): %s" % (len(keys), keys[:8]))
    except Exception as e:
        parts.append("Procs now: ps failed (%s)" % e)
    try:
        df = subprocess.run(["df", "-h", "/mnt/Vsio", "--output=pcent,avail"],
                            capture_output=True, text=True, timeout=20).stdout
        parts.append("Disk /mnt/Vsio now: " + " ".join(df.split()[2:4]))
    except Exception as e:
        parts.append("Disk now: df failed (%s)" % e)
    # chain tails: what finished / failed lately
    for pat in ("/mnt/Vsio/chain-*.log",):
        for p in sorted(glob.glob(pat))[-14:]:
            try:
                with open(p, errors="replace") as f:
                    lines = f.readlines()
                tail = "".join(lines[-12:]).strip()
                if tail:
                    parts.append("LOG %s tail:\n%s" % (p.split("/")[-1], tail))
            except Exception:
                pass
    return parts


def static_corpus():
    docs = []
    try:
        ag = open("/mnt/Vsio/MERNIK/AGENTS.md", errors="replace").read()
        # STATE block + last 6 letters only (fresh context, small index)
        head, _, rest = ag.partition("## Mailbox")
        docs.append(("AGENTS-STATE", head[-4000:]))
        letters = rest.split("### ")[1:7]
        for i, l in enumerate(letters):
            docs.append(("AGENTS-LETTER-%d" % i, l[:3000]))
    except Exception:
        pass
    return docs


def main():
    ensure_server()
    q = sys.argv[1] if len(sys.argv) > 1 else "what is happening now"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    live = live_status()
    idx_path = "/mnt/Vsio/MERNIK/provenance/lab_index.npz"
    titles, texts, E = [], live, None
    if os.path.exists(idx_path):
        try:
            z = np.load(idx_path, allow_pickle=True)
            E = np.array(z["E"], dtype=np.float64)
            titles = ["LIVE: " + d.split("\n")[0][:60] for d in live] + \
                     [str(n).split("/")[-1] for n in z["names"]]
            texts = live + [str(c) for c in z["texts"]]
        except Exception as e:
            print("(index unreadable, live-only: %s)" % e)
    if E is None:
        docs = static_corpus()
        titles = ["LIVE: " + d.split("\n")[0][:60] for d in live] + \
                 [t for t, _ in docs]
        texts = live + [d for _, d in docs]
        E = embed(texts, prefix="title: none | text: ")
    else:
        # embed live docs with the same prefix as the index
        Elive = embed(live, prefix="title: none | text: ")
        E = np.vstack([Elive, E])
    qv = embed([q], prefix="task: search result | query: ")[0]
    sims = E @ qv
    order = np.argsort(-sims)[:k]
    print("=== LIVE STATUS ===")
    for d in live[:4]:
        print("- " + d.split("\n")[0][:150])
    print("=== TOP-%d for: %s ===" % (k, q))
    for i in order:
        print("\n--- [%s] sim=%.3f ---\n%s" % (titles[i], sims[i], texts[i][:900]))


if __name__ == "__main__":
    main()
