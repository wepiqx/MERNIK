#!/usr/bin/env python3
"""MC-duel harness: HellaSwag slice scored via llama-server prompt logprobs.

Usage:
  python scripts/mc_duel.py --model A.gguf --tag A [--n 200 --port 28081]
  -> eval_results/mc_<tag>.json  (accuracy + per-item scores)

Score(option) = mean logprob of option tokens given ctx (length-normalized,
standard HellaSwag practice). No generation — prefill only, fast.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_BIN = os.environ.get(
    "LLAMA_SERVER", "/home/wepiqx/llama.cpp/build/bin/llama-server")
OUT_DIR = os.path.join(REPO_ROOT, "eval_results")


def start_server(model, port, ngl=99):
    log = open(f"/tmp/mc_server_{port}.log", "w")
    p = subprocess.Popen(
        [SERVER_BIN, "-m", model, "--port", str(port), "-ngl", str(ngl),
         "-c", "2048", "--log-disable"],
        stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            r = requests.get(f"{base}/health", timeout=2)
            if r.status_code == 200:
                return p
        except Exception:
            pass
        if p.poll() is not None:
            raise RuntimeError("server died, see /tmp/mc_server_%d.log" % port)
        time.sleep(2)
    raise RuntimeError("server timeout")


def score_option(base, ctx, ending):
    """Mean logprob of ending tokens conditioned on ctx."""
    r = requests.post(
        f"{base}/v1/completions",
        json={"prompt": ctx + ending, "max_tokens": 0, "logprobs": 1,
              "temperature": 0.0, "echo": True},
        timeout=120)
    r.raise_for_status()
    data = r.json()
    lp = data["choices"][0].get("logprobs", {})
    toks = lp.get("tokens", [])
    vals = lp.get("token_logprobs", [])
    if not toks or not vals:
        return None
    # find where ending starts: tokenize ctx alone to get prefix length
    r2 = requests.post(
        f"{base}/v1/completions",
        json={"prompt": ctx, "max_tokens": 0, "logprobs": 1,
              "temperature": 0.0, "echo": True},
        timeout=120)
    r2.raise_for_status()
    npre = len(r2.json()["choices"][0].get("logprobs", {}).get("tokens", []))
    tail = [v for v in vals[npre:] if v is not None]
    if not tail:
        return None
    return sum(tail) / len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--port", type=int, default=28081)
    ap.add_argument("--parquet",
                    default=os.path.join(REPO_ROOT, "eval_results",
                                         "hellaswag_val.parquet"))
    ap.add_argument("--ngl", type=int, default=99)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.parquet).head(args.n)
    base = f"http://127.0.0.1:{args.port}"
    srv = start_server(args.model, args.port, args.ngl)
    try:
        hits, total, items = 0, 0, []
        t0 = time.time()
        for i, row in df.iterrows():
            ctx, ends, lab = row["ctx"], list(row["endings"]), int(row["label"])
            try:
                scores = [score_option(base, ctx, e) for e in ends]
            except Exception as ex:
                print("item %d error: %s" % (i, ex), flush=True)
                continue
            if any(s is None for s in scores):
                continue
            pred = int(max(range(len(scores)), key=lambda k: scores[k]))
            hits += (pred == lab)
            total += 1
            items.append({"i": int(i), "pred": pred, "label": lab})
            if total % 25 == 0:
                print("  %d/%d acc=%.3f %.1f min" %
                      (total, len(df), hits / total,
                       (time.time() - t0) / 60), flush=True)
        acc = hits / total if total else 0.0
        out = {"tag": args.tag, "model": args.model, "n": total,
               "acc": acc, "items": items}
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "mc_%s.json" % args.tag), "w") as f:
            json.dump(out, f)
        print("FINAL %s acc=%.4f n=%d" % (args.tag, acc, total), flush=True)
    finally:
        srv.send_signal(signal.SIGTERM)


if __name__ == "__main__":
    main()
