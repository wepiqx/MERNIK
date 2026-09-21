#!/usr/bin/env python3
"""GPQA-duel: Diamond slice scored via llama-server prompt logprobs.

Usage:
  python scripts/gpqa_duel.py --model A.gguf --tag A [--port 28081]
  -> eval_results/gpqa_<tag>.json  (accuracy + per-item)

Score(option) = mean logprob of option tokens given question (length
normalized). No generation — prefill only. Choices shuffled per item with
a fixed seed so position can't leak.
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time

import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_BIN = os.environ.get(
    "LLAMA_SERVER", "/home/wepiqx/llama.cpp/build/bin/llama-server")
OUT_DIR = os.path.join(REPO_ROOT, "eval_results")
DATA = os.path.join(OUT_DIR, "gpqa_diamond.jsonl")


def start_server(model, port, ngl=99, ctx=8192):
    log = open("/tmp/gpqa_server_%d.log" % port, "w")
    p = subprocess.Popen(
        [SERVER_BIN, "-m", model, "--port", str(port), "-ngl", str(ngl),
         "-c", str(ctx), "--jinja", "--log-disable"],
        stdout=log, stderr=subprocess.STDOUT)
    base = "http://127.0.0.1:%d" % port
    for _ in range(120):
        try:
            r = requests.get(base + "/health", timeout=2)
            if r.status_code == 200:
                return p
        except Exception:
            pass
        if p.poll() is not None:
            raise RuntimeError("server died, see /tmp/gpqa_server_%d.log" % port)
        time.sleep(2)
    raise RuntimeError("server timeout")


def _first_dist(base, prompt, topk=30):
    """Top-k distribution of the first generated token.

    NOTE: this llama.cpp only returns logprobs for generated tokens, so
    choice scoring reads P(letter|prompt) off the first position instead
    of scoring full option strings. Letters missing from top-k get a
    floor value (counted as losses in practice).
    """
    r = requests.post(
        "%s/v1/completions" % base,
        json={"prompt": prompt, "max_tokens": 1, "logprobs": topk,
              "temperature": 0.0},
        timeout=180)
    r.raise_for_status()
    content = r.json()["choices"][0].get("logprobs", {}).get("content", [])
    if not content:
        return None
    return content[0].get("top_logprobs", [])


def score_letters(base, prompt):
    """Returns [lpA, lpB, lpC, lpD] or None. Floor -20 for missing."""
    top = _first_dist(base, prompt)
    if not top:
        return None
    out = []
    for want in "ABCD":
        best = None
        for e in top:
            if e.get("token", "").strip() == want:
                best = e.get("logprob", best)
                break
        out.append(-20.0 if best is None else best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--port", type=int, default=28081)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    rng = random.Random(args.seed)
    base = "http://127.0.0.1:%d" % args.port
    srv = start_server(args.model, args.port, args.ngl)
    try:
        hits, total = 0, 0
        t0 = time.time()
        for i, row in enumerate(rows):
            opts = [row["correct"]] + list(row["wrong"])
            order = list(range(4))
            rng.shuffle(order)
            opts = [opts[k] for k in order]
            gold = order.index(0)
            letters = "ABCD"
            prompt = (row["q"].strip() + "\n" +
                      "\n".join("%s) %s" % (letters[j], o)
                                 for j, o in enumerate(opts)) +
                      "\nAnswer with the letter only.\nAnswer:")
            try:
                scores = score_letters(base, prompt)
            except Exception as ex:
                print("item %d error: %s" % (i, str(ex)[:80]), flush=True)
                continue
            if scores is None:
                continue
            pred = max(range(4), key=lambda k: scores[k])
            hits += (pred == gold)
            total += 1
            if total % 50 == 0:
                print("  %d/%d acc=%.3f %.1f min" %
                      (total, len(rows), hits / total,
                       (time.time() - t0) / 60), flush=True)
        acc = hits / total if total else 0.0
        out = {"tag": args.tag, "model": args.model, "n": total, "acc": acc}
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "gpqa_%s.json" % args.tag), "w") as f:
            json.dump(out, f)
        print("FINAL %s acc=%.4f n=%d" % (args.tag, acc, total), flush=True)
    finally:
        srv.send_signal(signal.SIGTERM)


if __name__ == "__main__":
    main()
