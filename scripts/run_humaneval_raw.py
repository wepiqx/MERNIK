#!/usr/bin/env python3
"""HumanEval via raw /v1/completions (no chat template).

For models whose chat template is broken/missing (e.g. MiniCPM5-2B whose
jinja renders prompts the model answers with emptiness): classic base-model
HE protocol — prompt = problem["prompt"], raw completion, extract code.
Same sampling protocol otherwise (temp/top_p/top_k/min_p/presence via HE_*).

Server lifecycle: same as run_humaneval (HUMANEVAL_SERVE_MODEL self-serve
with auto-kill, or external HUMANEVAL_SERVER). Extra server flags via
HUMANEVAL_SERVER_ARGS.
"""
import json
import os
import signal
import socket
import subprocess
import time
from urllib.parse import urlparse
import requests
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

SERVER_URL = os.environ.get("HUMANEVAL_SERVER", "http://127.0.0.1:28082")
SERVE_MODEL = os.environ.get("HUMANEVAL_SERVE_MODEL")
SERVER_BIN = os.environ.get(
    "LLAMA_SERVER",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.environ.get(
    "HUMANEVAL_OUT",
    os.path.join(REPO_ROOT, "eval_results", "humaneval_raw.jsonl"),
)


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


SAMPLE = {
    "temperature": _f("HE_TEMP", 1.0),
    "top_p": _f("HE_TOP_P", 0.95),
    "top_k": int(os.environ.get("HE_TOP_K", 20)),
    "min_p": _f("HE_MIN_P", 0.0),
    "presence_penalty": _f("HE_PRESENCE", 0.0),
    "repetition_penalty": _f("HE_REPEAT", 1.0),
}


def _port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _serve_own_model():
    port = urlparse(SERVER_URL).port or 28082
    if _port_busy(port):
        raise SystemExit(f"refusing to serve: port {port} busy (strict queue)")
    log = open("/tmp/he_raw_serve_%d.log" % port, "w")
    extra = os.environ.get("HUMANEVAL_SERVER_ARGS", "").split()
    p = subprocess.Popen(
        [SERVER_BIN, "-m", SERVE_MODEL, "--port", str(port), "-ngl", "99",
         "-c", "8192", "--jinja", "--log-disable"] + extra,
        stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"serving {SERVE_MODEL} on {port}", flush=True)
                return p
        except Exception:
            pass
        if p.poll() is not None:
            raise SystemExit("own server died during load")
        time.sleep(5)
    p.send_signal(signal.SIGTERM)
    raise SystemExit(f"own server on {port} never healthy — killed")


def generate(prompt, max_tokens=None):
    if max_tokens is None:
        max_tokens = int(os.environ.get("HE_MAX_TOKENS", 1024))
    resp = requests.post(
        f"{SERVER_URL}/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens, **SAMPLE},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def extract_code(text):
    text = text.strip()
    if "```python" in text:
        text = text.split("```python")[1]
        if "```" in text:
            text = text.split("```")[0]
    return text.strip()


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    srv = _serve_own_model() if SERVE_MODEL else None
    try:
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=10)
            r.raise_for_status()
        except Exception as e:
            raise SystemExit(f"server not healthy at {SERVER_URL}: {e}")
        problems = read_problems()
        if os.environ.get("HE_LIMIT"):
            keep = sorted(problems)[:int(os.environ["HE_LIMIT"])]
            problems = {k: problems[k] for k in keep}
            print(f"HE_LIMIT: first {len(problems)} tasks only", flush=True)
        results = []
        total = len(problems)
        for i, (task_id, problem) in enumerate(sorted(problems.items())):
            if i % 10 == 0:
                try:
                    r = requests.get(f"{SERVER_URL}/health", timeout=10)
                    r.raise_for_status()
                except Exception as e:
                    raise SystemExit(
                        f"server died mid-run at task {i}/{total}: {e}. "
                        f"Results so far DISCARDED (not scored).")
            print(f"[{i+1}/{total}] {task_id} ... ", end="", flush=True)
            try:
                raw = generate(problem["prompt"])
                code = extract_code(raw) or raw
                results.append({"task_id": task_id, "completion": code})
                print("OK")
            except Exception as e:
                print(f"ERROR: {e}")
                results.append({"task_id": task_id, "completion": ""})
            time.sleep(0.1)
        write_jsonl(OUTPUT_FILE, results)
        print(f"\nSaved {len(results)} to {OUTPUT_FILE}")
        print("\n--- Evaluating pass@1 ---")
        r = evaluate_functional_correctness(sample_file=OUTPUT_FILE, k=[1],
                                            n_workers=4)
        print(f"pass@1: {r}")
    finally:
        if srv is not None:
            srv.send_signal(signal.SIGTERM)
            print("own server killed — GPU free", flush=True)


if __name__ == "__main__":
    main()
