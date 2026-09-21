#!/usr/bin/env python3
"""HumanEval runner using /v1/chat/completions with template.

Server lifecycle: point at a running server via HUMANEVAL_SERVER (manual
chain), or set HUMANEVAL_SERVE_MODEL=/path/to/model.gguf and the runner
serves it itself on HUMANEVAL_SERVER's port — and ALWAYS kills it at the
end (pass, fail or crash), so no ghost server eats VRAM overnight. A busy
port aborts LOUDLY instead of stealing the GPU (strict queue).
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
    os.path.join(REPO_ROOT, "eval_results", "humaneval_opts.jsonl"),
)


def _f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


# NeoHorse reported protocol (SGLang v0.5.17); override via env, e.g.
# HE_PRESENCE=0 if presence_penalty turns out harmful.
SAMPLE = {
    "temperature": _f("HE_TEMP", 1.0),
    "top_p": _f("HE_TOP_P", 0.95),
    "top_k": int(os.environ.get("HE_TOP_K", 20)),
    "min_p": _f("HE_MIN_P", 0.0),
    "presence_penalty": _f("HE_PRESENCE", 1.5),
    "repetition_penalty": _f("HE_REPEAT", 1.0),
}
THINKING = os.environ.get("HE_THINKING", "1") == "1"


def generate(problem, max_tokens=None):
    if max_tokens is None:
        max_tokens = int(os.environ.get("HE_MAX_TOKENS", 1024))
    resp = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python programmer. Complete the following function. Return ONLY Python code inside a fenced block ```python...```",
                },
                {"role": "user", "content": problem["prompt"]},
            ],
            "max_tokens": max_tokens,
            **SAMPLE,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_code(text):
    text = text.strip()
    if "```python" in text:
        text = text.split("```python")[1]
        if "```" in text:
            text = text.split("```")[0]
    return text.strip()


def _port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _serve_own_model():
    """Serve HUMANEVAL_SERVE_MODEL on SERVER_URL's port. Returns Popen."""
    port = urlparse(SERVER_URL).port or 28082
    if _port_busy(port):
        raise SystemExit(
            f"refusing to serve: port {port} busy (strict queue) — "
            f"other battery running, retry when free.")
    log = open("/tmp/he_serve_%d.log" % port, "w")
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
            raise SystemExit("own server died during load, see "
                             f"/tmp/he_serve_{port}.log")
        time.sleep(5)
    p.send_signal(signal.SIGTERM)
    raise SystemExit(f"own server on {port} never healthy — killed")


def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    srv = _serve_own_model() if SERVE_MODEL else None
    try:
        _battery()
    finally:
        if srv is not None:
            srv.send_signal(signal.SIGTERM)
            print("own server killed — GPU free", flush=True)


def _battery():
    # preflight: never silently score 164 empty completions against a dead
    # server (that failure mode already cost us one battery).
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=10)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(f"server not healthy at {SERVER_URL}: {e}")
    problems = read_problems()
    results = []
    total = len(problems)

    for i, (task_id, problem) in enumerate(sorted(problems.items())):
        # mid-run watchdog: a dead server must abort LOUDLY, never silently
        # score 100+ empties (that failure mode already cost us two batteries:
        # TD-SMAPE-1st and MSE-5100 were scored dead and had to be rerun).
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
            raw = generate(problem)
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
    r = evaluate_functional_correctness(sample_file=OUTPUT_FILE, k=[1], n_workers=4)
    print(f"pass@1: {r}")


if __name__ == "__main__":
    main()
