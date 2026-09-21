#!/usr/bin/env python3
"""HumanEval evaluation for MERNIK models.

Usage:
  # Start server in background:
  python eval_humaneval.py --model /path/to/model.gguf --mode serve
  
  # Generate completions (model must be serving):
  python eval_humaneval.py --model /path/to/model.gguf --mode generate --samples 1
  
  # Evaluate:
  python eval_humaneval.py --mode evaluate --samples-file /path/to/samples.jsonl
  
  # All in one:
  python eval_humaneval.py --model /path/to/model.gguf --mode all --samples 1
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

from human_eval.data import read_problems, write_jsonl, stream_jsonl
from human_eval.evaluation import evaluate_functional_correctness


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_BIN = os.environ.get(
    "LLAMA_SERVER", "/home/wepiqx/llama.cpp/build/bin/llama-server"
)
HOST = "127.0.0.1"
PORT = 28080
BASE_URL = f"http://{HOST}:{PORT}"
OUT_DIR = os.environ.get(
    "EVAL_OUT_DIR", os.path.join(REPO_ROOT, "eval_results")
)


def start_server(model_path: str) -> subprocess.Popen:
    os.makedirs(OUT_DIR, exist_ok=True)
    log_file = os.path.join(OUT_DIR, "server.log")
    cmd = [
        SERVER_BIN,
        "-m", model_path,
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", "99",
        "-c", "2048",
        "--no-kv-offload",
    ]
    print(f"Starting server: {' '.join(cmd)}")
    f = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_for_server(timeout: int = 120):
    for i in range(timeout):
        try:
            req = urllib.request.urlopen(f"{BASE_URL}/health", timeout=5)
            if req.status == 200:
                print(f"Server ready after {i}s")
                return True
        except:
            pass
        time.sleep(1)
    return False


def stop_server(proc: subprocess.Popen):
    if proc:
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait()


def generate_completion(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    data = json.dumps({
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": ["\nclass", "\ndef", "\nif __name__", "\n#"],
        "cache_prompt": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    result = json.loads(resp.read())
    return result["choices"][0]["text"]


def extract_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    lines = text.split("\n")
    code_lines = []
    for line in lines:
        if re.match(r'^(from |import |def |class |#|\s+|$)', line) or (code_lines and line.strip()):
            code_lines.append(line)
    return "\n".join(code_lines).strip()


def make_prompt(problem: dict) -> str:
    return (
        "<|im_start|>system\n"
        "You are an expert Python programmer. Complete the following function.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{problem['prompt']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "```python\n"
        f"{problem['prompt']}"
    )


def generate(problems: dict, samples: int, output: str):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    results = []
    total = len(problems)
    for i, (task_id, problem) in enumerate(sorted(problems.items())):
        prompt = make_prompt(problem)
        for s in range(samples):
            print(f"[{i+1}/{total}] sample {s+1}/{samples}: {task_id}")
            try:
                raw = generate_completion(prompt)
                code = extract_code(raw)
                if not code:
                    code = raw
                results.append({
                    "task_id": task_id,
                    "completion": code,
                })
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "task_id": task_id,
                    "completion": "",
                })
            time.sleep(0.1)
        has_results = sum(1 for r in results if r["task_id"] == task_id and r["completion"])
        print(f"  -> {has_results}/{samples} non-empty completions")

    write_jsonl(output, results)
    print(f"Saved {len(results)} completions to {output}")


def main():
    parser = argparse.ArgumentParser(description="HumanEval for MERNIK models")
    parser.add_argument("--model", help="Path to GGUF model")
    parser.add_argument("--mode", choices=["serve", "generate", "evaluate", "all"],
                        default="all", help="Mode")
    parser.add_argument("--samples", type=int, default=1, help="Samples per problem")
    parser.add_argument("--samples-file", help="Path to samples JSONL")
    parser.add_argument("--k", default="1", help="pass@k values")
    parser.add_argument("--n-workers", type=int, default=4, help="Eval workers")
    parser.add_argument("--tag", default="", help="Tag for output file naming")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(args.model or "unknown"))[0]
    tag = f"_{args.tag}" if args.tag else ""
    samples_file = args.samples_file or os.path.join(OUT_DIR, f"{model_name}{tag}_samples.jsonl")

    server_proc = None

    if args.mode in ("serve", "all"):
        if not args.model:
            print("--model required for serve mode")
            sys.exit(1)
        server_proc = start_server(args.model)
        if not wait_for_server():
            print("Server failed to start")
            stop_server(server_proc)
            sys.exit(1)

    if args.mode in ("generate", "all"):
        problems = read_problems()
        generate(problems, args.samples, samples_file)

    if args.mode in ("evaluate", "all"):
        if not os.path.exists(samples_file):
            print(f"Samples file not found: {samples_file}")
            sys.exit(1)
        # Check if we have completions
        count = sum(1 for _ in stream_jsonl(samples_file))
        if count == 0:
            print("No completions found!")
            sys.exit(1)
        print(f"Evaluating {count} completions from {samples_file}")
        results = evaluate_functional_correctness(
            sample_file=samples_file,
            k=list(map(int, args.k.split(","))),
            n_workers=args.n_workers,
        )
        print("Results:", results)

    if server_proc:
        stop_server(server_proc)


if __name__ == "__main__":
    main()
