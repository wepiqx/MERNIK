#!/usr/bin/env python3
"""Slow-ring verify for MERNIK builds: GPQA-recognition and HumanEval.

Called from main.py --verify (gpqa|he|all). Reuses scripts/ as-is:
  - gpqa: scripts/gpqa_duel.py manages its own llama-server
  - he: we serve the model on the protocol port, then run
    scripts/run_humaneval.py against it (never kill a foreign server:
    a busy port aborts LOUDLY instead of stealing the GPU).
"""
import os
import re
import signal
import socket
import subprocess
import sys
import time

import requests

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_BIN = os.environ.get(
    "LLAMA_SERVER",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"))
GPQA_PORT = int(os.environ.get("VERIFY_GPQA_PORT", 28081))
HE_PORT = int(os.environ.get("VERIFY_HE_PORT", 28082))


def tag_for(model_path):
    base = os.path.splitext(os.path.basename(model_path))[0]
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").lower()


def port_busy(port):
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_health(port, timeout=600):
    base = "http://127.0.0.1:%d" % port
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(base + "/health", timeout=5)
            if r.status_code == 200:
                return base
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError("server on %d never healthy" % port)


def run_gpqa(model, tag):
    if port_busy(GPQA_PORT):
        raise SystemExit(
            "VERIFY ABORTED: port %d busy (GPU queue) — other battery "
            "running, retry when free. Nothing was scored." % GPQA_PORT)
    cmd = [sys.executable, "-u",
           os.path.join(REPO_ROOT, "scripts", "gpqa_duel.py"),
           "--model", model, "--tag", tag, "--port", str(GPQA_PORT)]
    print("verify[gpqa]: %s" % " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        raise SystemExit("verify[gpqa] FAILED (exit=%d)" % rc)
    print("verify[gpqa]: done, see eval_results/gpqa_%s.json" % tag)


def run_he(model, tag):
    if port_busy(HE_PORT):
        raise SystemExit(
            "VERIFY ABORTED: port %d busy (GPU queue) — other battery "
            "running, retry when free. Nothing was scored." % HE_PORT)
    log = open("/tmp/verify_he_server_%d.log" % HE_PORT, "w")
    srv = subprocess.Popen(
        [SERVER_BIN, "-m", model, "--port", str(HE_PORT), "-ngl", "99",
         "-c", "8192", "--jinja", "--log-disable"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_health(HE_PORT)
        out = os.path.join(REPO_ROOT, "eval_results",
                           "humaneval_%s.jsonl" % tag)
        env = dict(os.environ)
        # protocol defaults (MERNIK.md); explicit env always wins
        env.setdefault("HE_PRESENCE", "0.0")
        env.setdefault("HE_MAX_TOKENS", "2048")
        env["HUMANEVAL_OUT"] = out
        env["HUMANEVAL_SERVER"] = "http://127.0.0.1:%d" % HE_PORT
        cmd = [sys.executable, "-u",
               os.path.join(REPO_ROOT, "scripts", "run_humaneval.py")]
        print("verify[he]: serving %s, out=%s" % (model, out), flush=True)
        rc = subprocess.call(cmd, env=env)
        if rc != 0:
            raise SystemExit("verify[he] FAILED (exit=%d), see %s" %
                             (rc, out))
        print("verify[he]: done, see %s{,_results.jsonl}" % out)
    finally:
        srv.send_signal(signal.SIGTERM)


def main_verify(model, which, tag=None):
    if not os.path.exists(model):
        raise SystemExit("verify: model not found: %s" % model)
    tag = tag or tag_for(model)
    print("verify: model=%s tag=%s" % (model, tag))
    if which in ("gpqa", "all"):
        run_gpqa(model, tag)
    if which in ("he", "all"):
        run_he(model, tag)
    print("verify: ALL DONE (%s)" % which)
