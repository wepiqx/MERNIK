"""protocol.py — single source of truth for the slow-ring protocol.

Every battery in the lab must run the SAME sampling, and every result file
must be able to prove it did. Before this module the numbers lived in four
places: two runners (with DIFFERENT defaults — one shipped
presence_penalty=1.5, which the README itself records as "breaks thinking
templates"), verify.py (correct, via setdefault), and the markdown ledgers.
Result files carried {task_id, completion} and nothing else, so a battery
run at presence 1.5 was indistinguishable from one run at 0.0.

Rule: import from here, never hardcode. A drift between the code default
and the documented protocol is now impossible to introduce silently.
"""
import json
import os
import platform
import subprocess
import sys
import time

SERVER_BIN = os.environ.get(
    "LLAMA_SERVER",
    os.path.expanduser("~/llama.cpp/build/bin/llama-server"))

# The verified protocol (README.md §4). Env overrides still win, so a
# deliberate A/B (temp 0.0 vs 1.0 — SRIQ duel) is possible and recorded.
PROTOCOL = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,   # 1.5 is the vendor recipe; it breaks thinking
    "repetition_penalty": 1.0,
    "max_tokens": 2048,        # 1024 truncates thinking models (MiMo: 11k+)
}

# Verdict-ring server settings. -c 8192 does NOT fit 8 GB VRAM with 6.4 GB of
# weights (proven: create_context fail) — the documented mitigation.
SERVER_ARGS = ["-ngl", "99", "-c", "8192", "--jinja", "--log-disable"]


def sample_params(environ=None):
    """Protocol dict with env overrides applied."""
    env = os.environ if environ is None else environ
    p = dict(PROTOCOL)

    def _f(name, default):
        try:
            return float(env.get(name, default))
        except ValueError:
            return float(default)

    p["temperature"] = _f("HE_TEMP", p["temperature"])
    p["top_p"] = _f("HE_TOP_P", p["top_p"])
    p["top_k"] = int(env.get("HE_TOP_K", p["top_k"]))
    p["min_p"] = _f("HE_MIN_P", p["min_p"])
    p["presence_penalty"] = _f("HE_PRESENCE", p["presence_penalty"])
    p["repetition_penalty"] = _f("HE_REPEAT", p["repetition_penalty"])
    p["max_tokens"] = int(env.get("HE_MAX_TOKENS", p["max_tokens"]))
    return p


def gpu_clocks():
    """GPU clock signature, for the sidecar.

    Spec from BIG (AGENTS.md 2026-09-28): this card is non-stock since
    Sep 23 23:50 and the OC survives reboots, so every later battery shares
    OC clocks and the earlier ones do not. A cross-era duel therefore
    carries a clock confound that nothing else in the record can see.

    WHICH number is the era key matters. Instantaneous clocks.gr drifts with
    load and temperature, so using them as the key would make every battery
    its own era and the check useless. The OC changes the MAXIMUM clocks, so
    the signature is (name, max.gr, max.mem) and the instantaneous values are
    recorded as diagnostics only.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,clocks.gr,clocks.mem,clocks.max.gr,clocks.max.mem,"
             "temperature.gpu,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        line = (out.stdout or "").strip().splitlines()
        if not line:
            return None
        name, gr, mem, grmax, memmax, temp, drv = [x.strip() for x in line[0].split(",")]
        return {
            "name": name,
            "gr": int(gr), "mem": int(mem),
            "gr_max": int(grmax), "mem_max": int(memmax),
            "temp": int(temp), "driver": drv,
            "signature": "%s gr_max=%d mem_max=%d" % (name, int(grmax), int(memmax)),
        }
    except Exception:
        return None


def _git_sha(repo_root):
    try:
        out = subprocess.run(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def battery_meta(server_url, serve_model, environ=None, extra=None):
    """Self-describing record of how a battery was run.

    Written to <results>.meta.json next to the sample file: a sidecar, not an
    extra key inside the sample jsonl — the human_eval checker owns that
    schema and must not be disturbed.
    """
    env = dict(os.environ if environ is None else environ)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    return {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "server_url": server_url,
        "serve_model": serve_model,
        "sample": sample_params(env),
        "server_bin": SERVER_BIN,
        "server_args": SERVER_ARGS + (extra or []),
        "clocks": gpu_clocks(),
        "git_sha": _git_sha(repo_root),
        "dirty": bool(subprocess.run(
            ["git", "-C", repo_root, "diff", "--quiet"],
            capture_output=True).returncode) if _git_sha(repo_root) else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "env_overrides": {k: v for k, v in env.items() if k.startswith("HE_")},
    }


def banner(meta):
    """One-line protocol stamp for the battery log (goes to stdout)."""
    s = meta["sample"]
    c = meta.get("clocks") or {}
    clock = (f"{c.get('signature')} now {c.get('gr')}/{c.get('mem')}MHz"
             if c else "gpu UNKNOWN")
    return (f"protocol: temp={s['temperature']} top_p={s['top_p']} "
            f"top_k={s['top_k']} min_p={s['min_p']} "
            f"presence={s['presence_penalty']} rep={s['repetition_penalty']} "
            f"max_tokens={s['max_tokens']} | {clock} | "
            f"git={meta.get('git_sha')} dirty={meta.get('dirty')}")


def write_meta(out_path, meta):
    side = os.path.splitext(out_path)[0] + ".meta.json"
    os.makedirs(os.path.dirname(side), exist_ok=True)
    with open(side, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    _append_ledger(out_path, meta)
    return side


# Append-only battery ledger, inside the repo, written by the runner itself.
# The chain logs were the only build→path record that ever existed and they
# lived in /tmp, so they died with the boot (2026-09-28: /tmp/opencode/chain.log
# gone, 48 of 59 batteries unidentifiable). The sidecar lives next to the
# results and can be lost with them; this line cannot, because it is written
# at battery time into a file the runner owns.
LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "provenance", "batteries.jsonl")


def _append_ledger(out_path, meta):
    try:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        rec = {
            "out": os.path.basename(out_path),
            "serve_model": meta.get("serve_model"),
            "started": meta.get("started"),
            "finished": meta.get("finished"),
            "n_tasks": meta.get("n_tasks"),
            "empties": meta.get("empties"),
            "sample": meta.get("sample"),
            "server_args": meta.get("server_args"),
            "clocks": (meta.get("clocks") or {}).get("signature"),
            "git_sha": meta.get("git_sha"),
            "dirty": meta.get("dirty"),
        }
        with open(LEDGER, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    except Exception as e:   # never lose a battery to a bookkeeping error
        print(f"protocol: WARNING could not append to {LEDGER}: {e}")
