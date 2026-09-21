"""Group damage sweep (battlefield teacher).

For each tied group / singleton: baseline (everything Q5_K + pins) vs variant
(group dropped to Q4_K). Label = PPL(variant) - PPL(baseline) — the measured
damage used to train the tiny importance net.

Usage:
    python group_damage_sweep.py --model M.gguf --imatrix I.gguf --out models/damage.jsonl [--resume]

~4.5 min/label on 1.7B (quant + PPL). ~130 labels ≈ 10h. Run overnight.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from classifier import build_groups
from config_generator import generate_flags
from quantizer import run_quantization

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.environ.get("LLAMA_QUANTIZE",
                     "/home/wepiqx/llama.cpp/build/bin/llama-quantize")
PPL = os.environ.get("LLAMA_PPL",
                     "/home/wepiqx/llama.cpp/build/bin/llama-perplexity")
WIKI = os.environ.get("WIKITEXT_DATA", "/mnt/Vsio/wikitext-2-raw/wiki.test.raw")


def run_ppl(model_path: str, ctx: int = 1024, n: int = 64,
            wiki: str | None = None, kld_base: str | None = None):
    """Returns (ppl, kld|None). With kld_base: also scores KLD vs ref logits."""
    cmd = [PPL, "-m", model_path, "-f", wiki or WIKI, "-ngl", "99",
           "-c", str(ctx), "-n", str(n), "-b", "512", "--seed", "7"]
    if kld_base:
        cmd += ["--kl-divergence", "--kl-divergence-base", kld_base]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    ppl, kld = None, None
    for line in (out.stderr + out.stdout).splitlines():
        if "Final estimate" in line:
            ppl = float(line.split("PPL =")[1].split("+/-")[0].strip())
        if "Mean" in line and "KLD" in line:
            try:
                kld = float(line.split("KLD:")[1].split("±")[0].strip())
            except (IndexError, ValueError):
                pass
    if ppl is None:
        raise RuntimeError(f"no PPL in output for {model_path}")
    if kld_base and kld is None:
        raise RuntimeError(f"no KLD in output for {model_path}")
    return ppl, kld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-tier", default="Q5_K")
    ap.add_argument("--drop-tier", default="Q4_K")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wiki-file", default=None,
                    help="PPL corpus (default: $WIKITEXT_DATA or home path)")
    ap.add_argument("--per-tensor", action="store_true",
                    help="v2: units are single tensors (finer Gnom labels), "
                         "not tied groups")
    ap.add_argument("--ppl-ctx", type=int, default=1024)
    ap.add_argument("--ppl-n", type=int, default=64,
                    help="v2: short protocol (-n 32) if damage is ctx-invariant")
    ap.add_argument("--kld-base", default=None,
                    help="KLD-teacher: path to ref logits .dat (from "
                         "--save-all-logits on unquantized model). When set, "
                         "each unit also scores KLD vs ref; damage_kld is "
                         "stored alongside PPL damage.")
    args = ap.parse_args()

    model = read_model(args.model)
    im = read_imatrix(args.imatrix)
    tg = detect_tied_groups(im)
    imp_table = build_importance_table(im, model)

    ne_map = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}
    for tname, info in imp_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # pool = everything the queue would fight over (no MTP here by construction
    # on dense models; keep pins as fixed like production)
    from constants import get_tensor_type, EMBD_PIN_TYPES, ROUTER_PIN_TYPES
    pool = set(ne_map.keys())
    for pin in (EMBD_PIN_TYPES, ROUTER_PIN_TYPES):
        pool = {n for n in pool
                if imp_table.get(n, {}).get("type", get_tensor_type(n)) not in pin}
    pool = {n for n in pool if "norm" not in n}

    os.makedirs(os.path.join(REPO_ROOT, "models"), exist_ok=True)
    if args.per_tensor:
        units = [[n] for n in sorted(pool)]  # v2: finer labels, no sharing
    else:
        groups = build_groups(tg, pool, ne_map, dict(ne_map))
        units = [g for _, (g, _, _) in sorted(groups.items())]
    print(f"units: {len(units)}", flush=True)

    done = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["unit"])

    def build_file_inner(drop: list | None, path: str):
        assignments = {}
        for n in ne_map:
            assignments[n] = args.base_tier
        if drop:
            for n in drop:
                assignments[n] = args.drop_tier
        flags = generate_flags(assignments, model, args.base_tier, 100000)
        flags["imatrix"] = args.imatrix
        ok = run_quantization(flags, args.model, path)
        if not ok:
            raise RuntimeError("quant failed")

    def _looks_valid(path: str) -> bool:
        """GGUF magic + sane size (>100MB for 1.7B quants)."""
        try:
            if os.path.getsize(path) < 100 * 1024 * 1024:
                return False
            with open(path, "rb") as f:
                return f.read(4) == b"GGUF"
        except OSError:
            return False

    def _wait_then_build(prev_fut, unit, path):
        prev_fut.result()
        build_file_inner(unit, path)

    build_file = build_file_inner

    tmp = os.path.join(REPO_ROOT, "models", ".sweep_tmp.gguf")
    kb = args.kld_base
    if "BASELINE" not in done:
        build_file(None, tmp)
        base_ppl, base_kld = run_ppl(tmp, args.ppl_ctx, args.ppl_n,
                                     args.wiki_file, kb)
        rec = {"unit": "BASELINE", "ppl": base_ppl}
        if kb:
            rec["kld"] = base_kld
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"BASELINE ppl={base_ppl:.4f}" +
              (f" kld={base_kld:.4f}" if kb else ""), flush=True)
    else:
        with open(args.out) as f:
            rows = [json.loads(l) for l in f
                    if json.loads(l)["unit"] == "BASELINE"]
            base_ppl = rows[0]["ppl"]
            base_kld = rows[0].get("kld")

    def unit_tag(unit):
        return "+".join(n.replace(".weight", "").split(".")[-1] + "@" +
                        n.split(".")[1] for n in unit
                        if n.startswith("blk.")) or "global"

    # Pipelined: quant(unit i+1) on CPU overlaps PPL(unit i) on GPU.
    # UNIQUE tmp file per unit: rotating buffers raced when a slow PPL
    # (loaded system) let a later build overwrite the file PPL still read.
    # ≤3 files alive at once (~3.4GB), deleted right after use.
    todo = [(unit_tag(u), u) for u in units if unit_tag(u) not in done]
    safe = lambda tag, i: re.sub(r"[^A-Za-z0-9_@+-]", "_", tag)[:60]
    ex = ThreadPoolExecutor(max_workers=2)
    try:
        if todo:
            tag0, unit0 = todo[0]
            buf0 = os.path.join(REPO_ROOT, "models",
                                f".sweep_0_{safe(tag0, 0)}.gguf")
            fut_q = ex.submit(build_file, unit0, buf0)
            pending = [(tag0, unit0, buf0, fut_q)]
            for i, (tag, unit) in enumerate(todo[1:], start=1):
                buf = os.path.join(REPO_ROOT, "models",
                                   f".sweep_{i}_{safe(tag, i)}.gguf")
                pending.append((tag, unit, buf,
                                ex.submit(_wait_then_build, pending[-1][3],
                                          unit, buf)))
            for idx, (tag, unit, buf, fq) in enumerate(pending):
                try:
                    fq.result()  # quant done (previous PPL overlapped it)
                except Exception as e:
                    print(f"[{idx + 1}/{len(pending)}] {tag} QUANT failed "
                          f"({type(e).__name__}), skipped", flush=True)
                    continue
                if not _looks_valid(buf):
                    print(f"[{idx + 1}/{len(pending)}] {tag} CORRUPT file, "
                          f"skipped", flush=True)
                    continue
                try:
                    p, k = run_ppl(buf, args.ppl_ctx, args.ppl_n,
                                   args.wiki_file, kb)
                except Exception as e:
                    print(f"[{idx + 1}/{len(pending)}] {tag} PPL failed "
                          f"({type(e).__name__}: {str(e)[:100]}), skipped",
                          flush=True)
                    continue
                finally:
                    try:
                        os.remove(buf)
                    except OSError:
                        pass
                rec = {"unit": tag, "tensors": unit,
                       "ppl": p, "damage": p - base_ppl}
                if kb:
                    rec["kld"] = k
                    rec["damage_kld"] = k - base_kld
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{idx + 1}/{len(pending)}] {tag} "
                      f"damage={p - base_ppl:+.4f}" +
                      (f" kld={k - base_kld:+.4f}" if kb else ""), flush=True)
    finally:
        ex.shutdown(wait=False)


if __name__ == "__main__":
    main()
