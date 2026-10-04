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
            wiki: str | None = None, kld_base: str | None = None,
            timeout: int = 1200, ngl: int = 99):
    """Returns (ppl, kld|None, extra dict).

    NOTE (build >=11051 scar): --kl-divergence REPLACES perplexity instead
    of augmenting it (no 'Final estimate' line). So with kld_base we run
    TWICE: plain (PPL Final, protocol-comparable) + KLD flags (Mean KLD).
    Costs 2x PPL time per unit — honesty over speed.

    The KLD pass prints a whole statistics section; we used to read ONE
    line (Mean KLD). Now also parsed, same run, zero extra compute:
    Maximum KLD (tail/detonation), Median KLD (skew vs mean), RMS dp
    (behavioral shift), Same-top-p (top-1 agreement = decisiveness).
    """
    cmd = [PPL, "-m", model_path, "-f", wiki or WIKI, "-ngl", str(ngl),
           "-c", str(ctx), "-n", str(n), "-b", "512", "--seed", "7"]
    out = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout)
    ppl, kld = None, None
    extra = {}
    for line in (out.stderr + out.stdout).splitlines():
        if "Final estimate" in line:
            ppl = float(line.split("PPL =")[1].split("+/-")[0].strip())
    if ppl is None:
        tail = "\n".join((out.stderr + out.stdout).splitlines()[-8:])
        raise RuntimeError(f"no PPL in output for {model_path}:\n{tail}")
    if kld_base:
        kcmd = cmd + ["--kl-divergence", "--kl-divergence-base", kld_base]
        out2 = subprocess.run(kcmd, capture_output=True, text=True,
                              timeout=timeout)
        for line in (out2.stderr + out2.stdout).splitlines():
            s = line.strip()
            if "Mean" in s and "KLD" in s and "Δp" not in s:
                try:
                    kld = float(s.split("KLD:")[1].split("±")[0].strip())
                except (IndexError, ValueError):
                    pass
            elif s.startswith("Maximum KLD:"):
                try:
                    extra["kld_max"] = float(s.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif s.startswith("Median") and "KLD" in s:
                try:
                    extra["kld_median"] = float(s.split(":")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif s.startswith("RMS"):
                try:
                    extra["dp_rms"] = float(s.split(":")[1].split("±")[0].strip()) / 100.0
                except (IndexError, ValueError):
                    pass
            elif s.startswith("Same top p:"):
                try:
                    extra["same_top"] = float(s.split(":")[1].split("±")[0].strip()) / 100.0
                except (IndexError, ValueError):
                    pass
        if kld is None:
            tail = "\n".join((out2.stderr + out2.stdout).splitlines()[-8:])
            raise RuntimeError(f"no KLD in output for {model_path}:\n{tail}")
    return ppl, kld, extra


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
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-PPL subprocess timeout, seconds "
                         "(slow backends: 3600+ for Vega)")
    ap.add_argument("--ngl", type=int, default=99,
                    help="GPU layers for PPL (big BF16 on small VRAM: "
                         "lower until context fits, e.g. 24-32 on 8GB)")
    ap.add_argument("--units", default=None,
                    help="comma-separated unit tags to measure (subset mode "
                         "for the active loop); default: all unfinished")
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
        units = [g for _, (g, _, _, *_) in sorted(groups.items())]
    print(f"units: {len(units)}", flush=True)

    # Failure ledger (mailbox 2026-09-28): a failed unit is a LABEL, not a
    # hole — but it must never enter the labels file, or --resume would
    # skip retrying it. Separate file, trainer reads both, resume untouched.
    fail_path = os.path.splitext(args.out)[0] + ".failures.jsonl"

    def log_failure(tag, stage, err, tensors):
        # HOLE vs TERMINAL (mailbox 2026-09-28): first failure of a unit is
        # a hole — logged aside, retried by --resume. A unit that fails
        # TWICE (this attempt + a prior hole) is TERMINAL: the measurement
        # is physically untakeable at this tier (censored observation).
        # TERMINAL goes INTO the labels file as a null record so --resume
        # never retries it and the trainer can see it (never impute it).
        prior = 0
        try:
            if os.path.exists(fail_path):
                with open(fail_path) as f:
                    for line in f:
                        try:
                            if json.loads(line).get("unit") == tag:
                                prior += 1
                        except (ValueError, AttributeError):
                            pass
        except OSError:
            pass
        if prior >= 1:
            rec = {"unit": tag, "tensors": tensors, "tier": args.drop_tier,
                   "damage": None, "terminated": f"{stage}:{err}"[:200],
                   "retryable": False}
            if kb:
                rec["damage_kld"] = None
            try:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"{tag} TERMINAL ({stage}) — censored label, "
                      f"will not retry", flush=True)
            except OSError:
                pass
            return
        try:
            with open(fail_path, "a") as f:
                f.write(json.dumps({"unit": tag, "status": "failed",
                                    "stage": stage,
                                    "error": str(err)[:200]}) + "\n")
        except OSError:
            pass

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
        # Serialization only (one quantize at a time: RAM). A dead
        # predecessor must NEVER poison this unit: 2026-10-03 scar —
        # one OOM-killed build cascaded RuntimeError through 148
        # downstream units that never even ran. Catch, then build anyway;
        # own failures surface via own future below.
        try:
            prev_fut.result()
        except Exception:
            pass
        build_file_inner(unit, path)

    build_file = build_file_inner

    tmp = os.path.join(REPO_ROOT, "models", ".sweep_tmp.gguf")
    # crash orphans: stale .sweep_* buffers from killed runs (else disk death)
    for fn in os.listdir(os.path.join(REPO_ROOT, "models")):
        if fn.startswith(".sweep_") and fn.endswith(".gguf"):
            try:
                os.remove(os.path.join(REPO_ROOT, "models", fn))
                print(f"orphan removed: {fn}", flush=True)
            except OSError:
                pass
    kb = args.kld_base
    if "BASELINE" not in done:
        build_file(None, tmp)
        base_ppl, base_kld, base_extra = run_ppl(tmp, args.ppl_ctx, args.ppl_n,
                                      args.wiki_file, kb,
                                      timeout=args.timeout, ngl=args.ngl)
        rec = {"unit": "BASELINE", "ppl": base_ppl}
        if kb:
            rec["kld"] = base_kld
            rec.update(base_extra)
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
            base_extra = {k: rows[0].get(k) for k in
                          ("kld_max", "kld_median", "dp_rms", "same_top")}

    def unit_tag(unit):
        return "+".join(n.replace(".weight", "").split(".")[-1] + "@" +
                        n.split(".")[1] for n in unit
                        if n.startswith("blk.")) or "global"

    # Pipelined: quant(unit i+1) on CPU overlaps PPL(unit i) on GPU.
    # UNIQUE tmp file per unit: rotating buffers raced when a slow PPL
    # (loaded system) let a later build overwrite the file PPL still read.
    # BOUNDED window (2026-10-03 scar): the old code submitted ALL builds
    # upfront — builders outran PPL, buffers + page cache grew all run,
    # and the tail died (OOM/pressure). At most MAX_INFLIGHT builds may
    # be ahead of consumption; the rest wait unsubmitted. ≤3 files alive
    # (~3.4GB per 1.7B unit), deleted right after use.
    MAX_INFLIGHT = 3
    todo = [(unit_tag(u), u) for u in units if unit_tag(u) not in done]
    if args.units:
        want = {s.strip() for s in args.units.split(",") if s.strip()}
        todo = [(t, u) for t, u in todo if t in want]
        print(f"subset mode: {len(todo)} units requested", flush=True)
    safe = lambda tag, i: re.sub(r"[^A-Za-z0-9_@+-]", "_", tag)[:60]
    ex = ThreadPoolExecutor(max_workers=2)
    submitted = {}  # todo_idx -> (tag, unit, buf, future)

    def ensure_submitted(idx):
        if idx < 0 or idx >= len(todo) or idx in submitted:
            return
        tag, unit = todo[idx]
        buf = os.path.join(REPO_ROOT, "models",
                            f".sweep_{idx}_{safe(tag, idx)}.gguf")
        prev = submitted[idx - 1][3] if idx - 1 in submitted else None
        if prev is None:
            fut = ex.submit(build_file, unit, buf)
        else:
            fut = ex.submit(_wait_then_build, prev, unit, buf)
        submitted[idx] = (tag, unit, buf, fut)

    try:
        if todo:
            for i in range(min(MAX_INFLIGHT, len(todo))):
                ensure_submitted(i)
            n_total = len(todo)
            for idx in range(n_total):
                ensure_submitted(idx)
                tag, unit, buf, fq = submitted[idx]
                try:
                    fq.result()  # quant done (previous PPL overlapped it)
                except Exception as e:
                    print(f"[{idx + 1}/{n_total}] {tag} QUANT failed "
                          f"({type(e).__name__}), skipped", flush=True)
                    log_failure(tag, "quant", f"{type(e).__name__}", unit)
                    try:
                        os.remove(buf)
                    except OSError:
                        pass
                    ensure_submitted(idx + MAX_INFLIGHT)
                    del submitted[idx]
                    continue
                if not _looks_valid(buf):
                    print(f"[{idx + 1}/{n_total}] {tag} CORRUPT file, "
                          f"skipped", flush=True)
                    log_failure(tag, "corrupt", "bad magic/size", unit)
                    try:
                        os.remove(buf)
                    except OSError:
                        pass
                    ensure_submitted(idx + MAX_INFLIGHT)
                    del submitted[idx]
                    continue
                try:
                    p, k, extra = run_ppl(buf, args.ppl_ctx, args.ppl_n,
                                    args.wiki_file, kb, timeout=args.timeout,
                                    ngl=args.ngl)
                except Exception as e:
                    print(f"[{idx + 1}/{n_total}] {tag} PPL failed "
                          f"({type(e).__name__}: {str(e)[:100]}), skipped",
                          flush=True)
                    log_failure(tag, "ppl", f"{type(e).__name__}: {str(e)[:100]}", unit)
                    ensure_submitted(idx + MAX_INFLIGHT)
                    del submitted[idx]
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
                    for ek in ("kld_max", "kld_median", "dp_rms", "same_top"):
                        if ek in extra and extra[ek] is not None:
                            rec[ek] = extra[ek]
                            b = (base_extra or {}).get(ek)
                            if b is not None:
                                rec["damage_" + ek] = extra[ek] - b
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[{idx + 1}/{n_total}] {tag} "
                      f"damage={p - base_ppl:+.4f}" +
                      (f" kld={k - base_kld:+.4f}" if kb else ""), flush=True)
                ensure_submitted(idx + MAX_INFLIGHT)
                del submitted[idx]
    finally:
        ex.shutdown(wait=False)


if __name__ == "__main__":
    main()
