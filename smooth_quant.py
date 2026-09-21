#!/usr/bin/env python3
"""smooth_quant: SmoothQuant-style preprocessing from in_sum2 (no extra calibration).

Per-input-channel scales s = rms_act^a / rms_w^(1-a) (geometric-mean
normalized), migrated offline:
  - linear input side:  W' = W * s          (exact)
  - compensation:       preceding RMSNorm g' = g / s, or paired-linear
                        output rows /s through SiLU (standard approx).
FFN-down input (post-SiLU, no norm) folds into up/gate output rows.
Attention Q/K outputs untouched (RoPE/softmax don't commute).

v1 scope: ffn_gate/up/down, attn_output, attn_qkv (input side only).
Verify numerically (block-output drift), then real quant duel.

Usage:
  python smooth_quant.py --model M-F16.gguf --imatrix M.imatrix.gguf \\
      --out M-SM.gguf --alpha 1.0 [--dry]
"""
import argparse
import os
import struct
import sys
import numpy as np
import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "scripts"))
from graft_mtp3 import pack_str, pack_field

FFN_TRIO = ("ffn_gate", "ffn_up", "ffn_down")


def layer_of(name):
    p = name.split(".")
    if len(p) >= 2 and p[0] in ("blk", "BLK") and p[1].isdigit():
        return int(p[1])
    return None


def base_of(name):
    """strip blk.N. prefix -> tensor kind, e.g. blk.3.ffn_gate.weight"""
    p = name.split(".")
    if len(p) >= 3 and p[0] in ("blk", "BLK") and p[1].isdigit():
        return ".".join(p[2:])
    return name


def preceding_norm(name, available):
    """norm tensor feeding this linear (arch-aware by availability)."""
    lyr = layer_of(name)
    if lyr is None:
        return None
    pre = f"blk.{lyr}."
    cands = []
    if "ffn_" in name:
        cands = [pre + "ffn_norm.weight", pre + "post_attention_norm.weight"]
    elif "attn_" in name or "qkv" in name:
        cands = [pre + "attn_norm.weight"]
    for c in cands:
        if c in available:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--skip-down", action="store_true",
                    help="forensics: skip ffn_down smoothing (isolate the "
                         "SiLU paired-fold path)")
    args = ap.parse_args()

    sys.path.insert(0, ".")
    from imatrix_reader import read_imatrix
    im = read_imatrix(args.imatrix)
    im_raw = {n: np.asarray(d["in_sum2_raw"], dtype=np.float64)
              for n, d in im["tensors"].items()
              if d.get("in_sum2_raw") is not None}

    r = gguf.GGUFReader(args.model)
    T = {t.name: t for t in r.tensors}
    from gguf.constants import GGMLQuantizationType as GQ
    assert all(int(t.tensor_type) in (int(GQ.F32), int(GQ.F16), int(GQ.BF16))
               for t in r.tensors if "norm" not in t.name), \
        "v1 needs F32/F16/BF16 source"

    import ml_dtypes

    def get(name):
        t = T[name]
        d = np.asarray(t.data)
        if d.dtype == np.uint8:  # BF16 raw
            return np.frombuffer(d.tobytes(),
                                 dtype=ml_dtypes.bfloat16).astype(np.float32)
        return np.asarray(d, dtype=np.float32)

    # plan: tensor -> (scales or None)
    # SHARED-INPUT RULE (scar 2026-09-19): gate+up read ONE norm output —
    # they MUST share a single joint s. Folding two different s into one
    # norm (v1 bug) breaks math in block 0 and compounds to PPL 523897.
    plan, folds = {}, {}
    joint_done = set()
    for name in T:
        b = base_of(name)
        if not any(k in b for k in ("ffn_gate", "ffn_up", "ffn_down",
                                    "attn_qkv")):
            continue
        if args.skip_down and "ffn_down" in b:
            continue
        # NOTE: attn_output excluded — softmax/V inside, no valid fold point
        if ".weight" not in name or name not in im_raw:
            continue
        W = get(name)
        e = im_raw[name]
        if e.shape[0] < W.shape[-1]:
            continue
        in_dim = W.shape[-1]
        rms_a = np.sqrt(np.maximum(e[:in_dim], 1e-12))
        rms_w = np.sqrt(np.maximum((W.astype(np.float64) ** 2).mean(axis=0),
                                   1e-24))
        s = (rms_a ** args.alpha) / (rms_w ** (1.0 - args.alpha) + 1e-24)
        s = s / np.exp(np.log(np.maximum(s, 1e-24)).mean())
        plan[name] = s
        # compensation target
        if b.startswith("ffn_down"):
            folds[name] = [n for n in
                           (f"blk.{layer_of(name)}.ffn_gate.weight",
                            f"blk.{layer_of(name)}.ffn_up.weight")
                           if n in T]
        else:
            nn = preceding_norm(name, T)
            folds[name] = [nn] if nn else []

    # joint resolution: gate+up share ONE input -> ONE s (geometric mean),
    # folded ONCE into their shared norm. v1 folded both -> PPL 523897.
    for name in list(plan):
        if ".ffn_gate.weight" not in name:
            continue
        up = name.replace("ffn_gate", "ffn_up")
        if up not in plan:
            continue
        sj = np.sqrt(np.maximum(plan[name], 1e-24) *
                     np.maximum(plan[up], 1e-24))
        plan[name] = sj
        plan[up] = sj
        # single fold target (shared norm); drop stale per-src entries
        nn = preceding_norm(name, T)
        folds[name] = [nn] if nn else []
        folds[up] = []
        joint_done.add(layer_of(name))

    print(f"smoothing plan: {len(plan)} tensors, alpha={args.alpha}, "
          f"joint gate+up blocks: {sorted(joint_done)}")
    if args.dry:
        for n in sorted(plan)[:8]:
            print("  smooth", n, "->", folds[n])
        return

    skipping = {n for n in plan if "attn_output" in base_of(n)}
    print(f"smoothing plan: {len(plan) - len(skipping)} tensors "
          f"(+folds), skipped: {len(skipping)} (attn_output)")

    # (drift check lives after transformed() def, before write)

    # STREAMING write (scar 2026-09-18: accumulating all arrays in a dict
    # ate 8 GB and OOM-killed the HE server mid-battery — one tensor in RAM
    # at a time from here on; plan/folds dicts hold only small scale vectors)
    def transformed(name):
        """Smoothed/folded array or None (= stream raw)."""
        t = T[name]
        d = np.asarray(t.data)
        if d.dtype == np.uint8:
            base = np.frombuffer(d.tobytes(),
                                 dtype=ml_dtypes.bfloat16).astype(np.float32)
        else:
            base = np.asarray(d, dtype=np.float32)
        b = base_of(name)
        out = None
        if name in plan and name not in skipping:
            W = base.astype(np.float64)
            s = plan[name]
            if W.shape[-1] == s.shape[0]:
                out = (W * s).astype(np.float32)
        for src, tgts in folds.items():
            if src in skipping or name not in tgts:
                continue
            s = plan[src].astype(np.float32)
            cur = base.astype(np.float32).copy() if out is None else out
            if "norm" in name:
                if cur.shape[0] == s.shape[0]:
                    cur = cur / s
            elif cur.shape[0] == s.shape[0]:
                cur = cur / s[:, None] if cur.ndim > 1 else cur / s
            out = cur
        return out

    # write copy with replaced tensors (F32 for norms stays F32; BF16 kept)
    parts = []
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue
        b = pack_field(k, f)
        if b is not None:
            parts.append(b)
    kv_raw = b"".join(parts)
    ti_raw, off, order = b"", 0, []
    for t in r.tensors:
        name = t.name
        dims = [int(d) for d in t.shape]
        ti_raw += pack_str(name)
        ti_raw += struct.pack("<I", len(dims))
        for d in dims:
            ti_raw += struct.pack("<Q", d)
        ti_raw += struct.pack("<I", int(t.tensor_type))
        ti_raw += struct.pack("<Q", off)
        ne = 1
        for d in dims:
            ne *= d
        bpe = 4 if int(t.tensor_type) == int(GQ.F32) else 2
        off += ne * bpe
        off += (32 - off % 32) % 32
        order.append(name)
    hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(order))
    hdr += struct.pack("<Q", len(parts)) + kv_raw + ti_raw
    hdr += b"\x00" * ((32 - len(hdr) % 32) % 32)
    # numeric drift check: full FFN block incl. norm, original vs
    # smoothed. Catches fold bugs BEFORE burning a quant (scar: PPL 523897
    # from double-folding one norm). Aborts the run if drift > 1e-2.
    def silu(x):
        return x / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def rmsnorm(x, g):
        return x / max(np.sqrt((x ** 2).mean()), 1e-12) * g

    drifts = []
    for L in sorted({layer_of(n) for n in plan
                     if layer_of(n) is not None})[:3]:
        names = {base_of(n): n for n in T if layer_of(n) == L}
        need = ["ffn_gate.weight", "ffn_up.weight"] + (
            [] if args.skip_down else ["ffn_down.weight"])
        nn = None
        for cand in (f"blk.{L}.ffn_norm.weight",
                     f"blk.{L}.post_attention_norm.weight"):
            if cand in T:
                nn = cand
                break
        if not all(k in names for k in need) or nn is None:
            continue
        go, uo = (get(names[k]).astype(np.float64) for k in need[:2])
        do = get(names[need[2]]).astype(np.float64) if len(need) > 2 else None
        gnorm = get(nn).astype(np.float64)
        tr = [transformed(names[k]) for k in need] + [transformed(nn)]
        if any(v is None for v in tr):
            print(f"  blk.{L}: transform missing — skip drift")
            continue
        if len(need) > 2:
            gs, us, ds, gn = (np.asarray(v, dtype=np.float64) for v in tr)
        else:
            gs, us, gn = (np.asarray(v, dtype=np.float64) for v in tr)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(go.shape[-1]).astype(np.float64)
        # layout: numpy (out,in) — verify by construction below
        if not (go.shape[-1] == x.shape[0]):
            print(f"  blk.{L}: layout mismatch {go.shape} vs x — abort math")
            continue
        x0 = rmsnorm(x, gnorm)
        h0 = silu(go @ x0) * (uo @ x0)
        x1 = rmsnorm(x, gn)
        h1 = silu(gs @ x1) * (us @ x1)
        if do is not None:
            ds = np.asarray(transformed(names[need[2]]), dtype=np.float64)
            y0, y1 = do @ h0, ds @ h1
        else:
            y0, y1 = h0, h1  # skip-down: compare pre-down activations
        drifts.append(float(np.abs(y1 - y0).max() /
                            max(np.abs(y0).max(), 1e-12)))
    if drifts:
        print(f"  block drift max-rel-err: {max(drifts):.2e} "
              f"(mean {sum(drifts) / len(drifts):.2e})")
        if max(drifts) > 1e-2:
            print("  DRIFT TOO BIG — aborting, fix folds first")
            sys.exit(2)

    with open(args.out, "wb") as fout:
        fout.write(hdr)
        for name in order:
            t = T[name]
            arr = transformed(name)
            if arr is not None:
                tt = int(t.tensor_type)
                if tt == int(GQ.BF16):
                    raw = np.ascontiguousarray(
                        arr).astype(ml_dtypes.bfloat16).tobytes()
                elif tt == int(GQ.F16):
                    raw = np.ascontiguousarray(
                        arr).astype(np.float16).tobytes()
                else:
                    raw = np.ascontiguousarray(
                        arr).astype(np.float32).tobytes()
            else:
                raw = np.ascontiguousarray(t.data).tobytes()
            fout.write(raw)
            fout.write(b"\x00" * ((32 - len(raw) % 32) % 32))
    print("wrote", args.out, os.path.getsize(args.out))


if __name__ == "__main__":
    main()
