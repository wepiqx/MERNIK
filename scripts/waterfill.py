#!/usr/bin/env python3
"""Water-filling allocator prototype.

Minimizes  sum_i G_i * 2^(-alpha * (b_i - b0_i))  over per-group tier choices,
subject to the same pinned costs and analytic byte accounting as the greedy
classifier. G_i is the Gnom predicted damage at the Q4_K reference; negative
predictions are treated as zero (no paid upgrade).

    python scripts/waterfill.py \
      --model /mnt/Vsio/assets/models/Ling-3.0-tiny-bf16.gguf \
      --imatrix /mnt/Vsio/shq-program/models/Ling-3.0-tiny-imatrix.gguf \
      --netpred models/netpred_ling_groups.json \
      --sizes 4500,5100,6500,9000 --alpha 1.0
"""
import argparse
import json
import math
import os
import shlex
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_inspect import read_model
from imatrix_reader import read_imatrix, build_importance_table, detect_tied_groups, combine_imatrix
from classifier import build_groups, _f32_map, _size_mib, _base_floor, _tier_index, _tier_at, K_QUANTS, MOE_PAD_TYPES
from constants import (
    TIER_ORDER, TIER_BPW, MTP_DEPLOY_TIER, EMBD_DEPLOY_TIER,
    EMBD_PIN_TYPES, ROUTER_PIN_TYPES, get_tensor_class, get_tensor_type,
    is_mtp_tensor,
)
import gnom
from config_generator import generate_flags
from quantizer import run_dry_run_ex
import preflight


def _model_groups(model, imatrix, netpred):
    features = model.get("features", {})
    has_mtp = features.get("has_mtp", False)
    n_layers = features.get("n_layers", 31)
    is_qat = features.get("is_qat", False)
    tensors = model.get("tensors", {})
    ne_map = {k: v["n_elements"] for k, v in tensors.items()}

    moe_d_ff = features.get("moe_intermediate_size", 0)
    padded_ne_map = dict(ne_map)
    if moe_d_ff > 0 and moe_d_ff % 256 != 0:
        aligned = ((moe_d_ff + 255) // 256) * 256
        for name, n_el in ne_map.items():
            ttype = get_tensor_type(name)
            if ttype in MOE_PAD_TYPES:
                padded_ne_map[name] = (n_el // moe_d_ff) * aligned

    f32_map = _f32_map(tensors)
    all_names = set(ne_map)
    mtp_names = {n for n in all_names if is_mtp_tensor(n, n_layers)} if has_mtp else set()
    rest = all_names - mtp_names
    embd_names = {n for n in rest if get_tensor_type(n) in EMBD_PIN_TYPES}
    rest -= embd_names
    router_names = {n for n in rest if get_tensor_type(n) in ROUTER_PIN_TYPES}
    rest -= router_names
    one_d_names = {n for n in rest if f32_map.get(n, False)}
    flex_names = rest - one_d_names

    group_registry = build_groups(
        detect_tied_groups(imatrix), flex_names, ne_map, padded_ne_map,
        f32_map=f32_map)

    fixed_cost = (
        sum(_size_mib(MTP_DEPLOY_TIER, ne_map.get(n, 0), f32_map.get(n, False)) for n in mtp_names)
        + sum(_size_mib(EMBD_DEPLOY_TIER, ne_map.get(n, 0), f32_map.get(n, False)) for n in embd_names)
        + sum(_size_mib("F16", ne_map.get(n, 0), f32_map.get(n, False)) for n in router_names)
        + sum(_size_mib("F32", ne_map.get(n, 0), True) for n in one_d_names)
    )
    return {
        "ne_map": ne_map, "padded_ne_map": padded_ne_map, "f32_map": f32_map,
        "mtp_names": mtp_names, "embd_names": embd_names,
        "router_names": router_names, "one_d_names": one_d_names,
        "flex_names": flex_names, "group_registry": group_registry,
        "fixed_cost": fixed_cost, "is_qat": is_qat,
        "importance": build_importance_table(imatrix, model),
    }


def _group_meta(g, importance, netpred, alpha, is_qat=False, max_tier=None):
    names, n_el, n_pad, f32 = g
    rep = names[0]
    info = importance.get(rep, {})
    ttype = info.get("type", get_tensor_type(rep))
    cls = get_tensor_class(ttype)
    floor = _base_floor(ttype, cls, False, "importance_mean" in info,
                        is_qat=is_qat)
    ceil = max_tier or "F16"
    lo = _tier_index(floor)
    hi = min(_tier_index(ceil), len(TIER_ORDER) - 1)
    ladder = TIER_ORDER[lo:hi + 1]
    gtag = gnom.group_tag(names)
    raw_g = gnom.table(netpred).get(gtag, 0.0)
    return {
        "names": names, "n_el": n_el, "n_pad": n_pad, "f32": f32,
        "rep": rep, "tag": gtag, "raw_g": raw_g, "g": max(0.0, raw_g),
        "ladder": ladder, "floor": floor, "ceiling": ceil,
        "b0": TIER_BPW[floor],
    }


def _cost(gm, tier):
    n = gm["n_pad"] if tier in K_QUANTS else gm["n_el"]
    return _size_mib(tier, n, gm["f32"])


def _damage(gm, tier, alpha):
    if gm["g"] <= 0.0:
        return 0.0
    return gm["g"] * 2.0 ** (-alpha * (TIER_BPW[tier] - gm["b0"]))


def _choose(gms, lam, alpha):
    tiers, cost, damage = [], 0.0, 0.0
    for gm in gms:
        if gm["g"] <= 0.0:
            t = gm["floor"]
            tiers.append(t)
            cost += _cost(gm, t)
            continue
        best = None
        for t in gm["ladder"]:
            c = _cost(gm, t)
            d = _damage(gm, t, alpha)
            val = d + lam * c
            if best is None or val < best[0]:
                best = (val, c, d, t)
        _, c, d, t = best
        tiers.append(t)
        cost += c
        damage += d
    return tiers, cost, damage


def _waterfill(gms, budget, alpha):
    if not gms:
        return [], 0.0, 0.0
    if budget <= 0:
        raise SystemExit("effective budget is <= 0")
    max_cost = sum(_cost(g, g["ladder"][-1]) for g in gms)
    min_cost = sum(_cost(g, g["floor"]) for g in gms)
    if min_cost > budget + 1e-9:
        raise SystemExit(f"Q4 floor {min_cost:.0f} MiB exceeds effective budget {budget:.0f} MiB")
    if max_cost <= budget + 1e-9:
        tiers = [g["ladder"][-1] for g in gms]
        damage = sum(_damage(g, t, alpha) for g, t in zip(gms, tiers))
        return tiers, max_cost, damage

    lo, hi = 0.0, 1.0
    _, hi_cost, _ = _choose(gms, hi, alpha)
    while hi_cost > budget and hi < 1e12:
        hi *= 10.0
        _, hi_cost, _ = _choose(gms, hi, alpha)
    best = None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        tiers, cost, damage = _choose(gms, mid, alpha)
        if cost > budget:
            lo = mid
        else:
            hi = mid
            if best is None or cost > best[1]:
                best = (tiers, cost, damage)
    if best is None:
        tiers, cost, damage = _choose(gms, hi, alpha)
        best = (tiers, cost, damage)
    tiers, cost, damage = best

    while True:
        slack = budget - cost
        if slack <= 1e-9:
            break
        pick = None
        for i, gm in enumerate(gms):
            if gm["g"] <= 0.0:
                continue
            cur = tiers[i]
            ci = _tier_index(cur)
            for cand in gm["ladder"]:
                if _tier_index(cand) <= ci:
                    continue
                dc = _cost(gm, cand) - _cost(gm, cur)
                if dc <= 0 or dc > slack:
                    continue
                dd = _damage(gm, cur, alpha) - _damage(gm, cand, alpha)
                if dd <= 0:
                    continue
                util = dd / dc
                if pick is None or util > pick[0]:
                    pick = (util, i, cand, dc, dd)
        if pick is None:
            break
        _, i, cand, dc, dd = pick
        tiers[i] = cand
        cost += dc
        damage -= dd
    return tiers, cost, damage


def _norm_tier(t, gm):
    if t in gm["ladder"]:
        return t
    if t in TIER_BPW:
        return min(gm["ladder"], key=lambda x: abs(TIER_BPW[x] - TIER_BPW[t]))
    return gm["floor"]


def _load_baseline(path, gms):
    if not path or not os.path.exists(path):
        return None
    raw = json.load(open(path))
    assigns = raw.get("assignments", {})
    tiers = []
    for gm in gms:
        t = _norm_tier(assigns.get(gm["rep"]), gm)
        tiers.append(t)
    return {"path": path, "tiers": tiers, "raw": raw}


def _hist(gms, tiers):
    by_count, by_mib = {}, {}
    for gm, t in zip(gms, tiers):
        by_count[t] = by_count.get(t, 0) + 1
        by_mib[t] = by_mib.get(t, 0.0) + _cost(gm, t)
    return by_count, by_mib


def _run_size(args, model, groups, size):
    gms = [_group_meta(g, groups["importance"], args.netpred, args.alpha,
                       is_qat=groups["is_qat"], max_tier=args.max_tier)
           for g in groups["group_registry"].values()]
    budget = size - groups["fixed_cost"]
    tiers, cost, damage = _waterfill(gms, budget, args.alpha)
    base = _load_baseline(args.baseline_for(size), gms)
    base_damage = None
    moved = None
    if base:
        base_damage = sum(_damage(gm, t, args.alpha) for gm, t in zip(gms, base["tiers"]))
        moved = sum(1 for a, b in zip(tiers, base["tiers"]) if a != b)

    assignments = {}
    for gm, t in zip(gms, tiers):
        for n in gm["names"]:
            assignments[n] = t
    for n in groups["mtp_names"]:
        assignments[n] = MTP_DEPLOY_TIER
    for n in groups["embd_names"]:
        assignments[n] = EMBD_DEPLOY_TIER
    for n in groups["router_names"]:
        assignments[n] = "F16"
    for n in groups["one_d_names"]:
        assignments[n] = "F32"
    by_count, by_mib = _hist(gms, tiers)

    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "architecture": model.get("architecture"),
        "utility": "waterfill",
        "alpha": args.alpha,
        "max_tier": args.max_tier,
        "target_mib": size,
        "fixed_cost_mib": round(groups["fixed_cost"], 3),
        "group_cost_mib": round(cost, 3),
        "total_mib": round(cost + groups["fixed_cost"], 3),
        "predicted_damage": damage,
        "baseline_damage": base_damage,
        "baseline_moved_groups": moved,
        "baseline_path": base["path"] if base else None,
        "coverage": sum(1 for g in gms if g["raw_g"] != 0.0),
        "n_groups": len(gms),
        "by_tier_count": by_count,
        "by_tier_mib": {k: round(v, 3) for k, v in by_mib.items()},
        "assignments": assignments,
        "groups": [
            {"tag": g["tag"], "rep": g["rep"], "names": g["names"],
             "tier": t, "raw_g": g["raw_g"],
             "floor": g["floor"], "ceiling": g["ceiling"],
             "cost_mib": round(_cost(g, t), 3)}
            for g, t in zip(gms, tiers)
        ],
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--imatrix", action="append", required=True)
    ap.add_argument("--netpred", required=True)
    ap.add_argument("--sizes", required=True, help="comma-separated MiB targets")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-tier", default=None)
    ap.add_argument("--baseline-dir", default="provenance/alloc_baselines")
    ap.add_argument("--baseline-prefix", default=None)
    ap.add_argument("--out", default="provenance/waterfill")
    ap.add_argument("--dump-alloc", default=None, help="write a single-size payload here")
    ap.add_argument("--config-out", default=None, help="write llama-quantize flags/shell here")
    ap.add_argument("--dry-run", action="store_true", help="run llama-quantize --dry-run")
    ap.add_argument("--base-type", default="Q4_K")
    args = ap.parse_args()

    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    if len(sizes) != 1 and args.dump_alloc:
        raise SystemExit("--dump-alloc needs exactly one --sizes entry")
    model = read_model(args.model)
    imatrix = combine_imatrix([read_imatrix(p) for p in args.imatrix])
    groups = _model_groups(model, imatrix, args.netpred)
    prefix = args.baseline_prefix or os.path.splitext(os.path.basename(args.model))[0].lower().replace("-", "_")

    def baseline_for(size):
        stem = f"{prefix}_gnom_td_{int(size)}.json"
        return os.path.join(args.baseline_dir, stem)
    args.baseline_for = baseline_for

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for size in sizes:
        payload = _run_size(args, model, groups, size)
        stem = f"{prefix}_waterfill_a{args.alpha:g}_{int(size)}"
        path = os.path.join(args.out, stem + ".json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        if args.dump_alloc:
            with open(args.dump_alloc, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        if args.config_out or args.dry_run:
            flags = generate_flags(payload["assignments"], model, args.base_type, size)
            flags["imatrix"] = args.imatrix[0]
            if args.config_out:
                rules = []
                for r in flags["tensor_type_rules"]:
                    rules += ["--tensor-type", r.split(" ", 1)[1].strip('"')]
                cmd = (["llama-quantize", "--imatrix", args.imatrix[0],
                        "--output-tensor-type", flags["output_tensor_type"],
                        "--token-embedding-type", flags["token_embedding_type"]]
                       + rules
                       + [args.model, args.config_out.replace(".json", ".gguf"),
                          flags["base_type"]])
                with open(args.config_out, "w") as f:
                    json.dump({"flags": flags, "command": cmd,
                               "shell": " ".join(shlex.quote(x) for x in cmd)},
                              f, indent=2)
            if args.dry_run:
                real, raw = run_dry_run_ex(flags, args.model)
                report = preflight.audit(payload["assignments"], raw,
                                         target_mib=size,
                                         est_mib=payload["total_mib"])
                payload["dry_run"] = {
                    "real_mib": report["real_mib"],
                    "bpw": report["bpw"],
                    "hist_mib": {k: round(v, 3) for k, v in report["hist_mib"].items()},
                    "mismatch_count": len(report["mismatches"]),
                }
                print(f"       dry-run real={report['real_mib']:.0f} MiB "
                      f"mismatch={len(report['mismatches'])}")
                with open(path, "w") as f:
                    json.dump(payload, f, indent=2, sort_keys=True)
        low = sum(v for k, v in payload["by_tier_mib"].items()
                  if k in ("IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS"))
        rows.append((size, payload["total_mib"], payload["predicted_damage"],
                     payload["baseline_damage"], payload["baseline_moved_groups"], low))
        print(f"{int(size):5d}  total={payload['total_mib']:7.0f}  "
              f"D={payload['predicted_damage']:.6f}  "
              f"base={payload['baseline_damage'] if payload['baseline_damage'] is not None else float('nan'):.6f}  "
              f"moved={payload['baseline_moved_groups'] if payload['baseline_moved_groups'] is not None else -1:3d}  "
              f"subQ4={low:.0f} MiB")
        print("       " + " ".join(f"{k}:{v:.0f}" for k, v in
                                   sorted(payload["by_tier_mib"].items(), key=lambda kv: -kv[1])))

    if len(sizes) > 1:
        with open(os.path.join(args.out, f"{prefix}_waterfill_a{args.alpha:g}_index.json"), "w") as f:
            json.dump({"model": args.model, "alpha": args.alpha,
                       "sizes": sizes, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
