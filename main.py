#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

from llama_inspect import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from classifier import optimal_classify, optimal_classify_topdown, compute_stats
from config_generator import generate_flags, format_flags
from quantizer import run_dry_run_ex, run_quantization
from constants import get_tensor_class, get_tensor_type
from legacy_modes import LEGACY_MODES
import gnom
import preflight


def _get_base_type(model: dict) -> str:
    is_qat = model.get("features", {}).get("is_qat", False)
    return "IQ4_XS" if is_qat else "Q5_K_M"


def main():
    parser = argparse.ArgumentParser(
        description="MERNIK: imatrix-driven hybrid quantization"
    )
    parser.add_argument("--model", help="Source GGUF model path")
    parser.add_argument("--imatrix", action="append", default=[],
                        help="Imatrix GGUF path (can be specified multiple times)")
    parser.add_argument("--imatrix-method", choices=["max", "mean"], default="max",
                        help="Legacy combine mode for multiple --imatrix files; "
                             "only used with --legacy-modes "
                             "--imatrix-legacy-combine (default: max)")
    parser.add_argument("--imatrix-legacy-combine", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--imatrix-tol", type=float, default=0.01,
                        help="Max tolerated relative importance spread across "
                             "--imatrix files before the lens counts as "
                             "truncated (default 1%%)")
    parser.add_argument("--size", type=float, default=6800,
                        help="Target file size in MiB (default: 6800 = ~6.6 GB)")
    parser.add_argument("--output", default=None, help="Output GGUF path")
    parser.add_argument("--run", action="store_true", help="Execute quantization")
    parser.add_argument("--show-config", action="store_true", help="Print config and exit")
    parser.add_argument("--dump-alloc", default=None, metavar="PATH",
                        help="Write final allocation + real dry-run geometry as JSON")
    parser.add_argument("--allow-q3-or-lower", dest="allow_q3",
                        action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dont-allow-q3", dest="allow_q3",
                        action="store_false",
                        help="Keep the Q4_K floor even when --size is smaller; "
                             "fail instead of dropping to IQ2/IQ1")
    parser.set_defaults(allow_q3=True)
    parser.add_argument("--top-down", action="store_true",
                         help="Start quantizable 2D groups at F16 and greedily "
                              "downgrade to fit --size; pins and 1D F32 stay "
                              "outside the queue")
    parser.add_argument("--legacy-1d", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--free-pins", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--pin-small", type=float, default=0.0, metavar="MIB",
                        help=argparse.SUPPRESS)
    parser.add_argument("--max-tier", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--utility", default="mse",
                        help="Utility: mse | smse | smape | gnom (default: mse). "
                             "Gnom is the active allocation path; archived modes "
                             "require --legacy-modes.")
    parser.add_argument("--legacy-modes", action="store_true",
                        help="Unlock archived utility modes and flags. Frozen for "
                             "reproducibility; not the active path. See LEGACY.md.")
    parser.add_argument("--netpred", default=None,
                        help="Gnom predicted-damage table (required by --utility gnom)")
    parser.add_argument("--hinge-w", type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--huber-delta", type=float, default=3e-4,
                        help=argparse.SUPPRESS)
    parser.add_argument("--pow-alpha", type=float, default=0.5,
                        help=argparse.SUPPRESS)
    parser.add_argument("--ssim-table", default="models/ssim_table.npz",
                        help=argparse.SUPPRESS)
    parser.add_argument("--frag-w", type=float, default=0.5,
                        help=argparse.SUPPRESS)
    parser.add_argument("--mix-base", choices=["mse", "rmse", "smape", "logcosh"],
                        default="smape", help=argparse.SUPPRESS)
    parser.add_argument("--mix-top-frac", type=float, default=0.2,
                        help=argparse.SUPPRESS)
    parser.add_argument("--relief", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--relief-thr", type=float, default=-0.5,
                        help=argparse.SUPPRESS)
    parser.add_argument("--squeeze", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--invert", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--invert-keep", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--verify", choices=["gpqa", "he", "all"], default=None,
                        help="Slow-ring verify of the build (GPU queue: aborts "
                             "LOUDLY if the port is busy, never steals). "
                             "With --run: verify the fresh quant. Without --run: "
                             "verify-only on existing --output (no --model/--imatrix needed)")
    parser.add_argument("--verify-tag", default=None,
                        help="Result tag (default: from --output basename)")
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip the dry-run audit (intended-vs-actual tier "
                             "diff, real size, real geometry)")
    parser.add_argument("--strict-preflight", action="store_true",
                        help="Abort if the binary will not write the assigned "
                             "tiers (default: warn loudly and continue)")

    args = parser.parse_args()

    core = {"mse", "smse", "smape", "gnom"}
    aliases = {"netpure": "gnom", "netfd": "gnom"}
    if args.utility in aliases:
        print(f"DEPRECATED: --utility {args.utility} -> gnom")
        if args.utility == "netfd" and not args.top_down:
            print("  netfd is top-down by construction; enabling --top-down")
            args.top_down = True
        args.utility = aliases[args.utility]
    if args.utility not in core:
        if args.utility not in LEGACY_MODES:
            print(f"main.py: error: unknown utility {args.utility!r}")
            sys.exit(1)
        if not args.legacy_modes:
            print(f"main.py: error: --utility {args.utility} is archived. "
                  f"Pass --legacy-modes to use it (reproducibility only); "
                  f"the active path is --utility gnom.")
            sys.exit(1)

    legacy_used = []
    if args.legacy_1d:
        legacy_used.append("--legacy-1d")
    if args.squeeze:
        legacy_used.append("--squeeze")
    if args.invert or args.invert_keep:
        legacy_used.append("--invert/--invert-keep")
    if args.relief is not None:
        legacy_used.append("--relief")
    if args.hinge_w != 0.0:
        legacy_used.append("--hinge-w")
    if args.huber_delta != 3e-4:
        legacy_used.append("--huber-delta")
    if args.pow_alpha != 0.5:
        legacy_used.append("--pow-alpha")
    if args.ssim_table != "models/ssim_table.npz":
        legacy_used.append("--ssim-table")
    if args.frag_w != 0.5:
        legacy_used.append("--frag-w")
    if args.mix_base != "smape" or args.mix_top_frac != 0.2:
        legacy_used.append("--mix-*")
    if args.free_pins:
        legacy_used.append("--free-pins")
    if args.pin_small > 0.0:
        legacy_used.append("--pin-small")
    if args.max_tier is not None:
        legacy_used.append("--max-tier")
    if args.imatrix_legacy_combine:
        legacy_used.append("--imatrix-legacy-combine")
    if legacy_used and not args.legacy_modes:
        print(f"main.py: error: archived flag(s) {', '.join(legacy_used)}. "
              f"Pass --legacy-modes to unlock them.")
        sys.exit(1)
    if args.legacy_modes:
        print("=" * 68)
        print("LEGACY MODES ENABLED — archived paths, frozen for")
        print("reproducibility. The active line is Gnom:")
        print("  --utility gnom [--top-down] --netpred models/...json")
        print("=" * 68)
    if args.utility == "gnom":
        if not args.netpred:
            print("main.py: error: --utility gnom needs --netpred")
            sys.exit(1)
        if not os.path.exists(args.netpred):
            print(f"main.py: error: Gnom table not found: {args.netpred}")
            sys.exit(1)

    if args.verify and not args.run:
        # verify-only: score an existing build, skip the whole classify path
        if not args.output or not os.path.exists(args.output):
            print("main.py: error: verify-only needs existing --output")
            sys.exit(1)
        from verify import main_verify
        main_verify(args.output, args.verify, args.verify_tag)
        return

    if not args.model or not args.imatrix:
        parser.print_usage()
        print("main.py: error: --model and --imatrix are required")
        sys.exit(1)
    if args.squeeze and args.top_down:
        parser.print_usage()
        print("main.py: error: --squeeze is bottom-up only (no --top-down)")
        sys.exit(1)

    target_mib = args.size

    print("=== MERNIK ===")
    print(f"Model:   {args.model}")
    if len(args.imatrix) == 1:
        print(f"Imatrix: {args.imatrix[0]}")
    else:
        print(f"Imatrix: {len(args.imatrix)} files ({args.imatrix_method})")
        for p in args.imatrix:
            print(f"  - {p}")
    print(f"Target:  {target_mib:.0f} MiB ({target_mib / 1024:.2f} GB)")
    if not args.allow_q3:
        print("  --dont-allow-q3: Q4_K floor enforced")
    else:
        print("  floor: Q4_K when it fits; IQ2/IQ1 only when --size is smaller")
    if args.free_pins:
        print("  --free-pins: EXPERIMENTAL, pins disabled")
    print()

    print("[1/4] Reading model...")
    model = read_model(args.model)
    print(f"  Architecture: {model['architecture']}")
    print(f"  Tensors: {model['n_tensors']}")
    print(f"  Features: {json.dumps(model['features'], indent=2)}")

    print("\n[2/4] Reading imatrix...")
    imatrix_list = [read_imatrix(p) for p in args.imatrix]
    for im in imatrix_list:
        print(f"  {im['path']}: {im['n_tensors']} tensors, datasets={im['meta'].get('imatrix.datasets', '?')}")

    # The binary reads ONE imatrix and re-derives importance from it. So with
    # several files, either both sides use the same one, or the build quietly
    # mixes two lenses: the queue allocating on max(A,B) while the binary
    # rounds with A alone (that is what quantizer.py used to do, with only a
    # warning). Writing our own merged imatrix was rejected on purpose — the
    # metadata (imatrix.datasets/chunk_count/chunk_size) has to be byte-exact
    # or the binary refuses it, and a subtly wrong imatrix degrades every
    # weight with no error anywhere. Merge upstream instead.
    from imatrix_reader import combine_imatrix, imatrix_divergence
    quant_imatrix = list(args.imatrix)
    if len(args.imatrix) > 1:
        div = imatrix_divergence(imatrix_list)
        worst = div[0][0] if div else 0.0
        print(f"  {len(args.imatrix)} imatrix files — llama-quantize accepts ONE. "
              f"Queue and binary will both use: {os.path.basename(args.imatrix[0])}")
        if worst > args.imatrix_tol:
            print(f"\n  !! LENS TRUNCATED: importance differs by up to "
                  f"{worst*100:.1f}% between files (tolerance "
                  f"{args.imatrix_tol*100:.1f}%).")
            for rel, name, b, v in div[:5]:
                print(f"     {name[:48]:48s} {b:12.1f} vs {v:12.1f}  ({rel*100:+.1f}%)")
            print("     The extra lenses will NOT reach the binary. To combine "
                  "them for real:\n       llama-imatrix -o merged.gguf --in-file A "
                  "--in-file B <model> -ngl 99\n     then pass --imatrix "
                  "merged.gguf alone.")
            if not args.imatrix_legacy_combine:
                print("     Refusing to build a split-brain file. "
                      "(--legacy-modes --imatrix-legacy-combine to override)")
                sys.exit(1)
            print("     --imatrix-legacy-combine: continuing anyway (legacy "
                  "behaviour, kept for reproducing 2026-09 builds)")
        else:
            print(f"  lenses agree within {worst*100:.2f}% — using one file for "
                  f"both sides loses nothing")
        if args.imatrix_legacy_combine:
            imatrix = combine_imatrix(imatrix_list, method=args.imatrix_method)
            print(f"  Combined for the queue ({args.imatrix_method}): "
                  f"{imatrix['n_tensors']} tensors")
        else:
            imatrix = combine_imatrix(imatrix_list[:1])
    else:
        imatrix = combine_imatrix(imatrix_list, method=args.imatrix_method)
        print(f"  Combined: {imatrix['n_tensors']} tensors")

    print("\n[3/4] Detecting tied groups...")
    tied_groups = detect_tied_groups(imatrix)
    print(f"  Found {len(tied_groups)} tied groups:")
    for g in tied_groups:
        if len(g) > 1:
            print(f"    TIED ({len(g)}): {g[0].replace('.weight', '')}  =  "
                  f"{g[1].replace('.weight', '')}")

    imp_table = build_importance_table(imatrix, model)
    if args.invert:
        # Adversarial control (operator's idea): the ranking runs upside
        # down — junk outbids kings for every rung. Tied groups, pins and
        # floors stay structural; only priority inverts. --invert-keep K
        # exempts the top-K groups (soft invert). Dedupe by id: the table
        # holds alias keys sharing one dict, double negation would cancel.
        keep = set()
        kept_groups = 0
        if args.invert_keep > 0:
            scored = []
            for _g in tied_groups:
                _vals = [imp_table[_n]["importance_mean"] for _n in _g
                         if _n in imp_table
                         and "importance_mean" in imp_table[_n]]
                if _vals:
                    scored.append((max(_vals), _g))
            scored.sort(reverse=True)
            for _, _g in scored[:args.invert_keep]:
                keep.update(_g)
            kept_groups = min(args.invert_keep, len(scored))
        _seen = set()
        for _n, _d in imp_table.items():
            if "importance_mean" in _d and _n not in keep \
                    and id(_d) not in _seen:
                _d["importance_mean"] = -_d["importance_mean"]
                _seen.add(id(_d))
        print(f"  INVERTED (top-{kept_groups} groups exempt): "
              "kings beg, junk builds skyscrapers")

    print("\n[4/4] Classifying tensors (%s)..." %
          ("top-down from F16" if args.top_down else "greedy imatrix-driven"))

    # Get both the tier map and the exact padding map from the classifier
    classify = optimal_classify_topdown if args.top_down else optimal_classify
    uopt = {"mode": args.utility, "top_down": args.top_down,
            "huber_delta": args.huber_delta, "ssim_table": args.ssim_table,
            "frag_w": args.frag_w, "netpred": args.netpred,
            "mix_base": args.mix_base, "pow_alpha": args.pow_alpha,
            "hinge_w": args.hinge_w}
    if args.utility == "mix":
        # kings = top-frac groups by max member importance; rep = g[0]
        # (same rep _gain sees). Ranked once here, not per call.
        scored = []
        for g in tied_groups:
            best = max(imp_table[n]["importance_mean"]
                       for n in g if n in imp_table)
            scored.append((best, g[0]))
        scored.sort(reverse=True)
        n_kings = max(1, int(len(scored) * args.mix_top_frac))
        uopt["mix_top"] = frozenset(rep for _, rep in scored[:n_kings])
        print(f"  MIX utility: {n_kings}/{len(scored)} king groups by mse, "
              f"rest by {args.mix_base}")
    relief = None
    if args.relief:
        relief = {}
        with open(args.relief) as f:
            for line in f:
                d = json.loads(line)
                if d["unit"] != "BASELINE":
                    relief[d["unit"]] = d["damage"]
        print(f"  BATTLEFIELD relief: {sum(1 for v in relief.values() if v < args.relief_thr)} groups pinned at Q4")
    if args.utility != "mse":
        tag = "LEGACY" if args.utility in LEGACY_MODES else "GNOM"
        print(f"  {tag}: {uopt}")
    if args.utility == "gnom":
        tags = [gnom.group_tag(g) for g in tied_groups]
        hit, total = gnom.coverage(args.netpred, tags)
        print(f"  Gnom coverage: {hit}/{total} tied groups in {args.netpred}")
        if hit < total:
            print(f"  WARNING: {total - hit} group(s) have no prediction; "
                  f"they are demoted first / never upgraded")
    assignments, padded_ne_map = classify(
        imp_table, tied_groups, model,
        target_size_mib=target_mib,
        allow_q3=args.allow_q3,
        free_pins=args.free_pins,
        uopt=uopt,
        legacy_1d=args.legacy_1d,
        **({"ceiling": args.max_tier} if (args.max_tier and not args.top_down) else {}),
        **({} if args.top_down else {
            "relief": relief,
            "relief_thr": args.relief_thr,
            "allowed_tiers": (["IQ1_S", "F16"] if args.squeeze else None),
            "pin_small": args.pin_small,
        }),
    )
    if args.squeeze:
        print("  SQUEEZE mode: dungeon IQ1_S or palace F16, nothing between")
    ne_map = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}
    for tname, info in imp_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # 1D tensors (norms, biases) are F32 in the artifact whatever the tier
    # says. Loud about it, because --pin-norms and the old norms-shield law
    # were both reasoning about precision the binary never spends.
    from classifier import _f32_map
    f32_map = _f32_map(model.get("tensors", {}))
    n_f32 = sum(1 for v in f32_map.values() if v)
    if n_f32 and not args.legacy_1d:
        mib_f32 = sum(ne_map.get(k, 0) for k, v in f32_map.items() if v) * 4 / 1024 / 1024
        print(f"  physical F32: {n_f32} tensors pinned outside the budget "
              f"({mib_f32:.1f} MiB) — 1D tensors and ssm_conv1d, which "
              f"llama.cpp never quantizes\n"
              f"    (use --legacy-1d to reproduce pre-2026-09-28 budgets)")

    # Pass padded_ne_map so the on-screen summary matches the real size
    _show_tier_summary(assignments, imp_table, ne_map, padded_ne_map, f32_map)

    from classifier import _LAST_RUN_INFO
    if _LAST_RUN_INFO.get("polish_steps"):
        print(f"  budget tail: {_LAST_RUN_INFO['polish_steps']} multi-rung step(s) "
              f"recovered, {_LAST_RUN_INFO['slack_mib']:.0f} MiB still unspent "
              f"(no ladder rung fits — the geometry, not a silent drop)")
    if not args.top_down:
        _show_ceiling(model, ne_map, f32_map, target_mib)

    # Fail fast: even the base floors don't fit the budget — no valid
    # config exists (greedy only upgrades, never downgrades).
    stats = compute_stats(assignments, ne_map, padded_ne_map, f32_map)
    if stats["total_mib"] > target_mib:
        hint = "Raise --size." if args.allow_q3 \
            else "Raise --size or drop --dont-allow-q3."
        print(f"\nERROR: base size {stats['total_mib']:.0f} MiB already exceeds "
              f"target {target_mib:.0f} MiB. " + hint)
        sys.exit(1)

    base_type = _get_base_type(model)
    flags = generate_flags(assignments, model, base_type, target_mib)
    flags["imatrix"] = quant_imatrix

    print(f"\nConfig (base={flags['base_type']}):")
    print(format_flags(flags))

    if args.show_config:
        if args.dump_alloc:
            _write_alloc_dump(args, model, assignments, stats, None)
        return

    print("\n--- Dry Run ---")
    dry_size, dry_raw = run_dry_run_ex(flags, args.model)
    _show_size_result(dry_size, target_mib)

    # Preflight: the dry run already told us what the binary will ACTUALLY
    # write, per tensor. It used to be discarded here. Free audit, and the
    # only place a shadowed --tensor-type rule can be caught before 30 min
    # of quantization and 6 GB of disk.
    report = None
    if not args.no_preflight and dry_raw:
        report = preflight.audit(assignments, dry_raw,
                                 target_mib=target_mib,
                                 est_mib=stats["total_mib"])
        if report["mismatches"] and args.strict_preflight:
            print("\nABORT: --strict-preflight — the binary disagrees with the "
                  "assignment map. Fix the rules (or drop the flag) before "
                  "spending the quantization.")
            sys.exit(1)

    if args.dump_alloc:
        _write_alloc_dump(args, model, assignments, stats, report)

    if not args.run:
        print("\nDry run only. Use --run to execute quantization.")
        return

    if not args.output:
        base = os.path.splitext(os.path.basename(args.model))[0]
        args.output = base + "-MERNIK.gguf"

    print(f"\n--- Running quantization: {args.output} ---")
    success = run_quantization(flags, args.model, args.output)
    if success:
        print("Done!")
    else:
        print("Failed!")
        sys.exit(1)

    if args.verify:
        from verify import main_verify
        main_verify(args.output, args.verify, args.verify_tag)


def _write_alloc_dump(args, model, assignments, stats, report):
    """Record the allocation snapshot for greedy-vs-waterfill comparisons."""
    real = None
    if report:
        real = {
            "real_mib": report["real_mib"],
            "bpw": report["bpw"],
            "hist_count": dict(report["hist_count"]),
            "hist_mib": {k: round(v, 3) for k, v in report["hist_mib"].items()},
            "mismatch_count": len(report["mismatches"]),
            "physical_count": len(report["physical"]),
            "unassigned_count": len(report["unassigned"]),
        }
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "model_size_bytes": os.path.getsize(args.model),
        "architecture": model.get("architecture"),
        "n_tensors": model.get("n_tensors"),
        "features": model.get("features", {}),
        "imatrix": args.imatrix,
        "utility": args.utility,
        "top_down": bool(args.top_down),
        "allow_q3": bool(args.allow_q3),
        "target_mib": args.size,
        "stats": {
            "by_tier_count": stats["by_tier_count"],
            "by_tier_mib": {k: round(v, 3) for k, v in stats["by_tier_mib"].items()},
            "total_mib": round(stats["total_mib"], 3),
            "tensor_count": stats["tensor_count"],
        },
        "real": real,
        "assignments": assignments,
        "command": sys.argv,
    }
    path = os.path.abspath(args.dump_alloc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"  allocation dump: {path}")


def _show_ceiling(model, ne_map, f32_map, target_mib):
    """Loud when --size is above what the tier policy can physically reach.

    Bottom-up caps every class at CLASS_MAX_TIER (Q8_0 for projections and
    FFN), so there is a hard ceiling on the output size: MiniCPM5-2B
    saturates at 2359.4 MiB, and --size 2600 silently produced a 2359 MiB
    file. On an 8 GB card that is the difference between a model that loads
    and one that does not.
    """
    from constants import CLASS_MAX_TIER, EMBD_DEPLOY_TIER, TIER_BPW, \
        get_tensor_class, get_tensor_type, EMBD_PIN_TYPES
    total, by_class = 0.0, {}
    for n, n_el in ne_map.items():
        ttype = get_tensor_type(n)
        if f32_map.get(n, False):
            tier = "F32"
        elif ttype in EMBD_PIN_TYPES:
            tier = EMBD_DEPLOY_TIER
        else:
            tier = CLASS_MAX_TIER.get(get_tensor_class(ttype), "Q8_0")
        bpw = 32.0 if tier == "F32" else TIER_BPW.get(tier, 0.0)
        total += n_el * bpw / 8 / 1024 / 1024
        by_class[get_tensor_class(ttype)] = by_class.get(get_tensor_class(ttype), 0) + \
            n_el * bpw / 8 / 1024 / 1024
    if target_mib and total < target_mib - 0.5:
        print(f"\n  !! SIZE ABOVE THE TIER CEILING: policy maxes out at "
              f"{total:.0f} MiB, you asked for {target_mib:.0f} MiB.")
        print(f"     The build will be ~{target_mib - total:.0f} MiB smaller than "
              f"requested, no matter what the queue does.")
        top = sorted(by_class.items(), key=lambda kv: -kv[1])[:3]
        print("     blocked by: " + ", ".join(f"{c} @ {CLASS_MAX_TIER.get(c, 'Q8_0')}"
                                              for c, _ in top))
        print("     Raise CLASS_MAX_TIER in constants.py if you really want the "
              "bytes (F16 costs 2x Q8_0).")
    return total


def _show_tier_summary(assignments, imp_table, ne_map, padded_ne_map=None, f32_map=None):
    stats = compute_stats(assignments, ne_map, padded_ne_map, f32_map)
    
    print("\n  Tier distribution:")
    for tier in sorted(stats["by_tier_count"].keys()):
        count = stats["by_tier_count"][tier]
        mib = stats["by_tier_mib"].get(tier, 0.0)
        print(f"    {tier}: {count} tensors ({mib:.1f} MiB)")
    print(f"  Total estimated size: {stats['total_mib']:.1f} MiB")

    by_class = defaultdict(Counter)
    for name, tier in assignments.items():
        by_class[get_tensor_class(get_tensor_type(name))][tier] += 1
    print("\n  Class x tier:")
    for cls in sorted(by_class):
        parts = " ".join(f"{t}x{c}" for t, c in sorted(by_class[cls].items()))
        print(f"    {cls:<14} {parts}")

    ranked = sorted(
        [(n, v) for n, v in imp_table.items()],
        key=lambda x: -x[1]["importance_mean"],
    )
    print("\n  Top 10 by importance:")
    for n, v in ranked[:10]:
        tier = assignments.get(n, "base")
        display = n.replace(".weight", "").replace(".bias", "")
        print(f"    {display[:52]:52s} imp={v['importance_mean']:10.0f} tier={tier}")


def _show_size_result(dry_size, target_mib):
    if dry_size:
        print(f"  Estimated size: {dry_size:.0f} MiB ({dry_size / 1024:.2f} GB)")
        diff = dry_size - target_mib
        if diff > 0:
            print(f"  ⚠  Over target by {diff:.0f} MiB")
        else:
            print(f"  ✓ Under target by {-diff:.0f} MiB")
    else:
        print("  ⚠  Could not parse size from dry-run output")


if __name__ == "__main__":
    main()
