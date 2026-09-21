#!/usr/bin/env python3

import argparse
import json
import os
import sys

from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from classifier import optimal_classify, optimal_classify_topdown, compute_stats
from config_generator import generate_flags, format_flags
from quantizer import run_dry_run, run_quantization
from constants import CLASS_HARD_FLOORS


def _get_base_type(model: dict) -> str:
    is_qat = model.get("features", {}).get("is_qat", False)
    return "IQ4_XS" if is_qat else "Q5_K_M"


def main():
    parser = argparse.ArgumentParser(
        description="MERNIK: imatrix-driven hybrid quantization"
    )
    parser.add_argument("--model", help="BF16 GGUF model path")
    parser.add_argument("--imatrix", action="append", default=[],
                        help="Imatrix GGUF path (can be specified multiple times)")
    parser.add_argument("--imatrix-method", choices=["max", "mean"], default="max",
                        help="How to combine multiple imatrix: max (conservative) or mean (default: max)")
    parser.add_argument("--size", type=float, default=6800,
                        help="Target file size in MiB (default: 6800 = ~6.6 GB)")
    parser.add_argument("--output", default=None, help="Output GGUF path")
    parser.add_argument("--run", action="store_true", help="Execute quantization")
    parser.add_argument("--show-config", action="store_true", help="Print config and exit")
    parser.add_argument("--verbose", action="store_true", help="Detailed output")
    parser.add_argument("--allow-q3-or-lower", action="store_true",
                        help="Allow Q3_K for low-importance tensors (risk of quality loss)")
    parser.add_argument("--top-down", action="store_true",
                         help="Start all quantizable tensors (norms included) at F16 "
                              "and greedily downgrade to fit --size")
    parser.add_argument("--pin-norms", action="store_true",
                        help="Top-down native norms shield: keep norms/small "
                             "tensors at F16 outside the budget")
    parser.add_argument("--free-pins", action="store_true",
                         help="EXPERIMENTAL: let output/token_embd/MTP/routers fight "
                              "for budget instead of fixed pins")
    parser.add_argument("--utility", choices=["mse", "rmse", "hybrid", "huber", "logcosh", "smape", "ssim", "smape_ssim", "smape_frag", "pw_ssim", "netdmg", "mix"],
                        default="mse",
                        help="BATTLEFIELD: utility metric for the queue "
                             "(default: mse)")
    parser.add_argument("--cv-w", type=float, default=1.0,
                        help="BATTLEFIELD hybrid: concentration discount weight")
    parser.add_argument("--linf-w", type=float, default=1.0,
                        help="BATTLEFIELD hybrid: worst-case-error boost weight")
    parser.add_argument("--huber-delta", type=float, default=3e-4,
                        help="BATTLEFIELD huber/logcosh: scale splitting small vs large deltas")
    parser.add_argument("--ssim-table", default="models/ssim_table.npz",
                        help="BATTLEFIELD ssim: path to ssim_probe.py table")
    parser.add_argument("--frag-w", type=float, default=0.5,
                        help="BATTLEFIELD smape_frag: fragility modulation weight")
    parser.add_argument("--ptable", default="models/ssim_ptable.npz",
                        help="BATTLEFIELD pw_ssim: perceptual damage table")
    parser.add_argument("--netpred", default="models/netpred.json",
                        help="BATTLEFIELD netdmg: Gnom predicted damage per group")
    parser.add_argument("--netdmg-w", type=float, default=0.5)
    parser.add_argument("--mix-base", choices=["mse", "rmse", "smape", "logcosh"],
                        default="smape",
                        help="MIX utility: gain for non-king groups "
                             "(kings always use mse)")
    parser.add_argument("--mix-top-frac", type=float, default=0.2,
                        help="MIX utility: fraction of top-importance groups "
                             "treated as kings (default: 0.2)")
    parser.add_argument("--relief", default=None,
                        help="BATTLEFIELD: damage.jsonl with measured relief; "
                             "groups below --relief-thr get ceiling Q4")
    parser.add_argument("--relief-thr", type=float, default=-0.5)
    parser.add_argument("--aggro", type=float, default=None,
                        help="[deprecated] Use --size instead")
    parser.add_argument("--verify", choices=["gpqa", "he", "all"], default=None,
                        help="Slow-ring verify of the build (GPU queue: aborts "
                             "LOUDLY if the port is busy, never steals). "
                             "With --run: verify the fresh quant. Without --run: "
                             "verify-only on existing --output (no --model/--imatrix needed)")
    parser.add_argument("--verify-tag", default=None,
                        help="Result tag (default: from --output basename)")
    parser.add_argument("--show-floors", action="store_true",
                        help="Print class hard floors and exit")

    args = parser.parse_args()

    if args.show_floors:
        _show_floors()
        return

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
    if args.allow_q3_or_lower:
        print("  --allow-q3-or-lower: low-importance tensors may go to Q3_K")
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

    from imatrix_reader import combine_imatrix
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

    print("\n[4/4] Classifying tensors (%s)..." %
          ("top-down from F16" if args.top_down else "greedy imatrix-driven"))

    # Получаем и маппинг тиров, и точную карту паддингов напрямую из классификатора
    classify = optimal_classify_topdown if args.top_down else optimal_classify
    uopt = {"mode": args.utility, "cv_w": args.cv_w, "linf_w": args.linf_w,
            "huber_delta": args.huber_delta, "ssim_table": args.ssim_table,
            "frag_w": args.frag_w, "ptable": args.ptable,
            "netpred": args.netpred, "netdmg_w": args.netdmg_w,
            "mix_base": args.mix_base}
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
        print(f"  BATTLEFIELD utility: {uopt}")
    assignments, padded_ne_map = classify(
        imp_table, tied_groups, model,
        target_size_mib=target_mib,
        allow_q3=args.allow_q3_or_lower,
        free_pins=args.free_pins,
        uopt=uopt,
        **({"pin_norms": args.pin_norms} if args.top_down else {
            "relief": relief,
            "relief_thr": args.relief_thr,
        }),
    )
    
    ne_map = {k: v["n_elements"] for k, v in model.get("tensors", {}).items()}
    for tname, info in imp_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # Передаем padded_ne_map для корректного вывода логов на экран
    _show_tier_summary(assignments, imp_table, ne_map, padded_ne_map)

    # Fail fast: even the base floors don't fit the budget — no valid
    # config exists (greedy only upgrades, never downgrades).
    stats = compute_stats(assignments, ne_map, padded_ne_map)
    if stats["total_mib"] > target_mib:
        hint = "Raise --size." if (args.top_down and args.allow_q3_or_lower) \
            else "Raise --size or pass --allow-q3-or-lower."
        print(f"\nERROR: base size {stats['total_mib']:.0f} MiB already exceeds "
              f"target {target_mib:.0f} MiB. " + hint)
        sys.exit(1)

    base_type = _get_base_type(model)
    flags = generate_flags(assignments, model, base_type, target_mib)
    flags["imatrix"] = args.imatrix

    print(f"\nConfig (base={flags['base_type']}):")
    print(format_flags(flags))

    if args.show_config:
        return

    print("\n--- Dry Run ---")
    dry_size = run_dry_run(flags, args.model)
    _show_size_result(dry_size, target_mib)

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


def _show_tier_summary(assignments, imp_table, ne_map, padded_ne_map=None):
    stats = compute_stats(assignments, ne_map, padded_ne_map)
    
    print("\n  Tier distribution:")
    for tier in sorted(stats["by_tier_count"].keys()):
        count = stats["by_tier_count"][tier]
        mib = stats["by_tier_mib"].get(tier, 0.0)
        print(f"    {tier}: {count} tensors ({mib:.1f} MiB)")
    print(f"  Total estimated size: {stats['total_mib']:.1f} MiB")

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


def _show_floors():
    print("  Class hard floors (never below without --allow-q3-or-lower):\n")
    max_n = max(len(c) for c in CLASS_HARD_FLOORS)
    for cls, floor in sorted(CLASS_HARD_FLOORS.items()):
        print(f"    {cls:<{max_n}}  →  {floor}")
    print(f"\n  Default floor (unknown class): Q4_K")
    print(f"  --allow-q3-or-lower enables Q3_K for: ffn_down, ffn_gate, ffn_up, attn_output, ssm_out")


if __name__ == "__main__":
    main()
