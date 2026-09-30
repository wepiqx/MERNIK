#!/usr/bin/env python3

import argparse
import json
import os
import sys

from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from classifier import optimal_classify, optimal_classify_topdown, compute_stats
from config_generator import generate_flags, format_flags
from quantizer import run_dry_run_ex, run_quantization
from constants import CLASS_HARD_FLOORS
import preflight


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
    parser.add_argument("--imatrix-legacy-combine", action="store_true",
                        help="LEGACY: with several --imatrix files, allocate on "
                             "the max/mean combination but hand llama-quantize "
                             "only the first one (the two importance sources "
                             "then disagree inside a single build)")
    parser.add_argument("--imatrix-tol", type=float, default=0.01,
                        help="Max tolerated relative importance spread across "
                             "--imatrix files before the lens counts as "
                             "truncated (default 1%%)")
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
    parser.add_argument("--legacy-1d", action="store_true",
                        help="LEGACY: treat 1D tensors (norms/biases) and "
                             "ssm_conv1d as quantizable, priced at F16. The "
                             "binary writes them as F32 regardless, so this "
                             "only reproduces pre-2026-09-28 tables — and the "
                             "pre-flight audit will (correctly) report every "
                             "one of them as a mismatch")
    parser.add_argument("--free-pins", action="store_true",
                         help="EXPERIMENTAL: let output/token_embd/MTP/routers fight "
                              "for budget instead of fixed pins")
    parser.add_argument("--max-tier", default=None,
                         help="EXPERIMENTAL: override CLASS_MAX_TIER for every "
                              "class at runtime (e.g. F16 to let kings rise "
                              "above Q8_0). Bottom-up only; top-down starts "
                              "at F16 by construction. Never touches "
                              "constants.py.")
    parser.add_argument("--invert", action="store_true",
                         help="ADVERSARIAL CONTROL: negate the importance "
                              "table — kings stay down, junk rises to the "
                              "skyscraper. If this still scores, importance "
                              "is bunk; if it collapses, importance is "
                              "causal. Operator's idea, bottom-up only.")
    parser.add_argument("--squeeze", action="store_true",
                         help="SQUEEZE mode (separate path): only IQ1_S or F16, "
                              "nothing between. Dungeon or palace per tensor; "
                              "the queue rescues kings to F16. Bottom-up only.")
    parser.add_argument("--utility", choices=["mse", "rmse", "hybrid", "huber", "logcosh", "smape", "ssim", "smape_ssim", "smape_frag", "pw_ssim", "netdmg", "mix", "smse", "srmse", "balance"],
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
    parser.add_argument("--no-preflight", action="store_true",
                        help="Skip the dry-run audit (intended-vs-actual tier "
                             "diff, real size, real geometry)")
    parser.add_argument("--strict-preflight", action="store_true",
                        help="Abort if the binary will not write the assigned "
                             "tiers (default: warn loudly and continue)")

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
                      "(--imatrix-legacy-combine to override)")
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
        # Adversarial control (operator's idea): the whole ranking runs
        # upside down — junk outbids kings for every rung. Tied groups,
        # pins and floors stay structural; only priority inverts.
        for _n, _d in imp_table.items():
            if "importance_mean" in _d:
                _d["importance_mean"] = -_d["importance_mean"]
        print("  INVERTED: kings beg, junk builds skyscrapers")

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
        legacy_1d=args.legacy_1d,
        **({"ceiling": args.max_tier} if (args.max_tier and not args.top_down) else {}),
        **({"pin_norms": args.pin_norms} if args.top_down else {
            "relief": relief,
            "relief_thr": args.relief_thr,
            "allowed_tiers": (["IQ1_S", "F16"] if args.squeeze else None),
        }),
    )
    if args.squeeze:
        print("  SQUEEZE mode: dungeon IQ1_S or palace F16, nothing between")
    if args.top_down and args.pin_norms:
        # SPEC (mailbox 2026-09-28): TDN is a provable no-op under the 1D
        # law — the shield protects only tensors already pinned F32. Warn
        # instead of letting anyone re-run that experiment blind.
        probe, _ = optimal_classify_topdown(
            imp_table, tied_groups, model,
            target_size_mib=target_mib,
            allow_q3=args.allow_q3_or_lower,
            free_pins=args.free_pins,
            uopt=uopt,
            legacy_1d=args.legacy_1d,
            pin_norms=False)
        diff = sum(1 for k, v in assignments.items()
                   if probe.get(k) != v)
        if diff == 0:
            print("  WARNING: --pin-norms changed 0 tensors vs plain "
                  "--top-down (1D law already pins them F32) — shield is "
                  "a no-op here, do not re-run this experiment")

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

    # Передаем padded_ne_map для корректного вывода логов на экран
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
        hint = "Raise --size." if (args.top_down and args.allow_q3_or_lower) \
            else "Raise --size or pass --allow-q3-or-lower."
        print(f"\nERROR: base size {stats['total_mib']:.0f} MiB already exceeds "
              f"target {target_mib:.0f} MiB. " + hint)
        sys.exit(1)

    base_type = _get_base_type(model)
    flags = generate_flags(assignments, model, base_type, target_mib)
    flags["imatrix"] = quant_imatrix

    print(f"\nConfig (base={flags['base_type']}):")
    print(format_flags(flags))

    if args.show_config:
        return

    print("\n--- Dry Run ---")
    dry_size, dry_raw = run_dry_run_ex(flags, args.model)
    _show_size_result(dry_size, target_mib)

    # Preflight: the dry run already told us what the binary will ACTUALLY
    # write, per tensor. It used to be discarded here. Free audit, and the
    # only place a shadowed --tensor-type rule can be caught before 30 min
    # of quantization and 6 GB of disk.
    if not args.no_preflight and dry_raw:
        report = preflight.audit(assignments, dry_raw,
                                 target_mib=target_mib,
                                 est_mib=stats["total_mib"])
        if report["mismatches"] and args.strict_preflight:
            print("\nABORT: --strict-preflight — the binary disagrees with the "
                  "assignment map. Fix the rules (or drop the flag) before "
                  "spending the quantization.")
            sys.exit(1)

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
