#!/usr/bin/env python3
"""
Sensitivity sweep: vary one tensor class tier, measure PPL.
Usage: python sensitivity_sweep.py --model M.gguf --imatrix I.gguf --class ffn_down --tiers Q3_K,IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0
"""

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from config_generator import generate_flags, format_flags
from quantizer import run_dry_run, run_quantization
from constants import (
    TIER_ORDER, get_tensor_class, strip_weight
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_PPL = os.environ.get(
    "LLAMA_PPL", "/home/wepiqx/llama.cpp/build/bin/llama-perplexity"
)
WIKITEXT_DATA = os.environ.get(
    "WIKITEXT_DATA", "/mnt/Vsio/wikitext-2-raw/wiki.test.raw"
)


def parse_tiers(tier_str):
    return [t.strip() for t in tier_str.split(",")]


def _tier_index(tier: str) -> int:
    return TIER_ORDER.index(tier) if tier in TIER_ORDER else -1


def build_assignments(model, target_class, tier, base_tier="Q5_K"):
    """Manually build assignments: all at base_tier, target_class at tier."""
    model_tensors = model.get("tensors", {})
    all_names = set()
    for tname in model_tensors:
        norm = strip_weight(tname)
        all_names.add(norm)

    assignments = {}
    for name in all_names:
        ttype = name.split(".")[-1] if "." in name else name
        cls = get_tensor_class(ttype)
        if cls == target_class:
            assignments[name] = tier
        else:
            assignments[name] = base_tier

    # Special: norms/ssm_params always F16
    for name in list(assignments.keys()):
        ttype = name.split(".")[-1] if "." in name else name
        cls = get_tensor_class(ttype)
        if cls in ("norms", "ssm_params"):
            assignments[name] = "F16"
        # MTP special handling
        if cls == "mtp":
            assignments[name] = base_tier
        # Embedding - keep at base_tier
        if cls == "embd":
            assignments[name] = base_tier

    return assignments


def main():
    parser = argparse.ArgumentParser(description="Sensitivity sweep for one tensor class")
    parser.add_argument("--model", required=True)
    parser.add_argument("--imatrix", required=True)
    parser.add_argument("--class", dest="target_class", required=True,
                        choices=["gate", "attn_proj", "ffn_gate_up", "ffn_down", "mtp", "embd"])
    parser.add_argument("--tiers", default="Q3_K,IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0")
    parser.add_argument("--base-tier", default="Q5_K")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to <repo>/output/sensitivity_sweep")
    parser.add_argument("--skip-ppl", action="store_true", help="Only dry-run, skip PPL")
    args = parser.parse_args()

    tiers = parse_tiers(args.tiers)
    if args.output_dir is None:
        args.output_dir = os.path.join(REPO_ROOT, "output", "sensitivity_sweep")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Sensitivity sweep: {args.target_class} ===")
    print(f"Tiers: {tiers}")
    print(f"Base tier for others: {args.base_tier}")

    model = read_model(args.model)
    imatrix = read_imatrix(args.imatrix)
    tied_groups = detect_tied_groups(imatrix)
    imp_table = build_importance_table(imatrix, model)

    results = []

    for tier in tiers:
        print(f"\n--- Testing {args.target_class} = {tier} ---")
        
        assignments = build_assignments(model, args.target_class, tier, args.base_tier)

        base_type = "Q5_K_M"
        flags = generate_flags(assignments, model, args.base_tier, 100000)
        flags["imatrix"] = args.imatrix

        # Dry run
        dry_size = run_dry_run(flags, args.model)
        print(f"  Dry-run size: {dry_size:.0f} MiB")

        if args.skip_ppl:
            results.append({"tier": tier, "size": dry_size, "ppl": None})
            continue

        # Quantize (CPU only to avoid OOM)
        out_path = os.path.join(args.output_dir, f"{args.target_class}_{tier}.gguf")
        print(f"  Quantizing to {out_path}...")
        
        # Use CPU for quantization to avoid GPU OOM
        success = run_quantization(flags, args.model, out_path)
        if not success:
            print(f"  FAILED")
            results.append({"tier": tier, "size": dry_size, "ppl": None, "error": "quant failed"})
            continue

        # PPL test (CPU)
        print(f"  Running PPL test (CPU)...")
        cmd = [
            LLAMA_PPL,
            "-m", out_path,
            "-f", WIKITEXT_DATA,
            "-c", "1024", "-ngl", "0"  # CPU only
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        ppl = None
        for line in result.stdout.split("\n"):
            if "Final estimate: PPL" in line:
                try:
                    ppl = float(line.split("PPL = ")[1].split(" ")[0])
                except:
                    pass
        
        print(f"  PPL: {ppl}")
        results.append({"tier": tier, "size": dry_size, "ppl": ppl})

        # Cleanup
        if os.path.exists(out_path):
            os.remove(out_path)

    # Summary
    print("\n=== RESULTS ===")
    print(f"{'Tier':<10} {'Size (MiB)':<12} {'PPL':<10}")
    for r in results:
        print(f"{r['tier']:<10} {r['size']:<12.0f} {r['ppl'] if r['ppl'] else 'FAILED':<10}")

    out_json = os.path.join(args.output_dir, f"sensitivity_{args.target_class}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()