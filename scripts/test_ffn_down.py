#!/usr/bin/env python3
"""Force ffn_down to specific tier, test PPL."""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_reader import read_model
from imatrix_reader import read_imatrix, detect_tied_groups, build_importance_table
from config_generator import generate_flags
from quantizer import run_dry_run, run_quantization
from constants import (
    get_tensor_class, get_tensor_type, strip_weight
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_PPL = os.environ.get(
    "LLAMA_PPL", "/home/wepiqx/llama.cpp/build/bin/llama-perplexity"
)
WIKITEXT_DATA = os.environ.get(
    "WIKITEXT_DATA", "/mnt/Vsio/wikitext-2-raw/wiki.test.raw"
)


def build_assignments(all_names, ffn_down_tier):
    assignments = {}
    for name in all_names:
        cls = get_tensor_class(get_tensor_type(name))
        if cls == "ffn_down":
            assignments[name] = ffn_down_tier
        elif cls in ("norms", "ssm_params"):
            assignments[name] = "F16"
        elif cls == "embd":
            assignments[name] = "Q5_K"
        elif cls == "mtp":
            assignments[name] = "Q5_K"
        else:
            assignments[name] = "Q5_K"
    return assignments


def test_tier(model, imatrix_path, model_path, all_names, out_dir, tier):
    print(f"\n=== Testing ffn_down = {tier} ===")
    assignments = build_assignments(all_names, tier)

    flags = generate_flags(assignments, model, "Q5_K_M", 100000)
    flags["imatrix"] = imatrix_path

    dry_size = run_dry_run(flags, model_path)
    print(f"  Dry-run: {dry_size:.0f} MiB")

    out_path = os.path.join(out_dir, f"ffn_down_{tier}.gguf")
    print(f"  Quantizing...")
    success = run_quantization(flags, model_path, out_path)
    if not success:
        print("  FAILED")
        return None

    print(f"  Running PPL (GPU)...")
    cmd = [
        LLAMA_PPL,
        "-m", out_path,
        "-f", WIKITEXT_DATA,
        "-c", "1024", "-ngl", "99"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

    ppl = None
    for line in result.stdout.split("\n"):
        if "Final estimate: PPL" in line:
            ppl = float(line.split("PPL = ")[1].split(" ")[0])
            break

    print(f"  PPL: {ppl}")
    os.remove(out_path)
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Force ffn_down to tier, test PPL")
    parser.add_argument("--model", required=True, help="BF16 GGUF model path")
    parser.add_argument("--imatrix", required=True, help="Imatrix GGUF path")
    parser.add_argument("--out-dir", default=None,
                        help="Defaults to <repo>/output/ffn_down_test")
    parser.add_argument("--tiers", default="Q3_K,IQ4_XS,Q4_K,Q5_K,Q6_K,Q8_0")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(REPO_ROOT, "output", "ffn_down_test")
    os.makedirs(args.out_dir, exist_ok=True)

    model = read_model(args.model)
    imatrix = read_imatrix(args.imatrix)
    tied_groups = detect_tied_groups(imatrix)
    imp_table = build_importance_table(imatrix, model)

    # Get all tensor names
    all_names = set()
    for tname in model.get("tensors", {}):
        all_names.add(strip_weight(tname))
    for tname in imatrix["tensors"]:
        all_names.add(strip_weight(tname))

    for tier in args.tiers.split(","):
        ppl = test_tier(model, args.imatrix, args.model, all_names,
                        args.out_dir, tier.strip())
        if ppl:
            print(f"  Result: {tier} -> PPL {ppl}")
        else:
            print(f"  Result: {tier} -> FAILED")


if __name__ == "__main__":
    main()
