#!/usr/bin/env python3
"""Streaming GGUF -> safetensors (sharded), low-RAM.

Usage: gguf_to_hf_stream.py --model M.gguf --out DIR [--shard-mib 4000]
Inverts gguf-py's TensorNameMap (prefers modern HF 'model.*' names).
Unmapped tensors keep GGUF names (reported at the end).
"""
import argparse
import json
import os

import numpy as np
import torch
from safetensors.torch import save_file

from gguf import GGUFReader
from gguf.constants import MODEL_ARCH
from gguf.tensor_mapping import get_tensor_name_map


def pick_hf_names(arch, n_blocks):
    m = get_tensor_name_map(arch, n_blocks)
    rev = {}
    for hf_name, (_, gguf_name) in dict(m.mapping).items():
        if gguf_name not in rev or (
                hf_name.startswith("model.") and not rev[gguf_name].startswith("model.")):
            rev[gguf_name] = hf_name
    return rev


def strip_affix(name):
    for suf in (".weight", ".bias"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-mib", type=float, default=4000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    r = GGUFReader(args.model)
    n_blocks = max(
        (int(n.split(".")[1]) for t in r.tensors
         for n in [t.name] if n.startswith("blk.") and n.split(".")[1].isdigit()),
        default=0) + 1
    rev = pick_hf_names(MODEL_ARCH.QWEN35, n_blocks)
    print("blocks=%d mapped-names=%d" % (n_blocks, len(rev)), flush=True)

    shard, shard_bytes, shard_idx, unmapped = {}, 0, 1, []
    index = {"metadata": {"total_size": 0}, "weight_map": {}}

    def flush():
        nonlocal shard, shard_bytes, shard_idx
        if not shard:
            return
        fn = "model-%05d-of-99999.safetensors" % shard_idx
        save_file(shard, os.path.join(args.out, fn))
        for k in shard:
            index["weight_map"][k] = fn
        print("shard %d: %d tensors %.0f MiB" % (
            shard_idx, len(shard), shard_bytes / 2 ** 20), flush=True)
        shard, shard_bytes, shard_idx = {}, 0, shard_idx + 1

    limit = args.shard_mib * 2 ** 20
    from gguf.constants import GGMLQuantizationType as GQ
    for i, t in enumerate(r.tensors):
        raw = np.ascontiguousarray(t.data)
        shape = tuple(reversed([int(d) for d in t.shape]))  # GGUF dims are reversed
        tt = int(t.tensor_type)
        if tt == int(GQ.BF16):
            import ml_dtypes
            ten = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(shape)
        elif tt == int(GQ.F32):
            ten = torch.frombuffer(raw, dtype=torch.float32).reshape(shape)
        elif tt == int(GQ.F16):
            ten = torch.frombuffer(raw, dtype=torch.float16).reshape(shape)
        else:
            raise ValueError("quantized tensor in source: %s type %s" % (t.name, tt))
        hf = rev.get(strip_affix(t.name))
        if hf is None:
            unmapped.append(t.name)
            hf = t.name
        else:
            if t.name.endswith(".weight"):
                hf += ".weight"
            elif t.name.endswith(".bias"):
                hf += ".bias"
        shard[hf] = ten
        shard_bytes += ten.nelement() * ten.element_size()
        if shard_bytes >= limit:
            flush()
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(r.tensors)), flush=True)
    flush()
    # rename shards to final count
    total = shard_idx - 1
    for j in range(1, total + 1):
        a = os.path.join(args.out, "model-%05d-of-99999.safetensors" % j)
        b = os.path.join(args.out, "model-%05d-of-%05d.safetensors" % (j, total))
        os.rename(a, b)
        for k, v in index["weight_map"].items():
            if v.endswith(os.path.basename(a)):
                index["weight_map"][k] = os.path.basename(b)
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    print("DONE shards=%d unmapped=%d" % (total, len(unmapped)))
    for u in unmapped[:20]:
        print("  UNMAPPED:", u)


if __name__ == "__main__":
    main()
