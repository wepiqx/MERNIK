#!/usr/bin/env python3
"""Graft MTP head (safetensors) onto BF16 GGUF via gguf-py. Robust version.

Usage: graft_mtp.py --bf16 SRC.gguf --mtp HEAD.safetensors --out OUT.gguf
"""
import argparse
import os

import numpy as np
from safetensors import safe_open

import gguf
from gguf.constants import GGUFValueType

NAME_MAP = {
    "mtp.pre_fc_norm_embedding.weight": "blk.32.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight":   "blk.32.pre_fc_norm_hidden.weight",
    "mtp.fc.weight":                    "blk.32.fc.weight",
    "mtp.layers.0.input_layernorm.weight":             "blk.32.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight":            "blk.32.attn_q.weight",
    "mtp.layers.0.self_attn.k_proj.weight":            "blk.32.attn_k.weight",
    "mtp.layers.0.self_attn.v_proj.weight":            "blk.32.attn_v.weight",
    "mtp.layers.0.self_attn.o_proj.weight":            "blk.32.attn_output.weight",
    "mtp.layers.0.self_attn.q_norm.weight":            "blk.32.self_attn_q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight":            "blk.32.self_attn_k_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight":    "blk.32.post_attention_layernorm.weight",
    "mtp.layers.0.mlp.gate_proj.weight":               "blk.32.ffn_gate.weight",
    "mtp.layers.0.mlp.up_proj.weight":                 "blk.32.ffn_up.weight",
    "mtp.layers.0.mlp.down_proj.weight":               "blk.32.ffn_down.weight",
    "mtp.norm.weight":                      "blk.32.norm.weight",
}

_ADD = {
    GGUFValueType.UINT8: "add_uint8",
    GGUFValueType.INT8: "add_int8",
    GGUFValueType.UINT16: "add_uint16",
    GGUFValueType.INT16: "add_int16",
    GGUFValueType.UINT32: "add_uint32",
    GGUFValueType.INT32: "add_int32",
    GGUFValueType.FLOAT32: "add_float32",
    GGUFValueType.BOOL: "add_bool",
    GGUFValueType.STRING: "add_string",
    GGUFValueType.UINT64: "add_uint64",
    GGUFValueType.INT64: "add_int64",
    GGUFValueType.FLOAT64: "add_float64",
}


def copy_field(writer, key, field):
    vals = field.parts
    if not vals:
        return
    t = field.types[0] if field.types else None
    if t == GGUFValueType.ARRAY:
        et = field.types[1] if len(field.types) > 1 else None
        fn = _ADD.get(et)
        if fn:
            getattr(writer, fn)(key, [v.tolist() if hasattr(v, "tolist") else v
                                      for v in vals])
        return
    fn = _ADD.get(t)
    if not fn:
        print("  SKIP kv:", key, t)
        return
    v = vals[0]
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8", "replace")
    try:
        getattr(writer, fn)(key, v)
    except Exception as e:
        print("  SKIP kv:", key, str(e)[:60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("Reading", args.bf16, flush=True)
    r = gguf.GGUFReader(args.bf16)
    w = gguf.GGUFWriter(args.out, arch="qwen35")
    for key, field in r.fields.items():
        copy_field(w, key, field)
    w.add_uint32("nextn_predict_layers", 1)

    print("Copying %d tensors..." % len(r.tensors), flush=True)
    from gguf.constants import GGMLQuantizationType as GQ
    for i, t in enumerate(r.tensors):
        raw = np.ascontiguousarray(t.data)
        shape = tuple(reversed([int(d) for d in t.shape]))
        if int(t.tensor_type) == int(GQ.BF16):
            import ml_dtypes
            arr = raw.view(ml_dtypes.bfloat16).reshape(shape).astype(np.float16)
        else:
            arr = np.asarray(raw).reshape(shape)
        w.add_tensor(t.name, np.ascontiguousarray(arr))
        if (i + 1) % 100 == 0:
            print("  %d/%d" % (i + 1, len(r.tensors)), flush=True)

    print("Grafting MTP head...", flush=True)
    with safe_open(args.mtp, framework="np") as f:
        for src, dst in NAME_MAP.items():
            arr = f.get_tensor(src)
            w.add_tensor(dst, np.ascontiguousarray(arr).astype(np.float16))
            print("  %s %s" % (dst, arr.shape), flush=True)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("Done:", os.path.getsize(args.out))


if __name__ == "__main__":
    main()
