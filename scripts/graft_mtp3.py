#!/usr/bin/env python3
"""Graft MTP head onto BF16 GGUF. v3: structs from gguf-py, data streamed.

Usage: graft_mtp3.py --bf16 SRC.gguf --mtp HEAD.safetensors --out OUT.gguf
"""
import argparse
import os
import struct

import numpy as np
import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy for safetensors)
from safetensors import safe_open

import gguf
from gguf.constants import GGUFValueType as VT

# Donor (Qwen3.5-MTP head) -> Ornith-style GGUF names (what llama.cpp
# looks up for qwen35-MTP). Extra donor tensors (fc, pre_fc_*, plain norm)
# have no counterpart in the working recipe and are DROPPED.
NAME_MAP = {
    "mtp.layers.0.input_layernorm.weight":             "blk.32.attn_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight":            "blk.32.attn_q.weight",
    "mtp.layers.0.self_attn.k_proj.weight":            "blk.32.attn_k.weight",
    "mtp.layers.0.self_attn.v_proj.weight":            "blk.32.attn_v.weight",
    "mtp.layers.0.self_attn.o_proj.weight":            "blk.32.attn_output.weight",
    "mtp.layers.0.self_attn.q_norm.weight":            "blk.32.attn_q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight":            "blk.32.attn_k_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight":    "blk.32.post_attention_norm.weight",
    "mtp.layers.0.mlp.gate_proj.weight":               "blk.32.ffn_gate.weight",
    "mtp.layers.0.mlp.up_proj.weight":                 "blk.32.ffn_up.weight",
    "mtp.layers.0.mlp.down_proj.weight":               "blk.32.ffn_down.weight",
}

NEXTN_SRC = {
    # donor head lacks nextn.* — borrow trained ones from a working MTP file
    # (same hidden size required). Frankendraft, but loads and drafts.
    "blk.32.nextn.enorm.weight": "blk.32.nextn.enorm.weight",
    "blk.32.nextn.hnorm.weight": "blk.32.nextn.hnorm.weight",
    "blk.32.nextn.shared_head_norm.weight": "blk.32.nextn.shared_head_norm.weight",
    "blk.32.nextn.eh_proj.weight": "blk.32.nextn.eh_proj.weight",
}

# Draft compute runs f32 on CPU: ALL 1D MTP tensors (norms) MUST be F32 —
# the quantizer keeps 1D at source precision, and f32×bf16 binary-ops crash
# (diagnosed 2026-09-15: Ornith norms are F32, ours were BF16).
# eh_proj stays F32 for the same reason (was BF16 donor).
DTYPE_OVERRIDE = {
    "blk.32.nextn.eh_proj.weight": 0,  # F32
}

TYPE_ID = {
    # gguf type id -> (writer id for TI)
    0: 0, 1: 1, 30: 30, 32: 32,
}

SCALAR = {VT.UINT8: ("<B", 1), VT.INT8: ("<b", 1), VT.UINT16: ("<H", 2),
          VT.INT16: ("<h", 2), VT.UINT32: ("<I", 4), VT.INT32: ("<i", 4),
          VT.FLOAT32: ("<f", 4), VT.BOOL: ("<?", 1), VT.UINT64: ("<Q", 8),
          VT.INT64: ("<q", 8), VT.FLOAT64: ("<d", 8)}


def pack_str(s):
    if isinstance(s, str):
        b = s.encode()
    elif isinstance(s, (bytes, bytearray)):
        b = bytes(s)
    else:
        import numpy as np
        a = np.asarray(s)
        if a.dtype == object:
            b = str(a.tolist()).encode()
        else:
            b = a.astype(np.uint8).tobytes()
    return struct.pack("<Q", len(b)) + b


def pack_val(t, v):
    if t == VT.STRING:
        return pack_str(v.tolist() if hasattr(v, "tolist") else v)
    fmt, _ = SCALAR[t]
    vv = v.tolist() if hasattr(v, "tolist") else v
    if t == VT.BOOL:
        vv = bool(vv)
    elif fmt in ("<B", "<b", "<H", "<h", "<I", "<i", "<Q", "<q"):
        vv = int(np.asarray(vv).ravel()[0]) if not isinstance(vv, int) else vv
    else:
        vv = float(np.asarray(vv).ravel()[0]) if not isinstance(vv, float) else vv
    return struct.pack(fmt, vv)


def pack_field(key, field):
    out = pack_str(key)
    types = list(field.types)
    try:
        vals = field.contents()
    except Exception as e:
        print("  SKIP kv %s: %s" % (key, str(e)[:50]))
        return None
    if types and types[0] == VT.ARRAY:
        et = types[1]
        if not isinstance(vals, (list, tuple)):
            vals = [vals]
        out += struct.pack("<I", 9)
        out += struct.pack("<I", int(et))
        out += struct.pack("<Q", len(vals))
        for v in vals:
            out += pack_val(et, v)
    else:
        t = types[0]
        out += struct.pack("<I", int(t))
        out += pack_val(t, vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--mtp", required=True)
    ap.add_argument("--nextn", default=None,
                    help="working MTP GGUF to borrow nextn.* tensors from")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("Reading header...", flush=True)
    r = gguf.GGUFReader(args.bf16)
    parts = []
    for k, f in r.fields.items():
        if k.startswith("GGUF."):
            continue  # reader-synthesized pseudo-fields, must not be stored
        if k == "qwen35.block_count":
            # MTP head counts as a block (working recipe: 32 trunk + 1 head)
            import struct as _st
            parts.append(pack_str(k) + _st.pack("<I", 4) + _st.pack("<I", 33))
            continue
        if k == "qwen35.attention.recurrent_layers":
            # extend to 33 entries (MTP head layer is not recurrent)
            import struct as _st2
            vals = list(f.contents()) + [False]
            b = pack_str(k) + _st2.pack("<I", 9) + _st2.pack("<I", 7)
            b += _st2.pack("<Q", len(vals))
            for v in vals:
                b += _st2.pack("<?", bool(v))
            parts.append(b)
            continue
        b = pack_field(k, f)
        if b is not None:
            parts.append(b)
    kv_raw = b"".join(parts)
    kv_raw += pack_str("qwen35.nextn_predict_layers") + struct.pack("<I", 4) + struct.pack("<I", 1)

    print("Rebuilding %d tensor infos..." % len(r.tensors), flush=True)
    from gguf.constants import GGMLQuantizationType as GQ
    TSZ = {int(GQ.F32): 4, int(GQ.F16): 2, int(GQ.BF16): 2, int(GQ.I8): 1,
           int(GQ.I16): 2, int(GQ.I32): 4, int(GQ.I64): 8, int(GQ.F64): 8,
           int(GQ.U8) if hasattr(GQ, "U8") else -1: 1}
    ti_raw = b""
    off = 0
    for t in r.tensors:
        ti_raw += pack_str(t.name)
        dims = [int(d) for d in t.shape]
        ti_raw += struct.pack("<I", len(dims))
        for d in dims:
            ti_raw += struct.pack("<Q", d)
        ti_raw += struct.pack("<I", int(t.tensor_type))
        ti_raw += struct.pack("<Q", off)
        ne = 1
        for d in dims:
            ne *= d
        off += ne * TSZ.get(int(t.tensor_type), 4)
        off += (32 - off % 32) % 32

    print("Loading MTP tensors...", flush=True)
    mtp = []  # (dst, arr, type_id)
    with safe_open(args.mtp, framework="np") as f:
        for src, dst in NAME_MAP.items():
            arr = np.ascontiguousarray(f.get_tensor(src))
            # keep donor dtype (BF16, id 30) like the working recipe,
            # EXCEPT 1D norms (must be F32 for draft compute);
            # GGUF stores dims reversed vs numpy
            if arr.ndim <= 1:
                mtp.append((dst, np.asarray(arr).astype(np.float32), 0))
                continue
            if str(arr.dtype) != "bfloat16":
                import ml_dtypes
                arr = np.asarray(arr).view(ml_dtypes.bfloat16)
            mtp.append((dst, arr, 30))
    if args.nextn:
        print("Borrowing nextn.* from", args.nextn, flush=True)
        from gguf.constants import GGMLQuantizationType as GQ2
        rn = gguf.GGUFReader(args.nextn)
        got = {t.name: t for t in rn.tensors}
        for src, dst in NEXTN_SRC.items():
            t = got[src]
            raw = np.ascontiguousarray(t.data)
            tt = int(t.tensor_type)
            shp = tuple(reversed([int(d) for d in t.shape]))
            if tt == int(GQ2.BF16):
                import ml_dtypes
                arr = raw.view(ml_dtypes.bfloat16).reshape(shp)
            elif tt == int(GQ2.F32):
                arr = raw.view(np.float32).reshape(shp)
            else:
                raise ValueError("nextn dtype %s" % tt)
            if dst in DTYPE_OVERRIDE:
                tid = DTYPE_OVERRIDE[dst]
                if tid == 0:
                    arr = np.asarray(arr).astype(np.float32)
                mtp.append((dst, arr, tid))
            else:
                mtp.append((dst, arr, tt))

    # MTP data starts after old data (which keeps relative offsets)
    old_data_sz = os.path.getsize(args.bf16) - r.data_offset
    off = old_data_sz
    for dst, arr, tid in mtp:
        data = arr.tobytes()
        dims = list(reversed(arr.shape))
        ti_raw += pack_str(dst)
        ti_raw += struct.pack("<I", len(dims))
        for d in dims:
            ti_raw += struct.pack("<Q", int(d))
        ti_raw += struct.pack("<I", tid)
        ti_raw += struct.pack("<Q", off)
        off += len(data) + (32 - len(data) % 32) % 32

    n_tensors = len(r.tensors) + len(mtp)
    n_kv = len(parts) + 1
    hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + struct.pack("<Q", n_kv)
    hdr += kv_raw + ti_raw
    hdr += b"\x00" * ((32 - len(hdr) % 32) % 32)
    print("header %d bytes, tensors %d" % (len(hdr), n_tensors), flush=True)

    print("Writing (streaming)...", flush=True)
    with open(args.out, "wb") as fout:
        fout.write(hdr)
        with open(args.bf16, "rb") as fb:
            fb.seek(r.data_offset)
            left = old_data_sz
            while left > 0:
                chunk = fb.read(min(256 * 1024 * 1024, left))
                if not chunk:
                    break
                fout.write(chunk)
                left -= len(chunk)
        for dst, arr, tid in mtp:
            data = arr.tobytes()
            fout.write(data)
            fout.write(b"\x00" * ((32 - len(data) % 32) % 32))
    print("Done:", os.path.getsize(args.out), flush=True)


if __name__ == "__main__":
    main()
