#!/usr/bin/env python3
"""Merge MTP head (safetensors) into BF16 GGUF — raw binary header rewrite."""

import argparse
import os, struct, shutil, gguf
import ml_dtypes
import numpy as np
from safetensors import safe_open

LLAMA_GGUF = os.environ.get(
    "LLAMA_GGUF", "/home/wepiqx/llama.cpp/build/bin/llama-gguf"
)

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

def pack_str(s):
    b = s.encode()
    return struct.pack('<Q', len(b)) + b

def main():
    parser = argparse.ArgumentParser(description="Merge MTP safetensors into BF16 GGUF")
    parser.add_argument("--bf16", required=True, help="Source BF16 GGUF path")
    parser.add_argument("--mtp", required=True, help="MTP head safetensors path")
    parser.add_argument("--out", required=True, help="Output GGUF path")
    args = parser.parse_args()

    BF16_PATH, MTP_PATH, OUT_PATH = args.bf16, args.mtp, args.out

    # --- 1. Read BF16 header bytes ---
    print("Reading BF16 header...")
    r = gguf.GGUFReader(BF16_PATH)
    data_offset = r.data_offset
    n_tensors = len(r.tensors)
    print(f"Tensors: {n_tensors}, data offset: {data_offset}")

    with open(BF16_PATH, 'rb') as f:
        hdr = f.read(data_offset)

    # --- 2. Parse KVs (from byte 24 to where TIs start) ---
    pos = 24
    orig_kv_count = struct.unpack('<Q', hdr[16:24])[0]
    for _ in range(orig_kv_count):
        klen = struct.unpack('<Q', hdr[pos:pos+8])[0]
        pos += 8 + klen
        vtype = struct.unpack('<I', hdr[pos:pos+4])[0]; pos += 4
        if vtype == 8:
            sl = struct.unpack('<Q', hdr[pos:pos+8])[0]; pos += 8 + sl
        elif vtype == 9:
            et = struct.unpack('<I', hdr[pos:pos+4])[0]; pos += 4
            ct = struct.unpack('<Q', hdr[pos:pos+8])[0]; pos += 8
            for _ in range(ct):
                if et == 8:
                    sl = struct.unpack('<Q', hdr[pos:pos+8])[0]; pos += 8 + sl
                elif et in (0, 1): pos += 1
                elif et in (2, 3): pos += 2
                elif et in (4, 5, 6, 7): pos += 4
                elif et in (10, 11, 12): pos += 8
                else:
                    print(f"UNKNOWN array etype {et} at {pos}")
                    break
        elif vtype in (0,1,2,3,4,5,6,12): pos += 4
        elif vtype in (7,10,11): pos += 8
        else:
            print(f"UNKNOWN vtype {vtype} at {pos-4}")
            break

    kv_raw_len = pos - 24
    ti_start = pos  # TIs start right after KVs, no padding
    print(f"KV raw: {kv_raw_len} bytes, TI at: {ti_start}")

    # --- 3. Parse existing TIs (between ti_start and data_offset) ---
    ti_data = hdr[ti_start:data_offset]
    ti_data_len = len(ti_data)
    print(f"TI section: {ti_data_len} bytes")

    # Find where raw TIs end (before padding)
    pos2 = 0
    for i in range(n_tensors):
        nl = struct.unpack('<Q', ti_data[pos2:pos2+8])[0]
        pos2 += 8 + nl
        nd = struct.unpack('<I', ti_data[pos2:pos2+4])[0]; pos2 += 4
        pos2 += 8 * nd
        pos2 += 4 + 8  # type + offset
    ti_raw_end = pos2
    ti_pad = ti_data_len - ti_raw_end
    print(f"TIs: {n_tensors} parsed, {ti_raw_end} raw + {ti_pad} pad")

    # --- 4. Load MTP tensors ---
    print("Loading MTP tensors...")
    mtp_entries = []  # (dst_name, dims, f16_data_bytes)
    with safe_open(MTP_PATH, framework='np') as f:
        for src_name, dst_name in NAME_MAP.items():
            arr = f.get_tensor(src_name)
            data = arr.astype(np.float32).astype(np.float16).tobytes()
            mtp_entries.append((dst_name, list(arr.shape), data))

    # --- 5. Build MTP TIs ---
    bf16_data_sz = os.path.getsize(BF16_PATH) - data_offset
    mtp_offset = bf16_data_sz
    mtp_ti_bytes = b''
    for name, dims, data in mtp_entries:
        ti = pack_str(name)
        ti += struct.pack('<I', len(dims))
        for d in dims: ti += struct.pack('<Q', d)
        ti += struct.pack('<I', 1)  # F16
        ti += struct.pack('<Q', mtp_offset)
        mtp_ti_bytes += ti
        dpad = (32 - len(data) % 32) % 32
        mtp_offset += len(data) + dpad

    # --- 6. Build new header ---
    raw_kv = hdr[24:24+kv_raw_len]  # original KVs
    nextn_kv = pack_str("nextn_predict_layers")
    nextn_kv += struct.pack('<I', 4)  # UINT32
    nextn_kv += struct.pack('<I', 1)

    new_kv = raw_kv + nextn_kv
    new_ti = ti_data[:ti_raw_end] + mtp_ti_bytes  # old raw TIs + MTP TIs
    # Pad TIs so data section aligns to 32 bytes from file start
    total_before_ti = 24 + len(new_kv)
    new_ti_pad = (32 - (total_before_ti + len(new_ti)) % 32) % 32
    new_ti += b'\x00' * new_ti_pad

    new_tc = n_tensors + len(mtp_entries)
    new_kvc = orig_kv_count + 1

    new_hdr = b'GGUF'
    new_hdr += struct.pack('<I', 3)
    new_hdr += struct.pack('<Q', new_tc)
    new_hdr += struct.pack('<Q', new_kvc)
    new_hdr += new_kv + new_ti

    print(f"New header: {len(new_hdr)} bytes")
    print(f"TC: {new_tc}, KVC: {new_kvc}")
    print(f"New data section at: {len(new_hdr)}")

    # --- 7. Write output ---
    with open(OUT_PATH, 'wb') as fout:
        fout.write(new_hdr)

        print("Copying BF16 data...")
        with open(BF16_PATH, 'rb') as fb:
            fb.seek(data_offset)
            shutil.copyfileobj(fb, fout)

        print("Writing MTP tensors...")
        for name, dims, data in mtp_entries:
            fout.write(data)
            pad = (32 - len(data) % 32) % 32
            fout.write(b'\x00' * pad)
            print(f"  {name}: {len(data)} bytes")

    sz = os.path.getsize(OUT_PATH)
    print(f"\nDone! {sz} bytes ({sz/1024/1024/1024:.2f} GB)")

    # Verify
    import subprocess
    res = subprocess.run(
        [LLAMA_GGUF, OUT_PATH, "r"],
        capture_output=True, text=True, timeout=30)
    for line in res.stdout.split('\n')[:25]:
        if any(x in line.lower() for x in ['blk.32', 'total', 'tensor', 'arch', 'name', 'layer']):
            print(line)
    if res.returncode != 0:
        print(f"VERIFY FAIL: {res.stderr[:300]}")


if __name__ == "__main__":
    main()
