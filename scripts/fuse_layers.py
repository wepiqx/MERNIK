#!/usr/bin/env python3
"""Layer-interleaved fusion of same-arch BF16 GGUFs (streaming).

Two donors (legacy): even blocks from A, odd blocks from B, globals from A.
  fuse_layers.py --a A.gguf --b B.gguf --out OUT.gguf

Three donors (monster): per-block donor map + optional soup-averaged backbone.
  fuse_layers.py --a OX.gguf --b ORN.gguf --c NEO.gguf --out M1.gguf \
      --map "15:b,19:b,23:b,27:b,31:c"
  fuse_layers.py --a OX.gguf --c NEO.gguf --out M3.gguf \
      --map "15:b,19:b" --soup "0-8" --soup-from "a,c"

Rules: same trunk required (extras like MTP head in any donor are ignored,
globals always from A). Soup (weight averaging) only for backbone blocks
where donors are near-identical — never for divergent blocks.
"""
import argparse
import os
import re
import struct

import numpy as np

import gguf
from gguf.constants import GGUFValueType as VT

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graft_mtp3 import pack_str, pack_field


def layer_of(name):
    p = name.split(".")
    if len(p) >= 2 and p[0] in ("blk", "BLK") and p[1].isdigit():
        return int(p[1])
    return None


def parse_blocks(spec):
    """'0-8,11,15' -> sorted set of ints."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            out.update(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.add(int(part))
    return out


def parse_map(spec):
    """'15:b,31:c' -> {15: 'b', 31: 'c'}."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        blk, _, donor = part.partition(":")
        blk = int(blk.strip())
        donor = donor.strip().lower()
        assert donor in ("a", "b", "c"), "map donor must be a/b/c: %r" % part
        out[blk] = donor
    return out


def soup_average(blobs, tensor_type):
    """Average raw tensor blobs across donors. BF16 decoded exactly."""
    from gguf.constants import GGMLQuantizationType as GQ
    if int(tensor_type) == int(GQ.BF16):
        import ml_dtypes
        acc = None
        for b in blobs:
            f = np.frombuffer(b, dtype=ml_dtypes.bfloat16).astype(np.float32)
            acc = f if acc is None else acc + f
        acc /= len(blobs)
        return acc.astype(ml_dtypes.bfloat16).tobytes()
    # F32 and friends: plain average
    acc = None
    for b in blobs:
        f = np.frombuffer(b, dtype=np.float32).astype(np.float64)
        acc = f if acc is None else acc + f
    acc /= len(blobs)
    return acc.astype(np.float32).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None, help="third donor (monster mode)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--swap", default=None,
                    help="comma list of block ids to take from B "
                         "(default: odd blocks from B, even from A)")
    ap.add_argument("--map", default=None,
                    help="block->donor map, e.g. '15:b,19:b,31:c' "
                         "(donors a/b/c; overrides --swap for listed blocks)")
    ap.add_argument("--soup", default=None,
                    help="blocks to weight-average, e.g. '0-8,11' "
                         "(backbone only — never divergent blocks)")
    ap.add_argument("--soup-from", default=None,
                    help="donors to average, e.g. 'a,c' (default: all given)")
    args = ap.parse_args()

    donors = {"a": gguf.GGUFReader(args.a), "b": gguf.GGUFReader(args.b)}
    if args.c:
        donors["c"] = gguf.GGUFReader(args.c)
    T = {k: {t.name: t for t in r.tensors} for k, r in donors.items()}
    ta = T["a"]
    # extras may sit on either side (e.g. MTP head in A, or in B):
    # fuse the common trunk, extras always come from A
    common = set(ta)
    for k in list(T):
        if k == "a":
            continue
        common &= set(T[k])
    print("common trunk: %d tensors (%d extras stay from A)" %
          (len(common), len(ta) - len(common)), flush=True)
    # shape gate across all donors (common trunk only)
    for n, t in ta.items():
        if n not in common:
            continue
        for k in T:
            if k == "a":
                continue
            assert [int(d) for d in T[k][n].shape] == [int(d) for d in t.shape], \
                "shape mismatch %s: a vs %s" % (n, k)

    mmap = parse_map(args.map) if args.map else {}
    for blk, d in mmap.items():
        assert d in donors, "map donor %r has no file (pass --%s)" % (d, d)
    soupset = parse_blocks(args.soup) if args.soup else set()
    sfrom = [s.strip().lower() for s in args.soup_from.split(",")] if args.soup_from \
        else sorted(donors)
    for d in sfrom:
        assert d in donors, "soup donor %r has no file" % d
    assert not (soupset & set(mmap)), "block both in --map and --soup: %s" % \
        sorted(soupset & set(mmap))

    parts = []
    ra = donors["a"]
    for k, f in ra.fields.items():
        if k.startswith("GGUF."):
            continue
        b = pack_field(k, f)
        if b is not None:
            parts.append(b)
    kv_raw = b"".join(parts)

    from gguf.constants import GGMLQuantizationType as GQ
    ti_raw, off, order = b"", 0, []
    swapset = set(int(x) for x in args.swap.split(",")) if args.swap else None
    donor_of_block = {}
    for name in [t.name for t in ra.tensors]:
        lyr = layer_of(name)
        if lyr is None:
            donor, soup = "a", False  # globals: always from A
        elif lyr in soupset:
            donor, soup = None, True
        elif lyr in mmap:
            donor, soup = mmap[lyr], False
        elif swapset is not None:
            donor, soup = ("b" if lyr in swapset else "a"), False
        elif len(donors) == 2:
            donor, soup = (("b" if (lyr or 0) % 2 == 1 else "a")), False
        else:
            # unmapped blocks default to the base (A) — never surprise-mix.
            # explicit tri-interleave via a full --map if wanted.
            donor, soup = "a", False
        if lyr is not None:
            donor_of_block.setdefault(lyr, donor if not soup else "soup(%s)" % "+".join(sfrom))
        t = ta[name]
        ti_raw += pack_str(name)
        dims = [int(d) for d in t.shape]
        ti_raw += struct.pack("<I", len(dims))
        for d in dims:
            ti_raw += struct.pack("<Q", d)
        ti_raw += struct.pack("<I", int(t.tensor_type))
        ti_raw += struct.pack("<Q", off)
        ne = 1
        for d in dims:
            ne *= d
        bpe = 2 if int(t.tensor_type) == int(GQ.BF16) else 4
        off += ne * bpe
        off += (32 - off % 32) % 32
        order.append((name, donor, soup))

    print("donor map: " + ", ".join(
        "%d:%s" % (b, donor_of_block[b]) for b in sorted(donor_of_block)), flush=True)
    print("globals: a, trunk tensors: %d" % len(ta), flush=True)

    n_tensors = len(ta)
    hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors)
    hdr += struct.pack("<Q", len(parts))
    hdr += kv_raw + ti_raw
    hdr += b"\x00" * ((32 - len(hdr) % 32) % 32)
    print("header %d bytes" % len(hdr), flush=True)

    with open(args.out, "wb") as fout:
        fout.write(hdr)
        # stream per tensor via reader mmap slices (soup averaged in RAM;
        # extras missing from a donor fall back to A)
        for name, donor, soup in order:
            if soup:
                holders = [d for d in sfrom if name in T[d]] or ["a"]
                blobs = [np.ascontiguousarray(T[d][name].data).tobytes()
                         for d in holders]
                raw = soup_average(blobs, ta[name].tensor_type)
            else:
                src = T[donor][name] if name in T[donor] else ta[name]
                raw = np.ascontiguousarray(src.data).tobytes()
            fout.write(raw)
            pad = (32 - len(raw) % 32) % 32
            fout.write(b"\x00" * pad)
    print("Done:", os.path.getsize(args.out), flush=True)


if __name__ == "__main__":
    main()
