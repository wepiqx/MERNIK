#!/usr/bin/env python3
"""Layer-interleaved fusion of same-arch BF16 GGUFs (streaming).

Two donors (legacy): even blocks from A, odd blocks from B, globals from A.
  fuse_layers.py --a A.gguf --b B.gguf --out OUT.gguf

Three donors (monster): per-block donor map + optional soup-averaged backbone.
  fuse_layers.py --a OX.gguf --b ORN.gguf --c NEO.gguf --out M1.gguf \
      --map "15:b,19:b,23:b,27:b,31:c"

Four donors: same, plus --d (e.g. --a M2.gguf --d MIMO.gguf --map "31:d").
  Building from an existing fusion (e.g. RINIQ-M2-BF16) as --a is smart:
  donor blocks already baked in, no need for the original BF16s.
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


def kind_of(name):
    """Tissue class of a tensor: ffn / attn / ssm / norm / glob.

    Same split as the weight/imatrix compasses: the soul lives in FFN
    (weight divergence ~2x attention), compatibility in attention.
    """
    p = name.split(".")
    if len(p) < 3 or p[0] not in ("blk", "BLK"):
        return "glob"
    rest = ".".join(p[2:])
    if "ffn" in rest:
        return "ffn"
    if "attn" in rest:
        return "attn"
    if "ssm" in rest:
        return "ssm"
    return "norm"


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
    """'15:b,31:c' -> {15: ('b', None), 31: ('c', None)} (whole block).
    Tissue mode: '15:b:ffn' -> {15: ('b', {'ffn'})} — only FFN-kind tensors
    of block 15 come from B, the rest of the block stays on A.
    Ranges: '15-17:b:ffn' expands to 15,16,17. Multi-tissue: '15:b:ffn+attn'.
    Kinds: ffn / attn / ssm / norm (see kind_of) + ln (any *norm* tensor,
    e.g. attn_norm/post_attention_norm/ssm_norm — the gain staging)."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        segs = [s.strip() for s in part.split(":")]
        blk_spec, donor = segs[0], segs[1].lower() if len(segs) > 1 else "a"
        assert donor in ("a", "b", "c", "d"), "map donor must be a/b/c/d: %r" % part
        kinds = None
        if len(segs) > 2 and segs[2]:
            kinds = frozenset(k.strip().lower() for k in segs[2].split("+"))
            assert kinds <= {"ffn", "attn", "ssm", "norm", "ln"}, \
                "map kinds must be ffn/attn/ssm/norm/ln: %r" % part
        m = re.fullmatch(r"(\d+)-(\d+)", blk_spec)
        blks = range(min(int(m.group(1)), int(m.group(2))),
                     max(int(m.group(1)), int(m.group(2))) + 1) if m \
            else (int(blk_spec),)
        for blk in blks:
            out[blk] = (donor, kinds)
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
    ap.add_argument("--d", default=None, help="fourth donor (e.g. MiMo blk31 onto M2 base)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--swap", default=None,
                    help="comma list of block ids to take from B "
                         "(default: odd blocks from B, even from A)")
    ap.add_argument("--map", default=None,
                    help="block->donor map, e.g. '15:b,19:b,31:c' "
                          "(donors a/b/c/d; overrides --swap for listed blocks). "
                          "TISSUE mode: '15:b:ffn' grafts only FFN-kind tensors "
                          "of block 15 (kinds: ffn/attn/ssm/norm, '+' for several; "
                          "'15-17:b:ffn' for ranges), rest stays on A")
    ap.add_argument("--soup", default=None,
                    help="blocks to weight-average, e.g. '0-8,11' "
                         "(backbone only — never divergent blocks)")
    ap.add_argument("--soup-from", default=None,
                    help="donors to average, e.g. 'a,c' (default: all given)")
    ap.add_argument("--dry", action="store_true",
                    help="print common trunk + donor map and exit "
                         "(no 18GB write)")
    ap.add_argument("--expect", default=None,
                    help="gate the resolved donor map, e.g. "
                         "'15:b,16:b,17:b,31:c' (normalised like --map: "
                         "ranges expanded, donors/kinds lowercased, kinds "
                         "sorted; tissue '15-17:b:ffn' allowed). Exits "
                         "non-zero with a diff on any mismatch, before a "
                         "single output byte is written. Silent when unused.")
    args = ap.parse_args()

    donors = {"a": gguf.GGUFReader(args.a), "b": gguf.GGUFReader(args.b)}
    if args.c:
        donors["c"] = gguf.GGUFReader(args.c)
    if args.d:
        donors["d"] = gguf.GGUFReader(args.d)
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
    for blk, (d, _k) in mmap.items():
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
            md, mk = mmap[lyr]
            if mk is None or kind_of(name) in mk or \
                    ("ln" in mk and "norm" in name):
                donor, soup = md, False
            else:
                donor, soup = "a", False  # tissue: rest of block stays home
        elif swapset is not None:
            donor, soup = ("b" if lyr in swapset else "a"), False
        elif mmap:
            # explicit --map means base-A-plus-overrides (scar 2026-09-23:
            # N1/N1m passed --map with 2 donors and silently got the legacy
            # odd/even interleave instead of a base — half-foreign models
            # misread as 3-block grafts). Unmapped blocks stay on A.
            donor, soup = "a", False
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

    for blk, (md, mk) in mmap.items():
        if mk is not None and blk in donor_of_block:
            donor_of_block[blk] = "%s(%s-only)" % (md, "+".join(sorted(mk)))
    print("donor map: " + ", ".join(
        "%d:%s" % (b, donor_of_block[b]) for b in sorted(donor_of_block)), flush=True)
    if args.expect is not None:
        # scar 2026-09-23: N1/N1m --map silently fell back to interleave and
        # nobody diffed the printed map against the recipe. --expect turns
        # that printed line into a gate: every expected block must be
        # EXPLICIT in --map and resolve to the expected donor/tissue.
        exp = parse_map(args.expect)
        def canon(blk, dk):
            d, k = dk
            return "%d:%s%s" % (blk, d, (":" + "+".join(sorted(k))) if k else "")
        errs = []
        for b in sorted(exp):
            if b not in mmap:
                errs.append("block %d: expected %s but not listed in --map "
                            "(would be silent fallback)" % (b, canon(b, exp[b])))
            elif b not in donor_of_block:
                errs.append("block %d: expected %s but block absent from "
                            "base model" % (b, canon(b, exp[b])))
            else:
                ed, ek = exp[b]
                want = ed if ek is None else \
                    "%s(%s-only)" % (ed, "+".join(sorted(ek)))
                if donor_of_block[b] != want:
                    errs.append("block %d: expected %s, resolved %s"
                                % (b, want, donor_of_block[b]))
        for b in sorted(set(mmap) - set(exp)):
            errs.append("extra --map block not in --expect: %s" % canon(b, mmap[b]))
        if errs:
            print("EXPECT MISMATCH (%d):" % len(errs), flush=True)
            for e in errs:
                print("  " + e, flush=True)
            sys.exit(2)
        print("expect: map matches (%d blocks)" % len(exp), flush=True)
    print("globals: a, trunk tensors: %d" % len(ta), flush=True)
    if args.dry:
        print("dry run: no output written", flush=True)
        return

    n_tensors = len(ta)
    hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors)
    hdr += struct.pack("<Q", len(parts))
    hdr += kv_raw + ti_raw
    hdr += b"\x00" * ((32 - len(hdr) % 32) % 32)
    print("header %d bytes" % len(hdr), flush=True)

    # SOUP SHAPE GUARD (the trunk gate at line ~176 covers every tensor
    # present in ALL donors by exact shape — including the cross-arch case,
    # which fails loudly there, never a silent corrupt file. The one hole:
    # a soup tensor missing from some donor is NOT in the common trunk, so
    # its holders were never shape-checked and soup_average would broadcast
    # garbage. Fail here, before a single output byte is written.)
    def _ne(t):
        n = 1
        for d in t.shape:
            n *= int(d)
        return n
    for name, _donor, _soup in order:
        if not _soup:
            continue
        want = _ne(ta[name])
        holders = [d for d in sfrom if name in T[d]] or ["a"]
        for d in holders:
            have = _ne(T[d][name])
            assert have == want, \
                "shape mismatch: soup %s from %s has %d elements, " \
                "base has %d" % (name, d, have, want)

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
