#!/usr/bin/env python3
"""Gate test for fuse_layers --expect, at the logic level, in milliseconds.

BIG's own verification was honest about its limit: the `--expect` cases
fused nothing (one file as both donors, resolution logic only). Mine ran the
real tool over 17 GB models, twelve times, which is too slow to be a test.
So this exercises the two pieces that actually decide a match — `parse_map`
normalisation and the resolved-vs-expected comparison — as pure functions,
with no models on disk at all.

The comparison block is reimplemented here DELIBERATELY, not imported: the
point of a golden net is that it keeps working if the tool's internals move.
If you change the semantics in fuse_layers.py, this file must be changed to
match ON PURPOSE, and the diff is the record that the gate changed.

    python tests/test_expect.py
"""
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from fuse_layers import parse_map   # noqa: E402  (BIG's file, read only)


def canon(blk, dk):
    d, k = dk
    return "%d:%s%s" % (blk, d, (":" + "+".join(sorted(k))) if k else "")


def compare(exp, mmap, donor_of_block):
    """Mirror of the check in fuse_layers.py:261-291. Returns a list of
    error strings; empty means the gate passes."""
    errs = []
    for b in sorted(exp):
        if b not in mmap:
            errs.append("not listed in --map (would be silent fallback)")
        elif b not in donor_of_block:
            errs.append("block absent from base model")
        else:
            ed, ek = exp[b]
            want = ed if ek is None else "%s(%s-only)" % (ed, "+".join(sorted(ek)))
            if donor_of_block[b] != want:
                errs.append("expected %s, resolved %s" % (want, donor_of_block[b]))
    for b in sorted(set(mmap) - set(exp)):
        errs.append("extra --map block not in --expect: %s" % canon(b, mmap[b]))
    return errs


def resolved(mmap, kind_only=None):
    """What the tool's own print shows for a resolved map: tissue blocks get
    the (kinds-only) suffix, whole blocks the bare donor."""
    out = {}
    for b, (d, k) in mmap.items():
        out[b] = d if k is None else "%s(%s-only)" % (d, "+".join(sorted(k)))
    return out


CASES = [
    # (name, map spec, expect spec, should_pass)
    ("match, whole block", "15:b", "15:b", True),
    ("range in map, expanded in expect", "15-17:b", "15:b,16:b,17:b", True),
    ("expect in reverse order", "15:b,31:c", "31:c,15:b", True),
    ("whitespace everywhere", "15:b, 31:c", " 15:b ,  31:c ", True),
    ("uppercase donor in map", "15:B", "15:b", True),
    ("reversed range in expect", "15-17:b", "17-15:b", True),
    ("tissue range, expanded", "15-17:b:ffn",
     "15:b:ffn,16:b:ffn,17:b:ffn", True),
    ("multi-tissue order-insensitive", "15:b:ffn+attn", "15:b:attn+ffn", True),

    # the scar class: something silently different
    ("donor mismatch", "15:b", "15:c", False),
    ("block absent from map", "15:b", "15:b,31:c", False),
    ("expect omits a map block", "15:b,31:c", "15:b", False),
    ("SILENT SUFFIX: map bare, expect tissue", "15:b", "15:b:ffn", False),
    ("tissue in map, expect bare", "15:b:ffn", "15:b", False),
    ("wrong tissue kind", "15:b:ffn", "15:b:attn", False),
    ("tissue against a different donor", "15:b:ffn", "15:c:ffn", False),
    ("empty expect, non-empty map", "15:b", "", False),
    ("expect with no map at all (legacy interleave)", None, "15:b", False),
    ("duplicate block in expect keeps the last", "15:c", "15:b,15:c", True),
    ("empty map and empty expect (vacuous pass)", None, "", True),
]


def main():
    fails = 0
    for name, mspec, espec, want_pass in CASES:
        mmap = parse_map(mspec) if mspec else {}
        exp = parse_map(espec) if espec else {}
        errs = compare(exp, mmap, resolved(mmap))
        ok = (not errs) == want_pass
        fails += 0 if ok else 1
        verdict = "PASS-GATE" if not errs else "BLOCKED  "
        note = errs[0][:44] if errs else ""
        print(f"{'ok  ' if ok else 'FAIL'}  {name:<42} {verdict} {note}")

    # the one property the gate exists to protect, stated as a property
    print("\nproperty check: a recipe can never pass while a block it names "
          "is missing from the map")
    leak = 0
    for mspec, espec in itertools.product(
            [None, "15:b", "15:b,31:c"], ["15:b", "15:b,31:c", "15:b,17:c"]):
        mmap = parse_map(mspec) if mspec else {}
        exp = parse_map(espec)
        missing = [b for b in exp if b not in mmap]
        if missing and not compare(exp, mmap, resolved(mmap)):
            leak += 1
    print(f"  {leak} leaks (0 required)")
    fails += leak

    print(f"\n{len(CASES) + 1 - fails} ok, {fails} failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
