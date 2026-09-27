#!/usr/bin/env python3
"""ledger.py — standings tables GENERATED from the artefacts, not typed.

Every standings table in the ledgers is hand-transcribed, and they have
already drifted apart: MTP.md says x1.53 where SAGA.md says 1.51x,
README §6 claims a 5-23% draft acceptance that MTP.md never measured, and
RINIQ-NEXT.md still has an older table reading "N4c ... HE running" for a
build whose verdict is printed six lines above it. Every one of those is a
typing error, and none of them is a measurement error.

This renders the same tables from eval_results + the manifest, so:
  * a number can only appear if the file says so
  * every number travels with its identity status, so a reader cannot
    mistake a contemporaneous measurement for an established build
  * `--check` diffs a ledger against the artefacts and lists drift

Usage:
    python scripts/ledger.py                     # standings
    python scripts/ledger.py --vs riniqn2        # crown duel, all 3 columns
    python scripts/ledger.py --check README.md   # what that ledger gets wrong
    python scripts/ledger.py --json
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest import (tags, collect, attach_provenance, PERMANENT, ROOT,
                      RESULTS)

LEDGER_DIR = ROOT

# The predecessor project's battery archive. MERNIK's own ledgers quote
# numbers from it (README §5 "Reference Results I/II" are all NeoHorse and
# OxCoder, produced under ASHQ1), and nothing in this repo pointed there —
# so `ledger.py --check` called them non-reproducible when they are merely
# filed elsewhere. Read-only, never used for the standings table: those
# builds belong to another era and another model family.
EXTERNAL_RESULTS = ["/mnt/Vsio/ASHQ1 battlefield/eval_results"]


def archive_hits(score):
    """External tags carrying the same N/164, for orphan resolution."""
    hits = []
    for d in EXTERNAL_RESULTS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".jsonl_results.jsonl"):
                continue
            ok = n = 0
            try:
                for line in open(os.path.join(d, f)):
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    n += 1
                    ok += bool(r.get("passed"))
            except Exception:
                continue
            if ok == score and n:
                hits.append("%s (%s)" % (f.replace("humaneval_", "")
                                         .replace(".jsonl_results.jsonl", ""),
                                         os.path.basename(d.rstrip("/"))))
    return hits


def load_all():
    recs = [collect(t) for t in tags()]
    attach_provenance(recs)
    return recs


def _n(rec, key):
    v = rec["counts"].get(key)
    return int(v.split("/")[0]) if v else None


def mcnemar_p(a, b, xa, xb):
    common = sorted(set(a) & set(b))
    ab = sum(1 for t in common if a[t] and not b[t])
    ba = sum(1 for t in common if b[t] and not a[t])
    n = ab + ba
    if n == 0:
        return 1.0, ab, ba, len(common)
    k = min(ab, ba)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return p, ab, ba, len(common)


def _vecs(tag, results_dir=RESULTS):
    """Per-task vectors for a tag, the three columns the way duel.py reads
    them (human_eval for HE, evalplus for HE+, completions for empties)."""
    p = os.path.join(results_dir, "humaneval_%s" % tag)
    out = {}
    f = p + ".jsonl_results.jsonl"
    if os.path.exists(f):
        out["base"] = {json.loads(l)["task_id"]: bool(json.loads(l)["passed"])
                       for l in open(f) if l.strip() and json.loads(l).get("passed") is not None}
    f = p + "_eval_results.json"
    if os.path.exists(f):
        ev = json.load(open(f)).get("eval", {})
        out["plus"] = {t: (r[0] if isinstance(r, list) else r).get("plus_status") == "pass"
                       for t, r in ev.items()
                       if (r[0] if isinstance(r, list) else r).get("plus_status") is not None}
    f = p + ".jsonl"
    if os.path.exists(f):
        out["empties"] = {json.loads(l)["task_id"]: not json.loads(l)["completion"].strip()
                          for l in open(f) if l.strip()}
    return out


def standings(recs, limit=None):
    rows = [r for r in recs if r["counts"].get("HE")]
    rows.sort(key=lambda r: -_n(r, "HE"))
    if limit:
        rows = rows[:limit]
    head = (f"{'build':<26} {'HE':>9} {'HE+':>9} {'empty':>7}  identity")
    lines = [head, "-" * len(head)]
    for r in rows:
        idn = r["identity"]
        mark = "ok" if idn in ("verified", "reconstructed") else PERMANENT
        lines.append(f"{r['tag']:<26} {r['counts']['HE']:>9} "
                     f"{r['counts'].get('HE+','-'):>9} "
                     f"{r['counts'].get('empties','-'):>7}  {mark}")
    return "\n".join(lines)


def duel(recs, ref, other):
    by = {r["tag"]: r for r in recs}
    for t in (ref, other):
        if t not in by:
            raise SystemExit("ledger: unknown tag %r" % t)
    va, vb = _vecs(ref), _vecs(other)
    print(f"{ref}  vs  {other}\n")
    sig = 0
    for col, label in (("base", "HE  "), ("plus", "HE+ "), ("empties", "empty")):
        if col not in va or col not in vb:
            print(f"  {label} n/a")
            continue
        xa = sum(1 for t in va[col] if va[col][t])
        xb = sum(1 for t in vb[col] if vb[col][t])
        p, ab, ba, n = mcnemar_p(va[col], vb[col], xa, xb)
        s = p < 0.05
        sig += s
        extra = "  (fewer is better)" if col == "empties" else ""
        print(f"  {label} {xa:3d}/{n} vs {xb:3d}/{n}   p={p:.4f}  "
              f"{'SIGNIFICANT' if s else 'noise'}{extra}")
    print(f"\n  VERDICT: {sig} of 3 columns separate them."
          + ("  Crown may move." if sig == 3 else "  Crown stays."))
    for t in (ref, other):
        print(f"    {t:<24} identity: {by[t]['identity']}")


def check(recs, ledger):
    path = os.path.join(LEDGER_DIR, ledger) if not os.path.isabs(ledger) else ledger
    if not os.path.exists(path):
        raise SystemExit("ledger: no such file %s" % path)
    text = open(path, encoding="utf-8", errors="replace").read()
    quoted = set(re.findall(r"\d{1,3}/164", text))
    have = {}
    for r in recs:
        he = r["counts"].get("HE")
        if he:
            have.setdefault(he, []).append(r["tag"])

    unlogged = [r["tag"] for r in recs
                if r["counts"].get("HE") and r["counts"]["HE"] not in quoted]
    orphan = sorted(q for q in quoted if q not in have)

    print(f"{ledger}: {len(quoted)} distinct N/164 values quoted, "
          f"{len(have)} exist in the artefacts")
    print(f"\n  scored here, NOT quoted in that ledger: {len(unlogged)}")
    for t in unlogged:
        print(f"     {t:<28} {next(r for r in recs if r['tag']==t)['counts']['HE']}")
    print(f"\n  quoted in that ledger, NOT reproducible from THIS repo: {len(orphan)}")
    for q in sorted(orphan, key=lambda s: -int(s.split("/")[0])):
        n = int(q.split("/")[0])
        hits = archive_hits(n)
        if hits:
            print(f"     {q:>9}  -> filed in the ASHQ1 archive: {', '.join(hits)}")
        else:
            print(f"     {q:>9}  -> NOT FOUND anywhere, including the archive")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--vs", nargs=2, metavar=("REF", "OTHER"))
    ap.add_argument("--check", metavar="LEDGER")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()

    recs = load_all()

    if args.vs:
        duel(recs, *args.vs)
        return
    if args.check:
        check(recs, args.check)
        return
    if args.json:
        print(json.dumps(recs, indent=2))
        return
    print(standings(recs, args.top or None))
    perm = sum(1 for r in recs if r["identity"] == PERMANENT)
    print(f"\n{len(recs)} batteries; {perm} permanently unverifiable — their "
          f"scores are contemporaneous measurements, the served build is not.")


if __name__ == "__main__":
    main()
