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
from collections import Counter

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


COL_P = ("HE", "HE+", "empty")


def load_vectors(recs, results_dir=RESULTS):
    """Read every battery's three per-task vectors ONCE. 1711 pairs × 3
    columns of file reads would take minutes; this takes about one second."""
    out = {}
    for r in recs:
        out[r["tag"]] = _vecs(r["tag"], results_dir)
    return out


def pair_verdict(va, vb):
    """Three-column paired test. Returns (verdict, per-column detail).

    verdict: 'SIGNIFICANT' every column separates them, 'MIXED' some do
    (columns disagree — say which, never average), 'NOISE' none do.
    """
    details, sig = {}, []
    for col in ("base", "plus", "empties"):
        if col not in va or col not in vb:
            continue
        a, b = va[col], vb[col]
        common = sorted(set(a) & set(b))
        if not common:
            continue
        xa = sum(1 for t in common if a[t])
        xb = sum(1 for t in common if b[t])
        p, ab, ba, n = mcnemar_p(a, b, xa, xb)
        hit = p < 0.05
        if hit:
            sig.append(col)
        details[COL_P[("base", "plus", "empties").index(col)]] = {
            "A": xa, "B": xb, "n": n, "delta": xa - xb,
            "p": p, "A_only": ab, "B_only": ba, "sig": hit}
    if not details:
        return "NO-DATA", {}
    if len(sig) == len(details):
        v = "SIGNIFICANT"
    elif sig:
        v = "MIXED"
    else:
        v = "NOISE"
    return v, details


def pairs_report(recs, vectors, min_delta=3, family_filter=None):
    """All within-family pairs, ranked. Cross-family pairs are deliberately
    NOT computed: comparing a MiMo build to a Prism build measures the base
    model, not the allocation — the same objection that killed the ASHQ1
    archive idea. The count of suppressed pairs is reported, not hidden."""
    from manifest import family_of
    fams = {}
    for r in recs:
        if not r["counts"].get("HE"):
            continue
        fams.setdefault(family_of(r["tag"]) or "?", []).append(r["tag"])

    rows, suppressed = [], 0
    for fam, tags in sorted(fams.items()):
        if family_filter and fam.lower() != family_filter.lower():
            continue
        tags.sort()
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = tags[i], tags[j]
                v, det = pair_verdict(vectors.get(a, {}), vectors.get(b, {}))
                if v == "NO-DATA":
                    continue
                d = det.get("HE", {}).get("delta", 0)
                rows.append({"family": fam, "A": a, "B": b, "verdict": v,
                             "delta": d, "cols": det})
    all_tags = [t for ts in fams.values() for t in ts]
    suppressed = len(all_tags) * (len(all_tags) - 1) // 2 - len(rows)

    rows.sort(key=lambda r: (-abs(r["delta"]), r["A"]))
    by_verdict = Counter(r["verdict"] for r in rows)
    sig_rows = [r for r in rows if r["verdict"] == "SIGNIFICANT"]
    mixed_rows = [r for r in rows if r["verdict"] == "MIXED"]

    head = (f"{'family':<7} {'A':<22} {'B':<22} {'dHE':>4} {'p(HE)':>7} "
            f"{'dHE+':>5} {'p(HE+)':>7}  verdict")
    print(head)
    print("-" * len(head))
    for r in rows:
        if abs(r["delta"]) < min_delta and r["verdict"] == "NOISE":
            continue
        c = r["cols"]
        d1 = c.get("HE", {}).get("delta", 0)
        d2 = c.get("HE+", {}).get("delta", 0)
        p1 = c.get("HE", {}).get("p", float("nan"))
        p2 = c.get("HE+", {}).get("p", float("nan"))
        mark = {"SIGNIFICANT": "***", "MIXED": "*  ", "NOISE": "   "}[r["verdict"]]
        print(f"{r['family']:<7} {r['A']:<22} {r['B']:<22} {d1:>+4d} {p1:>7.4f} "
              f"{d2:>+5d} {p2:>7.4f}  {mark} {r['verdict']}")

    print(f"\n{len(rows)} within-family pairs computed, {suppressed} cross-family "
          f"pairs suppressed by design (they would measure the base model).")
    print(f"  SIGNIFICANT (3/3 columns): {by_verdict.get('SIGNIFICANT', 0)}")
    print(f"  MIXED (columns disagree):    {by_verdict.get('MIXED', 0)}")
    print(f"  NOISE (0/3):                 {by_verdict.get('NOISE', 0)}")
    if sig_rows:
        smallest = min(abs(r["delta"]) for r in sig_rows)
        print(f"\n  Smallest |delta| that reached 3/3 significance: {smallest} tasks "
              f"({smallest/164*100:.1f}pp) — the instrument's real resolution, "
              f"measured from the lab's own data.")
    big_noise = [r for r in rows if r["verdict"] == "NOISE" and abs(r["delta"]) >= 8]
    if big_noise:
        print(f"  ...and {len(big_noise)} pairs with |delta| >= 8 tasks that are "
              f"still NOISE — the exact width of the noise band.")

    mix = [r for r in rows if r["verdict"] == "MIXED"]
    if mix:
        print("\n  MIXED broken down by the EXACT set of significant columns "
              "(alpha=0.05, two-sided exact McNemar):")
        buckets = Counter()
        for r in mix:
            sig = tuple(c for c in COL_P if r["cols"].get(c, {}).get("sig"))
            buckets[sig or ("none",)] += 1
        for sig, cnt in buckets.most_common():
            label = " + ".join(sig)
            print(f"     only {label:<14} {cnt:3d} pairs")
        empt_only = [r for r in mix
                     if r["cols"].get("empty", {}).get("sig")
                     and not r["cols"].get("HE", {}).get("sig")
                     and not r["cols"].get("HE+", {}).get("sig")]
        if empt_only:
            print(f"     -> in {len(empt_only)} of them empties is the ONLY column "
                  f"that speaks.")

    print("\n" + "=" * 72)
    print("Where only empties separates two builds, capability is silent and")
    print("decisiveness talks — the verdict column cannot rank those, and the")
    print("empties column is the single loudest signal in the lab.  — BIG,")
    print("  asked verbatim to be put here; his line, my footer.")
    return rows


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
    ap.add_argument("--pairs", action="store_true",
                    help="every within-family pair, three columns, verdict")
    ap.add_argument("--family", default=None,
                    help="restrict --pairs to one family (mimo/riniq/prism/...)")
    ap.add_argument("--min-delta", type=int, default=3,
                    help="--pairs: hide |d| below this for NOISE rows")
    args = ap.parse_args()

    recs = load_all()

    if args.vs:
        duel(recs, *args.vs)
        return
    if args.check:
        check(recs, args.check)
        return
    if args.pairs:
        pairs_report(recs, load_vectors(recs), args.min_delta, args.family)
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
