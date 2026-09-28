#!/usr/bin/env python3
"""manifest.py — which build produced which number.

The one thing no artefact in the lab records: the result files are named
`humaneval_<tag>.*` and nothing inside them says which .gguf was served.
That identity is the foundation of every verdict — if a tag names the wrong
build, the paired duels still compute perfectly and still mean nothing.

Three tiers of evidence, never guessed:
  verified    - <tag>.meta.json exists and carries serve_model (protocol.py
                stamps it since 2026-09-28)
  inferred    - no sidecar, but the tag is corroborated by the ledger prose
  unverified  - neither. The number is reported with a flag, not dropped.

Usage:
    python scripts/manifest.py                 # human table
    python scripts/manifest.py --json          # machine-readable
    python scripts/manifest.py --audit         # integrity checks only
    python scripts/manifest.py --tag mimo4500  # one battery in detail
"""
import argparse
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval_results")
PROVENANCE = os.path.join(ROOT, "provenance")

# ASHQ1 is the ARCHIVE this project grew out of (operator-confirmed: treat it
# as archive, MERNIK is the active repo). MERNIK's ledgers quote numbers from
# it — README §5 "Reference Results I/II" are all NeoHorse/OxCoder builds made
# there — so `ledger.py --check` would call those rows non-reproducible when
# they are merely filed elsewhere. This is a SOFT hint for that one report:
# best-effort, read-only, and if the path ever disappears the tool degrades to
# "not found" instead of failing. Not a dependency, and deliberately NOT a
# battery source here: those builds are another era and another model family.
EXTERNAL_RESULTS = ["/mnt/Vsio/ASHQ1 battlefield/eval_results"]
LEDGERS = ["README.md", "FUSION.md", "README-ZOO.md", "MIMO.md",
           "RINIQ-NEXT.md", "MTP.md", "SAGA.md", "DIARY.md"]




# ------------------------------------------------------------ decisions
# Settled in AGENTS.md on 2026-09-28 by both owners. Recorded here so the
# question cannot be re-opened from a code-reading alone.
DECISIONS = {
    "2026-09-28-permanent-unverifiable": (
        "The only build->path record that ever existed was "
        "/tmp/opencode/chain.log, destroyed by a reboot; /mnt/Vsio/chain-*.log "
        "survived but covers 3 sessions. Batteries outside those sessions are "
        "marked PERMANENT-UNVERIFIABLE, not pending: no further source is "
        "coming. Their scores stand as contemporaneous ledger measurements "
        "(counted and written the same day); their served model does not "
        "stand at all. Effect size is independent evidence against a label "
        "error — a rename cannot conjure 41 tasks (MiMo-4500 SMSE vs SMAPE, "
        "3/3 columns)."),
    "2026-09-28-three-key-identity": (
        "Identity requires THREE independent keys: exact score AND battery "
        "start time AND tag-family agreement with the served path. A bare "
        "score match is coincidence trafficking — counts repeat across "
        "sessions (117, 119, 130, 104 each appear twice, days apart)."),
}

PERMANENT = "permanent-unverifiable"


# ------------------------------------------------------------ chain logs
# The only build→path record that ever existed. The runner prints
#   serving <full path> on <port>      (self-serve mode)
#   battery start: YYYY-MM-DD HH:MM:SS
#   pass@1: {'pass@1': np.float64(0.7865853658536586)}
# so each battery can be tied to a served file by TWO independent keys: the
# timestamp window and the exact score. A chain-log hit is `reconstructed`,
# not `verified` — it is evidence, and it is cited, never invented.
SERVE_RE = re.compile(r"serving\s+(\S+)\s+on\s+(\d+)")
START_RE = re.compile(r"battery start:\s*([0-9-]+\s+[0-9:]+)")
SCORE_RE = re.compile(r"pass@1:?\s*\{?'?pass@1'?:\s*np\.float64\(([0-9.]+)\)")
BATTERY_WINDOW_S = 6 * 3600


def parse_chain_logs(directory=PROVENANCE):
    """-> [{served, started, score, n_passed, source}] one per battery."""
    if not os.path.isdir(directory):
        return []
    out = []
    for fn in sorted(os.listdir(directory)):
        if not fn.endswith(".log"):
            continue
        path = os.path.join(directory, fn)
        served = None
        cur = None
        for line in open(path, encoding="utf-8", errors="replace"):
            m = SERVE_RE.search(line)
            if m:
                served = m.group(1)
                continue
            m = START_RE.search(line)
            if m:
                cur = {"served": served, "started": m.group(1),
                       "score": None, "source": "provenance/" + fn}
                out.append(cur)
                continue
            if cur is not None and cur["score"] is None:
                m = SCORE_RE.search(line)
                if m:
                    try:
                        cur["score"] = float(m.group(1))
                    except ValueError:
                        pass
    for b in out:
        if b["score"]:
            b["n_passed"] = round(b["score"] * 164)
    return [b for b in out if b["score"]]


# Tag family -> token that MUST appear in the served path. The tag name is
# independent evidence: if a chain-log hit serves a different family, that is
# a contradiction to report, never an identity to claim. Without this check
# the matcher happily bound mimo5100bal (117/164) to a PrismCoder file
# because the score and the timestamp happened to line up.
FAMILY = {
    "mimo": "MiMo", "sriq": "SRIQ", "prism": "Prism", "riniq": "RINIQ",
    "gemma": "gemma", "ling": "Ling", "minicpm": "MiniCPM", "minim": "MiniCPM",
    "fuse": "RINIQ", "trim": "RINIQ", "ox": "OxCoder",
}


def family_of(tag):
    t = tag.lower()
    for pref, token in FAMILY.items():
        if t.startswith(pref):
            return token
    return None


def _mtime(path):
    import datetime
    return datetime.datetime.fromtimestamp(os.path.getmtime(path))


def _parse_ts(s):
    import datetime
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def resolve_from_logs(tag, rec, batteries, results_dir=RESULTS):
    """Two-key match: score within one task AND start before the artefact
    mtime, inside the window. Returns (battery, how) or (None, reason)."""
    he = rec["counts"].get("HE")
    if not he or not batteries:
        return None, "no score" if he else "no HE"
    n = int(he.split("/")[0])
    comp = rec["files"].get("completions")
    mt = _mtime(comp) if comp else None
    by_score, by_time = [], []
    for b in batteries:
        if b["n_passed"] == n:
            by_score.append(b)
            st = _parse_ts(b["started"])
            if st and mt and 0 <= (mt - st).total_seconds() <= BATTERY_WINDOW_S:
                by_time.append(b)
    both = [b for b in by_score if b in by_time]
    fam = family_of(tag)
    if not both:
        if by_score:
            # A bare score is NOT identity: counts repeat across sessions
            # (117, 119, 130, 104 all occur in more than one era, days
            # apart). Report the coincidence, claim nothing.
            return None, ("unverified: %d log(s) carry the same %d/164 but from a "
                          "different session — count coincidence, not identity"
                          % (len(by_score), n))
        return None, "no surviving log carries this battery"
    how = "score+time"
    if len(both) > 1:
        return None, ("%d logs match score+time — ambiguous, not claiming"
                      % len(both))
    b = both[0]
    served = b["served"] or "?"
    if fam and fam.lower() not in served.lower():
        return b, ("CONFLICT: log says %s but the tag family is %s (%s)"
                   % (os.path.basename(served), fam, how))
    st = _parse_ts(b["started"])
    delta = (mt - st).total_seconds() / 60 if (st and mt) else float("nan")
    return b, ("%s: %s (start %s, %+.0f min before the jsonl, %d/164, "
               "family %s ok)" % (b["source"], os.path.basename(served),
                                  b["started"], delta, b["n_passed"], fam))


def tags(results_dir=RESULTS):
    """Battery tags are the COMPLETIONS file stems. `*.jsonl_results.jsonl`
    ends in .jsonl too and is the scored copy of the same battery, not a
    battery of its own."""
    out = set()
    for f in os.listdir(results_dir):
        if f.endswith(".jsonl_results.jsonl"):
            continue
        m = re.match(r"humaneval_(.+)\.jsonl$", f)
        if m:
            out.add(m.group(1))
    return sorted(out)


def _count_passed_jsonl(path):
    n = ok = 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if r.get("passed"):
            ok += 1
    return n, ok


def _empty_count(path):
    n = e = 0
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        n += 1
        if not r["completion"].strip():
            e += 1
    return n, e


def _evalplus(path, key):
    ev = json.load(open(path)).get("eval", {})
    n = ok = 0
    for _task, recs in ev.items():
        r = recs[0] if isinstance(recs, list) else recs
        if not isinstance(r, dict) or r.get(key) is None:
            continue
        n += 1
        ok += (r[key] == "pass")
    return n, ok


def collect(tag, results_dir=RESULTS):
    p = os.path.join(results_dir, "humaneval_%s" % tag)
    rec = {"tag": tag, "files": {}, "counts": {}, "identity": "unverified",
           "serve_model": None, "meta": None}
    for kind, suffix in (("completions", ".jsonl"),
                         ("human_eval", ".jsonl_results.jsonl"),
                         ("evalplus", "_eval_results.json")):
        f = p + suffix
        if os.path.exists(f):
            rec["files"][kind] = os.path.relpath(f, ROOT)
    if "human_eval" in rec["files"]:
        n, ok = _count_passed_jsonl(p + ".jsonl_results.jsonl")
        rec["counts"]["HE"] = f"{ok}/{n}"
    if "evalplus" in rec["files"]:
        n, ok = _evalplus(p + "_eval_results.json", "base_status")
        rec["counts"]["HE+base"] = f"{ok}/{n}"
        n2, ok2 = _evalplus(p + "_eval_results.json", "plus_status")
        rec["counts"]["HE+"] = f"{ok2}/{n2}"
    if "completions" in rec["files"]:
        n, e = _empty_count(p + ".jsonl")
        rec["counts"]["empties"] = f"{e}/{n}"

    side = p + ".meta.json"
    if os.path.exists(side):
        meta = json.load(open(side))
        rec["meta"] = os.path.relpath(side, ROOT)
        rec["serve_model"] = meta.get("serve_model")
        rec["git_sha"] = meta.get("git_sha")
        if meta.get("serve_model"):
            rec["identity"] = "verified"
    return rec



def attach_provenance(recs, directory=PROVENANCE, results_dir=RESULTS):
    batteries = parse_chain_logs(directory)
    for r in recs:
        b, how = resolve_from_logs(r["tag"], r, batteries, results_dir)
        if b is not None and not how.startswith("CONFLICT"):
            r["identity"] = "reconstructed"
            r["serve_model"] = b["served"]
            r["identity_note"] = how
            continue
        if b is not None:
            r["identity"] = "conflict"
            r["serve_model"] = b["served"]
            r["identity_note"] = how
            continue
        if r["identity"] == "verified":
            continue
        r["identity"] = PERMANENT
        r["identity_note"] = how + " — settled 2026-09-28, no further source"
    return batteries


# ---------------------------------------------------------------- audit
def audit(recs):
    """Checks that need no external record — pure internal consistency."""
    issues = []
    for r in recs:
        tag = r["tag"]
        c = r["counts"]
        he = c.get("HE")
        hb = c.get("HE+base")
        # 1. The two harnesses disagree by ~1 task on identical completions
        #    (FUSION.md footnote). A bigger split means the files under one
        #    tag are not from the same battery.
        if he and hb:
            a = int(he.split("/")[0]); b = int(hb.split("/")[0])
            if abs(a - b) > 3:
                issues.append((tag, "harness-gap",
                               f"HE={he} vs HE+base={hb} — more than the known "
                               f"~1-task split; these files may not be one battery"))
        # 2. column task counts must agree
        nums = {k: int(v.split("/")[1]) for k, v in c.items()}
        if len(set(nums.values())) > 1:
            issues.append((tag, "task-count",
                           f"columns disagree on task count: {c}"))
        # 3. completions present but no scored result
        if "completions" in r["files"] and "HE" not in c:
            issues.append((tag, "unscored", "completions saved, never scored"))
        # 4. identity — a settled fact, not an open issue
        if r["identity"] == "conflict":
            issues.append((tag, "conflict",
                           r.get("identity_note", "tag family contradicts the log")))
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--results", default=RESULTS,
                    help="directory of humaneval_* artefacts (default "
                         "eval_results); lets the manifest read a copy or an "
                         "archive without writing to either")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.tag:
        r = collect(args.tag, args.results)
        attach_provenance([r], results_dir=args.results)
        print(json.dumps(r, indent=2))
        return

    recs = [collect(t, args.results) for t in tags(args.results)]
    batteries = attach_provenance(recs, results_dir=args.results)
    issues = audit(recs)

    if args.json:
        print(json.dumps({"batteries": recs, "issues":
                          [{"tag": t, "kind": k, "detail": d} for t, k, d in issues]},
                         indent=2))
        return

    if args.audit:
        if not issues:
            print("audit: clean")
            return
        by_kind = {}
        for t, k, d in issues:
            by_kind.setdefault(k, []).append((t, d))
        for k, items in sorted(by_kind.items()):
            print(f"\n{k}: {len(items)}")
            for t, d in items[:12]:
                print(f"  {t:34s} {d}")
            if len(items) > 12:
                print(f"  ... and {len(items)-12} more")
        return

    verified = sum(1 for r in recs if r["identity"] == "verified")
    recon = sum(1 for r in recs if r["identity"] == "reconstructed")
    conflict = sum(1 for r in recs if r["identity"] == "conflict")
    perm = sum(1 for r in recs if r["identity"] == PERMANENT)
    print(f"batteries: {len(recs)}   verified(sidecar): {verified}   "
          f"reconstructed(chain log): {recon}   CONFLICT: {conflict}   "
          f"{PERMANENT}: {perm}")
    print(f"chain-log batteries parsed: {len(batteries)}")
    print(f"\n{'tag':<26} {'identity':<14} {'HE':>8}  served model / note")
    for r in recs:
        c = r["counts"]
        note = r["serve_model"] or r.get("identity_note", "")
        if r["identity"] == "reconstructed":
            note = os.path.basename(note)
        if r["identity"] == PERMANENT:
            note = "score stands; served model not established (settled)"
        print(f"{r['tag']:<26} {r['identity']:<24} {c.get('HE','-'):>8}  {note}")

    kinds = {}
    for _t, k, _d in issues:
        kinds[k] = kinds.get(k, 0) + 1
    if kinds:
        print("\naudit: " + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
              + "   (--audit for detail)")

    if perm:
        print("\n" + "=" * 72)
        print(f"{perm} batteries are PERMANENTLY unverifiable (not pending).")
        for key, text in DECISIONS.items():
            print(f"  [{key}] {text}")



if __name__ == "__main__":
    main()
