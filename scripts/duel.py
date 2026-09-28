#!/usr/bin/env python3
"""duel.py — paired significance for two HE batteries, ALL THREE columns.

The protocol says a claim is unverified unless all three columns carry it
(README §1: PPL canary / KLD rank / tasks verdict; in practice the three
verdict-side columns are HE pass@1, HE+ rescore, and empties). A single
column decides nothing on its own: the lab's own noise band is ~±3.6pp, so
a ±1-2pp "crown" sits inside it. This tool answers the only question that
matters: how many of the three columns actually separate the two arms.

Usage:
    python scripts/duel.py A B [C ...]      # 2 = duel, 3+ = ranking
    python scripts/duel.py --column base A B   # focus one column

Accepted inputs (schema auto-detected, harness reported):
  *.jsonl_results.jsonl  human_eval runner -> passed (bool)  == HE column
  *_eval_results.json    EvalPlus rescore  -> base_status / plus_status
  <tag>.jsonl            completions       -> "" == empty completion

HARNESS LAW (FUSION.md, HE+ ledger footnote): EvalPlus base and the
human_eval runner disagree by ~0.5pp on the SAME completions. A duel across
harnesses measures the harness, not the build — refused, not averaged.
"""
import json
import math
import os
import sys

COLUMNS = ("base", "plus", "empties")
LABEL = {"base": "HE  ", "plus": "HE+ ", "empties": "empty"}


def _tag_of(path):
    """Reduce any result artefact to its battery tag, so the three columns
    can each find their own file inside the same family."""
    for suf in (".jsonl_results.jsonl", "_eval_results.json", ".jsonl", ".json"):
        if path.endswith(suf):
            return path[: -len(suf)]
    return path


def _roots(tag):
    """Accept a bare tag as well as a full artefact path. Result files are
    written with a `humaneval_` prefix, so both spellings have to resolve."""
    out = [tag]
    if not tag.startswith("humaneval_"):
        out.append("humaneval_" + tag)
    return out


def _pick(tag, column):
    """(file, kind) for one column inside a battery family. kind decides the
    schema: 'human' = .jsonl_results.jsonl (passed), 'evalplus' =
    _eval_results.json (base_status/plus_status), 'raw' = completions."""
    if column == "empties":
        names, kinds = (".jsonl",), ("raw",)
    elif column == "plus":
        names, kinds = ("_eval_results.json",), ("evalplus",)
    else:
        names = (".jsonl_results.jsonl", "_eval_results.json")
        kinds = ("human", "evalplus")
    for root in _roots(tag):
        for name, kind in zip(names, kinds + names[len(kinds):]):
            f = root + name
            if os.path.exists(f):
                return f, kind
    return None, None


def load(path, column):
    """Return {task_id: value} where value is a pass, plus the harness name."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl") and column != "empties":
        out = {}
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("passed") is None:   # never bool(None): null is not a fail
                continue
            out[r["task_id"]] = bool(r["passed"])
        if not out:
            raise ValueError("no passed/ results in %s" % path)
        return out, "human_eval"
    if path.endswith(".jsonl") and column == "empties":
        out = {}
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l)
            out[r["task_id"]] = not r["completion"].strip()
        return out, "completions"

    ev = json.load(open(path)).get("eval", {})
    out, harness = {}, "unknown"
    for task, recs in ev.items():
        r = recs[0] if isinstance(recs, list) else recs
        if not isinstance(r, dict):
            continue
        if column == "plus" and r.get("plus_status") is not None:
            harness = "evalplus+"
            out[task] = (r["plus_status"] == "pass")
        elif column != "plus" and r.get("base_status") is not None:
            harness = "evalplus"
            out[task] = (r["base_status"] == "pass")
        elif column != "plus" and r.get("passed") is not None:
            harness = "human_eval"
            out[task] = bool(r["passed"])
    if not out:
        raise ValueError("no %r results in %s" % (column, path))
    return out, harness


def arm(path, column):
    tag = _tag_of(path)
    f, kind = _pick(tag, column)
    if f is None:
        raise SystemExit("duel: no %r results for tag %r" % (column, tag))
    return load(f, column)


def wilson(x, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = x / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / den, (c + m) / den)


def mcnemar(b, c):
    """Exact two-sided binomial on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    argv = sys.argv[1:]
    only = None
    if "--column" in argv:
        i = argv.index("--column")
        only = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cols = [only] if only in COLUMNS else list(COLUMNS)


    arms = []
    for p in argv:
        loaded = {c: arm(p, c) for c in cols}
        arms.append((p, loaded))

    # Harness check per column, on the column that actually carries results.
    for c in cols:
        harnesses = {loaded[c][1] for _, loaded in arms}
        if len(harnesses) > 1:
            raise SystemExit(
                "duel: REFUSED — %s arms come from different harnesses %s.\n"
                "       The same completions score ~0.5pp apart across harnesses;\n"
                "       a cross-harness duel measures the harness, not the build."
                % (c, sorted(harnesses)))

    for c in cols:
        base = arms[0][1][c][0]
        n = len(base)
        print("\n=== %s  (n=%d, harness=%s) ===" % (LABEL[c], n, arms[0][1][c][1]))
        for path, loaded in arms:
            vec = loaded[c][0]
            common = sorted(set(vec) & set(base))
            x = sum(1 for t in common if vec[t])
            if c == "empties":
                print(f"  {os.path.basename(path):<42} {x:3d}/{len(common)} empty "
                      f"({x/len(common):5.1%})   fewer is better")
            else:
                lo, hi = wilson(x, len(common))
                print(f"  {os.path.basename(path):<42} {x:3d}/{len(common)} = "
                      f"{x/len(common):6.2%}  CI95 [{lo:.2%}, {hi:.2%}]")
        if len(arms) == 2:
            a, b = arms[0][1][c][0], arms[1][1][c][0]
            common = sorted(set(a) & set(b))
            ab = sum(1 for t in common if a[t] and not b[t])
            ba = sum(1 for t in common if b[t] and not a[t])
            p = mcnemar(ab, ba)
            print(f"  paired: A-only={ab}  B-only={ba}  McNemar exact p={p:.4f}"
                  f"  ->  {'SIGNIFICANT' if p < 0.05 else 'noise'}")

    if len(arms) > 2:
        print("\n=== ranking (pairwise McNemar needed before any crown moves) ===")
        for c in cols:
            common = set(arms[0][1][c][0])
            for _, loaded in arms[1:]:
                common &= set(loaded[c][0])
            ranked = sorted(arms, key=lambda x: -sum(1 for t in common if x[1][c][0][t]))
            print(f"  {LABEL[c]}: " + "  ".join(
                f"{os.path.basename(p)[:28]}={sum(1 for t in common if ld[c][0][t])}"
                for p, ld in ranked))
        return

    # The only verdict that means anything: how many columns separate them.
    sig = []
    for c in cols:
        a, b = arms[0][1][c][0], arms[1][1][c][0]
        common = sorted(set(a) & set(b))
        if not common:
            continue
        ab = sum(1 for t in common if a[t] and not b[t])
        ba = sum(1 for t in common if b[t] and not a[t])
        if mcnemar(ab, ba) < 0.05:
            sig.append(c)
    print("\n" + "=" * 62)
    if not sig:
        print("VERDICT: 0 of 3 columns separate these arms. NOISE.")
        print("         A crown must not move on this pair.")
    elif len(sig) == len(cols):
        print("VERDICT: all %d columns significant — the effect is real." % len(cols))
    else:
        print("VERDICT: %d of %d columns significant (%s)."
              % (len(sig), len(cols), ", ".join(LABEL[s] for s in sig)))
        print("         Columns disagree — treat the claim as UNVERIFIED and")
        print("         say which column carries it, do not average them.")


if __name__ == "__main__":
    main()
