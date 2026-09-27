"""preflight.py — audit what llama-quantize ACTUALLY decided, before paying for it.

The dry run already happens on every invocation (quantizer.run_dry_run) and
costs ~1 second, but its output was reduced to a single size number and the
rest thrown away. That discarded text is the only place where the binary
reports, per tensor, the type it will ACTUALLY write — after tensor-type
regex matching (std::regex_search, first match wins) and after
tensor_type_fallback(). Everything downstream assumes MERNIK's assignment
map survives into the file; nothing ever checked.

This module closes that loop for free:

  1. intended tier vs the type the binary chose, per tensor
     -> catches shadowed/unanchored --tensor-type rules and shape fallbacks
  2. real quant size vs MERNIK's analytic estimate and vs --size
     -> the size bias of the model (norms are physically F32; the estimate
        assumes F16, so it systematically undercounts)
  3. real geometry (tier histogram in true MiB) before the 30-minute write

Nothing here changes the allocation. It only reports disagreement.
"""
import re
from collections import Counter

# One line per tensor in --dry-run. The two LLAMA_LOG_INFO calls concatenate:
#   [  12/  427] blk.5.ffn_gate.weight - [11008, 4096], type = BF16, size = 85.94 MiB -> 35.16 MiB (Q5_K)
# Tensors that are not quantized print only the source size.
TENSOR_RE = re.compile(
    r"\[\s*(?P<idx>\d+)/\s*(?P<n>\d+)\]\s+(?P<name>\S+)\s+-\s+"
    r"\[(?P<shape>[^\]]*)\],\s*type\s*=\s*(?P<src>\S+?),\s*"
    r"size\s*=\s*(?P<src_mib>[\d.]+)\s*MiB"
    r"(?:\s*->\s*(?P<dst_mib>[\d.]+)\s*MiB\s*\((?P<dst>\w+)\))?"
)

TOTAL_RE = re.compile(r"quant size\s*=\s*([\d.]+)\s*MiB\s*\(([\d.]+)\s*BPW\)")
MODEL_RE = re.compile(r"model size\s*=\s*([\d.]+)\s*MiB")

OVERRIDE_RE = re.compile(
    r"(?P<name>\S+)\s+-\s+applying manual override:\s*(?P<old>\S+)\s*->\s*(?P<new>\S+)"
)


def parse(raw):
    """Parse --dry-run output into per-tensor facts + totals."""
    tensors, overrides = {}, {}
    for m in TENSOR_RE.finditer(raw):
        name = m.group("name")
        tensors[name] = {
            "src_type": m.group("src"),
            "dst_type": m.group("dst") or m.group("src"),
            "src_mib": float(m.group("src_mib")),
            "dst_mib": float(m.group("dst_mib") or m.group("src_mib")),
            "shape": m.group("shape"),
            "quantized": m.group("dst") is not None,
        }
    for m in OVERRIDE_RE.finditer(raw):
        overrides[m.group("name")] = (m.group("old"), m.group("new"))
    total = TOTAL_RE.search(raw)
    model = MODEL_RE.search(raw)
    return {
        "tensors": tensors,
        "overrides": overrides,
        "total_mib": float(total.group(1)) if total else None,
        "bpw": float(total.group(2)) if total else None,
        "model_mib": float(model.group(1)) if model else None,
    }


# Tiers the binary may physically use where MERNIK's model says otherwise.
# llama.cpp keeps 1D tensors (norms, biases) at F32 regardless of the rules;
# classifier.py already notes this for top-down. Not a bug — but it makes
# the analytic estimate a floor, not a prediction.
_PHYSICAL = {"F32"}


def _norm(t):
    """Type names differ only in case: llama.cpp prints q5_K / f32."""
    return (t or "").upper()


def audit(assignments, raw, target_mib=None, est_mib=None, verbose_limit=25):
    """Compare MERNIK's intent with the binary's decision. Returns a report dict."""
    info = parse(raw)
    got = info["tensors"]

    mismatches, physical, unassigned = [], [], []
    for name, want in assignments.items():
        if name not in got:
            continue
            # not reported by dry-run: kept as-is (embd/output via dedicated
            # flags are still reported, so this is rare)
        actual = got[name]["dst_type"]
        if _norm(actual) == _norm(want):
            continue
        if _norm(actual) in _PHYSICAL and _norm(want) in ("F16", "F32"):
            physical.append((name, want, actual))
        else:
            mismatches.append((name, want, actual))

    for name in got:
        if name not in assignments:
            unassigned.append((name, got[name]["dst_type"]))

    real = info["total_mib"]
    bias = None
    if real is not None and est_mib:
        bias = (real - est_mib) / est_mib * 100.0

    hist_c, hist_m = Counter(), Counter()
    for name, t in got.items():
        key = _norm(t["dst_type"])
        hist_c[key] += 1
        hist_m[key] += t["dst_mib"]

    report = {
        "n_parsed": len(got),
        "n_assigned": len(assignments),
        "mismatches": mismatches,
        "physical": physical,
        "unassigned": unassigned,
        "real_mib": real,
        "est_mib": est_mib,
        "bias_pct": bias,
        "target_mib": target_mib,
        "bpw": info["bpw"],
        "hist_count": hist_c,
        "hist_mib": hist_m,
        "overrides": info["overrides"],
    }
    _print(report, verbose_limit)
    return report


def _print(r, limit):
    print("\n  --- PREFLIGHT (llama-quantize --dry-run, what the binary will do) ---")
    print(f"  tensors reported: {r['n_parsed']}   assigned by MERNIK: {r['n_assigned']}")

    if r["real_mib"] is not None:
        line = f"  real quant size: {r['real_mib']:.0f} MiB"
        if r["bpw"]:
            line += f"  ({r['bpw']:.2f} BPW)"
        if r["target_mib"]:
            d = r["real_mib"] - r["target_mib"]
            line += f"   target {r['target_mib']:.0f} MiB -> " + (
                f"OVER by {d:.0f}" if d > 0 else f"under by {-d:.0f}")
        print(line)
    if r["est_mib"] is not None and r["bias_pct"] is not None:
        print(f"  MERNIK estimate:  {r['est_mib']:.0f} MiB   "
              f"bias {r['bias_pct']:+.2f}%  (est {'under' if r['bias_pct'] > 0 else 'over'}shoots)")

    if r["mismatches"]:
        print(f"\n  !! {len(r['mismatches'])} tensor(s) the binary will NOT write as "
              f"assigned (rule shadowing / unanchored regex / shape fallback):")
        for name, want, actual in r["mismatches"][:limit]:
            print(f"     {name[:52]:52s} want={want:8s} got={actual}")
        if len(r["mismatches"]) > limit:
            print(f"     ... and {len(r['mismatches']) - limit} more")
    else:
        print("  assignment map survived intact (no shadowed rules)")

    if r["physical"]:
        print(f"  note: {len(r['physical'])} 1D/norm tensor(s) physically F32 "
              f"though the model says F16 — expected, and it makes the "
              f"estimate a floor")

    if r["unassigned"]:
        print(f"  note: {len(r['unassigned'])} tensor(s) not in the assignment map "
              f"(binary default): {', '.join(n.split('.')[-2] + '=' + t for n, t in r['unassigned'][:6])}")

    if r["hist_count"]:
        print("  real geometry (by MiB):")
        for t in sorted(r["hist_mib"], key=lambda k: -r["hist_mib"][k]):
            print(f"    {t:10s} {r['hist_count'][t]:4d}  {r['hist_mib'][t]:8.1f} MiB")
