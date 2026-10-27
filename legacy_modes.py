"""Archived utility modes.

These are historical lenses, kept only so old recipes and ledgers remain
reproducible. They are hidden from the CLI unless --legacy-modes is passed,
and Gnom is the active allocation path. New work should not add modes here.

Deleted outright (dead by construction / lost duel / unused):
  hybrid     — concentration discount was never causal
  pw_ssim    — circular: importance x importance
  netdmg     — Gnom modulator, lost its duel; superseded by gnom
"""
import math
import warnings
from typing import Any, Dict, List

import numpy as np

LEGACY_MODES = (
    "rmse", "huber", "logcosh", "srmse", "ssim", "smape_ssim",
    "smape_frag", "mix", "balance", "logratio", "powalpha", "recovery",
)


def apply_hinge(gain: float, next_tier: str, hw: float) -> float:
    if hw <= 0:
        return gain
    from constants import TIER_BPW
    tau = 0.8
    bw = float(TIER_BPW.get(next_tier, 8.0))
    return gain * (1.0 + hw / (1.0 + math.exp((bw - 4.0) / tau)))


def legacy_gain(mode: str, cur_tier: str, next_tier: str,
                g_names: List[str], tensor_importance: Dict[str, float],
                uopt: Dict[str, Any]) -> float:
    from classifier import _gain_mode, _mse_delta, _mse_eff, _scale
    uopt = uopt or {}

    if mode == "mix":
        top = uopt.get("mix_top") or frozenset()
        rep = g_names[0] if g_names else ""
        base = "mse" if rep in top else uopt.get("mix_base", "smape")
        return _scale(base, cur_tier, next_tier, uopt) * _gain_mode(
            base, cur_tier, next_tier, g_names, tensor_importance, uopt)

    if mode == "balance":
        return _gain_mode("smse", cur_tier, next_tier, g_names,
                          tensor_importance, uopt)

    if mode == "logratio":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        if en <= 0 or ec <= 0:
            return 0.0
        return math.log(ec / en)

    if mode == "powalpha":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        a = float(uopt.get("pow_alpha", 0.5))
        return (ec ** a - en ** a) if ec > 0 and en > 0 else 0.0

    if mode == "recovery":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        if ec <= 0 or en > ec:
            return 0.0
        return (ec - en) / ec

    if mode == "rmse":
        return math.sqrt(_mse_eff(cur_tier)) - math.sqrt(_mse_eff(next_tier))

    if mode == "huber":
        d = _mse_delta(cur_tier, next_tier)
        s = uopt.get("huber_delta", 3e-4)
        if d < 0:
            return d
        return d * d / (2 * s) if d < s else s * (d / s - 0.5)

    if mode == "logcosh":
        d = _mse_delta(cur_tier, next_tier)
        s = uopt.get("huber_delta", 3e-4)
        if d < 0:
            return d
        return math.log(math.cosh(d / s)) * s

    if mode == "srmse":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        if ec + en <= 0:
            return 0.0
        rel = 2.0 * (ec - en) / (ec + en)
        rba = math.sqrt(ec) - math.sqrt(en)
        if rel <= 0.0 or rba <= 0.0:
            return 0.0
        s_ref = _scale("smape", cur_tier, next_tier, uopt)
        r_ref = _scale("rmse", cur_tier, next_tier, uopt)
        if s_ref <= 0.0 or r_ref <= 0.0:
            return 0.0
        return math.sqrt((rel / s_ref) * (rba / r_ref))

    if mode in ("ssim", "smape_ssim"):
        return _ssim_gain(cur_tier, next_tier, g_names, tensor_importance,
                          uopt.get("ssim_table"), relative=(mode == "smape_ssim"))

    if mode == "smape_frag":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        base = 0.0 if ec + en <= 0 else 2.0 * (ec - en) / (ec + en)
        frag = _group_fragility(g_names, uopt.get("ssim_table"))
        return base * (1.0 + uopt.get("frag_w", 0.5) * (frag - 1.0))

    return 0.0


def _ssim_gain(cur_tier: str, next_tier: str, g_names: List[str],
               tensor_importance: Dict[str, float],
               table_path: str | None, relative: bool = False) -> float:
    from constants import TIER_BPW
    from classifier import _mse_delta
    table = _ssim_table(table_path)
    if not table:
        return _mse_delta(cur_tier, next_tier)

    def s_of(name: str, tier: str) -> float | None:
        row = table.get(name)
        if row is None:
            return None
        if tier in row:
            return row[tier]
        if tier == "F16":
            return 1.0
        b = TIER_BPW.get(tier)
        pts = sorted((TIER_BPW[t], s) for t, s in row.items() if t in TIER_BPW)
        if b is None or not pts:
            return None
        if b <= pts[0][0]:
            b0, s0 = pts[0]
            b1, s1 = pts[1] if len(pts) > 1 else (b0 + 1.0, 1.0)
            s = s0 + (s1 - s0) * (b - b0) / (b1 - b0)
            return max(0.0, min(1.0, s))
        for (b0, s0), (b1, s1) in zip(pts, pts[1:]):
            if b0 <= b <= b1:
                f = (b - b0) / (b1 - b0) if b1 > b0 else 0.0
                return s0 + (s1 - s0) * f
        return pts[-1][1]

    num, den = 0.0, 0.0
    for n in g_names:
        t = tensor_importance.get(n, 0.0)
        sc, sn = s_of(n, cur_tier), s_of(n, next_tier)
        if sc is None or sn is None:
            continue
        if relative:
            dc, dn = 1.0 - sc, 1.0 - sn
            g = 0.0 if dc + dn <= 0 else 2.0 * (dc - dn) / (dc + dn)
        else:
            g = sn - sc
        num += t * g
        den += t
    return num / den if den > 0 else 0.0


_SSIM_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}
_FRAG_CACHE: Dict[str, float] = {}


def _group_fragility(g_names: List[str], table_path: str | None) -> float:
    table = _ssim_table(table_path)
    if not table:
        return 1.0
    key = (table_path or "") + "|mean"
    if key not in _FRAG_CACHE:
        vals = [1.0 - row["Q4_K"] for row in table.values() if "Q4_K" in row]
        _FRAG_CACHE[key] = sum(vals) / len(vals) if vals else 1.0
    gmean = _FRAG_CACHE[key]
    ds = [1.0 - table[n]["Q4_K"] for n in g_names
          if n in table and "Q4_K" in table[n]]
    if not ds or gmean <= 0:
        return 1.0
    return (sum(ds) / len(ds)) / gmean


def _ssim_table(path: str | None) -> Dict[str, Dict[str, float]]:
    if not path or path in _SSIM_CACHE:
        return _SSIM_CACHE.get(path, {})
    try:
        z = np.load(path, allow_pickle=False)
    except Exception as e:
        warnings.warn(f"ssim table {path} unreadable ({e}); MSE fallback, "
                      f"duels compare blind", RuntimeWarning)
        _SSIM_CACHE[path] = {}
        return {}
    tiers = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    table = {}
    for name in z.files:
        arr = z[name]
        table[name] = {t: float(arr[i][0]) for i, t in enumerate(tiers)}
    _SSIM_CACHE[path] = table
    return table
