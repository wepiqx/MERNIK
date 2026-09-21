"""Experimental / retired zoo utility modes (kept for reuse, out of the hot path).

Moved here during the MERNIK cleanup: huber, hybrid (dead by construction),
pw_ssim (circular: importance x importance), netdmg (Gnom modulator, lost duel).
Core modes (mse/rmse/logcosh/smape/ssim/smape_ssim/smape_frag) stay in
classifier._gain. Dispatch via experimental_gain() -> None if not experimental.
"""
from typing import Dict, List, Any
import numpy as np

LINF_ERR = {
    "F16": 0.0, "Q8_0": 0.02, "Q6_K": 0.05, "Q5_K": 0.09,
    "Q4_K": 0.16, "IQ4_NL": 0.20, "IQ4_XS": 0.24,
    "Q3_K": 0.40, "IQ3_M": 0.38, "IQ3_S": 0.42, "IQ3_XXS": 0.48,
    "IQ2_S": 0.65, "IQ2_XS": 0.70, "IQ2_XXS": 0.78, "IQ1_S": 1.0,
}

_NETPRED_CACHE: Dict[str, Dict[str, float]] = {}
_PDMG_CACHE: Dict[str, Dict[str, Dict[str, float]]] = {}

EXPERIMENTAL_MODES = ("huber", "hybrid", "pw_ssim", "netdmg")


def experimental_gain_modes():
    return EXPERIMENTAL_MODES


def _group_netdmg_z(g_names: List[str], path: str | None) -> float:
    """Gnom's predicted damage for this group, z-scored. 0.0 if unknown."""
    if not path or path not in _NETPRED_CACHE:
        if not path:
            return 0.0
        try:
            import json as _json
            with open(path) as f:
                _NETPRED_CACHE[path] = {k: float(v) for k, v in
                                        _json.load(f).items()}
        except Exception:
            _NETPRED_CACHE[path] = {}
    table = _NETPRED_CACHE[path]
    if not table:
        return 0.0
    tag = "+".join(n.replace(".weight", "").split(".")[-1] + "@" +
                   n.split(".")[1] for n in g_names
                   if n.startswith("blk.")) or "global"
    if tag not in table:
        return 0.0
    vals = np.array(sorted(table.values()))
    mu, sd = vals.mean(), vals.std() + 1e-12
    return (table[tag] - mu) / sd


def _pdamage_table(path: str | None) -> Dict[str, Dict[str, float]]:
    """Perceptual (activation-weighted) damage per tensor per tier."""
    if not path or path in _PDMG_CACHE:
        return _PDMG_CACHE.get(path, {})
    try:
        z = np.load(path, allow_pickle=False)
    except Exception:
        _PDMG_CACHE[path] = {}
        return {}
    tiers = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    _PDMG_CACHE[path] = {n: {t: float(z[n][i]) for i, t in enumerate(tiers)}
                         for n in z.files}
    return _PDMG_CACHE[path]


def _pdamage_gain(cur_tier: str, next_tier: str, g_names: List[str],
                  tensor_importance: Dict[str, float],
                  table_path: str | None) -> float:
    """Importance-weighted mean perceptual-damage drop over members."""
    from classifier import _mse_delta
    from constants import TIER_BPW
    table = _pdamage_table(table_path)
    if not table:
        return _mse_delta(cur_tier, next_tier)

    def d_of(name: str, tier: str) -> float | None:
        row = table.get(name)
        if row is None:
            return None
        if tier in row:
            return row[tier]
        if tier == "F16":
            return 0.0
        b = TIER_BPW.get(tier)
        pts = sorted((TIER_BPW[t], d) for t, d in row.items() if t in TIER_BPW)
        if b is None or not pts:
            return None
        if b <= pts[0][0]:
            b0, d0 = pts[0]
            b1, d1 = pts[1] if len(pts) > 1 else (b0 + 1.0, 0.0)
            d = d0 + (d1 - d0) * (b - b0) / (b1 - b0)
            return max(0.0, d)
        for (b0, d0), (b1, d1) in zip(pts, pts[1:]):
            if b0 <= b <= b1:
                f = (b - b0) / (b1 - b0) if b1 > b0 else 0.0
                return d0 + (d1 - d0) * f
        return pts[-1][1]

    num, den = 0.0, 0.0
    for n in g_names:
        t = tensor_importance.get(n, 0.0)
        dc, dn = d_of(n, cur_tier), d_of(n, next_tier)
        if dc is None or dn is None:
            continue
        num += t * (dc - dn)
        den += t
    return num / den if den > 0 else 0.0


def experimental_gain(cur_tier: str, next_tier: str, g_names: List[str],
                      tensor_importance: Dict[str, float],
                      uopt: Dict[str, Any]) -> float | None:
    """Gain for experimental modes, None if mode is a core one."""
    import math
    from classifier import _mse_delta, _mse_eff
    mode = (uopt or {}).get("mode", "mse")
    if mode == "huber":
        d = _mse_delta(cur_tier, next_tier)
        s = (uopt or {}).get("huber_delta", 3e-4)
        if d < 0:
            return d
        return d * d / (2 * s) if d < s else s * (d / s - 0.5)
    if mode == "netdmg":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        base = 0.0 if ec + en <= 0 else 2.0 * (ec - en) / (ec + en)
        z = _group_netdmg_z(g_names, (uopt or {}).get("netpred"))
        return base * (1.0 + (uopt or {}).get("netdmg_w", 0.5) * z)
    if mode == "pw_ssim":
        return _pdamage_gain(cur_tier, next_tier, g_names,
                             tensor_importance, (uopt or {}).get("ptable"))
    if mode == "hybrid":
        gain = _mse_delta(cur_tier, next_tier)
        cv_w = (uopt or {}).get("cv_w", 0.0)
        linf_w = (uopt or {}).get("linf_w", 0.0)
        if cv_w and len(g_names) > 1:
            imps = [tensor_importance.get(n, 0.0) for n in g_names]
            mean = sum(imps) / len(imps)
            if mean > 0:
                var = sum((x - mean) ** 2 for x in imps) / len(imps)
                gain = gain / (1.0 + cv_w * var / mean ** 2)
        if linf_w:
            lc = LINF_ERR.get(cur_tier, 0.5)
            ln = LINF_ERR.get(next_tier, 0.5)
            if lc > 0:
                gain = gain * (1.0 + linf_w * (lc - ln) / lc)
        return gain
    return None
