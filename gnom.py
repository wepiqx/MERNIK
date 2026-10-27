"""Gnom: the learned per-group damage predictor and the queue's priority.

One source of truth for:
  - loading the netpred table and its z-score statistics;
  - the group tag the predictor was trained on;
  - the gain handed to the allocator.

`gnom` is the Gnom mode kept in the CLI. `netpure` and `netfd` are archived
aliases: netpure = bottom-up (z-scored priority), netfd = top-down (raw
damage, F16 blend).
"""
from typing import Dict, List, Tuple

_CACHE: Dict[str, Tuple[Dict[str, float], float, float]] = {}


def group_tag(g_names: List[str]) -> str:
    """The predictor's unit key for a tied group (same rule as training)."""
    return "+".join(
        n.replace(".weight", "").split(".")[-1] + "@" + n.split(".")[1]
        for n in g_names if n.startswith("blk.")
    ) or "global"


def _load(path: str | None) -> Tuple[Dict[str, float], float, float]:
    if not path:
        return {}, 0.0, 1.0
    if path in _CACHE:
        return _CACHE[path]
    try:
        import json
        with open(path) as f:
            raw = {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        raw = {}
    if raw:
        vals = sorted(raw.values())
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        sd = var ** 0.5
    else:
        mu, sd = 0.0, 1.0
    _CACHE[path] = (raw, mu, sd + 1e-12)
    return _CACHE[path]


def table(path: str | None) -> Dict[str, float]:
    return _load(path)[0]


def coverage(path: str | None, tags: List[str]) -> Tuple[int, int]:
    t = table(path)
    return sum(1 for tag in tags if tag in t), len(tags)


def gain(cur_tier: str, next_tier: str, g_names: List[str],
         uopt: Dict) -> float:
    """Gnom priority for a step.

    top-down: raw predicted damage (negative = safer than average; the
    downgrade queue demotes those first). bottom-up: z-scored priority
    (zero = average; negative groups stay at the floor).
    """
    uopt = uopt or {}
    raw, mu, sd = _load(uopt.get("netpred"))
    if not raw:
        return 0.0
    v = raw.get(group_tag(g_names), 0.0)
    if uopt.get("top_down"):
        return v
    return (v - mu) / sd


def is_gnom(mode: str) -> bool:
    return mode in ("gnom", "netpure", "netfd")
