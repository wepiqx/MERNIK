import heapq
import warnings
import numpy as np
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field

from constants import (
    TIER_ORDER, TIER_BPW, GGUF_OVERHEAD_FACTOR, CLASS_MAX_TIER,
    CAN_Q3, ALLOW_LOWER_FLOOR, MTP_DEPLOY_TIER, EMBD_DEPLOY_TIER,
    EMBD_PIN_TYPES, ROUTER_PIN_TYPES, TOXICITY_SUB4, get_tensor_class, get_tensor_type,
    is_mtp_tensor,
)

# Выносим делитель в константу (8 бит * 1024 байт * 1024 кбайт)
BITS_IN_MIB = 8 * 1024 * 1024.0

# Предвычисляем множители размеров для каждого тира (ускорение математики)
TIER_SIZE_MULTIPLIER = {
    tier: (bpw / BITS_IN_MIB) * GGUF_OVERHEAD_FACTOR 
    for tier, bpw in TIER_BPW.items()
}

# K-quants используют блоки по 256 элементов и требуют выравнивания (padding) 3D-тензоров MoE
K_QUANTS = {"Q3_K", "Q4_K", "Q5_K", "Q6_K"}
MOE_PAD_TYPES = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_down"}

MSE_BPW = {
    "IQ1_S": 1.5625, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125,
    "IQ2_S": 2.5,
    "IQ3_XXS": 3.0625,
    "Q3_K": 3.4375, "IQ3_S": 3.44,
    # IQ4_NL gets +0.15 effective bpw over its real 4.5bpw twin IQ4_XS:
    # K-quant bonus. Must stay strictly between IQ4_XS and Q4_K, otherwise
    # the IQ4_XS<->IQ4_NL steps become zero-gain and strand both greedy
    # chains (upgrades refuse paid zero-gain, downgrades stall).
    "IQ4_XS": 4.25, "IQ4_NL": 4.40, "Q4_K": 4.50,
    "Q5_K": 5.50, "Q6_K": 6.5625, "Q8_0": 8.50, "F16": 16.0,
}

@dataclass(order=True, slots=True)
class UpgradeItem:
    """Элемент очереди апгрейдов. Сравнивается только по neg_utility."""
    neg_utility: float
    group_id: int = field(compare=False)
    next_tier: str = field(compare=False)
    cost_delta: float = field(compare=False)


def _tier_index(tier: str) -> int:
    if tier not in TIER_ORDER:
        raise ValueError(f"Unknown tier: {tier}")
    return TIER_ORDER.index(tier)

def _tier_at(idx: int) -> str:
    if not (0 <= idx < len(TIER_ORDER)):
        raise IndexError(f"Tier index {idx} out of range")
    return TIER_ORDER[idx]

def _size_mib(tier: str, n_elements: int) -> float:
    """Размер тензора в MiB с учётом оверхеда GGUF."""
    if n_elements <= 0:
        return 0.0
    return n_elements * TIER_SIZE_MULTIPLIER.get(tier, 0.0)

@lru_cache(maxsize=64)
def _mse_eff(tier: str) -> float:
    """Effective MSE with sub-4-bit toxicity penalty (see TOXICITY_SUB4)."""
    base = 2 ** (-2 * MSE_BPW[tier])
    if TIER_BPW[tier] < 4.0:
        base *= TOXICITY_SUB4
    return base


@lru_cache(maxsize=128)
def _mse_delta(cur_tier: str, next_tier: str) -> float:
    return _mse_eff(cur_tier) - _mse_eff(next_tier)


# ---------------------------------------------------------------- battlefield
# Experimental hybrid utility: MSE is not the only lens. Modes:
#   mse    — baseline (current behavior, uopt ignored)
#   rmse   — sqrt scale: compresses dynamic range, cheap small steps rank higher
#   hybrid — ΔMSE, discounted for concentrated groups (bits wasted on junk
#            members) and boosted for large worst-case-error drops.
# Concentration rationale: group timp [100,0,0] vs [34,33,33], same sum and
# cost — the first wastes 2/3 of the upgrade on zero-importance members, so
# its gain is divided by (1 + cv_w * cv^2).



def _gain(cur_tier: str, next_tier: str, g_names: List[str],
          tensor_importance: Dict[str, float], uopt: Dict[str, Any]) -> float:
    """Quality gain of cur->next step under the selected utility metric."""
    import math
    mode = (uopt or {}).get("mode", "mse")
    if mode == "mix":
        # Per-group utility: kings (top importance rank) evaluated by MSE
        # (spread/protect), everyone else by mix_base (default: sacrifice).
        # mix_top is a set of group rep names, built once by the caller.
        # SCALE WARNING (scar 2026-09-18): raw formulas live on different
        # scales (mse Δ ~1e-3, smape Δ ~1) — compared raw, smape junk
        # outbids mse kings ~800:1 and mix collapses to pure smape.
        # _scale() below normalizes every formula to O(1) first.
        top = (uopt or {}).get("mix_top") or frozenset()
        rep = g_names[0] if g_names else ""
        mode = "mse" if rep in top else (uopt or {}).get("mix_base", "smape")
        return _scale(mode, cur_tier, next_tier, uopt or {}) * _gain_mode(
            mode, cur_tier, next_tier, g_names, tensor_importance, uopt or {})
    # non-mix: raw formula, behavior unchanged (zoo baselines intact)
    return _gain_mode(mode, cur_tier, next_tier, g_names,
                      tensor_importance, uopt or {})


_scale_cache: Dict[str, float] = {}


def _scale(mode: str, cur_tier: str, next_tier: str, uopt: Dict[str, Any]) -> float:
    """Normalize a utility formula to O(1).

    Reference = |raw Q4_K→Q5_K delta| for that mode (cached). Without this,
    bounded formulas (smape ≤ 2) outbid absolute ones (mse Δ ~1e-3) ~800:1
    and any mix collapses to the bounded side. cur/next args kept for
    future per-step references; currently unused.
    """
    if mode not in _scale_cache:
        ref = abs(_gain_mode(mode, "Q4_K", "Q5_K", [], {}, uopt))
        _scale_cache[mode] = 1.0 / ref if ref > 0 else 1.0
    return _scale_cache[mode]


def _gain_mode(mode: str, cur_tier: str, next_tier: str,
               g_names: List[str], tensor_importance: Dict[str, float],
               uopt: Dict[str, Any]) -> float:
    """Single-formula gain (mode already resolved, never 'mix')."""
    import math
    assert mode != "mix", "mix must be resolved before _gain_mode"
    from experimental import experimental_gain, experimental_gain_modes
    if mode == "rmse":
        return math.sqrt(_mse_eff(cur_tier)) - math.sqrt(_mse_eff(next_tier))
    if mode == "logcosh":
        # Gain reshaper on the MSE delta: linear-ish for small steps,
        # compressed for large ones (outlier-heavy steps stop dominating).
        d = _mse_delta(cur_tier, next_tier)
        s = (uopt or {}).get("huber_delta", 3e-4)  # splits the typical Δ range
        if d < 0:
            return d
        return math.log(math.cosh(d / s)) * s
    if mode in experimental_gain_modes():
        return experimental_gain(cur_tier, next_tier, g_names,
                                 tensor_importance, uopt)
    if mode == "smape":
        # Relative lens: symmetric relative error drop, bounded in [0, 2].
        # Favors steps that slash the *remaining* error proportionally,
        # regardless of absolute scale.
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        if ec + en <= 0:
            return 0.0
        return 2.0 * (ec - en) / (ec + en)
    if mode == "balance":
        # Kings need the people, people need kings (scar 2026-09-22):
        # geometric blend of the relative lens (SMAPE barbell shape) and
        # the absolute lens (MSE weight + live sub-4 toxicity). Both halves
        # O(1)-normalized first (MIX lesson), so neither outbids 800:1.
        # Toxicity survives here because the MSE half is divided by a
        # CONSTANT reference — unlike SMAPE's (ec+en) denominator, which
        # cancels it. TOX multipliers reverted (dead code, wrong direction).
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        if ec + en <= 0:
            return 0.0
        rel = 2.0 * (ec - en) / (ec + en)
        aba = ec - en
        if rel <= 0.0 or aba <= 0.0:
            return 0.0
        s_ref = _scale("smape", cur_tier, next_tier, uopt or {})
        m_ref = _scale("mse", cur_tier, next_tier, uopt or {})
        if s_ref <= 0.0 or m_ref <= 0.0:
            return 0.0
        return math.sqrt((rel / s_ref) * (aba / m_ref))
    if mode == "ssim":
        # Measured structural gain, per group MEMBER (not rep): members of a
        # tied group share importance but not weights, so ΔSSIM differs per
        # member — a genuine per-group signal. No toxicity fiction: SSIM
        # measures the sub-4 collapse directly (that's the hypothesis).
        return _ssim_gain(cur_tier, next_tier, g_names, tensor_importance,
                          (uopt or {}).get("ssim_table"), relative=False)
    if mode == "smape_ssim":
        # Relative lens on the measured signal: symmetric relative structural
        # recovery per member, importance-weighted. The zoo final boss.
        return _ssim_gain(cur_tier, next_tier, g_names, tensor_importance,
                          (uopt or {}).get("ssim_table"), relative=True)
    if mode == "smape_frag":
        # Don't replace the champion — modulate it. SMAPE-on-MSE decides the
        # shape; per-group structural fragility (measured Q4 damage, mean over
        # members, normalized) reweights the vote. Fragile groups get boosted
        # upgrades, oak groups get discounted ones.
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        base = 0.0 if ec + en <= 0 else 2.0 * (ec - en) / (ec + en)
        frag = _group_fragility(g_names, (uopt or {}).get("ssim_table"))
        return base * (1.0 + (uopt or {}).get("frag_w", 0.5) * (frag - 1.0))
    gain = _mse_delta(cur_tier, next_tier)
    return gain


def _ssim_gain(cur_tier: str, next_tier: str, g_names: List[str],
               tensor_importance: Dict[str, float],
               table_path: str | None, relative: bool = False) -> float:
    """Importance-weighted mean ΔSSIM over group members.

    Returns an average (not a sum): the caller multiplies by Σtimp, so the
    result is Σt·ΔS — members with more importance contribute more gain.
    Tiers outside the measured table are bpw-interpolated on the measured
    curve; tensors missing from the table contribute 0 (pinned anyway).
    """
    from constants import TIER_BPW
    table = _ssim_table(table_path)
    if not table:
        return _mse_delta(cur_tier, next_tier)  # table not ready: mse fallback

    def s_of(name: str, tier: str) -> float | None:
        row = table.get(name)
        if row is None:
            return None
        if tier in row:
            return row[tier]
        if tier == "F16":
            return 1.0
        # bpw-interpolate on the measured curve
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
        return pts[-1][1]  # above measured range: clamp to best

    num, den = 0.0, 0.0
    for n in g_names:
        t = tensor_importance.get(n, 0.0)
        sc, sn = s_of(n, cur_tier), s_of(n, next_tier)
        if sc is None or sn is None:
            continue
        if relative:
            # Relative lens on DAMAGE (1-S), not similarity: damage spans an
            # order of magnitude (vs similarity stuck in [0.987, 1]), so the
            # relative rescaling actually bites — same trick that made SMAPE
            # win on MSE. Positive when damage drops.
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
    """Mean measured Q4 damage over group members, normalized by global mean.

    Returns 1.0 when the table is missing (modulation becomes a no-op).
    """
    table = _ssim_table(table_path)
    if not table:
        return 1.0
    key = table_path + "|mean"
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
    except Exception:
        _SSIM_CACHE[path] = {}
        return {}
    tiers = ["Q4_K", "Q5_K", "Q6_K", "Q8_0"]
    table = {}
    for name in z.files:
        arr = z[name]
        table[name] = {t: float(arr[i][0]) for i, t in enumerate(tiers)}
    _SSIM_CACHE[path] = table
    return table


def _push_upgrade(group_id: int,
                   group_registry: Dict[int, Tuple[List[str], int, int]],
                   assignments: Dict[str, str],
                   tensor_importance: Dict[str, float],
                   upgrade_queue: List[UpgradeItem],
                   importance_table: Dict[str, Any],
                   ceiling: str | None = None,
                   uopt: Dict[str, Any] | None = None):
    
    # Храним 3 элемента: имена, реальный размер, размер с padding
    g_names, g_elements, g_elements_padded = group_registry[group_id]
    rep_name = g_names[0]
    cur_tier = assignments[rep_name]
    cur_idx = _tier_index(cur_tier)

    rep_info = importance_table.get(rep_name, {})
    ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(rep_name)
    cls = get_tensor_class(ttype)
    max_tier = ceiling or CLASS_MAX_TIER.get(cls, "Q8_0")

    if cur_idx >= _tier_index(max_tier) or cur_idx >= len(TIER_ORDER) - 1:
        return

    next_tier = _tier_at(cur_idx + 1)
    
    # Выбираем размер в зависимости от того, относится ли тир к K-quants
    cur_size_g = g_elements_padded if cur_tier in K_QUANTS else g_elements
    next_size_g = g_elements_padded if next_tier in K_QUANTS else g_elements
    
    cost_delta = _size_mib(next_tier, next_size_g) - _size_mib(cur_tier, cur_size_g)

    quality_delta = _gain(cur_tier, next_tier, g_names, tensor_importance, uopt or {})
    if quality_delta < 0:
        return
    if cost_delta < 0:
        return

    if cost_delta == 0:
        # Free step: genuine free upgrade (IQ4_NL→Q4_K, same real bpw) or
        # zero-cost transit across the IQ4_NL/IQ4_XS fiction boundary.
        utility_per_mb = float('inf')
    else:
        if quality_delta == 0:
            return  # paying real MB for zero modeled gain
        total_g_imp = sum(tensor_importance.get(n, 0) for n in g_names)
        utility_per_mb = (total_g_imp * quality_delta) / cost_delta

    heapq.heappush(upgrade_queue, UpgradeItem(-utility_per_mb, group_id, next_tier, cost_delta))


def _base_floor(ttype: str, cls: str, allow_q3: bool, has_imatrix: bool,
                is_qat: bool = False) -> str:
    """Minimum tier for a tensor: bottom-up starts here, top-down stops here."""
    if allow_q3 and (ttype in CAN_Q3 or cls in CAN_Q3) and has_imatrix:
        return ALLOW_LOWER_FLOOR
    if is_qat:
        return "Q4_K" if cls == "attn_proj" else "IQ4_XS"
    return "Q4_K"


def compute_initial_assignments(non_mtp_names: Set[str], mtp_names: Set[str], 
                                importance_table: Dict, allow_q3: bool, is_qat: bool = False) -> Dict[str, str]:
    assignments = {name: MTP_DEPLOY_TIER for name in mtp_names}

    for name in non_mtp_names:
        rep_info = importance_table.get(name, {})
        has_imatrix = "importance_mean" in rep_info
        ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(name)
        cls = get_tensor_class(ttype)

        # output/token_embd are pinned in optimal_classify (never reach here
        # via non_mtp_names) — kept out of the upgrade budget entirely.
        if cls in ("norms", "ssm_params"):
            assignments[name] = "F16"
        else:
            assignments[name] = _base_floor(ttype, cls, allow_q3, has_imatrix, is_qat)

    return assignments


def build_groups(tied_groups: List[List[str]], non_mtp_names: Set[str], 
                 ne_map: Dict[str, int], padded_ne_map: Dict[str, int]) -> Dict[int, Tuple[List[str], int, int]]:
    group_registry = {}
    assigned_tensors = set()
    
    for group_idx, group in enumerate(tied_groups):
        clean_group = [n for n in group if n in non_mtp_names]
        if clean_group:
            g_elements = sum(ne_map.get(n, 0) for n in clean_group)
            g_elements_padded = sum(padded_ne_map.get(n, 0) for n in clean_group)
            group_registry[group_idx] = (clean_group, g_elements, g_elements_padded)
            assigned_tensors.update(clean_group)

    unassigned_tensors = non_mtp_names - assigned_tensors
    next_group_idx = len(group_registry)
    
    for name in unassigned_tensors:
        group_registry[next_group_idx] = ([name], ne_map.get(name, 0), padded_ne_map.get(name, 0))
        next_group_idx += 1
            
    return group_registry


def optimal_classify(importance_table: dict, tied_groups: list, model: dict, 
                     target_size_mib: float, allow_q3: bool = False,
                     free_pins: bool = False,
                     uopt: Dict[str, Any] | None = None,
                     relief: Dict[str, float] | None = None,
                     relief_thr: float = -0.5) -> Tuple[dict, dict]:
    """relief: {sweep_unit_tag: Q5->Q4 damage}. Groups with damage below
    relief_thr get ceiling Q4 (their measured sweet spot): the queue never
    upgrades them above Q4, and the saved budget flows to other groups."""
    if target_size_mib <= 0:
        raise ValueError("target_size_mib must be positive")
    uopt = uopt or {"mode": "mse"}

    features = model.get("features", {})
    has_mtp = features.get("has_mtp", False) and not free_pins
    n_layers = features.get("n_layers", 31)
    is_qat = features.get("is_qat", False)
    model_tensors = model.get("tensors", {})

    ne_map = {k: v["n_elements"] for k, v in model_tensors.items()}
    for tname, info in importance_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # --- Вычисление MoE Padding (Целочисленное выравнивание) ---
    moe_d_ff = features.get("moe_intermediate_size", 0)
    padded_ne_map = dict(ne_map)
    if moe_d_ff > 0 and moe_d_ff % 256 != 0:
        aligned_d_ff = ((moe_d_ff + 255) // 256) * 256
        for name, n_el in ne_map.items():
            ttype = importance_table.get(name, {}).get("type", get_tensor_type(name))
            if ttype in MOE_PAD_TYPES:
                padded_ne_map[name] = (n_el // moe_d_ff) * aligned_d_ff
    # -----------------------------------------------------------

    all_names = set(ne_map.keys())
    mtp_names = {n for n in all_names if is_mtp_tensor(n, n_layers)} if has_mtp else set()
    non_mtp_names = all_names - mtp_names

    # Pinned output/token_embd: fixed tier, excluded from budget and upgrades
    # (llama.cpp writes them at the output/token types unconditionally).
    # --free-pins: experiment — they join the normal pool instead.
    embd_names = {
        n for n in non_mtp_names
        if importance_table.get(n, {}).get("type", get_tensor_type(n)) in EMBD_PIN_TYPES
    } if not free_pins else set()
    non_mtp_names -= embd_names

    # Pinned MoE routers: always F16, outside the budget.
    router_names = {
        n for n in non_mtp_names
        if importance_table.get(n, {}).get("type", get_tensor_type(n)) in ROUTER_PIN_TYPES
    } if not free_pins else set()
    non_mtp_names -= router_names

    tensor_importance = {}
    for name in non_mtp_names:
        raw_imp = importance_table.get(name, {}).get("importance_mean", 0.0)
        tensor_importance[name] = raw_imp

    assignments = compute_initial_assignments(non_mtp_names, mtp_names, importance_table, allow_q3, is_qat)
    for n in embd_names:
        assignments[n] = EMBD_DEPLOY_TIER
    for n in router_names:
        assignments[n] = "F16"
    group_registry = build_groups(tied_groups, non_mtp_names, ne_map, padded_ne_map)

    # Relief ceilings: measured sweet spots from the damage sweep.
    def _unit_tag(unit):
        return "+".join(n.replace(".weight", "").split(".")[-1] + "@" +
                        n.split(".")[1] for n in unit
                        if n.startswith("blk.")) or "global"
    relief_ceiling: Dict[int, str] = {}
    if relief:
        for g_id, (g_names, _, _) in group_registry.items():
            d = relief.get(_unit_tag(g_names))
            if d is not None and d < relief_thr:
                relief_ceiling[g_id] = "Q4_K"

    mtp_cost = sum(_size_mib(MTP_DEPLOY_TIER, ne_map.get(n, 0)) for n in mtp_names)
    embd_cost = sum(_size_mib(EMBD_DEPLOY_TIER, ne_map.get(n, 0)) for n in embd_names)
    router_cost = sum(_size_mib("F16", ne_map.get(n, 0)) for n in router_names)
    effective_target = target_size_mib - mtp_cost - embd_cost - router_cost
    
    current_size = sum(
        _size_mib(assignments[n], padded_ne_map.get(n, ne_map.get(n, 0)) if assignments[n] in K_QUANTS else ne_map.get(n, 0))
        for n in non_mtp_names
    )

    if current_size > effective_target:
        warnings.warn(f"Initial size {current_size:.1f} MiB already exceeds target {effective_target:.1f} MiB", RuntimeWarning)

    upgrade_queue = []
    for g_id in group_registry:
        _push_upgrade(g_id, group_registry, assignments, tensor_importance, upgrade_queue, importance_table, ceiling=relief_ceiling.get(g_id), uopt=uopt)

    while upgrade_queue:
        item = heapq.heappop(upgrade_queue)
        g_id, next_tier, cost_delta = item.group_id, item.next_tier, item.cost_delta
        
        if cost_delta > 0 and current_size + cost_delta > effective_target:
            continue
            
        for n in group_registry[g_id][0]:
            assignments[n] = next_tier
        current_size += cost_delta
        
        _push_upgrade(g_id, group_registry, assignments, tensor_importance, upgrade_queue, importance_table, ceiling=relief_ceiling.get(g_id), uopt=uopt)

    return assignments, padded_ne_map


# ---------------------------------------------------------------- top-down
# Crazy mode: everything quantizable starts at F16 (norms included) and is
# greedily downgraded — cheapest quality-loss-per-MB first — until under target.


@dataclass(order=True, slots=True)
class DowngradeItem:
    """Downgrade candidate. Min-heap on loss_per_mb (cheapest loss first)."""
    loss_per_mb: float
    group_id: int = field(compare=False)
    next_tier: str = field(compare=False)
    saved: float = field(compare=False)


def _push_downgrade(group_id: int,
                    group_registry: Dict[int, Tuple[List[str], int, int]],
                    group_floors: Dict[int, str],
                    assignments: Dict[str, str],
                    tensor_importance: Dict[str, float],
                    downgrade_queue: List[DowngradeItem],
                    uopt: Dict[str, Any] | None = None):
    g_names, g_elements, g_elements_padded = group_registry[group_id]
    rep_name = g_names[0]
    cur_tier = assignments[rep_name]
    cur_idx = _tier_index(cur_tier)
    floor_idx = _tier_index(group_floors[group_id])

    if cur_idx <= floor_idx:
        return

    next_tier = _tier_at(cur_idx - 1)

    cur_size_g = g_elements_padded if cur_tier in K_QUANTS else g_elements
    next_size_g = g_elements_padded if next_tier in K_QUANTS else g_elements

    saved = _size_mib(cur_tier, cur_size_g) - _size_mib(next_tier, next_size_g)
    if saved < 0:
        return  # padding quirk: downgrade would grow — stall here

    loss = _gain(next_tier, cur_tier, g_names, tensor_importance, uopt or {})  # quality increase reversed
    if loss < 0:
        return

    total_g_imp = sum(tensor_importance.get(n, 0) for n in g_names)
    if saved == 0:
        # Transit step (e.g. Q4_K→IQ4_NL: same bpw, must pass through to reach
        # deeper tiers). No saving, so defer to the end with inf utility —
        # it only fires if real savings elsewhere weren't enough.
        utility = float("inf")
    elif loss == 0:
        # Free savings under the effective-MSE fiction (IQ4_NL→IQ4_XS):
        # take first.
        utility = 0.0
    else:
        utility = (total_g_imp * loss) / saved
    heapq.heappush(downgrade_queue,
                   DowngradeItem(utility, group_id, next_tier, saved))


def optimal_classify_topdown(importance_table: dict, tied_groups: list, model: dict,
                             target_size_mib: float, allow_q3: bool = False,
                             free_pins: bool = False,
                             uopt: Dict[str, Any] | None = None,
                             pin_norms: bool = False) -> Tuple[dict, dict]:
    """Top-down classification: start at F16, downgrade to fit.

    pin_norms: keep norms/small tensors at F16 outside the budget (the
    native norms shield — post-hoc forcing disrupts the greedy path).
    """
    if target_size_mib <= 0:
        raise ValueError("target_size_mib must be positive")
    uopt = uopt or {"mode": "mse"}
    if target_size_mib <= 0:
        raise ValueError("target_size_mib must be positive")

    features = model.get("features", {})
    has_mtp = features.get("has_mtp", False) and not free_pins
    n_layers = features.get("n_layers", 31)
    is_qat = features.get("is_qat", False)
    model_tensors = model.get("tensors", {})

    ne_map = {k: v["n_elements"] for k, v in model_tensors.items()}
    for tname, info in importance_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # --- MoE padding (same as bottom-up) ---
    moe_d_ff = features.get("moe_intermediate_size", 0)
    padded_ne_map = dict(ne_map)
    if moe_d_ff > 0 and moe_d_ff % 256 != 0:
        aligned_d_ff = ((moe_d_ff + 255) // 256) * 256
        for name, n_el in ne_map.items():
            ttype = importance_table.get(name, {}).get("type", get_tensor_type(name))
            if ttype in MOE_PAD_TYPES:
                padded_ne_map[name] = (n_el // moe_d_ff) * aligned_d_ff
    # --------------------------------------

    all_names = set(ne_map.keys())
    mtp_names = {n for n in all_names if is_mtp_tensor(n, n_layers)} if has_mtp else set()
    rest = all_names - mtp_names
    embd_names = {
        n for n in rest
        if importance_table.get(n, {}).get("type", get_tensor_type(n)) in EMBD_PIN_TYPES
    } if not free_pins else set()
    # No pinning here: literally every other tensor (norms, 1D, biases —
    # anything) starts at F16 and fights for budget — except MoE routers,
    # which stay F16 outside the budget. Note the estimate assumes the binary
    # quantizes the rest too, while llama.cpp physically keeps 1D/*_norm.weight
    # at F16 — estimate vs binary will diverge by that amount.
    flex_names = rest - embd_names
    router_names = {
        n for n in flex_names
        if importance_table.get(n, {}).get("type", get_tensor_type(n)) in ROUTER_PIN_TYPES
    } if not free_pins else set()
    flex_names -= router_names
    # Native norms shield: norms/small tensors stay F16 outside the budget
    # (post-hoc forcing was proven to disrupt the greedy path — TDN-SMAPE).
    norm_names = {
        n for n in flex_names
        if "norm" in n or ne_map.get(n, 10 ** 9) < 100000
    } if pin_norms else set()
    flex_names -= norm_names

    tensor_importance = {}
    for name in flex_names:
        tensor_importance[name] = importance_table.get(name, {}).get("importance_mean", 0.0)

    assignments = {n: MTP_DEPLOY_TIER for n in mtp_names}
    for n in embd_names:
        assignments[n] = EMBD_DEPLOY_TIER
    for n in router_names:
        assignments[n] = "F16"
    for n in norm_names:
        assignments[n] = "F16"
    for n in flex_names:
        assignments[n] = "F16"

    # Groups over flex tensors only, with per-group floors
    group_registry = build_groups(tied_groups, flex_names, ne_map, padded_ne_map)
    group_floors = {}
    for g_id, (g_names, _, _) in group_registry.items():
        floors = []
        for n in g_names:
            rep_info = importance_table.get(n, {})
            ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(n)
            cls = get_tensor_class(ttype)
            floors.append(_tier_index(_base_floor(
                ttype, cls, allow_q3, "importance_mean" in rep_info, is_qat)))
        group_floors[g_id] = _tier_at(min(floors))

    fixed_cost = (
        sum(_size_mib(MTP_DEPLOY_TIER, ne_map.get(n, 0)) for n in mtp_names)
        + sum(_size_mib(EMBD_DEPLOY_TIER, ne_map.get(n, 0)) for n in embd_names)
        + sum(_size_mib("F16", ne_map.get(n, 0)) for n in router_names)
        + sum(_size_mib("F16", ne_map.get(n, 0)) for n in norm_names)
    )
    effective_target = target_size_mib - fixed_cost
    current_size = sum(_size_mib("F16", ne_map.get(n, 0)) for n in flex_names)

    downgrade_queue: List[DowngradeItem] = []
    for g_id in group_registry:
        _push_downgrade(g_id, group_registry, group_floors, assignments,
                        tensor_importance, downgrade_queue, uopt=uopt)

    while current_size > effective_target and downgrade_queue:
        item = heapq.heappop(downgrade_queue)
        for n in group_registry[item.group_id][0]:
            assignments[n] = item.next_tier
        current_size -= item.saved
        _push_downgrade(item.group_id, group_registry, group_floors, assignments,
                        tensor_importance, downgrade_queue, uopt=uopt)

    if current_size > effective_target:
        warnings.warn(f"Top-down hit all floors at {current_size:.1f} MiB, "
                      f"still over target {effective_target:.1f} MiB", RuntimeWarning)
        return assignments, padded_ne_map

    # ---- Phase 2: spend leftover slack on mirrored upgrades ----
    # Same imp×ΔMSE/cost metric as bottom-up (not raw importance), ceiling F16:
    # the last downgraded (most precious) groups are the first to recover.
    upgrade_queue: List[UpgradeItem] = []
    for g_id in group_registry:
        _push_upgrade(g_id, group_registry, assignments, tensor_importance,
                      upgrade_queue, importance_table, ceiling="F16", uopt=uopt)
    while upgrade_queue:
        item = heapq.heappop(upgrade_queue)
        if item.cost_delta > 0 and current_size + item.cost_delta > effective_target:
            continue
        for n in group_registry[item.group_id][0]:
            assignments[n] = item.next_tier
        current_size += item.cost_delta
        _push_upgrade(item.group_id, group_registry, assignments, tensor_importance,
                      upgrade_queue, importance_table, ceiling="F16", uopt=uopt)

    return assignments, padded_ne_map


def compute_stats(assignments: dict, ne_map: dict = None, padded_ne_map: dict = None) -> dict:
    """Собирает статистику по тирам с точным учётом K_QUANTS padding."""
    stats = {"by_tier_count": {}, "by_tier_mib": {}, "total_mib": 0.0, "tensor_count": 0}
    for name, tier in assignments.items():
        if not isinstance(tier, str):
            continue
        stats["tensor_count"] += 1
        stats["by_tier_count"][tier] = stats["by_tier_count"].get(tier, 0) + 1
        
        if ne_map:
            if padded_ne_map and tier in K_QUANTS:
                elements = padded_ne_map.get(name, ne_map.get(name, 0))
            else:
                elements = ne_map.get(name, 0)
                
            size = _size_mib(tier, elements)
            stats["by_tier_mib"][tier] = stats["by_tier_mib"].get(tier, 0.0) + size
            stats["total_mib"] += size
            
    return stats
