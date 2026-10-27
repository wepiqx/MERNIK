import heapq
import math
import warnings
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field

import gnom
from legacy_modes import LEGACY_MODES, apply_hinge, legacy_gain
from constants import (
    TIER_ORDER, TIER_BPW, GGUF_OVERHEAD_FACTOR, CLASS_MAX_TIER,
    CAN_Q3, ALLOW_LOWER_FLOOR, EXTREME_FLOOR, MTP_DEPLOY_TIER, EMBD_DEPLOY_TIER,
    EMBD_PIN_TYPES, ROUTER_PIN_TYPES, TOXICITY_SUB4, F32_BPW,
    FORCE_F32_TYPES,
    get_tensor_class, get_tensor_type, is_mtp_tensor,
)

# Keep the divisor in a constant (8 bits * 1024 bytes * 1024 KiB)
BITS_IN_MIB = 8 * 1024 * 1024.0

# Filled in by the last classify() call so main.py can report the tail of the
# budget without threading a third return value through everything.
_LAST_RUN_INFO: Dict[str, float] = {"slack_mib": 0.0, "polish_steps": 0}

# Precompute size multipliers per tier (faster math)
TIER_SIZE_MULTIPLIER = {
    tier: (bpw / BITS_IN_MIB) * GGUF_OVERHEAD_FACTOR 
    for tier, bpw in TIER_BPW.items()
}

# K-quants use 256-element blocks and need padding for 3D MoE tensors
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
    """Upgrade queue item. Compared only by neg_utility."""
    neg_utility: float
    group_id: int = field(compare=False)
    next_tier: str = field(compare=False)
    cost_delta: float = field(compare=False)


def _tier_index(tier: str, order: list | None = None) -> int:
    order = order or TIER_ORDER
    if tier not in order:
        raise ValueError(f"Unknown tier: {tier}")
    return order.index(tier)

def _tier_at(idx: int, order: list | None = None) -> str:
    order = order or TIER_ORDER
    if not (0 <= idx < len(order)):
        raise IndexError(f"Tier index {idx} out of range")
    return order[idx]

# SQUEEZE mode: binary allocation — every tensor is either dungeon or
# palace, nothing in between. Separate path, other modes untouched.

def _size_mib(tier: str, n_elements: int, force_f32: bool = False) -> float:
    """Tensor size in MiB including GGUF overhead.

    1D tensors (norms, biases) are F32 in the file whatever the tier says —
    llama.cpp does not quantize 1D. Accounting them at the assigned tier
    both under-counts the budget and hides the fact that the assignment is
    a no-op, so they are priced at F32 bits instead.
    """
    if n_elements <= 0:
        return 0.0
    if force_f32:
        return n_elements * (F32_BPW / BITS_IN_MIB) * GGUF_OVERHEAD_FACTOR
    if tier not in TIER_SIZE_MULTIPLIER:
        # Silent zeros in budget arithmetic caused silent regressions
        # (review 2026-10-08): a typo'd tier must fail loud, never vanish.
        raise ValueError(f"Unknown tier in budget accounting: {tier!r}")
    return n_elements * TIER_SIZE_MULTIPLIER[tier]

def _f32_map(model_tensors: Dict[str, Any]) -> Dict[str, bool]:
    """Tensors the binary will write as F32 whatever tier we assign.

    Two proven sources: 1D tensors (norms, biases — llama.cpp does not
    quantize rank-1), and the types in FORCE_F32_TYPES (ssm_conv1d, caught by
    preflight on RINIQ-M2). Anything assigned here is pinned outside the
    budget: keeping it in the queue only produces decisions that never land.
    """
    out = {}
    for k, v in model_tensors.items():
        shape = v.get("shape") or []
        out[k] = bool(v.get("rank1", len(shape) <= 1)) or \
            get_tensor_type(k) in FORCE_F32_TYPES
    return out


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


# ---------------------------------------------------------------- utility
# Core: mse (baseline), smse (SMAPE x MSE), smape, gnom.
# Archived lenses live in legacy_modes.py and are hidden from the CLI.

def _gain(cur_tier: str, next_tier: str, g_names: List[str],
          tensor_importance: Dict[str, float], uopt: Dict[str, Any]) -> float:
    """Quality gain of cur->next step under the selected utility metric."""
    uopt = uopt or {}
    g = _gain_mode((uopt.get("mode", "mse")), cur_tier, next_tier, g_names,
                   tensor_importance, uopt)
    hw = float(uopt.get("hinge_w", 0.0) or 0.0)
    if hw > 0:
        g = apply_hinge(g, next_tier, hw)
    return g


_scale_cache: Dict[str, float] = {}


def _scale(mode: str, cur_tier: str, next_tier: str, uopt: Dict[str, Any]) -> float:
    """Normalize a utility formula to O(1) (legacy mix/srmse only)."""
    uopt = uopt or {}
    key = (mode, float(uopt.get("huber_delta", 3e-4)))
    if key not in _scale_cache:
        ref = abs(_gain_mode(mode, "Q4_K", "Q5_K", [], {}, uopt))
        _scale_cache[key] = 1.0 / ref if ref > 0 else 1.0
    return _scale_cache[key]


def _gain_mode(mode: str, cur_tier: str, next_tier: str,
               g_names: List[str], tensor_importance: Dict[str, float],
               uopt: Dict[str, Any]) -> float:
    """Single-formula gain, core modes and Gnom only."""
    if mode in LEGACY_MODES:
        return legacy_gain(mode, cur_tier, next_tier, g_names,
                           tensor_importance, uopt)
    if gnom.is_gnom(mode):
        return gnom.gain(cur_tier, next_tier, g_names, uopt)
    if mode == "smape":
        ec, en = _mse_eff(cur_tier), _mse_eff(next_tier)
        if ec + en <= 0:
            return 0.0
        return 2.0 * (ec - en) / (ec + en)
    if mode == "smse":
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
    return _mse_delta(cur_tier, next_tier)


def _step_cost(cur_tier: str, next_tier: str, g_elements: int,
               g_elements_padded: int) -> float:
    """MiB delta of one ladder step, honouring K-quant padding."""
    cur_n = g_elements_padded if cur_tier in K_QUANTS else g_elements
    next_n = g_elements_padded if next_tier in K_QUANTS else g_elements
    return _size_mib(next_tier, next_n) - _size_mib(cur_tier, cur_n)


def _best_step_within(group_id, group_registry, assignments,
                      tensor_importance, importance_table, slack,
                      ceiling=None, uopt=None, order=None):
    """Best ladder step this group can take with `slack` MiB left.

    The greedy drain only ever offers cur->cur+1, and a step that does not
    fit is dropped from the queue for good (the budget only shrinks from
    there). So the tail of the budget goes unspent: a group whose next rung
    is 400 MiB stays at its floor with 300 MiB idle, even though a higher
    rung might cost less than the near one. This walks the whole reachable
    window and returns the best value-per-MiB step inside it.
    """
    order = order or TIER_ORDER
    g_names, g_elements, g_elements_padded, g_f32 = group_registry[group_id]
    if g_f32 or slack <= 0:
        return None
    rep_name = g_names[0]
    cur_tier = assignments[rep_name]
    cur_idx = _tier_index(cur_tier, order)

    rep_info = importance_table.get(rep_name, {})
    ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(rep_name)
    max_tier = ceiling or CLASS_MAX_TIER.get(get_tensor_class(ttype), "Q8_0")
    top_idx = min(_tier_index(max_tier, order), len(order) - 1)
    if cur_idx >= top_idx:
        return None

    total_g_imp = sum(tensor_importance.get(n, 0) for n in g_names)
    best = None
    idx = cur_idx + 1
    while idx <= top_idx:
        cand = _tier_at(idx, order)
        cost = _step_cost(cur_tier, cand, g_elements, g_elements_padded)
        if cost > slack:
            idx += 1
            continue
        gain = _gain(cur_tier, cand, g_names, tensor_importance, uopt or {})
        if gain < 0:
            idx += 1
            continue
        if cost == 0:
            util = float("inf")
        elif gain == 0:
            idx += 1
            continue
        else:
            util = (total_g_imp * gain) / cost
        if best is None or util > best[0]:
            best = (util, cand, cost)
        idx += 1
    return best


def _polish_slack(assignments, group_registry, current_size, effective_target,
                  tensor_importance, importance_table, uopt=None, order=None,
                  ceiling_of=None):
    """Spend the tail of the budget that the single-rung greedy left idle.

    Returns (current_size, steps_taken). Terminates when no group can use the
    remaining slack. Bounded: each iteration raises some group's tier, and
    tiers only move up, so it cannot cycle.
    """
    order = order or TIER_ORDER
    taken = 0
    while True:
        slack = effective_target - current_size
        if slack <= 0:
            break
        best = None
        for g_id in group_registry:
            cand = _best_step_within(
                g_id, group_registry, assignments, tensor_importance,
                importance_table, slack,
                ceiling=(ceiling_of(g_id) if ceiling_of else None),
                uopt=uopt, order=order)
            if cand is None:
                continue
            util, tier, cost = cand
            if best is None or util > best[0]:
                best = (util, g_id, tier, cost)
        if best is None:
            break
        _, g_id, tier, cost = best
        for n in group_registry[g_id][0]:
            assignments[n] = tier
        current_size += cost
        taken += 1
    # main.py reports this: leftover slack that no rung can absorb is a
    # property of the geometry (the next step is too big), not a silent bug.
    _LAST_RUN_INFO["slack_mib"] = max(0.0, effective_target - current_size)
    _LAST_RUN_INFO["polish_steps"] = taken
    return current_size, taken


def _push_upgrade(group_id: int,
                   group_registry: Dict[int, Tuple[List[str], int, int]],
                   assignments: Dict[str, str],
                   tensor_importance: Dict[str, float],
                   upgrade_queue: List[UpgradeItem],
                   importance_table: Dict[str, Any],
                   ceiling: str | None = None,
                   uopt: Dict[str, Any] | None = None,
                   order: list | None = None):
    order = order or TIER_ORDER
    # Keep 4 fields: names, real size, padded size, 1D flag
    g_names, g_elements, g_elements_padded, g_f32 = group_registry[group_id]
    if g_f32:
        return  # F32 in the file whatever we assign — no decision to make
    rep_name = g_names[0]
    cur_tier = assignments[rep_name]
    cur_idx = _tier_index(cur_tier, order)

    rep_info = importance_table.get(rep_name, {})
    ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(rep_name)
    cls = get_tensor_class(ttype)
    max_tier = ceiling or CLASS_MAX_TIER.get(cls, "Q8_0")

    if cur_idx >= _tier_index(max_tier, order) or cur_idx >= len(order) - 1:
        return

    next_tier = _tier_at(cur_idx + 1, order)

    # Single source of truth for step cost (K_QUANTS → padded inside).
    cost_delta = _step_cost(cur_tier, next_tier, g_elements, g_elements_padded)

    quality_delta = _gain(cur_tier, next_tier, g_names, tensor_importance, uopt or {})
    # Safety floor: a group below Q3_K must be allowed to climb out even when
    # Gnom predicts zero gain for the next rung. Missing predictions used to
    # strand 37 tensors at IQ2_XXS on Ling and leave hundreds of MiB unused at
    # 5100+; sub-Q3 is a risk zone, not a place to park slack.
    q3_idx = _tier_index("Q3_K", order)
    sub_q3 = cur_idx < q3_idx
    if quality_delta < 0:
        if not sub_q3:
            return
        quality_delta = 0.0
    if cost_delta < 0:
        return

    if cost_delta == 0:
        # Free step: genuine free upgrade (IQ4_NL→Q4_K, same real bpw) or
        # zero-cost transit across the IQ4_NL/IQ4_XS fiction boundary.
        utility_per_mb = float('inf')
    else:
        if quality_delta == 0:
            if not sub_q3:
                return  # paying real MB for zero modeled gain
            utility_per_mb = 0.0  # safety floor gets first claim on slack
        else:
            total_g_imp = sum(tensor_importance.get(n, 0) for n in g_names)
            utility_per_mb = (total_g_imp * quality_delta) / cost_delta

    heapq.heappush(upgrade_queue, UpgradeItem(-utility_per_mb, group_id, next_tier, cost_delta))


def _base_floor(ttype: str, cls: str, allow_q3: bool, has_imatrix: bool,
                is_qat: bool = False, low_floor: str | None = None) -> str:
    """Minimum tier for a tensor: bottom-up starts here, top-down stops here."""
    if allow_q3 and (ttype in CAN_Q3 or cls in CAN_Q3) and has_imatrix:
        return low_floor or ALLOW_LOWER_FLOOR
    if is_qat:
        return "Q4_K" if cls == "attn_proj" else "IQ4_XS"
    return "Q4_K"


def compute_initial_assignments(non_mtp_names: Set[str], mtp_names: Set[str],
                                importance_table: Dict, allow_q3: bool, is_qat: bool = False,
                                floor: str | None = None,
                                f32_map: Dict[str, bool] | None = None,
                                low_floor: str | None = None) -> Dict[str, str]:
    f32_map = f32_map or {}
    assignments = {name: MTP_DEPLOY_TIER for name in mtp_names}

    for name in non_mtp_names:
        rep_info = importance_table.get(name, {})
        has_imatrix = "importance_mean" in rep_info
        ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(name)
        cls = get_tensor_class(ttype)

        # output/token_embd are pinned in optimal_classify (never reach here
        # via non_mtp_names) — kept out of the upgrade budget entirely.
        if f32_map.get(name, False):
            # norms/biases: llama.cpp writes 1D as F32 regardless of the
            # rules. Assigning anything else is a decision that never lands
            # (preflight caught 75 of them doing exactly that on MiniCPM5-2B).
            assignments[name] = "F32"
        elif cls in ("norms", "ssm_params"):
            assignments[name] = "F16"
        else:
            assignments[name] = floor or _base_floor(ttype, cls, allow_q3, has_imatrix, is_qat, low_floor)

    return assignments


def build_groups(tied_groups: List[List[str]], non_mtp_names: Set[str],
                 ne_map: Dict[str, int], padded_ne_map: Dict[str, int],
                 f32_map: Dict[str, bool] | None = None) -> Dict[int, Tuple[List[str], int, int, bool]]:
    f32_map = f32_map or {}
    group_registry = {}
    assigned_tensors = set()

    for group_idx, group in enumerate(tied_groups):
        clean_group = [n for n in group if n in non_mtp_names]
        if clean_group:
            g_elements = sum(ne_map.get(n, 0) for n in clean_group)
            g_elements_padded = sum(padded_ne_map.get(n, 0) for n in clean_group)
            g_f32 = all(f32_map.get(n, False) for n in clean_group)
            group_registry[group_idx] = (clean_group, g_elements, g_elements_padded, g_f32)
            assigned_tensors.update(clean_group)

    unassigned_tensors = non_mtp_names - assigned_tensors
    next_group_idx = len(group_registry)

    for name in unassigned_tensors:
        group_registry[next_group_idx] = ([name], ne_map.get(name, 0),
                                           padded_ne_map.get(name, 0),
                                           f32_map.get(name, False))
        next_group_idx += 1

    return group_registry


def optimal_classify(importance_table: dict, tied_groups: list, model: dict,
                     target_size_mib: float, allow_q3: bool = False,
                     free_pins: bool = False,
                     uopt: Dict[str, Any] | None = None,
                     relief: Dict[str, float] | None = None,
                     relief_thr: float = -0.5,
                     allowed_tiers: list | None = None,
                     legacy_1d: bool = False,
                     ceiling: str | None = None,
                     pin_small: float = 0.0) -> Tuple[dict, dict]:
    """relief: {sweep_unit_tag: Q5->Q4 damage}. Groups with damage below
    relief_thr get ceiling Q4 (their measured sweet spot): the queue never
    upgrades them above Q4, and the saved budget flows to other groups.
    ceiling: global per-group ceiling override (e.g. F16) used when no
    relief/squeeze ceiling applies — runtime override for CLASS_MAX_TIER,
    constants.py untouched.
    allowed_tiers: restrict the ladder (SQUEEZE mode, e.g. ["IQ1_S", "F16"]
    = dungeon or palace, nothing between). Floor/ceiling follow the list;
    relief ceilings are ignored in squeeze mode."""
    if target_size_mib <= 0:
        raise ValueError("target_size_mib must be positive")
    uopt = dict(uopt or {"mode": "mse"})
    uopt["top_down"] = False
    order = allowed_tiers or TIER_ORDER
    squeeze = allowed_tiers is not None

    features = model.get("features", {})
    has_mtp = features.get("has_mtp", False) and not free_pins
    n_layers = features.get("n_layers", 31)
    is_qat = features.get("is_qat", False)
    model_tensors = model.get("tensors", {})

    ne_map = {k: v["n_elements"] for k, v in model_tensors.items()}
    for tname, info in importance_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # 1D tensors (norms, biases) are F32 in the file no matter what tier the
    # rules ask for. They are pinned and priced at F32 so the budget matches
    # the artifact and the queue stops spending decisions on them.
    f32_map = {} if legacy_1d else _f32_map(model_tensors)

    # --- MoE padding (integer alignment) ---
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

    # --pin-small: tensors under pin_small MiB (F16 weight) join the F16 pin,
    # outside the budget. Rationale: sub-MiB tensors cost nothing to protect
    # and carry control-plane roles (k/v heads, gates). 1D tensors land F32
    # by the binary's law regardless; the preflight says so loudly.
    small_names = set()
    if pin_small > 0 and not free_pins:
        small_names = {
            n for n in non_mtp_names
            if ne_map.get(n, 0) * 2.0 / (1024 * 1024) < pin_small
        }
        non_mtp_names -= small_names

    tensor_importance = {}
    for name in non_mtp_names:
        raw_imp = importance_table.get(name, {}).get("importance_mean", 0.0)
        tensor_importance[name] = raw_imp

    group_registry = build_groups(tied_groups, non_mtp_names, ne_map, padded_ne_map,
                                  f32_map=f32_map)

    # Relief ceilings: measured sweet spots from the damage sweep.
    def _unit_tag(unit):
        return "+".join(n.replace(".weight", "").split(".")[-1] + "@" +
                        n.split(".")[1] for n in unit
                        if n.startswith("blk.")) or "global"
    relief_ceiling: Dict[int, str] = {}
    if relief:
        for g_id, (g_names, _, _, _) in group_registry.items():
            d = relief.get(_unit_tag(g_names))
            if d is not None and d < relief_thr:
                relief_ceiling[g_id] = "Q4_K"

    mtp_cost = sum(_size_mib(MTP_DEPLOY_TIER, ne_map.get(n, 0), f32_map.get(n, False)) for n in mtp_names)
    embd_cost = sum(_size_mib(EMBD_DEPLOY_TIER, ne_map.get(n, 0), f32_map.get(n, False)) for n in embd_names)
    router_cost = sum(_size_mib("F16", ne_map.get(n, 0), f32_map.get(n, False)) for n in router_names)
    small_cost = sum(_size_mib("F16", ne_map.get(n, 0), f32_map.get(n, False)) for n in small_names)
    effective_target = target_size_mib - mtp_cost - embd_cost - router_cost - small_cost

    def _initial_assignments(allow_low: bool,
                             low_floor: str | None = None) -> Dict[str, str]:
        a = compute_initial_assignments(
            non_mtp_names, mtp_names, importance_table, allow_low, is_qat,
            floor=(order[0] if squeeze else None), f32_map=f32_map,
            low_floor=low_floor)
        for n in embd_names:
            a[n] = EMBD_DEPLOY_TIER
        for n in router_names:
            a[n] = "F16"
        for n in small_names:
            a[n] = "F16"
        return a

    def _size_of(a: Dict[str, str]) -> float:
        return sum(
            _size_mib(a[n],
                      padded_ne_map.get(n, ne_map.get(n, 0)) if a[n] in K_QUANTS else ne_map.get(n, 0),
                      f32_map.get(n, False))
            for n in non_mtp_names
        )

    # Adaptive floor: preserve the Q4_K start whenever it fits, so large
    # budgets keep their old distribution. Drop to IQ2/IQ1 only when the
    # budget cannot hold the Q4_K floor. --dont-allow-q3 pins allow_q3=False
    # and therefore refuses the fallback by construction.
    assignments = _initial_assignments(False)
    current_size = _size_of(assignments)
    if allow_q3 and not squeeze and current_size > effective_target:
        # Stage the fallback: IQ2 first, IQ1 only if even that cannot fit.
        assignments = _initial_assignments(True, ALLOW_LOWER_FLOOR)
        current_size = _size_of(assignments)
        if current_size > effective_target:
            assignments = _initial_assignments(True, EXTREME_FLOOR)
            current_size = _size_of(assignments)

    if current_size > effective_target:
        warnings.warn(f"Initial size {current_size:.1f} MiB already exceeds target {effective_target:.1f} MiB", RuntimeWarning)

    upgrade_queue = []
    for g_id in group_registry:
        _push_upgrade(g_id, group_registry, assignments, tensor_importance, upgrade_queue, importance_table, ceiling=(order[-1] if squeeze else (relief_ceiling.get(g_id) or ceiling)), uopt=uopt, order=order)

    while upgrade_queue:
        item = heapq.heappop(upgrade_queue)
        g_id, next_tier, cost_delta = item.group_id, item.next_tier, item.cost_delta

        if cost_delta > 0 and current_size + cost_delta > effective_target:
            continue

        for n in group_registry[item.group_id][0]:
            assignments[n] = next_tier
        current_size += cost_delta

        _push_upgrade(item.group_id, group_registry, assignments, tensor_importance, upgrade_queue, importance_table, ceiling=(order[-1] if squeeze else (relief_ceiling.get(item.group_id) or ceiling)), uopt=uopt, order=order)

    # Tail of the budget: the drain only offers cur->cur+1, so whatever is
    # left when the cheapest single rungs stop fitting stays unspent.
    if not squeeze:
        current_size, polished = _polish_slack(
            assignments, group_registry, current_size, effective_target,
            tensor_importance, importance_table, uopt=uopt, order=order,
            ceiling_of=lambda g_id: (relief_ceiling.get(g_id) or ceiling))

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
    g_names, g_elements, g_elements_padded, g_f32 = group_registry[group_id]
    if g_f32:
        return  # F32 in the file whatever we assign — no decision to make
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
    _mode = (uopt or {}).get("mode")
    if loss < 0 and _mode not in ("gnom", "netfd", "netpure"):
        return

    total_g_imp = sum(tensor_importance.get(n, 0) for n in g_names)
    if _mode in ("gnom", "netfd", "netpure"):
        # Gnom's prediction IS the loss here; the imatrix lens is not
        # re-imported. Weight 1.0 -> utility = damage per byte saved.
        total_g_imp = 1.0
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
                             legacy_1d: bool = False,
                             low_floor: str | None = None) -> Tuple[dict, dict]:
    """Top-down classification: start at F16, downgrade to fit."""
    if target_size_mib <= 0:
        raise ValueError("target_size_mib must be positive")
    uopt = dict(uopt or {"mode": "mse"})
    uopt["top_down"] = True

    features = model.get("features", {})
    has_mtp = features.get("has_mtp", False) and not free_pins
    n_layers = features.get("n_layers", 31)
    is_qat = features.get("is_qat", False)
    model_tensors = model.get("tensors", {})

    ne_map = {k: v["n_elements"] for k, v in model_tensors.items()}
    for tname, info in importance_table.items():
        if tname not in ne_map:
            ne_map[tname] = info["n_elements"]

    # 1D tensors are F32 in the file whatever the tier says. They are pinned
    # here (outside the budget) and priced at F32 — top-down used to feed
    # them into the downgrade queue, "freeing" megabytes that the binary
    # never gave back, and generating rules that cannot take effect.
    f32_map = {} if legacy_1d else _f32_map(model_tensors)

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
    # No pinning here: every other 2D tensor starts at F16 and fights for
    # budget — except MoE routers (F16) and 1D tensors (F32, physically:
    # llama.cpp does not quantize 1D, verified via --dry-run).
    flex_names = rest - embd_names
    router_names = {
        n for n in flex_names
        if importance_table.get(n, {}).get("type", get_tensor_type(n)) in ROUTER_PIN_TYPES
    } if not free_pins else set()
    flex_names -= router_names
    one_d_names = {n for n in flex_names if f32_map.get(n, False)}
    flex_names -= one_d_names
    tensor_importance = {}
    for name in flex_names:
        tensor_importance[name] = importance_table.get(name, {}).get("importance_mean", 0.0)

    assignments = {n: MTP_DEPLOY_TIER for n in mtp_names}
    for n in embd_names:
        assignments[n] = EMBD_DEPLOY_TIER
    for n in router_names:
        assignments[n] = "F16"
    for n in one_d_names:
        assignments[n] = "F32"
    for n in flex_names:
        assignments[n] = "F16"

    # Groups over flex tensors only, with per-group floors
    group_registry = build_groups(tied_groups, flex_names, ne_map, padded_ne_map,
                                  f32_map=f32_map)
    group_floors = {}
    for g_id, (g_names, _, _, _) in group_registry.items():
        floors = []
        for n in g_names:
            rep_info = importance_table.get(n, {})
            ttype = rep_info["type"] if "type" in rep_info else get_tensor_type(n)
            cls = get_tensor_class(ttype)
            floors.append(_tier_index(_base_floor(
                ttype, cls, allow_q3, "importance_mean" in rep_info, is_qat,
                low_floor)))
        group_floors[g_id] = _tier_at(min(floors))

    fixed_cost = (
        sum(_size_mib(MTP_DEPLOY_TIER, ne_map.get(n, 0), f32_map.get(n, False)) for n in mtp_names)
        + sum(_size_mib(EMBD_DEPLOY_TIER, ne_map.get(n, 0), f32_map.get(n, False)) for n in embd_names)
        + sum(_size_mib("F16", ne_map.get(n, 0), f32_map.get(n, False)) for n in router_names)
        + sum(_size_mib("F32", ne_map.get(n, 0), True) for n in one_d_names)
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
        used = low_floor or ALLOW_LOWER_FLOOR
        if allow_q3 and _tier_index(used) > _tier_index(EXTREME_FLOOR):
            return optimal_classify_topdown(
                importance_table, tied_groups, model, target_size_mib,
                allow_q3=True, free_pins=free_pins, uopt=uopt,
                legacy_1d=legacy_1d, low_floor=EXTREME_FLOOR)
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

    # Same tail-of-budget gap as bottom-up: single-rung steps only.
    current_size, polished = _polish_slack(
        assignments, group_registry, current_size, effective_target,
        tensor_importance, importance_table, uopt=uopt, order=TIER_ORDER,
        ceiling_of=lambda g_id: "F16")

    return assignments, padded_ne_map


def compute_stats(assignments: dict, ne_map: dict = None, padded_ne_map: dict = None,
                  f32_map: dict = None) -> dict:
    """Collect per-tier stats with exact K_QUANTS padding accounting."""
    f32_map = f32_map or {}
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

            size = _size_mib(tier, elements, f32_map.get(name, False))
            stats["by_tier_mib"][tier] = stats["by_tier_mib"].get(tier, 0.0) + size
            stats["total_mib"] += size
            
    return stats
