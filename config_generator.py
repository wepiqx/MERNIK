"""Assignment map -> llama-quantize --tensor-type flags ("built to last" edition).

Contract with the binary, verified against llama-quant.cpp (build 11571):
- each --tensor-type "PATTERN=TIER" splits at the FIRST '='; the pattern is
  LOWERCASED by the binary, then compiled as std::regex (ECMAScript);
- per tensor, patterns are tried IN CLI ORDER via regex_search (substring);
  the FIRST hit wins and skips the standard mixture logic; NO hit falls back
  to the mixture logic — never to our assignment;
- tensor names carry their suffix (".weight"/".bias"); patterns match it by
  substring, so rules name the full suffix explicitly.

Generation (exact by construction):
- one rule per (suffix, tier) group over blk layers, with exact layer
  enumerations ([a-b] within one digit, (?:a|b|...) otherwise, \\d+ only
  for a full 0..max span); suffix = everything after "blk.LAYER.",
  so 5-part names (nextn.hnorm.weight) get firing rules, not leftovers;
- anchored global rules for non-blk tensors (output/token_embd travel via
  --output-tensor-type/--token-embedding-type flags, not rules);
- order = fewest matched tensors first (provably most-specific-first),
  tie-break by pattern string (deterministic). This is what makes
  tissue-prefix shadowing (attn_q rule eating attn_qkv/attn_q_norm tensors)
  impossible: the narrower rule always precedes;
- simulate-and-assert: every rule-carried tensor is matched in-process
  (Python re.search, same substring semantics) and MUST first-hit its
  intended tier. Any shadowing class — present or future — fails LOUD
  here, before a single binary byte. Allowed regex constructs are exactly
  (?:...) [...] \\. \\d | ^ (?=...) — shared by ECMAScript and Python re.
"""
import re

# 1D tensors are F32 in the file whatever the rules say — emitting a rule
# for them is a no-op (kept out of rules AND out of the simulation).
_FLAG_CARRIED = ("output.weight", "token_embd.weight")


def _is_contiguous(lst, low, high):
    if not lst:
        return False
    return len(lst) == (high - low + 1)


def _group_ranges(lst):
    if not lst:
        return
    start = lst[0]
    end = lst[0]
    for i in range(1, len(lst)):
        if lst[i] == end + 1:
            end = lst[i]
        else:
            yield (start, end)
            start = end = lst[i]
    yield (start, end)


def _range_to_regex(start: int, end: int) -> str:
    """Convert a range of layer numbers [start, end] to a valid regex."""
    if start == end:
        return str(start)
    # For single-digit ranges, use character class
    if end <= 9:
        return f"[{start}-{end}]"
    # Enumerate all numbers as pipe alternatives
    alt = "|".join(str(i) for i in range(start, end + 1))
    return f"(?:{alt})"


def _suffix(tname: str):
    """Grouping key tissue: everything after 'blk.LAYER.' (or global).

    Minimum 3 parts, not 4: newer conversions write some tensors without
    a trailing .weight (Ling-3.0-tiny writes blk.0.ssm_a, 2026-10-23).
    Under the 4-part gate those fell to 'global' and produced anchored
    '(?=' rules that llama-quantize cannot parse (it splits the rule at
    the first '=' — the '=' inside the lookahead)."""
    parts = tname.split(".")
    if (len(parts) >= 3 and parts[0] == "blk" and parts[1].isdigit()):
        return ("blk", int(parts[1]), ".".join(parts[2:]))
    return ("global", tname)


def _max_layer(assignments, model):
    layers = [k[1] for k in (_suffix(n) for n in assignments)
              if k[0] == "blk"]
    if layers:
        return max(layers)
    return model.get("features", {}).get("n_layers", 31)


def verify_rules(assignments: dict, model: dict, rules,
                 flag_carried=_FLAG_CARRIED):
    """Simulate first-hit-wins over every rule-carried tensor.

    Returns the simulated map. Raises RuntimeError listing mismatches —
    a mismatch means the binary would write a tier nobody assigned.
    """
    from classifier import _f32_map
    f32 = _f32_map(model.get("tensors", {}))
    compiled = [(re.compile(p), t) for p, t in rules]
    bad, sim = [], {}
    for tname, tier in assignments.items():
        if f32.get(tname, False) or tname in flag_carried:
            continue
        hit = next((t for rx, t in compiled if rx.search(tname)), None)
        sim[tname] = hit
        if hit != tier:
            bad.append((tname, tier, hit))
    if bad:
        show = "\n".join(f"  {n}: intended {w}, rules give {g}"
                         for n, w, g in bad[:8])
        raise RuntimeError(
            f"rule simulation mismatch on {len(bad)} tensors "
            f"(first-hit-wins != assignment):\n{show}")
    return sim


def generate_flags(
    assignments: dict,
    model: dict,
    base_type: str,
    target_size_mib: float = None,
) -> dict:
    tensors = model.get("tensors", {})
    from classifier import _f32_map
    f32 = _f32_map(tensors)
    # Normally Q5_K pins (see EMBD_DEPLOY_TIER); with --free-pins the
    # classifier may assign output/token_embd other tiers — honor them here
    # since the binary applies these two flags unconditionally.
    output_type = assignments.get("output.weight", "Q5_K")
    token_embd_type = assignments.get("token_embd.weight", "Q5_K")

    max_layer = _max_layer(assignments, model)
    names = [n for n in assignments
             if not f32.get(n, False) and n not in _FLAG_CARRIED]

    # Group blk tensors by (suffix, tier)
    type_tier_layers = {}
    for tname in names:
        key = _suffix(tname)
        if key[0] != "blk":
            continue
        _, layer, suffix = key
        kk = (suffix, assignments[tname])
        type_tier_layers.setdefault(kk, []).append(layer)

    rules = []  # (pattern, tier, match_count)
    for (suffix, tier), layers in type_tier_layers.items():
        layers = sorted(set(layers))
        if len(layers) >= 8 and _is_contiguous(layers, 0, max_layer):
            lay = r"\d+"
        else:
            parts = []
            for start, end in _group_ranges(layers):
                parts.append(_range_to_regex(start, end)
                             if start != end else str(start))
            lay = parts[0] if len(parts) == 1 else f"(?:{'|'.join(parts)})"
        pattern = f"blk\\.({lay})\\.{re.escape(suffix)}"
        rules.append((pattern, tier))

    # Global anchored rules for non-blk tensors (not flag-carried)
    seen = set()
    for tname in names:
        key = _suffix(tname)
        if key[0] != "global":
            continue
        # anchor on a dot boundary with a lookahead: the documented
        # *norm* scar (an unanchored "norm" rule re-tiered every *norm*
        # tensor in the model).
        pattern = f"(?:^|\\.){re.escape(tname)}(?=\\.|$)"
        if pattern not in seen:
            seen.add(pattern)
            rules.append((pattern, assignments[tname]))

    # Provable most-specific-first: fewest matched tensors first.
    # re.search here mirrors the binary's regex_search (substring).
    scored = []
    for pattern, tier in rules:
        rx = re.compile(pattern)
        n = sum(1 for n in names if rx.search(n))
        scored.append((n, pattern, tier))
    scored.sort(key=lambda x: (x[0], x[1]))
    ordered = [(p, t) for _, p, t in scored]

    # The gate: intended map must survive its own rules, every tensor.
    verify_rules(assignments, model, ordered)

    flags = {
        "imatrix": None,
        "output_tensor_type": output_type,
        "token_embedding_type": token_embd_type,
        "tensor_type_rules": [f'--tensor-type "{p}={t}"'
                              for p, t in ordered],
        "base_type": base_type,
        "target_size_mib": target_size_mib,
    }
    return flags


def format_flags(flags: dict) -> str:
    lines = []
    lines.append("  --output-tensor-type " + flags["output_tensor_type"])
    lines.append("  --token-embedding-type " + flags["token_embedding_type"])
    for rule in flags["tensor_type_rules"]:
        lines.append("  " + rule)
    return "\n".join(lines)
