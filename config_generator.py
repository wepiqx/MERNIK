import re
from constants import QUANT_RANK


def get_regex_priority(regex: str) -> int:
    """Higher = more specific = should come first (first-match-wins)."""
    score = 0

    if "nextn" in regex:
        score += 200
    if re.search(r"(blk|BLK)\\.3[0-2]\\.", regex):
        score += 100
    if re.search(r"(blk|BLK)\\.0\\.", regex):
        score += 90
    if re.search(r"(blk|BLK)\\.31\\.", regex):
        score += 80
    # Layer-range group: (blk|BLK)\.( — more specific than \d+
    if r"(blk|BLK)\.(" in regex:
        score += 50
    elif r"(blk|BLK)\.\d" in regex:
        score += 30
    if regex.startswith(".*"):
        score -= 50
    if regex.endswith(r"\.weight"):
        score += 10

    return score


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


def generate_flags(
    assignments: dict,
    model: dict,
    base_type: str,
    target_size_mib: float = None,
) -> dict:
    is_qat = model.get("features", {}).get("is_qat", False)
    # Normally Q5_K pins (see EMBD_DEPLOY_TIER); with --free-pins the
    # classifier may assign output/token_embd other tiers — honor them here
    # since the binary applies these two flags unconditionally.
    output_type = assignments.get("output.weight", "Q5_K")
    token_embd_type = assignments.get("token_embd.weight", "Q5_K")

    max_layer = model.get("features", {}).get("n_layers", 31)

    rules = []

    # Group blk tensors by (ttype, tier)
    type_tier_layers = {}
    for tname, tier in assignments.items():
        parts = tname.split(".")
        if len(parts) >= 3 and parts[0] in ("blk", "BLK"):
            try:
                layer = int(parts[1])
            except ValueError:
                continue
            ttype = parts[2]
            key = (ttype, tier)
            if key not in type_tier_layers:
                type_tier_layers[key] = []
            type_tier_layers[key].append(layer)

    # Generate rules for blk tensor groups
    for (ttype, tier), layers in sorted(
        type_tier_layers.items(),
        key=lambda x: -QUANT_RANK.get(x[0][1], 0),
    ):
        layers = sorted(set(layers))

        if len(layers) >= 8 and _is_contiguous(layers, 0, max_layer):
            pattern = f"(blk|BLK)\\.\\d+\\.{ttype}={tier}"
        else:
            parts = []
            for start, end in _group_ranges(layers):
                if start == end:
                    parts.append(str(start))
                else:
                    parts.append(_range_to_regex(start, end))
            desc = "|".join(parts)
            pattern = f"(blk|BLK)\\.({desc})\\.{ttype}={tier}"

        prio = get_regex_priority(pattern) + (
            10 if tier == "Q8_0" else 5 if tier == "Q6_K" else 0
        ) + (5 if len(layers) == 1 else 0) + (3 if "ffn_down" in ttype else 0)

        rules.append((pattern, prio))

    # Generate rules for global tensors (non-blk)
    prefix = model.get("features", {}).get("prefix", "blk")
    for tname, tier in assignments.items():
        parts = tname.split(".")
        if len(parts) >= 2 and parts[0].lower() == prefix.lower():
            continue
        ttype = parts[0] if len(parts) >= 1 else tname
        if ttype in ("token_embd", "output"):
            continue
        if ttype == tname and "." in tname:
            # e.g. "nextn.eh_proj" without blk prefix
            pass
        # Check that this global tensor wasn't already handled as a blk tensor
        pattern = f".*{re.escape(ttype)}.*={tier}"
        prio = get_regex_priority(pattern) + (5 if tier == "Q8_0" else 0)
        # Deduplicate (same pattern may appear from different names)
        if not any(p == pattern for p, _ in rules):
            rules.append((pattern, prio))

    rules.sort(key=lambda x: -x[1])

    flags = {
        "imatrix": None,
        "output_tensor_type": output_type,
        "token_embedding_type": token_embd_type,
        "tensor_type_rules": [f'--tensor-type "{r[0]}"' for r in rules],
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
