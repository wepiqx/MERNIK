import re

# Ordered from worst to best quality
TIER_ORDER = [
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S",
    "IQ3_XXS", "Q3_K", "IQ3_S",
    "IQ4_XS", "IQ4_NL", "Q4_K", "Q5_K", "Q6_K", "Q8_0", "F16",
]

# Exact bits per weight from ggml block structs (ggml_type_sizef * 8)
# Does NOT include GGUF overhead — GGUF_OVERHEAD_FACTOR is applied separately
TIER_BPW = {
    "IQ1_S": 1.5625,
    "IQ2_XXS": 2.0625,
    "IQ2_XS": 2.3125,
    "IQ2_S": 2.5,
    "IQ3_XXS": 3.0625,
    "Q3_K": 3.4375,
    "IQ3_S": 3.44,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "Q4_K": 4.5,
    "Q5_K": 5.5,
    "Q6_K": 6.5625,
    "Q8_0": 8.5,
    "F16": 16.0,
}
# 1D tensors (norms, biases) are written as F32 by llama.cpp no matter what
# --tensor-type says — verified against --dry-run: 85 assigned tensors in,
# 85 F32 out, on MiniCPM5-2B, including ones assigned Q4_K/IQ4_XS. The ladder
# has no rung for this, so the budget is accounted at F32 bits directly.
F32_BPW = 32.0

# Same story, found by preflight on RINIQ-M2 (qwen35): 24 ssm_conv1d tensors
# (shape [4, 8192]) came back F32 while assigned F16. Small in bytes, but it
# means the tier map was making decisions the binary discards — same class of
# phantom as 1D norms, different reason (tensor category, not rank).
FORCE_F32_TYPES = {"ssm_conv1d"}
GGUF_OVERHEAD_FACTOR = 1.0

# Quant quality rank: higher = better (same order as TIER_ORDER)
QUANT_RANK = {tier: i for i, tier in enumerate(TIER_ORDER)}

# Per-class hard floor — display-only (--show-floors). The greedy never
# downgrades below its base, so floors are not enforced in code.
CLASS_HARD_FLOORS = {
    "gate": "Q8_0",
    "attn_proj": "Q8_0",
    "ffn_gate_up": "IQ4_XS",
    "ffn_down": "Q6_K",
    "norms": "F16",
    "ssm_params": "F16",
    "mtp": "IQ4_XS",
    "embd": "Q5_K",
}

# Per-class max tier — never exceed this (for greedy upgrades)
# Deep layers can be upgraded to Q8_0 for important tensors
CLASS_MAX_TIER = {
    "gate": "Q8_0",
    "attn_proj": "Q8_0",
    "ffn_gate_up": "Q8_0",
    "ffn_down": "Q8_0",
    "norms": "F16",
    "ssm_params": "F16",
    "mtp": "Q8_0",
    "embd": "Q8_0",
}

# Classes that can go to Q3_K when --allow-q3-or-lower is set
# (matched against tensor *type*: ffn_gate, ffn_up, ffn_down, attn_output, ssm_out)
CAN_Q3 = {"ffn_gate", "ffn_up", "ffn_down", "attn_output", "ssm_out"}
ALLOW_LOWER_FLOOR = "IQ2_XXS"  # lowest starting tier for --allow-q3-or-lower

# Tier for MTP head deployment
MTP_DEPLOY_TIER = "Q8_0"

# Sub-4-bit toxicity: tensors sitting below 4.0 real bpw poison PPL far worse
# than the theoretical MSE suggests (verified empirically). Their effective
# MSE is inflated, so the queue escapes sub-4 eagerly (bottom-up) and enters
# it reluctantly (top-down). Tunable.
TOXICITY_SUB4 = 2.0

# --- TOXICITY IS A TAIL, NOT A STEP (measured 2026-09-28) ------------------
# The multiplier above is a single number applied to EVERY tensor below
# 4.0 bpw. Paired measurement on the same 112 Spark-1.7B units, Q5_K->Q4_K
# against Q5_K->Q3_K (gnom/labels/spark17_damage_q4.jsonl vs _q3.jsonl):
#
#   damage @Q4:  mean +0.120  median +0.042  std 0.773  P(>0)=60.7%  P(>1)= 7.1%
#   damage @Q3:  mean +0.090  median +0.192  std 1.521  P(>0)=61.6%  P(>1)=19.6%
#   paired diff: mean -0.031  median +0.093  std 1.605  P(>0)=59.8%
#
# Read that carefully: the mean difference is ZERO (step cancelled out), the
# MEDIAN is positive, and the top-1 unit (0.9% of all units) carries 58% of
# the net damage. Crossing into sub-4 does not make the typical tensor worse
# by a factor — it makes ~1 in 5 tensors detonate where 1 in 14 did before.
# The mean is silent because a few large negatives cancel a heavy positive
# tail. So the effect is a change in the TAIL, not in the mean.
#
# What that means for the queue (spec for the classifier owner — constants.py
# holds the data, classifier.py consumes it; no behaviour change here):
#
#   1. kind="step" stays the default so every existing table reproduces.
#   2. kind="tail" is NOT a different constant. If detonation is
#      unpredictable — which is exactly what Gnom-0.2 established — then no
#      reweighting of the tiers can capture it, and the expectation is the
#      same number. What the tail changes is the OBJECTIVE: expected loss is
#      dominated by a few groups, so a queue that maximises expected gain is
#      not the queue you want; one that bounds worst case is. That is a new
#      lens, not a new constant, and it is a classifier.py decision.
#   3. The measurement that can settle it is tail P(detonation | group), and
#      that needs paired multi-tier labels on the same units — one drop tier
#      identifies nothing (pool_kld.json has a single level: 5.5 -> 4.5 bpw).
TOXICITY_MODEL = {
    "kind": "step",                 # "step" (legacy) | "tail"
    "sub4_bpw": 4.0,
    "multiplier": TOXICITY_SUB4,
    "evidence": {                   # spark17, 112 paired units, Q5 base
        "q4": {"mean": 0.1203, "median": 0.0417, "std": 0.7726,
               "p_gt_0": 0.607, "p_gt_1": 0.071},
        "q3": {"mean": 0.0896, "median": 0.1917, "std": 1.5206,
               "p_gt_0": 0.616, "p_gt_1": 0.196},
        "paired": {"mean": -0.0307, "median": 0.0934, "std": 1.6054,
                   "p_gt_0": 0.598},
        "top1_share_of_net_damage": 0.58,
    },
    "tail": {
        "detonation_threshold": 1.0,     # damage above this = "detonates"
        "p_detonation": {"q4": 0.071, "q3": 0.196},   # ~2.8x across sub-4
        "usable_without_a_predictor": False,
    },
}


def toxicity_multiplier(tier_bpw: float, model: dict | None = None) -> float:
    """Effective-MSE inflation for a tier, per TOXICITY_MODEL.

    kind="step" reproduces the legacy constant exactly (unchanged default).
    kind="tail" is intentionally NOT implemented as a per-group inflation:
    see the spec above — without a predictor of which groups detonate, the
    expected value is identical and only the objective would change.
    """
    model = model or TOXICITY_MODEL
    if tier_bpw >= model["sub4_bpw"]:
        return 1.0
    return float(model["multiplier"])

# Tier for output/token_embd — must match --output-tensor-type /
# --token-embedding-type in generate_flags (llama.cpp applies those
# unconditionally, explicit --tensor-type rules can't override output.weight)
EMBD_DEPLOY_TIER = "Q5_K"

# Tensor types pinned to EMBD_DEPLOY_TIER (binary writes them at the
# output/token types no matter the assignment)
EMBD_PIN_TYPES = {"output", "token_embd"}

TENSOR_CLASS = {
    # Qwen 3.5 hybrid
    "attn_gate": "gate",
    "ssm_alpha": "gate",
    "ssm_beta": "gate",
    "attn_q": "attn_proj",
    "attn_k": "attn_proj",
    "attn_v": "attn_proj",
    "attn_qkv": "attn_proj",
    "attn_output": "attn_proj",
    "ffn_gate": "ffn_gate_up",
    "ffn_up": "ffn_gate_up",
    "ffn_down": "ffn_down",
    "ssm_out": "ffn_down",
    "ssm_conv1d": "norms",
    "router": "norms",
    "ssm_dt": "ssm_params",
    "ssm_a": "ssm_params",
    "nextn": "mtp",
    "ffn_gate_exps": "ffn_gate_up",
    "ffn_up_exps": "ffn_gate_up",
    "ffn_down_exps": "ffn_down",
    "ffn_gate_shexp": "ffn_gate_up",  # Ling shared experts (always active — never CAN_Q3)
    "ffn_up_shexp": "ffn_gate_up",
    "ffn_down_shexp": "ffn_down",
    "ffn_gate_inp": "norms",
    # Ling MLA low-rank factors → attention projections
    "attn_q_a": "attn_proj",
    "attn_q_b": "attn_proj",
    "attn_kv_a_mqa": "attn_proj",
    "attn_k_b": "attn_proj",
    "attn_v_b": "attn_proj",

    # Standard llama.cpp tensor names
    "q_proj": "attn_proj",
    "k_proj": "attn_proj",
    "v_proj": "attn_proj",
    "o_proj": "attn_proj",
    "gate_proj": "ffn_gate_up",
    "up_proj": "ffn_gate_up",
    "down_proj": "ffn_down",
}

ARCH_FEATURES = {
    "qwen35": {
        "has_qkv": True,
        "has_ssm": True,
        "has_mtp": True,
        "has_moe": False,
        "is_qat": False,
        "prefix": "blk",
        "n_layers": 32,
    },
    "mellum2": {
        "has_qkv": False,
        "has_ssm": False,
        "has_mtp": False,
        "has_moe": True,
        "is_qat": False,
        "prefix": "blk",
        "n_layers": 28,
    },
    "gemma4": {
        "has_qkv": False,
        "has_ssm": False,
        "has_mtp": False,
        "has_moe": False,
        "is_qat": True,
        "prefix": "blk",
        "n_layers": 48,
    },
    "granite": {
        "has_qkv": False,  # separate attn_q/k/v projections, like gemma4 — but NOT qat
        "has_ssm": False,
        "has_mtp": False,
        "has_moe": False,
        "is_qat": False,
        "prefix": "blk",
        "n_layers": 40,
    },
    "spark2_5": {
        "has_qkv": True,  # fused q_k_v_proj (tied groups like qwen35)
        "has_ssm": False,
        "has_mtp": False,
        "has_moe": False,
        "is_qat": False,
        "prefix": "blk",
        "n_layers": 36,  # per imatrix (36 blocks); overwritten by estimate from file
    },
    "bailingmoe3": {
        "has_qkv": False,  # separate projections + MLA low-rank factors
        "has_ssm": True,   # KDA layers carry ssm_* tensors (ssm_params class)
        "has_mtp": False,
        "has_moe": True,
        "is_qat": False,
        "prefix": "blk",
        "n_layers": 24,
        "moe_intermediate_size": 512,  # 512 % 256 == 0 → no K-quant padding needed
    },
}

# general.architecture aliases that differ from preset keys.
METADATA_ARCH_MAP = {
    "mellum": "mellum2",
}

# MoE routers — always F16, outside the budget. Corrupting the router
# corrupts expert choice for every token (verified disaster).
ROUTER_PIN_TYPES = {"ffn_gate_inp", "exp_probs_b", "ffn_gate_tid2eid"}


def strip_weight(name: str) -> str:
    return name.lstrip(".").removesuffix(".weight").removesuffix(".bias")


def get_tensor_type(name: str) -> str:
    parts = strip_weight(name).split(".")
    if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
        return parts[2] if len(parts) >= 3 else "unknown"
    if "token_embd" in name:
        return "token_embd"
    if name.startswith("output") and "norm" not in name:
        return "output"
    return name


def get_tensor_class(ttype: str) -> str:
    if ttype in TENSOR_CLASS:
        return TENSOR_CLASS[ttype]
    if "norm" in ttype or "scale" in ttype:
        return "norms"
    if ttype.startswith("ssm_"):
        return "ssm_params"
    if ttype in ("token_embd", "output", "embed_tokens", "lm_head",
                 "vision_embedder", "audio_embedder"):
        return "embd"
    return "unknown"


def is_mtp_tensor(name: str, n_layers: int = 32) -> bool:
    if "nextn" in name:
        return True
    if n_layers > 40:
        return False  # Deep models (Gemma4, 48 layers) use separate drafter, not in-model MTP
    layer = get_layer_number(name)
    return layer is not None and layer >= n_layers


def get_layer_number(name: str) -> int | None:
    parts = strip_weight(name).split(".")
    if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None
