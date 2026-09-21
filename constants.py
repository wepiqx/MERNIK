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
