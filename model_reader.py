import gguf
import numpy as np
from constants import ARCH_FEATURES, METADATA_ARCH_MAP


def _metadata_arch(reader) -> str | None:
    """Read general.architecture straight from GGUF metadata (most reliable)."""
    try:
        field = reader.fields.get("general.architecture")
        if field is None:
            return None
        raw = field.parts[-1]
        return bytes(np.asarray(raw).tobytes()).decode("utf-8", errors="ignore")
    except Exception:
        return None


def detect_architecture(tensors: dict, metadata_arch: str | None = None) -> str:
    """Detect model architecture from tensor names."""
    if metadata_arch:
        key = metadata_arch.strip().lower()
        if key in METADATA_ARCH_MAP:
            return METADATA_ARCH_MAP[key]
        if key in ARCH_FEATURES:
            return key
    names = list(tensors.keys())

    has_ssm = any("ssm_" in n for n in names)
    has_qkv = any("attn_qkv" in n for n in names)
    has_nextn = any("nextn" in n for n in names)
    has_moe = any("exps" in n for n in names)
    has_separate_qkv = any("attn_q.weight" in n for n in names)
    has_gemma_specific = any(
        t in n for t in ("layer_output_scale", "post_attention_norm", "post_ffw_norm")
        for n in names
    )

    if has_ssm and has_qkv:
        return "qwen35"
    if has_moe:
        return "mellum2"
    # Исправление: заменяем неопределённую has_blk_attn на уже существующий признак
    if has_separate_qkv or has_gemma_specific:
        return "gemma4"
    return "unknown"


def _detect_prefix(tensors: dict) -> str:
    for name in tensors:
        if name.startswith("BLK."):
            return "BLK"
    return "blk"


def _estimate_layers(tensors: dict) -> int:
    max_layer = 0
    for name in tensors:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
            try:
                layer = int(parts[1])
                if layer > max_layer:
                    max_layer = layer
            except ValueError:
                pass
    return max_layer + 1  # layers are 0-indexed


def _detect_mtp_layer(tensors: dict) -> int | None:
    """Find the MTP head layer index (one past the last regular layer)."""
    for name in tensors:
        if "nextn" in name:
            parts = name.split(".")
            if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
                try:
                    return int(parts[1])
                except ValueError:
                    pass
    for name in tensors:
        if name.startswith("blk.32.") or name.startswith("BLK.32."):
            return 32
    return None


def read_model(path: str) -> dict:
    """Parse BF16 GGUF, return model info."""
    r = gguf.GGUFReader(path)
    tensors = {}
    meta = {}

    for k, v in r.fields.items():
        # Безопасное извлечение данных (список/число, а не raw numpy)
        try:
            data = v.data
            if isinstance(data, np.ndarray):
                data = data.tolist()
            elif isinstance(data, (np.generic,)):
                data = data.item()
            meta[k] = data
        except Exception:
            meta[k] = str(v)

    for t in r.tensors:
        shape = list(t.shape)
        name = t.name
        n_elements = int(np.prod(shape))
        tensors[name] = {
            "shape": shape,
            "n_elements": n_elements,
            "size_mib": n_elements * 2 / 1024 / 1024,
        }

    arch = detect_architecture(tensors, _metadata_arch(r))
    arch_features = ARCH_FEATURES.get(arch, {}).copy()

    prefix = _detect_prefix(tensors)
    raw_layers = _estimate_layers(tensors)

    has_moe = any("exps" in n for n in tensors)
    if has_moe:
        arch_features["has_moe"] = True

    has_nextn = any("nextn" in n for n in tensors)
    # blk.32 is MTP only in ~32-layer models (Qwen). Skip for deeper models (Gemma4, 48 layers).
    has_blk32 = raw_layers <= 33 and any(
        n.startswith("blk.32.") or n.startswith("BLK.32.") for n in tensors
    )
    arch_features["has_mtp"] = has_nextn or has_blk32

    # The MTP head lives one past the last regular layer — exclude it from
    # n_layers so is_mtp_tensor(layer >= n_layers) catches the whole head,
    # not just nextn.* tensors.
    n_layers = raw_layers
    if arch_features["has_mtp"]:
        mtp_layer = _detect_mtp_layer(tensors)
        if mtp_layer is not None and mtp_layer == raw_layers - 1:
            n_layers = raw_layers - 1

    if arch == "mellum2" and arch_features.get("moe_intermediate_size", 0) == 0:
        arch_features["moe_intermediate_size"] = 896

    arch_features["prefix"] = prefix
    if n_layers > 0:
        arch_features["n_layers"] = n_layers

    return {
        "path": path,
        "architecture": arch,
        "features": arch_features,
        "tensors": tensors,
        "n_tensors": len(tensors),
        "meta": meta,
    }
