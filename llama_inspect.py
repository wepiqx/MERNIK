"""Model inventory through llama.cpp itself.

The C++ loader owns the GGUF contract: architecture names, metadata keys and
tensor shapes.  Instead of keeping our own detector and GGUF parser, run
``llama-quantize --dry-run`` and read back what llama.cpp loaded.  It prints
the metadata table plus every tensor with its shape and source type, without
reading tensor data into RAM (Ling BF16: ~0.2 s, ~77 KiB of output).

This module is intentionally a thin adapter.  If llama.cpp cannot load the
file, the allocator cannot quantize it either, and the error is surfaced
loudly instead of being guessed around.
"""
import os
import re
import subprocess

QUANTIZER_PATH = os.environ.get(
    "LLAMA_QUANTIZE", "/home/wepiqx/llama.cpp/build/bin/llama-quantize")

# llama.cpp has one architecture name that is inherently quantization-aware;
# gemma4 QAT checkpoints are already in their deploy precision.
_QAT_ARCHES = {"gemma4"}

_KV_RE = re.compile(
    r"^\s*llama_model_loader: - kv\s+\d+:\s+(.+?)\s+"
    r"(str|u32|i32|f32|bool|arr\[[^\]]+\])\s+=\s+(.*)$")
_TENSOR_RE = re.compile(
    r"^\[\s*\d+/\s*(\d+)\]\s+(.+?)\s+-\s+\[([^\]]*)\],\s+"
    r"type\s*=\s*(\S+),\s+size\s*=\s*([0-9.]+)\s*MiB")
_COUNT_RE = re.compile(
    r"llama_model_loader: loaded meta data with (\d+) key-value pairs "
    r"and (\d+) tensors")


def _parse_meta(kind: str, raw: str):
    raw = raw.strip()
    if kind.startswith("arr["):
        return raw
    if kind in ("u32", "i32"):
        try:
            return int(raw)
        except ValueError:
            return raw
    if kind == "f32":
        try:
            return float(raw)
        except ValueError:
            return raw
    if kind == "bool":
        return raw.lower() == "true"
    return raw


def _meta_int(meta: dict, key: str, default: int = 0) -> int:
    value = meta.get(key, default)
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _detect_prefix(tensors: dict) -> str:
    for name in tensors:
        if name.startswith("BLK."):
            return "BLK"
    return "blk"


def _estimate_layers(tensors: dict) -> int:
    max_layer = -1
    for name in tensors:
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] in ("blk", "BLK"):
            try:
                max_layer = max(max_layer, int(parts[1]))
            except ValueError:
                pass
    return max_layer + 1


def _detect_mtp_layer(tensors: dict) -> int | None:
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


def _run_dry_run(path: str) -> str:
    cmd = [QUANTIZER_PATH, "--dry-run", path, "Q4_K"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"llama-quantize not found: {QUANTIZER_PATH}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"llama-quantize --dry-run timed out for {path}") from exc
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(
            f"llama-quantize --dry-run failed for {path} "
            f"(exit {proc.returncode}):\n{tail}")
    return output


def read_model(path: str) -> dict:
    """Return the allocator inventory that llama.cpp reports for ``path``."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    text = _run_dry_run(path)
    meta = {}
    tensors = {}
    expected_tensors = None

    for line in text.splitlines():
        count = _COUNT_RE.search(line)
        if count:
            expected_tensors = int(count.group(2))
            continue
        match = _KV_RE.match(line)
        if match:
            key, kind, raw = match.groups()
            meta[key] = _parse_meta(kind, raw)
            continue
        match = _TENSOR_RE.match(line)
        if not match:
            continue
        _total, name, shape_raw, source_type, source_mib = match.groups()
        shape = [int(part) for part in shape_raw.split(",") if part.strip()]
        n_elements = 1
        for dim in shape:
            n_elements *= dim
        tensors[name] = {
            "shape": shape,
            "n_elements": n_elements,
            # llama.cpp prints four dimensions; trailing ones are padding.
            "rank1": len(shape) <= 1 or all(dim == 1 for dim in shape[1:]),
            "size_mib": n_elements * 2 / 1024 / 1024,
            "source_type": source_type,
            "source_size_mib": float(source_mib),
        }

    if expected_tensors is not None and len(tensors) != expected_tensors:
        raise RuntimeError(
            f"llama-quantize reported {expected_tensors} tensors but "
            f"{len(tensors)} inventory lines were parsed for {path}")
    if not tensors:
        raise RuntimeError(f"llama-quantize returned no tensors for {path}")

    raw_arch = meta.get("general.architecture")
    if not raw_arch:
        raise RuntimeError(
            f"{path} has no general.architecture; llama.cpp cannot load it")
    arch = str(raw_arch).strip().lower()

    prefix = _detect_prefix(tensors)
    raw_layers = _meta_int(meta, f"{arch}.block_count", 0)
    if raw_layers <= 0:
        raw_layers = _estimate_layers(tensors)

    expert_count = _meta_int(meta, f"{arch}.expert_count", 0)
    has_moe = expert_count > 0 or any("exps" in name for name in tensors)
    moe_intermediate_size = (
        _meta_int(meta, f"{arch}.expert_feed_forward_length", 0)
        if has_moe else 0)

    nextn_layers = _meta_int(meta, f"{arch}.nextn_predict_layers", 0)
    has_nextn = any("nextn" in name for name in tensors)
    # blk.32 is MTP only in ~32-layer models (Qwen). Skip for deeper models.
    has_blk32 = raw_layers <= 33 and any(
        name.startswith("blk.32.") or name.startswith("BLK.32.")
        for name in tensors)
    has_mtp = nextn_layers > 0 or has_nextn or has_blk32

    # The MTP head lives one past the last regular layer -- exclude it from
    # n_layers so is_mtp_tensor(layer >= n_layers) catches the whole head.
    n_layers = raw_layers
    if has_mtp:
        mtp_layer = _detect_mtp_layer(tensors)
        if mtp_layer is not None and mtp_layer == raw_layers - 1:
            n_layers = raw_layers - 1

    features = {
        "has_moe": has_moe,
        "moe_intermediate_size": moe_intermediate_size,
        "has_mtp": has_mtp,
        "is_qat": arch in _QAT_ARCHES,
        "prefix": prefix,
        "n_layers": n_layers,
    }

    return {
        "path": path,
        "architecture": arch,
        "features": features,
        "tensors": tensors,
        "n_tensors": len(tensors),
        "meta": meta,
    }


if __name__ == "__main__":
    import json
    import sys

    model = read_model(sys.argv[1])
    summary = {
        "architecture": model["architecture"],
        "features": model["features"],
        "n_tensors": model["n_tensors"],
    }
    print(json.dumps(summary, indent=2))
