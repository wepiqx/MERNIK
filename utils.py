def parse_quant_size(output: str) -> float | None:
    """Extract quant size from full output (prefers quant over model size)."""
    import re
    # Try quant size first across all lines
    m = re.search(r"quant size\s*=\s*([0-9.]+)\s*MiB", output)
    if m:
        return float(m.group(1))
    # Fall back to model size
    m = re.search(r"model size\s*=\s*([0-9.]+)\s*MiB", output)
    if m:
        return float(m.group(1))
    return None
