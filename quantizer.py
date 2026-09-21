import subprocess
import re
import os
import warnings
from utils import parse_quant_size


QUANTIZER_PATH = os.environ.get(
    "LLAMA_QUANTIZE", "/home/wepiqx/llama.cpp/build/bin/llama-quantize"
)


def _build_cmd(flags: dict, model_in: str, model_out: str, dry_run: bool = False) -> list:
    """Build llama-quantize command. Options before positional args."""
    cmd = [QUANTIZER_PATH]

    if dry_run:
        cmd.append("--dry-run")

    if flags.get("imatrix"):
        imatrix = flags["imatrix"]
        if isinstance(imatrix, list):
            if len(imatrix) > 1:
                warnings.warn(
                    "llama-quantize accepts a single imatrix: passing only the "
                    f"first of {len(imatrix)} files ({imatrix[0]}). The importance "
                    "table used files combined, so sizes may diverge slightly.",
                    RuntimeWarning,
                )
            imatrix = imatrix[0]  # llama-quantize accepts one imatrix; we combined in importance table
        cmd.extend(["--imatrix", imatrix])
    cmd.extend(["--output-tensor-type", flags["output_tensor_type"]])
    cmd.extend(["--token-embedding-type", flags["token_embedding_type"]])

    for rule in flags["tensor_type_rules"]:
        parts = rule.split(" ", 1)
        if len(parts) == 2:
            cmd.extend(["--tensor-type", parts[1].strip('"')])

    # Positional args: model_in model_out type
    cmd.append(model_in)
    cmd.append(model_out)
    cmd.append(flags["base_type"])

    return cmd


def run_dry_run(flags: dict, model_in: str) -> float | None:
    """Run llama-quantize --dry-run and return quant size in MiB."""
    cmd = _build_cmd(flags, model_in, "/dev/null", dry_run=True)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",  # binary may print non-UTF8 bytes on unknown-arch errors
        timeout=300,
    )

    output = (result.stdout or "") + (result.stderr or "")
    size = parse_quant_size(output)
    if size is not None:
        return size

    print("STDERR:", result.stderr[:2000])
    return None


def run_quantization(flags: dict, model_in: str, model_out: str, dry_run: bool = False) -> bool:
    """Run actual quantization or dry run."""
    cmd = _build_cmd(flags, model_in, model_out, dry_run=dry_run)

    print("Running:", " ".join(cmd[:6]) + " ...")
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=3600)

    if dry_run:
        output = (result.stdout or "") + (result.stderr or "")
        return parse_quant_size(output)

    print(result.stderr[:500])
    success = result.returncode == 0
    if success:
        import os
        size_mib = os.path.getsize(model_out) / 1024 / 1024
        print(f"Done: {model_out} ({size_mib:.0f} MiB)")
    return success
