import gguf
import numpy as np
from typing import List


def read_imatrix(path: str) -> dict:
    """Parse imatrix GGUF, return per-tensor importance data."""
    r = gguf.GGUFReader(path)
    raw = {}
    meta = {}

    for k, v in r.fields.items():
        try:
            meta[k] = v.data
        except:
            meta[k] = str(v)

    for t in r.tensors:
        name = t.name
        arr = np.array(t.data, dtype=np.float64)

        if name.endswith(".in_sum2"):
            base = name[:-8]
            if base not in raw:
                raw[base] = {}
            raw[base]["in_sum2"] = arr
        elif name.endswith(".counts"):
            base = name[:-7]
            if base not in raw:
                raw[base] = {}
            raw[base]["counts"] = float(np.mean(arr))

    result = {}
    for base, data in raw.items():
        if "in_sum2" not in data:
            continue
        arr = data["in_sum2"]
        result[base] = {
            "importance_mean": float(np.mean(arr)),
            "importance_sum": float(np.sum(arr)),
            "importance_max": float(np.max(arr)),
            "importance_min": float(np.min(arr)),
            "n_elements": arr.size,
            "in_sum2_raw": arr,
        }

    return {
        "path": path,
        "tensors": result,
        "n_tensors": len(result),
        "meta": meta,
    }


def combine_imatrix(imatrix_list: List[dict], method: str = "max") -> dict:
    """Combine multiple imatrix into one by aggregating importance.
    
    Args:
        imatrix_list: List of imatrix dicts from read_imatrix()
        method: "max", "mean", or "weighted_mean"
    """
    if not imatrix_list:
        return {}
    if len(imatrix_list) == 1:
        return imatrix_list[0]
    
    # Get all tensor names across all imatrix
    all_names = set()
    for im in imatrix_list:
        all_names.update(im["tensors"].keys())
    
    combined_tensors = {}
    for name in all_names:
        vals = []
        in_sum2_raw = None
        n_elements = 0
        
        for im in imatrix_list:
            if name in im["tensors"]:
                t = im["tensors"][name]
                vals.append(t["importance_mean"])
                if in_sum2_raw is None and "in_sum2_raw" in t:
                    in_sum2_raw = t["in_sum2_raw"]
                n_elements = max(n_elements, t["n_elements"])
        
        if not vals:
            continue
        
        if method == "max":
            imp_mean = max(vals)
        elif method == "mean":
            imp_mean = sum(vals) / len(vals)
        elif method == "weighted_mean":
            # Could weight by n_elements or dataset size
            imp_mean = sum(vals) / len(vals)
        else:
            imp_mean = max(vals)
        
        combined_tensors[name] = {
            "importance_mean": imp_mean,
            "importance_sum": imp_mean * n_elements,
            "importance_max": max(v.get("importance_max", 0) for im in imatrix_list if name in im["tensors"] for v in [im["tensors"][name]]),
            "importance_min": min(v.get("importance_min", float('inf')) for im in imatrix_list if name in im["tensors"] for v in [im["tensors"][name]]),
            "n_elements": n_elements,
            "in_sum2_raw": in_sum2_raw,
        }
    
    # Merge metadata
    combined_meta = {}
    for im in imatrix_list:
        for k, v in im["meta"].items():
            if k not in combined_meta:
                combined_meta[k] = v
    
    return {
        "path": "+".join(im["path"] for im in imatrix_list),
        "tensors": combined_tensors,
        "n_tensors": len(combined_tensors),
        "meta": combined_meta,
    }


def detect_tied_groups(imatrix: dict, atol: float = 1e-5) -> list:
    """Find tied tensor groups (identical importance arrays)."""
    names = sorted(imatrix["tensors"].keys())
    tied_groups = []
    visited = set()

    for i, n1 in enumerate(names):
        if n1 in visited:
            continue
        group = [n1]
        arr1 = imatrix["tensors"][n1].get("in_sum2_raw")
        if arr1 is None:
            tied_groups.append(group)
            visited.add(n1)
            continue
        for j in range(i + 1, len(names)):
            n2 = names[j]
            if n2 in visited:
                continue
            arr2 = imatrix["tensors"][n2].get("in_sum2_raw")
            if arr2 is None:
                continue
            if arr1.shape == arr2.shape and np.allclose(arr1, arr2, atol=atol):
                group.append(n2)
                visited.add(n2)
        tied_groups.append(group)
        visited.add(n1)

    return tied_groups


def build_importance_table(imatrix: dict, model: dict) -> dict:
    """Build unified importance table, merging imatrix with model tensor info."""
    table = {}
    for tname, info in imatrix["tensors"].items():
        ttype = _imatrix_type(tname)
        table[tname] = {
            "importance_mean": info["importance_mean"],
            "importance_sum": info["importance_sum"],
            "importance_max": info["importance_max"],
            "importance_min": info["importance_min"],
            "n_elements": info["n_elements"],
            "type": ttype,
        }
    # Also index by name without trailing dot (safety for old code)
    for tname, info in list(table.items()):
        if tname.endswith("."):
            alt = tname.rstrip(".")
            table[alt] = info
    return table


def _imatrix_type(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        return parts[2]
    return name
