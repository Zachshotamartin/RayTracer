"""Retained independent Monte Carlo checks, not noise-free ground truth."""

import json
import numpy as np

from ..diagnostics import reference_regions
from ..io import digest, safe_path


def reference_errors(target, alternate, features=None):
    linear = np.mean((target.astype(np.float64) - alternate) ** 2, axis=-1)
    log = np.mean(
        (np.log1p(target.astype(np.float64)) - np.log1p(alternate.astype(np.float64))) ** 2, axis=-1
    )
    regions = {}
    for name, mask in reference_regions(target, features).items():
        regions[name] = {
            "pixels": int(mask.sum()),
            "linear_mse": float(linear[mask].mean()) if mask.any() else None,
            "log_mse": float(log[mask].mean()) if mask.any() else None,
            "linear_squared_error_p95": float(np.quantile(linear[mask], 0.95))
            if mask.any()
            else None,
        }
    return {"linear_mse": float(linear.mean()), "log_mse": float(log.mean()), "regions": regions}


def validate_reference_check(root, receipt, rows):
    """Verify pairing and retained pixels before accepting an existing check receipt."""
    if receipt.get("schema_version") != 2:
        raise ValueError("Reference check has no retained-image schema")
    matches = [r for r in rows if r["reference"] == receipt["reference"]]
    if not matches:
        raise ValueError("Reference check is not paired with this dataset")
    row = matches[0]
    if (
        any(
            receipt[key] != row[key]
            for key in (
                "reference_sha256",
                "reference_samples",
                "target_seed",
                "scene_sha256",
                "split",
            )
        )
        or receipt["samples"] <= row["reference_samples"]
    ):
        raise ValueError("Reference check identity/sample budget mismatch")
    if receipt["seed"] in {row["target_seed"], *(r["input_seed"] for r in matches)}:
        raise ValueError("Reference check seed is not independent")
    path = safe_path(root, receipt["path"])
    reference = safe_path(root, receipt["reference"])
    if (
        not path.is_file()
        or digest(path) != receipt["sha256"]
        or digest(reference) != receipt["reference_sha256"]
    ):
        raise ValueError("Reference check checksum mismatch")
    with np.load(path, allow_pickle=False) as saved, np.load(reference, allow_pickle=False) as ref:
        alternate, target = saved["target"], ref["target"]
        if (
            alternate.shape != target.shape
            or not np.isfinite(alternate).all()
            or np.any(alternate < 0)
        ):
            raise ValueError("Invalid reference check shape/radiance")
        errors = reference_errors(target, alternate, ref["features"] if "features" in ref else None)
        if any(receipt.get(key) != value for key, value in errors.items()):
            raise ValueError("Reference check metrics disagree with retained images")


def validate_reference_checks(root, rows, info):
    retained, legacy, seen = 0, 0, set()
    for path in sorted((root / "reference_checks").glob("*.json")):
        receipt = json.loads(path.read_text())
        if "schema_version" not in receipt:
            legacy += 1
            continue
        validate_reference_check(root, receipt, rows)
        if receipt["reference"] in seen:
            raise ValueError("Duplicate independent reference check")
        seen.add(receipt["reference"])
        retained += 1
    expected = info["estimate"].get("reference_check_images")
    if expected is not None and retained != expected:
        raise ValueError(f"Incomplete retained reference checks: {retained}/{expected}")
    return {"retained_images": retained, "legacy_scalar_only": legacy}
