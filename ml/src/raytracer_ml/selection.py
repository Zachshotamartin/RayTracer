"""Validation-only measurements and explicit, fail-closed eligibility constraints."""

import math
import numpy as np
from scipy.ndimage import binary_dilation
from .diagnostics import gradient, edge_magnitude, reconstruction_region_metrics
from .metrics import image_metrics
from .io import display


def structural_scores(prediction, target, metrics=None):
    p, t = display(prediction), display(target)
    mask = binary_dilation(edge_magnitude(t) > 0.04, iterations=2)
    px, py = gradient(p)
    tx, ty = gradient(t)
    error = np.mean(np.abs(px - tx) + np.abs(py - ty), axis=-1)
    return {
        "ssim": (metrics if metrics is not None else image_metrics(prediction, target))["ssim"],
        "edge": float(error[mask].mean()) if mask.any() else 0.0,
    }


def quality_scores(prediction, target, features, model_config):
    regions = reconstruction_region_metrics(prediction, target, features, model_config)
    metrics = image_metrics(prediction, target)
    return {
        **structural_scores(prediction, target, metrics),
        "psnr": metrics["psnr"],
        "hdr_mse": float(np.mean((prediction.astype(np.float64) - target) ** 2)),
        "model_hdr_mse": regions["model_region_linear_mse"],
    }


def aggregate_scores(scores):
    result = {}
    for key in {key for row in scores for key in row}:
        values = [row[key] for row in scores if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    if any("temporal_log_mae" in row for row in scores):
        result["temporal_transitions"] = sum(
            row.get("temporal_log_mae") is not None for row in scores
        )
    return result


def validate_selection_config(config):
    if not config:
        return
    allowed = {
        "ssim_tolerance",
        "edge_ratio",
        "ssim_penalty",
        "edge_penalty",
        "hdr_ratio",
        "model_hdr_ratio",
        "preservation_ratio",
        "temporal_ratio",
        "min_preservation_views",
        "min_temporal_transitions",
        "psnr_gain",
        "psnr_min",
        "ssim_min",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown selection constraints: {sorted(set(config) - allowed)}")
    for key, value in config.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Selection {key} must be finite and numeric")
        if key.startswith("min_"):
            if type(value) is not int or value < 1:
                raise ValueError(f"Selection {key} must be a positive integer")
        elif value < 0 or (key.endswith("ratio") and value == 0):
            raise ValueError(f"Invalid selection {key}")
    for coverage, gate in (
        ("min_preservation_views", "preservation_ratio"),
        ("min_temporal_transitions", "temporal_ratio"),
    ):
        if coverage in config and gate not in config:
            raise ValueError(f"{coverage} requires {gate}")


def constraint_report(measured, baseline, config):
    validate_selection_config(config)
    if not config:
        return {}
    specs = [
        ("ssim", "ssim", "min", None),
        ("edge", "edge", "ratio", config.get("edge_ratio", 1.05)),
    ]
    for gate, metric in (
        ("hdr_ratio", "hdr_mse"),
        ("model_hdr_ratio", "model_hdr_mse"),
        ("preservation_ratio", "preservation_log_mae"),
        ("temporal_ratio", "temporal_log_mae"),
    ):
        if gate in config:
            specs.append((gate, metric, "ratio", config[gate]))
    result = {}
    for name, metric in (("psnr_min", "psnr"), ("ssim_min", "ssim"), ("psnr_gain", "psnr")):
        if name not in config:
            continue
        value = measured.get(metric)
        reference = baseline.get(metric) if name == "psnr_gain" else None
        present = value is not None and math.isfinite(value)
        if name == "psnr_gain":
            present = present and reference is not None and math.isfinite(reference)
        limit = (
            (reference + config[name] if name == "psnr_gain" else config[name]) if present else None
        )
        result[name] = dict(
            value=value,
            baseline=reference,
            limit=limit,
            passed=bool(present and value >= limit),
            missing=not present,
        )
    for name, metric, comparison, threshold in specs:
        value, reference = measured.get(metric), baseline.get(metric)
        present = all(v is not None and math.isfinite(v) for v in (value, reference))
        limit = (
            (
                reference - config.get("ssim_tolerance", 0.005)
                if comparison == "min"
                else reference * threshold
            )
            if present
            else None
        )
        result[name] = dict(
            value=value,
            baseline=reference,
            limit=limit,
            passed=bool(present and (value >= limit if comparison == "min" else value <= limit)),
            missing=not present,
        )
    for gate, metric, option in (
        ("preservation_ratio", "preservation_views", "min_preservation_views"),
        ("temporal_ratio", "temporal_transitions", "min_temporal_transitions"),
    ):
        if gate in config:
            count, minimum = measured.get(metric, 0), config.get(option, 1)
            result[metric] = dict(
                value=count, minimum=minimum, passed=bool(count is not None and count >= minimum)
            )
    return result


def checkpoint_score(loss, measured, baseline, config):
    if not config:
        return float(loss), True
    ssim_gap = baseline["ssim"] - measured["ssim"] - config.get("ssim_tolerance", 0.005)
    edge_ratio = measured["edge"] / max(1e-8, baseline["edge"])
    edge_gap = edge_ratio - config.get("edge_ratio", 1.05)
    score = (
        loss
        + config.get("ssim_penalty", 1) * max(0, ssim_gap)
        + config.get("edge_penalty", 0.05) * max(0, edge_gap)
    )
    constraints = constraint_report(measured, baseline, config)
    return float(score), all(c["passed"] for c in constraints.values())
