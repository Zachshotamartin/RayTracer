"""Cheap validation-only structural measurements used during checkpoint selection."""

import numpy as np
from scipy.ndimage import binary_dilation
from .diagnostics import gradient, edge_magnitude
from .metrics import image_metrics
from .io import display


def structural_scores(prediction, target):
    p, t = display(prediction), display(target)
    mask = binary_dilation(edge_magnitude(t) > 0.04, iterations=2)
    px, py = gradient(p)
    tx, ty = gradient(t)
    error = np.mean(np.abs(px - tx) + np.abs(py - ty), axis=-1)
    return {
        "ssim": image_metrics(prediction, target)["ssim"],
        "edge": float(error[mask].mean()) if mask.any() else 0.0,
    }


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
    return float(score), bool(ssim_gap <= 0 and edge_gap <= 0)
