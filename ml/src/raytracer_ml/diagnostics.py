"""Reference-defined detail/HDR regions and scene-group uncertainty summaries.

All masks are derived from the independent target/target guides, never the candidate.
Image edges are radiometric proxies; geometry edges are reported separately.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter, minimum_filter
from .io import display


def gradient(image):
    image = np.asarray(image, dtype=np.float64)
    dx = np.diff(image, axis=1, append=image[:, -1:])
    dy = np.diff(image, axis=0, append=image[-1:])
    return dx, dy


def edge_magnitude(image):
    dx, dy = gradient(image)
    return np.sqrt(np.mean(dx * dx + dy * dy, axis=-1))


def reference_regions(target, features=None):
    color = display(target)
    magnitude = edge_magnitude(color)
    edge = magnitude > 0.04
    luminance = np.asarray(target) @ np.array([0.2126, 0.7152, 0.0722])
    masks = {
        "image_edge": edge,
        "flat": ~binary_dilation(magnitude > 0.01, iterations=3),
        "highlight": luminance > 1,
        "dark": (luminance > 0) & (luminance < 0.03),
    }
    for width in (1, 2, 4):
        masks[f"edge_{width}px"] = binary_dilation(edge, iterations=width)
    if features is not None and features.shape[1:] == target.shape[:2]:
        normal = features[6:9].transpose(1, 2, 0)
        distance = features[9] / np.maximum(features[10], 1e-6)
        depth = np.log1p(distance)[..., None]
        masks["geometry_edge"] = (edge_magnitude(normal) > 0.15) | (edge_magnitude(depth) > 0.03)
        masks["albedo_edge"] = edge_magnitude(features[3:6].transpose(1, 2, 0)) > 0.05
        masks["mixed_support"] = (features[11] > 0) & (features[11] < 0.999999)
        masks["unsupported"] = features[11] < 0.999999
    return masks


def detail_metrics(prediction, target, features=None, annotations=None):
    masks = reference_regions(target, features)
    p, t = display(prediction), display(target)
    pdx, pdy = gradient(p)
    tdx, tdy = gradient(t)
    derivative_error = np.mean(np.abs(pdx - tdx) + np.abs(pdy - tdy), axis=-1)
    linear_error = np.mean((np.asarray(prediction, dtype=np.float64) - target) ** 2, axis=-1)
    residual = p.astype(np.float64) - t
    result = {}
    for name, mask in masks.items():
        result[f"{name}_pixels"] = int(mask.sum())
        result[f"{name}_linear_mse"] = float(linear_error[mask].mean()) if mask.any() else None
        result[f"{name}_gradient_mae"] = (
            float(derivative_error[mask].mean()) if mask.any() else None
        )
    luminance_weights = np.array([0.2126, 0.7152, 0.0722])
    pl, tl = prediction @ luminance_weights, target @ luminance_weights
    for name in ("highlight", "dark"):
        mask = masks[name]
        result[f"{name}_energy_relative_bias"] = (
            float((pl[mask].sum() - tl[mask].sum()) / max(1e-8, tl[mask].sum()))
            if mask.any()
            else None
        )
    flat = masks["flat"]
    result["flat_display_residual_variance"] = float(np.var(residual[flat])) if flat.any() else None
    # Locations within one output pixel count as matched anti-aliased image edges.
    # This is not semantic thin-object recall, which needs explicit annotations.
    expected, actual = masks["image_edge"], edge_magnitude(p) > 0.04
    result["image_edge_recall_1px"] = (
        float((distance_transform_edt(~actual)[expected] <= 1).mean())
        if expected.any() and actual.any()
        else (0.0 if expected.any() else None)
    )
    result["image_edge_precision_1px"] = (
        float((distance_transform_edt(~expected)[actual] <= 1).mean())
        if actual.any() and expected.any()
        else (0.0 if actual.any() else None)
    )
    result["image_edge_displacement_p95_px"] = (
        float(np.percentile(distance_transform_edt(~actual)[expected], 95))
        if expected.any() and actual.any()
        else None
    )
    band = masks["edge_2px"]
    low, high = minimum_filter(t, size=(3, 3, 1)), maximum_filter(t, size=(3, 3, 1))
    halo = np.maximum(0, p - high) + np.maximum(0, low - p)
    result["halo_display_mae"] = float(halo[band].mean()) if band.any() else None
    result["linear_energy_relative_bias"] = float((pl.sum() - tl.sum()) / max(1e-8, tl.sum()))
    if annotations is not None and annotations.shape == target.shape:
        thin = binary_dilation(annotations[..., 0] > 0.05, iterations=1)
        result["thin_pixels"] = int(thin.sum())
        result["thin_linear_mse"] = float(linear_error[thin].mean()) if thin.any() else None
        result["thin_gradient_mae"] = float(derivative_error[thin].mean()) if thin.any() else None
        thin_edges = expected & thin
        result["thin_edge_recall_1px"] = (
            float((distance_transform_edt(~actual)[thin_edges] <= 1).mean())
            if actual.any() and thin_edges.any()
            else (0.0 if thin_edges.any() else None)
        )
    from skimage.feature import corner_harris, corner_peaks

    # Reference-image corner localization, distinct from exact geometric vertices.
    reference_corners = corner_peaks(
        corner_harris(t.mean(axis=2)), min_distance=2, threshold_rel=0.05, exclude_border=2
    )
    predicted_corners = corner_peaks(
        corner_harris(p.mean(axis=2)), min_distance=2, threshold_rel=0.05, exclude_border=2
    )
    corner_mask = np.zeros(t.shape[:2], dtype=bool)
    if len(predicted_corners):
        corner_mask[tuple(predicted_corners.T)] = True
    result["reference_image_corners"] = len(reference_corners)
    result["corner_displacement_p95_px"] = (
        float(np.percentile(distance_transform_edt(~corner_mask)[tuple(reference_corners.T)], 95))
        if len(reference_corners) and len(predicted_corners)
        else None
    )
    return result


def grouped_summary(rows, method, key, repeats=1000, seed=9026):
    """Bootstrap independent scene groups, retaining within-group correlated views."""
    groups = {}
    values = []
    for row in rows:
        value = row["metrics"][method].get(key)
        if value is not None and np.isfinite(value):
            groups.setdefault(row.get("group", row["id"].split("-v")[0]), []).append(value)
            values.append(value)
    if not values:
        return {
            "images": 0,
            "groups": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "group_mean_ci95": None,
        }
    group_values = np.array([np.mean(group) for group in groups.values()])
    rng = np.random.default_rng(seed)
    draws = rng.choice(group_values, (repeats, len(group_values)), replace=True).mean(axis=1)
    return {
        "images": len(values),
        "groups": len(groups),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "group_mean": float(group_values.mean()),
        "group_mean_ci95": np.percentile(draws, [2.5, 97.5]).tolist() if len(groups) > 1 else None,
    }
