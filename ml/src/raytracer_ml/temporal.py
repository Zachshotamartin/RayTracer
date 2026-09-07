"""Geometry-validated history reprojection; cumulative static passes are not new frames."""

import copy
import numpy as np
from .io import identity


def history_key(scene):
    value = copy.deepcopy(scene)
    value.pop("camera", None)
    value.pop("name", None)
    return identity(value)


def reproject(
    position, features, scene, previous_position, previous_features, previous_scene, previous_rgb
):
    if features.shape[0] >= 27:
        indices, weights, mask = reprojection_map(
            position, features, scene, previous_position, previous_features, previous_scene
        )
        history = np.sum(previous_rgb.reshape(-1, 3)[indices] * weights[..., None], axis=0)
        return history.astype(np.float32), mask[..., None]
    height, width = position.shape[:2]
    history = np.zeros((height, width, 3), dtype=np.float32)
    mask = np.zeros((height, width, 1), dtype=np.float32)
    if (
        history_key(scene) != history_key(previous_scene)
        or previous_position.shape != position.shape
    ):
        return history, mask
    camera = previous_scene["camera"]
    current = scene["camera"]
    origin = np.asarray(camera["lookfrom"], dtype=np.float64)
    if np.linalg.norm(origin - np.asarray(current["lookfrom"])) > 3:
        return history, mask
    w = origin - np.asarray(camera["lookat"])
    w /= np.linalg.norm(w)
    u = np.cross(np.asarray(camera.get("up", [0, 1, 0])), w)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    delta = position - origin
    distance = -(delta @ w)
    tangent = np.tan(np.deg2rad(camera["vfov"]) / 2)
    safe = np.maximum(distance, 1e-8)
    px = ((delta @ u) / (safe * tangent * (width / height)) + 1) * width / 2 - 0.5
    py = (-(delta @ v) / (safe * tangent) + 1) * height / 2 - 0.5
    ix = np.floor(np.clip(px + 0.5, -1, width)).astype(np.int64)
    iy = np.floor(np.clip(py + 0.5, -1, height)).astype(np.int64)
    valid = (
        (distance > 0)
        & (ix >= 0)
        & (ix < width)
        & (iy >= 0)
        & (iy < height)
        & (features[11] >= 0.999999)
    )
    ix = np.clip(ix, 0, width - 1)
    iy = np.clip(iy, 0, height - 1)
    old_position = previous_position[iy, ix]
    normal = features[6:9].transpose(1, 2, 0)
    old_normal = previous_features[6:9].transpose(1, 2, 0)[iy, ix]
    normal_dot = np.sum(normal * old_normal, axis=2) / (
        np.maximum(np.linalg.norm(normal, axis=2) * np.linalg.norm(old_normal, axis=2), 1e-8)
    )
    tolerance = 0.02 * np.maximum(1, np.linalg.norm(delta, axis=2))
    valid &= (
        (previous_features[11, iy, ix] >= 0.999999)
        & (normal_dot > 0.9)
        & (np.linalg.norm(old_position - position, axis=2) < tolerance)
    )
    history[valid] = previous_rgb[iy[valid], ix[valid]]
    mask[valid] = 1
    return history, mask


def append_history(features, history, mask):
    return np.concatenate(
        [features, history.transpose(2, 0, 1), mask.transpose(2, 0, 1)], axis=0
    ).astype(np.float32)


def reprojection_map(
    position, features, scene, previous_position, previous_features, previous_scene
):
    """Four-tap, center-geometry validated interpolation, usable with model gradients.

    Weights are normalized only over compatible visible surfaces. A partial
    footprint has lower confidence rather than borrowing another surface's color.
    Camera/scene changes are determined without consulting reference radiance.
    """
    h, width = position.shape[:2]
    indices = np.zeros((4, h, width), dtype=np.int64)
    weights = np.zeros((4, h, width), dtype=np.float32)
    confidence = np.zeros((h, width), dtype=np.float32)
    if (
        features.shape[0] < 27
        or previous_features.shape[0] < 27
        or previous_position.shape != position.shape
        or features.shape[1:] != (h, width)
        or previous_features.shape[1:] != (h, width)
        or history_key(scene) != history_key(previous_scene)
    ):
        return indices, weights, confidence
    camera, current = previous_scene["camera"], scene["camera"]
    origin = np.asarray(camera["lookfrom"], dtype=np.float64)
    now = np.asarray(current["lookfrom"], dtype=np.float64)
    w = origin - np.asarray(camera["lookat"])
    forward = now - np.asarray(current["lookat"])
    distance_to_target = np.linalg.norm(w)
    if distance_to_target < 1e-8 or np.linalg.norm(forward) < 1e-8:
        return indices, weights, confidence
    w /= distance_to_target
    forward /= np.linalg.norm(forward)
    if (
        np.linalg.norm(now - origin) > 0.35 * distance_to_target
        or w @ forward < np.cos(np.deg2rad(20))
        or abs(np.log(current["vfov"] / camera["vfov"])) > 0.18
    ):
        return indices, weights, confidence
    u = np.cross(np.asarray(camera.get("up", [0, 1, 0])), w)
    u /= max(1e-8, np.linalg.norm(u))
    now_u = np.cross(np.asarray(current.get("up", [0, 1, 0])), forward)
    now_u /= max(1e-8, np.linalg.norm(now_u))
    if u @ now_u < np.cos(np.deg2rad(20)):
        return indices, weights, confidence
    v = np.cross(w, u)
    delta = position.astype(np.float64) - origin
    depth = -(delta @ w)
    tangent = np.tan(np.deg2rad(camera["vfov"]) / 2)
    px = ((delta @ u) / (np.maximum(depth, 1e-8) * tangent * width / h) + 1) * width / 2 - 0.5
    py = (-(delta @ v) / (np.maximum(depth, 1e-8) * tangent) + 1) * h / 2 - 0.5
    ix = np.floor(np.clip(px, -1, width)).astype(np.int64)
    iy = np.floor(np.clip(py, -1, h)).astype(np.int64)
    fx, fy = px - ix, py - iy
    supported = (
        (features[26] > 0.5)
        & (features[23] > 0)
        & (np.abs(features[10] - features[11]) < 1e-6)
        & (depth > 0)
        & (px >= -0.5)
        & (px < width - 0.5)
        & (py >= -0.5)
        & (py < h - 0.5)
    )
    normal = features[20:23].transpose(1, 2, 0).astype(np.float64)
    albedo = features[17:20].transpose(1, 2, 0)
    tolerance = np.maximum(1e-5, 1.5 * 2 * tangent * depth / h)
    for k, (dx, dy) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        x, y = ix + dx, iy + dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < h)
        x, y = np.clip(x, 0, width - 1), np.clip(y, 0, h - 1)
        indices[k] = y * width + x
        old_normal = previous_features[20:23].transpose(1, 2, 0)[y, x].astype(np.float64)
        cosine = np.sum(normal * old_normal, axis=-1) / np.maximum(
            1e-8, np.linalg.norm(normal, axis=-1) * np.linalg.norm(old_normal, axis=-1)
        )
        compatible = (
            supported
            & inside
            & (previous_features[26, y, x] > 0.5)
            & (np.abs(previous_features[10, y, x] - previous_features[11, y, x]) < 1e-6)
            & (cosine > 0.95)
            & (np.linalg.norm(previous_position[y, x] - position, axis=-1) < tolerance)
            & (
                np.max(np.abs(previous_features[17:20].transpose(1, 2, 0)[y, x] - albedo), axis=-1)
                < 0.1
            )
        )
        weight = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
        weights[k] = np.where(compatible, weight, 0)
    total = weights.sum(axis=0)
    # Less than a quarter of a pixel's footprint is too little reliable evidence.
    valid = total >= 0.25
    weights /= np.maximum(total, 1e-8)[None]
    weights *= valid[None]
    confidence[:] = np.where(valid, total, 0)
    return indices, weights, confidence


def warp_tensor(rgb, indices, weights):
    """Differentiable CHW history warp using the same geometry map as evaluation."""
    import torch

    lookup = torch.from_numpy(indices).to(rgb.device)
    coefficients = torch.from_numpy(weights).to(device=rgb.device, dtype=rgb.dtype)
    return (rgb.reshape(3, -1)[:, lookup] * coefficients[None]).sum(dim=1)
