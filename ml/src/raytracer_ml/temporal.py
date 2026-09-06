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
