"""Schema 1: C++ packs these same 17 raw channels, transforms live in the model."""

import json
from pathlib import Path
import numpy as np
from .io import read_pfm

CHANNELS = [
    "r",
    "g",
    "b",
    "albedo_r",
    "albedo_g",
    "albedo_b",
    "normal_x",
    "normal_y",
    "normal_z",
    "depth",
    "coverage",
    "support",
    "variance_r",
    "variance_g",
    "variance_b",
    "samples",
    "variance_valid",
]
SCHEMA_VERSION = 1


def load_features(directory):
    directory = Path(directory)
    meta = json.loads((directory / "features.json").read_text())
    if meta["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported feature schema")
    arrays = [
        read_pfm(directory / f"{name}.pfm")
        for name in ["radiance", "albedo", "normal", "geometry", "variance"]
    ]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays) or shape[:2] != (meta["height"], meta["width"]):
        raise ValueError("Feature dimensions disagree")
    count = np.full((*shape[:2], 1), meta["samples"], dtype=np.float32)
    valid = np.full_like(count, meta["samples"] > 1)
    return np.concatenate([*arrays, count, valid], axis=2).transpose(2, 0, 1).copy()


def validate_features(x):
    if x.ndim != 3 or x.shape[0] != len(CHANNELS) or not np.isfinite(x).all():
        raise ValueError("Expected finite 17xHxW input")
    if np.any(x[:6] < 0) or np.any(x[3:6] > 1.00001) or np.any(np.abs(x[6:9]) > 1.00001):
        raise ValueError("Radiance/albedo/normal outside schema range")
    if np.any(x[9:16] < -1e-7) or np.any(x[10:12] > 1.00001) or np.any(x[11] > x[10] + 1e-6):
        raise ValueError("Invalid depth, masks, or variance")
    count = x[15]
    if (
        not np.all(count == count.flat[0])
        or count.flat[0] < 1
        or count.flat[0] != int(count.flat[0])
    ):
        raise ValueError("Sample count must be a positive uniform integer")
    if not np.array_equal(x[16], (count > 1).astype(np.float32)):
        raise ValueError("Invalid variance availability mask")
    if count.flat[0] == 1 and np.any(x[12:15] != 0):
        raise ValueError("Single-sample variance must be unavailable and zero-filled")
