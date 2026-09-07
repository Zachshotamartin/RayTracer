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
BOUNDARY_CHANNELS = [
    "center_albedo_r",
    "center_albedo_g",
    "center_albedo_b",
    "center_normal_x",
    "center_normal_y",
    "center_normal_z",
    "center_depth",
    "depth_variance",
    "normal_spread",
    "center_support",
]


def channel_names(schema=1, temporal=False):
    if schema not in (1, 2):
        raise ValueError("Unsupported feature schema")
    return (
        CHANNELS
        + (BOUNDARY_CHANNELS if schema == 2 else [])
        + (["history_r", "history_g", "history_b", "history_valid"] if temporal else [])
    )


def model_schema(config):
    return config.get("feature_schema", 2 if config.get("kind") in ("guided", "refine") else 1)


def load_features(directory, schema=1):
    directory = Path(directory)
    meta = json.loads((directory / "features.json").read_text())
    if (
        schema not in (1, 2)
        or meta["schema_version"] not in (1, 2)
        or meta["schema_version"] < schema
    ):
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
    if meta["schema_version"] == 2:
        sampling = read_pfm(directory / "sampling.pfm")
        count, valid = sampling[..., :1], sampling[..., 1:2]
    result = [*arrays, count, valid]
    if schema == 2:
        result += [
            read_pfm(directory / f"{name}.pfm")
            for name in ("center_albedo", "center_normal", "boundary")
        ]
        result += [sampling[..., 2:3]]
    return np.concatenate(result, axis=2).transpose(2, 0, 1).copy()


def validate_features(x):
    if x.ndim != 3 or x.shape[0] not in (17, 27) or not np.isfinite(x).all():
        raise ValueError("Expected finite 17/27xHxW input")
    if np.any(x[:6] < 0) or np.any(x[3:6] > 1.00001) or np.any(np.abs(x[6:9]) > 1.00001):
        raise ValueError("Radiance/albedo/normal outside schema range")
    if np.any(x[9:16] < -1e-7) or np.any(x[10:12] > 1.00001) or np.any(x[11] > x[10] + 1e-6):
        raise ValueError("Invalid depth, masks, or variance")
    count = x[15]
    if (
        np.any(count < 1)
        or np.any(count != np.floor(count))
        or (x.shape[0] == 17 and not np.all(count == count.flat[0]))
    ):
        raise ValueError("Sample count must be positive integers (uniform in schema 1)")
    if not np.array_equal(x[16], (count > 1).astype(np.float32)):
        raise ValueError("Invalid variance availability mask")
    if np.any(x[12:15, count == 1] != 0):
        raise ValueError("Single-sample variance must be unavailable and zero-filled")
    if x.shape[0] == 27:
        if (
            np.any(x[17:20] < 0)
            or np.any(x[17:20] > 1.00001)
            or np.any(np.abs(x[20:23]) > 1.00001)
            or np.any(x[23:] < -1e-6)
            or np.any(x[25:] > 1.00001)
        ):
            raise ValueError("Invalid boundary feature range")
