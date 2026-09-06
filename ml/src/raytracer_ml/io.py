"""Strict float image IO and atomic, content-addressed experiment records."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    return value


def git_revision():
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def read_pfm(path):
    with Path(path).open("rb") as f:
        if f.readline().strip() != b"PF":
            raise ValueError("Expected RGB PFM")
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        if not (0 < width * height <= 16777216 and width > 0 and height > 0) or scale == 0:
            raise ValueError("Invalid PFM dimensions/scale")
        data = np.frombuffer(f.read(), dtype="<f4" if scale < 0 else ">f4")
        if data.size != width * height * 3:
            raise ValueError("Truncated or oversized PFM")
        image = data.reshape(height, width, 3)[::-1].astype(np.float32) * abs(scale)
        if not np.isfinite(image).all():
            raise ValueError("Non-finite PFM")
        return image.copy()


def write_pfm(path, value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError("PFM requires finite HxWx3")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"PF\n{value.shape[1]} {value.shape[0]}\n-1.0\n".encode())
        f.write(value[::-1].astype("<f4").tobytes())


def display(value, exposure=0):
    x = np.maximum(np.asarray(value, dtype=np.float64) * 2**exposure, 0)
    mapped = np.clip(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    return np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * mapped ** (1 / 2.4) - 0.055)


def write_png(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(display(value) * 255).astype("uint8")).save(path)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def manifest(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path escapes the dataset")
    return path
