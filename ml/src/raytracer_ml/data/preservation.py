"""Deterministic measured near-clean validation, independent of training augmentation."""

import numpy as np
import torch
from .arrays import load_example
from ..augment import fuse_measurements
from ..io import manifest, safe_path


def view_key(row):
    return (
        row["group"],
        row["configuration"],
        row["scene_sha256"],
        row["target_seed"],
        row["reference_sha256"],
    )


def measured_pairs(rows, samples):
    views, result = {}, {}
    for row in rows:
        views.setdefault(view_key(row), []).append(row)
    for key, candidates in views.items():
        candidates = sorted(candidates, key=lambda r: r["id"])
        pairs = [
            (a, b)
            for i, a in enumerate(candidates)
            for b in candidates[i + 1 :]
            if a["samples"] + b["samples"] == samples
            and a["input_seed"] != b["input_seed"]
            and a["input_seed"] != a["target_seed"]
            and b["input_seed"] != b["target_seed"]
            and a["reference_samples"] > samples
            and b["reference_samples"] > samples
        ]
        if not pairs:
            raise ValueError("Data lacks independent measurements for near-clean pairs")
        result[key] = pairs
    return result


class PreservationValidation:
    def __init__(self, root, samples=96):
        if type(samples) is not int or not 1 < samples < 128:
            raise ValueError("Preservation validation samples must be below the 128-spp bypass")
        self.root = root
        rows = [r for r in manifest(root / "manifest.jsonl") if r["split"] == "val"]
        if not rows or any(r["scale"] != 1 or r.get("feature_schema", 1) != 2 for r in rows):
            raise ValueError("Preservation validation requires schema-2 same-resolution val data")
        self.pairs = [pairs[0] for _, pairs in sorted(measured_pairs(rows, samples).items())]

    def measure(self, model, device):
        measured, raw = [], []
        with torch.inference_mode():
            for a, b in self.pairs:
                x = fuse_measurements(
                    load_example(self.root, a)["features"], load_example(self.root, b)["features"]
                )
                # Only independent input measurements enter the network. No target injection.
                prediction = model(torch.from_numpy(x[None]).to(device))[0].cpu().numpy()
                with np.load(safe_path(self.root, a["reference"]), allow_pickle=False) as ref:
                    target = np.log1p(ref["target"].astype(np.float64).transpose(2, 0, 1))
                measured.append(
                    float(np.abs(np.log1p(prediction.astype(np.float64)) - target).mean())
                )
                raw.append(float(np.abs(np.log1p(x[:3].astype(np.float64)) - target).mean()))
        return (
            {"preservation_log_mae": float(np.mean(measured)), "preservation_views": len(measured)},
            {"preservation_log_mae": float(np.mean(raw)), "preservation_views": len(raw)},
        )
