"""Validate artifact integrity, renderer pairing, and group leakage before training."""

import json
from pathlib import Path
import numpy as np
from ..io import manifest, safe_path, digest, identity
from ..preprocessing import validate_features
from ..contracts import EXAMPLE_VALIDATOR


def validate(root):
    root = Path(root)
    rows = manifest(root / "manifest.jsonl")
    info = json.loads((root / "dataset.json").read_text())
    if not rows:
        raise ValueError("Empty dataset")
    ids, groups, configurations, checked_refs = set(), {}, {}, set()
    scene_splits = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        error = next(EXAMPLE_VALIDATOR.iter_errors(r), None)
        if error:
            raise ValueError(f"Example schema: {error.message}")
        if r["schema_version"] != 1 or r["id"] in ids or r["split"] not in counts:
            raise ValueError("Invalid schema, duplicate example, or split")
        ids.add(r["id"])
        counts[r["split"]] += 1
        for mapping, key in [(groups, r["group"]), (configurations, r["configuration"])]:
            if key in mapping and mapping[key] != r["split"]:
                raise ValueError("Dataset group/configuration leaks across splits")
            mapping[key] = r["split"]
        if r["input_seed"] == r["target_seed"] or r["reference_samples"] <= r["samples"]:
            raise ValueError("Targets need independent seeds and more samples")
        if r["scene_sha256"] in scene_splits and scene_splits[r["scene_sha256"]] != r["split"]:
            raise ValueError("Identical scene/camera leaks across splits")
        scene_splits[r["scene_sha256"]] = r["split"]
        if identity(r["scene"]) != r["scene_sha256"]:
            raise ValueError("Scene configuration hash mismatch")
        path, reference = safe_path(root, r["path"]), safe_path(root, r["reference"])
        if digest(path) != r["sha256"] or digest(reference) != r["reference_sha256"]:
            raise ValueError("Dataset file checksum mismatch")
        with np.load(path, allow_pickle=False) as data:
            x = data["features"]
            validate_features(x)
            if x.shape[0] != {1: 17, 2: 27}.get(r.get("feature_schema", 1)):
                raise ValueError("Manifest feature schema differs from arrays")
            if x[15, 0, 0] != r["samples"] or data["atrous"].shape != (x.shape[1], x.shape[2], 3):
                raise ValueError("Sample count/baseline dimensions disagree")
            if (
                not np.isfinite(data["atrous"]).all()
                or np.any(data["atrous"] < 0)
                or data["position"].shape != (x.shape[1], x.shape[2], 3)
                or not np.isfinite(data["position"]).all()
            ):
                raise ValueError("Invalid baseline/position")
            shape = x.shape[1:]
        if reference not in checked_refs:
            with np.load(reference, allow_pickle=False) as data:
                target = data["target"]
                if (
                    target.shape != (shape[0] * r["scale"], shape[1] * r["scale"], 3)
                    or not np.isfinite(target).all()
                    or np.any(target < 0)
                ):
                    raise ValueError("Invalid reference shape/radiance")
            checked_refs.add(reference)
    if any(n == 0 for n in counts.values()):
        raise ValueError("All three splits must contain examples")
    expected = info["estimate"]["examples"]
    if len(rows) != expected:
        raise ValueError(f"Incomplete dataset: {len(rows)}/{expected} examples")
    return {
        "examples": len(rows),
        "groups": len(groups),
        "splits": counts,
        "references": len(checked_refs),
        "manifest_sha256": digest(root / "manifest.jsonl"),
    }
