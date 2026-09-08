"""Validate artifact integrity, renderer pairing, and group leakage before training."""

import json
from collections import Counter
from pathlib import Path
import numpy as np
from ..io import manifest, safe_path, digest, identity
from ..preprocessing import validate_features
from ..contracts import EXAMPLE_VALIDATOR
from .arrays import load_example
from .reference_checks import validate_reference_checks


def validate(root):
    root = Path(root)
    rows = manifest(root / "manifest.jsonl")
    info = json.loads((root / "dataset.json").read_text())
    if not rows:
        raise ValueError("Empty dataset")
    ids, groups, configurations, checked_refs = set(), {}, {}, {}
    parent_splits, coverage = {}, {s: Counter() for s in ("train", "val", "test")}
    scene_splits = {}
    shared_checked = set()
    counts = {"train": 0, "val": 0, "test": 0}
    for r in rows:
        error = next(EXAMPLE_VALIDATOR.iter_errors(r), None)
        if error:
            raise ValueError(f"Example schema: {error.message}")
        if r["schema_version"] != 1 or r["id"] in ids or r["split"] not in counts:
            raise ValueError("Invalid schema, duplicate example, or split")
        ids.add(r["id"])
        counts[r["split"]] += 1
        if "parent_configuration" in r:
            key = r["parent_configuration"]
            if key in parent_splits and parent_splits[key] != r["split"]:
                raise ValueError("Resolution/camera variants leak across splits")
            parent_splits[key] = r["split"]
        if r["scale"] != info["config"].get("scale", 1):
            raise ValueError("Manifest scale differs from dataset configuration")
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
        if "shared_guides" in r:
            shared = safe_path(root, r["shared_guides"])
            key = (shared, r["shared_guides_sha256"])
            if key not in shared_checked:
                if digest(shared) != r["shared_guides_sha256"]:
                    raise ValueError("Shared guide checksum mismatch")
                shared_checked.add(key)
        data = load_example(root, r)
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
        if (r["stats"].get("height"), r["stats"].get("width")) != shape:
            raise ValueError("Native input dimensions disagree with renderer metadata")
        target_shape = (shape[0] * r["scale"], shape[1] * r["scale"], 3)
        if (r["reference_stats"].get("height"), r["reference_stats"].get("width")) != target_shape[
            :2
        ]:
            raise ValueError("Native reference dimensions disagree with renderer metadata")
        coverage[r["split"]][f"{shape[1]}x{shape[0]}->{target_shape[1]}x{target_shape[0]}"] += 1
        if reference not in checked_refs:
            with np.load(reference, allow_pickle=False) as data:
                target = data["target"]
                if (
                    target.shape != target_shape
                    or not np.isfinite(target).all()
                    or np.any(target < 0)
                ):
                    raise ValueError("Invalid reference shape/radiance")
            checked_refs[reference] = target.shape
        elif checked_refs[reference] != target_shape:
            raise ValueError("Shared reference is incompatible with input dimensions")
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
        "resolution_coverage": {s: dict(c) for s, c in coverage.items()},
        "reference_checks": validate_reference_checks(root, rows, info),
        "manifest_sha256": digest(root / "manifest.jsonl"),
    }
