"""Explicit external-cohort registration without weakening training provenance."""

import json
from pathlib import Path
from .data.validate import validate
from .io import digest, identity, manifest, write_json


def register(checkpoint, training_root, evaluation_root, output):
    from .train import load_checkpoint

    state = load_checkpoint(checkpoint)
    training, evaluation = validate(training_root), validate(evaluation_root)
    if state["manifest_sha256"] != training["manifest_sha256"]:
        raise ValueError("Training manifest does not match checkpoint")
    training_rows = [
        r for r in manifest(Path(training_root) / "manifest.jsonl") if r["split"] == "train"
    ]
    external_rows = manifest(Path(evaluation_root) / "manifest.jsonl")
    geometries = {identity(r["scene"].get("objects", [])) for r in training_rows}
    overlaps = [
        r["id"] for r in external_rows if identity(r["scene"].get("objects", [])) in geometries
    ]
    if overlaps:
        raise ValueError("External unseen-geometry cohort overlaps training geometry")
    result = {
        "schema_version": 1,
        "checkpoint_sha256": digest(checkpoint),
        "training_manifest_sha256": training["manifest_sha256"],
        "evaluation_manifest_sha256": evaluation["manifest_sha256"],
        "overlapping_geometry": 0,
        "cohort": "external-unseen-geometry",
        "note": "Different geometry hashes do not establish a different scene family.",
    }
    write_json(output, result)
    return result


def authorize(state, checkpoint, evaluation, registration=None):
    return authorize_digests(state["manifest_sha256"], digest(checkpoint), evaluation, registration)


def authorize_digests(training_manifest, checkpoint_sha256, evaluation, registration=None):
    """Apply the same provenance contract to PyTorch and exported native models."""
    if training_manifest == evaluation["manifest_sha256"]:
        return "original-dataset"
    if registration is None:
        raise ValueError("Checkpoint dataset differs; register an external evaluation first")
    record = json.loads(Path(registration).read_text())
    required = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "training_manifest_sha256": training_manifest,
        "evaluation_manifest_sha256": evaluation["manifest_sha256"],
        "overlapping_geometry": 0,
        "cohort": "external-unseen-geometry",
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("Invalid or stale external evaluation registration")
    return record["cohort"]
