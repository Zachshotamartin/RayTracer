"""Export measured, sample-aligned features for the unmodified OIDN training toolkit."""

import json
import os
from pathlib import Path

import numpy as np

from .data.arrays import load_example
from .data.validate import validate
from .io import digest, identity, manifest, safe_path, write_json


def sample_names(row):
    # Budgets of a single noise realization share a reference, as OIDN expects.
    # A separate seed gets a separate sample group, without changing our scene split.
    key = identity([row["group"], row["configuration"], row["input_seed"]])
    prefix = f"view-{key}"
    return f"{prefix}_{row['samples']:06d}spp", f"{prefix}_ref"


def write_exr(path, pixels):
    try:
        import OpenImageIO as oiio
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional oidn-training dependencies to export EXRs"
        ) from exc
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or not np.isfinite(pixels).all():
        raise ValueError("Expected finite HxWx3 EXR pixels")
    path = Path(path)
    temporary = path.with_name(path.stem + ".tmp.exr")
    spec = oiio.ImageSpec(pixels.shape[1], pixels.shape[0], 3, oiio.FLOAT)
    spec.attribute("compression", "zip")
    output = oiio.ImageOutput.create(str(temporary))
    if not output or not output.open(str(temporary), spec):
        raise RuntimeError(f"Cannot create EXR: {path}")
    try:
        if not output.write_image(np.ascontiguousarray(pixels)):
            raise RuntimeError(f"Cannot write EXR: {path}")
    finally:
        output.close()
    os.replace(temporary, path)


def export_dataset(root, output, splits=("train", "val")):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "val", "test"}:
        raise ValueError("Choose unique train/val/test splits")
    audit = validate(root)
    rows = [r for r in manifest(root / "manifest.jsonl") if r["split"] in splits]
    if any(r["scale"] != 1 for r in rows):
        raise ValueError("OIDN spatial training needs same-resolution inputs and targets")
    descriptor = {
        "schema_version": 1,
        "source_manifest_sha256": audit["manifest_sha256"],
        "splits": list(splits),
        "features": ["hdr", "alb", "nrm"],
        "clean_aux": False,
        "encoding": "linear-float32-exr",
        "guide_policy": "sample-aligned means; no center/reference guide substitution",
    }
    if output.exists():
        receipt = output / "export.json"
        if not receipt.exists():
            raise ValueError(
                "Output exists without a completed export receipt; use a new directory"
            )
        saved = json.loads(receipt.read_text())
        if saved["descriptor"] != descriptor:
            raise ValueError("Export configuration or dataset changed")
        for relative, checksum in saved["files"].items():
            if digest(safe_path(output, relative)) != checksum:
                raise ValueError("Exported EXR checksum mismatch")
        return saved

    output.mkdir(parents=True)
    files, records, targets, destinations, group_targets = {}, [], {}, set(), {}
    for row in rows:
        split = "valid" if row["split"] == "val" else row["split"]
        directory = output / split
        directory.mkdir(exist_ok=True)
        noisy, target = sample_names(row)
        if (split, noisy) in destinations:
            raise ValueError("Duplicate noise/budget within an OIDN sample group")
        destinations.add((split, noisy))
        group_key = (split, target)
        if group_targets.get(group_key, row["reference_sha256"]) != row["reference_sha256"]:
            raise ValueError("OIDN sample group has inconsistent references")
        group_targets[group_key] = row["reference_sha256"]
        data = load_example(root, row)
        features = data["features"]
        source_target = safe_path(root, row["reference"])
        target_file = directory / f"{target}.hdr.exr"
        target_key = (split, row["reference_sha256"])
        if not target_file.exists():
            if target_key in targets:
                os.link(targets[target_key], target_file)
            else:
                with np.load(source_target, allow_pickle=False) as reference:
                    write_exr(target_file, reference["target"])
                targets[target_key] = target_file
            files[str(target_file.relative_to(output))] = digest(target_file)
        for feature, start in (("hdr", 0), ("alb", 3), ("nrm", 6)):
            path = directory / f"{noisy}.{feature}.exr"
            write_exr(path, features[start : start + 3].transpose(1, 2, 0))
            files[str(path.relative_to(output))] = digest(path)
        records.append(
            {
                "id": row["id"],
                "group": row["group"],
                "split": row["split"],
                "input": f"{split}/{noisy}",
                "target": f"{split}/{target}",
                "samples": row["samples"],
                "input_seed": row["input_seed"],
                "source_sha256": row["sha256"],
                "reference_sha256": row["reference_sha256"],
            }
        )
    receipt = {
        "descriptor": descriptor,
        "examples": len(records),
        "records": records,
        "files": files,
        "unique_references": len(targets),
    }
    write_json(output / "export.json", receipt)
    return receipt
