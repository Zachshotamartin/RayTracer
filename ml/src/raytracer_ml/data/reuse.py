"""Build a small manifest over existing rendered arrays. No renderer is called.

Payloads are hard-linked on the same SSD, not copied or regenerated. Source
receipts are snapshotted; the loader derives aligned pairs at read time.
"""

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path

from filelock import FileLock

from ..io import digest, identity, manifest, safe_path, write_json


def derived_row(source, namespace):
    r = deepcopy(source)
    factor = 2 if r["scale"] == 1 else 1
    if r.get("feature_schema") != 2 or r["scale"] not in (1, 2):
        raise ValueError("Reuse requires spatial schema-2 paired renders")
    h, w = r["stats"]["height"], r["stats"]["width"]
    if h % factor or w % factor:
        raise ValueError("Reuse source dimensions must divide exactly")
    h, w = h // factor, w // factor
    spec = {
        "source": namespace,
        "source_id": r["id"],
        "factor": factor,
        "top": 0,
        "left": 0,
        "height": h,
        "width": w,
        "turns": 0,
    }
    if factor == 2:
        # One deterministic view per saved camera: no inflation of the example
        # count by cloning crops. Training draws additional crops every epoch.
        variant = int(identity(r["configuration"])[:8], 16) % 4
        if variant == 1:
            spec["width"] = min(h, w)
            spec["left"] = (w - spec["width"]) // 2
        elif variant == 2:
            spec["turns"] = 1
        elif variant == 3:
            spec["height"] = max(16, h * 2 // 3)
            spec["top"] = (h - spec["height"]) // 2
        r["samples"] *= 4
        r["cohort"] = "synthetic-area2"
        r["scale"] = 2
        oh, ow = spec["height"], spec["width"]
        if spec["turns"] % 2:
            oh, ow = ow, oh
        r["stats"].update(
            height=oh,
            width=ow,
            samples=r["samples"],
            render_seconds=None,
            pipeline_seconds=None,
            process_seconds=None,
            measurement="derived-from-existing-pixels",
        )
        r["reference_stats"].update(height=oh * 2, width=ow * 2)
    else:
        r["cohort"] = "native-2x"
    r["reuse"] = spec
    # Keep the original geometry group: all camera/crop variants inherit its
    # original split, even if the same data is submitted under another name.
    for key in ("id", "configuration", "parent_configuration"):
        if key in r:
            r[key] = namespace + "-" + r[key]
    for key in ("path", "reference", "shared_guides"):
        if key in r:
            r[key] = f"sources/{namespace}/{r[key]}"
    return r


def build_reuse(sources, output):
    output = Path(output).resolve()
    sources = [Path(p).resolve() for p in sources]
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("Provide distinct existing datasets")
    if any(output == p or p in output.parents or output in p.parents for p in sources):
        raise ValueError("Reuse output must be separate from source datasets")
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / ".reuse.lock", timeout=0):
        if (output / "dataset.json").exists():
            raise ValueError("Reuse dataset already built; use its existing manifest")
        rows, receipts, linked, skipped = [], [], set(), []

        def link(root, relative, dest):
            src = safe_path(root, relative)
            dest = safe_path(output, dest)
            if str(dest) in linked:
                return
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if not os.path.samefile(src, dest):
                    raise ValueError("Existing reuse artifact is not the original hard link")
            else:
                # Intentionally no copy fallback: never duplicate a large image
                # collection because the destination is on a different volume.
                os.link(src, dest)
            linked.add(str(dest))

        for index, root in enumerate(sources):
            namespace = f"s{index}"
            info = json.loads((root / "dataset.json").read_text())
            if info.get("reuse"):
                raise ValueError("Nested derived datasets are not supported")
            original = manifest(root / "manifest.jsonl")
            views = {}
            for row in original:
                views.setdefault(row["configuration"], []).append(row)
            expected = len(info["config"]["budgets"]) * info["config"]["noise_realizations"]
            selected = []
            for view, candidates in views.items():
                if len(candidates) != expected:
                    skipped.append(
                        {
                            "source": namespace,
                            "view": view,
                            "reason": "interrupted incomplete view",
                            "rows": len(candidates),
                        }
                    )
                    continue
                selected.extend(candidates)
            if not selected:
                raise ValueError("Source has no complete saved views")
            snapshot = output / "sources" / namespace
            snapshot.mkdir(parents=True, exist_ok=True)
            write_json(snapshot / "dataset.json", info)
            (snapshot / "manifest.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in selected)
            )
            for source_row in selected:
                row = derived_row(source_row, namespace)
                for key in ("path", "reference", "shared_guides"):
                    if key in source_row:
                        link(root, source_row[key], row[key])
                rows.append(row)
            references = {r["reference"] for r in selected}
            checks = 0
            for p in sorted((root / "reference_checks").glob("*.json")):
                check = json.loads(p.read_text())
                if check.get("schema_version") == 2 and check["reference"] in references:
                    link(root, check["path"], f"sources/{namespace}/{check['path']}")
                    write_json(snapshot / "reference_checks" / p.name, check)
                    checks += 1
            receipts.append(
                {
                    "namespace": namespace,
                    "source": str(root),
                    "source_manifest_sha256": digest(root / "manifest.jsonl"),
                    "snapshot_manifest_sha256": digest(snapshot / "manifest.jsonl"),
                    "dataset_sha256": digest(snapshot / "dataset.json"),
                    "examples": len(selected),
                    "reference_checks": checks,
                }
            )
            print(
                json.dumps(
                    {
                        "phase": "reusing-existing-files",
                        "source": str(root),
                        "examples": len(selected),
                        "new_rays": 0,
                    }
                ),
                flush=True,
            )
        counts = Counter(r["split"] for r in rows)
        cohorts = Counter(r["cohort"] for r in rows)
        temporary = output / "manifest.jsonl.tmp"
        temporary.write_text("".join(json.dumps(r) + "\n" for r in rows))
        temporary.replace(output / "manifest.jsonl")
        info = {
            "schema_version": 1,
            "config": {"scale": 2, "feature_schema": 2},
            "estimate": {"examples": len(rows)},
            "reuse": {
                "schema_version": 1,
                "sources": receipts,
                "skipped_incomplete_views": skipped,
                "new_rays": 0,
                "new_reference_renders": 0,
                "storage": "hard-linked original arrays; on-demand transforms",
                "synthetic_policy": "2x2 area-reduced noisy measurements; actual contributing path count; stratified variance with conservative one-path estimate; selected existing center probes; untouched or cropped/rotated high-quality targets",
                "native_validation_required": True,
            },
            "splits": dict(counts),
            "cohorts": dict(cohorts),
        }
        write_json(output / "dataset.json", info)
        return {
            "examples": len(rows),
            "splits": dict(counts),
            "cohorts": dict(cohorts),
            "hardlinked_files": len(linked),
            "new_rays": 0,
            "manifest_sha256": digest(output / "manifest.jsonl"),
        }


def validate_reuse(root, rows, info):
    """Bind every derived row to an immutable source receipt and inherited split."""
    from .reference_checks import validate_reference_check

    expected, checks = {}, 0
    for receipt in info["reuse"]["sources"]:
        source = safe_path(root, f"sources/{receipt['namespace']}")
        if (
            digest(source / "manifest.jsonl") != receipt["snapshot_manifest_sha256"]
            or digest(source / "dataset.json") != receipt["dataset_sha256"]
        ):
            raise ValueError("Reuse source receipt changed")
        original = manifest(source / "manifest.jsonl")
        for row in original:
            derived = derived_row(row, receipt["namespace"])
            if derived["id"] in expected:
                raise ValueError("Duplicate reuse source")
            expected[derived["id"]] = identity(derived)
        paths = sorted((source / "reference_checks").glob("*.json"))
        if len(paths) != receipt["reference_checks"]:
            raise ValueError("Missing retained source reference checks")
        for path in paths:
            validate_reference_check(source, json.loads(path.read_text()), original)
            checks += 1
    if len(expected) != len(rows) or any(expected.get(r["id"]) != identity(r) for r in rows):
        raise ValueError("Derived manifest differs from source rows or inherited splits")
    return {
        "retained_images": checks,
        "legacy_scalar_only": 0,
        "scope": "unchanged original reference checks; derived crops retain provenance",
    }
