"""Bounded, resumable paired rendering. A validated example is the commit unit."""

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from filelock import FileLock

from ..io import digest, identity, read_pfm, save_arrays, write_json, git_revision
from ..preprocessing import load_features, validate_features
from .scenes import configurations


def sampling_seed(*parts):
    return int(identity(parts)[:15], 16)


def renderer(binary, scene, width, samples, seed, directory, cfg, remaining, features=False):
    directory.mkdir(parents=True, exist_ok=True)
    scene_path = directory / "scene.json"
    write_json(scene_path, scene)
    command = [
        str(binary),
        "--headless",
        "--quiet",
        "--scene-file",
        str(scene_path),
        "--width",
        str(width),
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--depth",
        str(cfg["depth"]),
        "--threads",
        str(cfg["threads"]),
        "--output",
        str(directory / "image.pfm"),
    ]
    if features:
        command += ["--features", str(directory / "features")]
    start = time.perf_counter()
    run = subprocess.run(command, capture_output=True, text=True, timeout=max(0.1, remaining))
    elapsed = time.perf_counter() - start
    if run.returncode:
        raise RuntimeError(f"Renderer failed ({run.returncode}): {run.stderr[-4000:]}")
    stats = json.loads(run.stdout)
    if stats["samples"] != samples:
        raise RuntimeError("Renderer returned an incomplete image")
    stats["process_seconds"] = elapsed
    return read_pfm(directory / "image.pfm"), stats


def generate(cfg, binary, root, dry_run=False):
    root, binary = Path(root).resolve(), Path(binary).resolve()
    if not binary.is_file():
        raise ValueError("Renderer binary does not exist")
    budgets = cfg["budgets"]
    if (
        not budgets
        or budgets != sorted(set(budgets))
        or any(type(n) is not int or n < 1 or n > 100000 for n in budgets)
    ):
        raise ValueError("Budgets must be sorted distinct positive integers")
    for name in ["width", "noise_realizations", "reference_samples", "depth", "threads"]:
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"Invalid {name}")
    if cfg.get("schema_version") != 1 or cfg.get("scale", 1) not in (1, 2):
        raise ValueError("Unsupported data schema or scale")
    if cfg["reference_samples"] <= max(budgets) or cfg["max_seconds"] <= 0 or cfg["max_gib"] <= 0:
        raise ValueError("References need more samples than inputs and positive resource caps")
    if cfg["width"] % 16:
        raise ValueError("Dataset width must be a multiple of 16 for aligned 16:9 pairs")
    items = list(configurations(cfg))
    feature_schema = int(cfg.get("feature_schema", 1))
    if feature_schema not in (1, 2):
        raise ValueError("Unsupported feature schema")
    # External asset paths would make the inline scene hash incomplete. Dataset
    # suites use embedded meshes; ordinary CLI scenes may still load OBJ files.
    if any("path" in o for item in items for o in item["scene"].get("objects", [])):
        raise ValueError("Dataset mesh assets must be embedded in scene JSON")
    check_indices = list(range(min(len(items), cfg.get("reference_checks", 0))))
    if cfg.get("reference_check_strategy") == "stratified":
        chosen = {}
        for i, item in enumerate(items):
            chosen.setdefault((item["split"], item.get("stratum", "room")), i)
        check_indices = list(chosen.values())
    count = len(items) * len(budgets) * cfg["noise_realizations"]
    estimate = {
        "configurations": len(items),
        "examples": count,
        "reference_images": len(items),
        "uncompressed_input_gib": count
        * cfg["width"]
        * int(cfg["width"] * 9 / 16)
        * (30 if feature_schema == 2 else 20)
        * 4
        / 2**30,
        "max_seconds": cfg["max_seconds"],
        "max_gib": cfg["max_gib"],
    }
    if dry_run:
        return estimate
    if cfg.get("require_clean_source", False):
        status = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if status.stdout.strip():
            raise ValueError("Commit source changes before generating this reproducible dataset")
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".generation.lock"), timeout=0):
        contract = {"schema_version": 1, "config": cfg, "renderer_sha256": digest(binary)}
        fingerprint = identity(contract)
        info = root / "dataset.json"
        if info.exists() and json.loads(info.read_text())["fingerprint"] != fingerprint:
            raise ValueError(
                "Dataset configuration/renderer changed; choose a new artifact directory"
            )
        if not info.exists():
            write_json(
                info,
                {
                    **contract,
                    "fingerprint": fingerprint,
                    "renderer_commit": git_revision(),
                    "estimate": estimate,
                    "created_unix": time.time(),
                },
            )
        deadline = time.monotonic() + cfg["max_seconds"]
        records = []
        persisted = 0
        disk_checks = 0
        # Individual example JSON/checksums are the resumable commit units.
        # Rebuild the append-only manifest from those records instead of rewriting
        # all previous rows after each example (quadratic IO on large datasets).
        (root / "manifest.jsonl").write_text("")

        def guard():
            nonlocal disk_checks
            if time.monotonic() >= deadline:
                raise TimeoutError("Dataset generation time cap reached; rerun to resume")
            disk_checks += 1
            if disk_checks % 16 != 1:
                return
            used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
            reserve = 16 * cfg["width"] ** 2 * 4 * 64 * cfg.get("scale", 1) ** 2
            if (
                used + reserve > cfg["max_gib"] * 2**30
                or shutil.disk_usage(root).free < reserve * 2
            ):
                raise RuntimeError("Dataset disk cap/free-space reserve reached")

        def persist():
            nonlocal persisted
            with (root / "manifest.jsonl").open("a") as stream:
                for record in records[persisted:]:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
            persisted = len(records)

        for index, item in enumerate(items):
            guard()
            ident = item["id"]
            reference = root / "references" / f"{ident}.npz"
            reference_info = reference.with_suffix(".json")
            ref_seed = sampling_seed(cfg["seed"], ident, "target")
            with tempfile.TemporaryDirectory(prefix=".render-", dir=root) as temporary:
                work = Path(temporary)
                if (
                    not reference.exists()
                    or not reference_info.exists()
                    or json.loads(reference_info.read_text())["sha256"] != digest(reference)
                ):
                    target, stats = renderer(
                        binary,
                        item["scene"],
                        cfg["width"] * cfg.get("scale", 1),
                        cfg["reference_samples"],
                        ref_seed,
                        work / "target",
                        cfg,
                        deadline - time.monotonic(),
                        feature_schema == 2,
                    )
                    extra = {}
                    if feature_schema == 2:
                        extra["features"] = load_features(work / "target/features", 2)
                        extra["position"] = read_pfm(work / "target/features/position.pfm")
                        extra["annotations"] = read_pfm(work / "target/features/annotations.pfm")
                    save_arrays(reference, target=target, **extra)
                    write_json(
                        reference_info,
                        {"sha256": digest(reference), "seed": ref_seed, "stats": stats},
                    )
                ref_record = json.loads(reference_info.read_text())
                if index in check_indices:
                    checked = root / "reference_checks" / f"{ident}.json"
                    if not checked.exists():
                        check_samples = int(
                            cfg.get("reference_check_samples", 4 * cfg["reference_samples"])
                        )
                        if check_samples <= cfg["reference_samples"]:
                            raise ValueError("Reference check needs a higher sample budget")
                        alternate, stats = renderer(
                            binary,
                            item["scene"],
                            cfg["width"] * cfg.get("scale", 1),
                            check_samples,
                            sampling_seed(cfg["seed"], ident, "reference-check"),
                            work / "check",
                            cfg,
                            deadline - time.monotonic(),
                        )
                        with np.load(reference) as data:
                            target = data["target"]
                        write_json(
                            checked,
                            {
                                "samples": check_samples,
                                "independent_seed": True,
                                "linear_mse": float(np.mean((target - alternate) ** 2)),
                                "log_mse": float(
                                    np.mean((np.log1p(target) - np.log1p(alternate)) ** 2)
                                ),
                                "stats": stats,
                            },
                        )
                for realization in range(cfg["noise_realizations"]):
                    seed = sampling_seed(cfg["seed"], ident, "input", realization)
                    for samples in budgets:
                        guard()
                        example_id = f"{ident}-n{realization}-s{samples}"
                        output = root / "examples" / f"{example_id}.npz"
                        record_path = output.with_suffix(".json")
                        if output.exists() and record_path.exists():
                            record = json.loads(record_path.read_text())
                            if (
                                record["sha256"] == digest(output)
                                and record["reference_sha256"] == ref_record["sha256"]
                            ):
                                records.append(record)
                                continue
                        run_dir = work / "input"
                        if run_dir.exists():
                            shutil.rmtree(run_dir)
                        try:
                            raw, stats = renderer(
                                binary,
                                item["scene"],
                                cfg["width"],
                                samples,
                                seed,
                                run_dir,
                                cfg,
                                deadline - time.monotonic(),
                                True,
                            )
                            features = load_features(run_dir / "features", feature_schema)
                            validate_features(features)
                            if not np.array_equal(features[:3].transpose(1, 2, 0), raw):
                                raise ValueError("Feature RGB differs from raw render")
                            save_arrays(
                                output,
                                features=features,
                                atrous=read_pfm(run_dir / "features/atrous.pfm"),
                                position=read_pfm(run_dir / "features/position.pfm"),
                            )
                            record = {
                                "schema_version": 1,
                                "feature_schema": feature_schema,
                                "id": example_id,
                                "configuration": ident,
                                "group": item["group"],
                                "split": item["split"],
                                "cohort": item["cohort"],
                                "frame": item["frame"],
                                "stratum": item.get("stratum", "room"),
                                "geometry_sha256": identity(item["scene"].get("objects", [])),
                                "scene": item["scene"],
                                "scene_sha256": identity(item["scene"]),
                                "path": str(output.relative_to(root)),
                                "sha256": digest(output),
                                "reference": str(reference.relative_to(root)),
                                "reference_sha256": ref_record["sha256"],
                                "input_seed": seed,
                                "target_seed": ref_seed,
                                "samples": samples,
                                "reference_samples": cfg["reference_samples"],
                                "scale": cfg.get("scale", 1),
                                "stats": stats,
                                "reference_stats": ref_record["stats"],
                            }
                            write_json(record_path, record)
                            records.append(record)
                            persist()
                            print(f"{len(records)}/{count} {example_id}", flush=True)
                        except Exception as error:
                            with (root / "failures.jsonl").open("a") as log:
                                log.write(
                                    json.dumps(
                                        {"id": example_id, "error": str(error), "time": time.time()}
                                    )
                                    + "\n"
                                )
                            persist()
                            raise
        persist()
        write_json(
            root / "splits.json",
            {
                s: sorted({r["group"] for r in records if r["split"] == s})
                for s in ["train", "val", "test"]
            },
        )
        return {
            **estimate,
            "completed": len(records),
            "manifest_sha256": digest(root / "manifest.jsonl"),
        }
