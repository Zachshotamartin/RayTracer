"""Bounded train/validation controls with retained, independent reference images.

A small diagnostic cohort is separate from the immutable source dataset. It is
never a release benchmark, and test groups cannot enter its selector or fitter.
"""

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .data.generate import renderer, sampling_seed
from .data.reference_checks import reference_errors
from .diagnostics import reconstruction_region, reference_regions
from .io import (
    digest,
    git_revision,
    identity,
    manifest,
    read_pfm,
    safe_path,
    save_arrays,
    write_json,
)
from .learning_health import LearningHealth
from .losses import reconstruction_loss, validate_loss_config
from .models import build_model
from .preprocessing import load_features, validate_features
from .selection import quality_scores
from .train import device_for, save_checkpoint, synchronize


def sources():
    package = Path(__file__).parent
    # Fingerprint the complete Python implementation, not just the experiment driver.
    return {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob("*.py"))}


def select_views(rows):
    selected = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        if row["split"] not in ("train", "val"):
            continue
        if row.get("feature_schema") != 2 or row.get("scale") != 1:
            raise ValueError("Cohort requires spatial schema-2 source views")
        key = row["split"], row.get("stratum", "scene")
        selected.setdefault(key, row)
    if {key[0] for key in selected} != {"train", "val"}:
        raise ValueError("Cohort requires both training and validation strata")
    chosen = list(selected.values())
    if len({r["group"] for r in chosen}) != len(chosen):
        raise ValueError("Cohort strata must use distinct scene groups")
    return chosen


def render_cohort(root, output, binary, cfg):
    root, output, binary = Path(root).resolve(), Path(output).resolve(), Path(binary).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Use a new cohort directory outside the source dataset")
    for key in ("width", "threads", "reference_samples", "check_samples", "noise_realizations"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"Invalid cohort {key}")
    if cfg["width"] % 16 or cfg["max_seconds"] <= 0 or cfg["max_gib"] <= 0:
        raise ValueError("Cohort needs aligned dimensions and positive resource caps")
    if not cfg["budgets"] or not all(
        type(s) is int and 0 < s < cfg["reference_samples"] for s in cfg["budgets"]
    ):
        raise ValueError("Invalid cohort sample budgets")
    if cfg["check_samples"] <= cfg["reference_samples"]:
        raise ValueError("Independent check must have a larger sample budget")
    info = json.loads((root / "dataset.json").read_text())
    if digest(binary) != info["renderer_sha256"]:
        raise ValueError("Cohort renderer must match the original dataset renderer")
    chosen = select_views(manifest(root / "manifest.jsonl"))
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + cfg["max_seconds"]
    record = dict(
        schema_version=1,
        state="running",
        config=cfg,
        source_manifest_sha256=digest(root / "manifest.jsonl"),
        renderer_sha256=digest(binary),
        source_hashes=sources(),
        code_commit=git_revision(),
        selection="First configuration per train/val stratum in sorted source ids; no test access",
        selected=[
            dict(
                id=r["configuration"],
                split=r["split"],
                group=r["group"],
                stratum=r.get("stratum"),
                scene_sha256=r["scene_sha256"],
            )
            for r in chosen
        ],
        views=[],
        examples=[],
    )
    receipt = output / "cohort.json"

    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError("Cohort wall-time cap reached")
        used = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        if used >= cfg["max_gib"] * 2**30 or shutil.disk_usage(output).free < 8 * 2**30:
            raise RuntimeError("Cohort storage/free-space cap reached")

    def persist():
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(receipt, record)

    def archive(relative, **arrays):
        guard()
        path = output / relative
        save_arrays(path, **arrays)
        return dict(path=relative, sha256=digest(path))

    persist()
    try:
        for row in chosen:
            if identity(row["scene"]) != row["scene_sha256"]:
                raise ValueError("Source scene hash mismatch")
            ident = row["configuration"]
            view = dict(
                id=ident,
                split=row["split"],
                group=row["group"],
                stratum=row.get("stratum"),
                scene=row["scene"],
                scene_sha256=row["scene_sha256"],
            )
            rcfg = dict(depth=info["config"]["depth"], threads=cfg["threads"])
            with tempfile.TemporaryDirectory(prefix=".render-", dir=output) as temp:
                work = Path(temp)
                used_seeds = {row["target_seed"], row["input_seed"]}
                for name, samples in (
                    ("reference", cfg["reference_samples"]),
                    ("check", cfg["check_samples"]),
                ):
                    guard()
                    seed = sampling_seed(cfg["seed"], ident, "research-cohort", name)
                    if seed in used_seeds:
                        raise ValueError("Reference seed collision")
                    used_seeds.add(seed)
                    image, stats = renderer(
                        binary,
                        row["scene"],
                        cfg["width"],
                        samples,
                        seed,
                        work / name,
                        rcfg,
                        deadline - time.monotonic(),
                        name == "reference",
                    )
                    arrays = dict(target=image)
                    if name == "reference":
                        arrays["features"] = load_features(work / name / "features", 2)
                        target, guides = image, arrays["features"]
                    view[name] = dict(
                        **archive(f"{name}/{ident}.npz", **arrays),
                        seed=seed,
                        samples=samples,
                        stats=stats,
                    )
                view["reference_disagreement"] = reference_errors(target, image, guides)
                record["views"].append(view)
                persist()
                for noise in range(cfg["noise_realizations"]):
                    for samples in cfg["budgets"]:
                        guard()
                        seed = sampling_seed(
                            cfg["seed"], ident, "research-cohort", "input", noise, samples
                        )
                        if seed in used_seeds:
                            raise ValueError("Input seed collision")
                        used_seeds.add(seed)
                        raw, stats = renderer(
                            binary,
                            row["scene"],
                            cfg["width"],
                            samples,
                            seed,
                            work / "input",
                            rcfg,
                            deadline - time.monotonic(),
                            True,
                        )
                        features = load_features(work / "input/features", 2)
                        validate_features(features)
                        if not np.array_equal(features[:3].transpose(1, 2, 0), raw):
                            raise ValueError("Cohort feature RGB mismatch")
                        example_id = f"{ident}-n{noise}-s{samples}"
                        record["examples"].append(
                            dict(
                                id=example_id,
                                view=ident,
                                split=row["split"],
                                group=row["group"],
                                noise=noise,
                                samples=samples,
                                seed=seed,
                                **archive(
                                    f"examples/{example_id}.npz",
                                    features=features,
                                    atrous=read_pfm(work / "input/features/atrous.pfm"),
                                ),
                                stats=stats,
                            )
                        )
                        persist()
            print(
                json.dumps(
                    dict(
                        view=ident,
                        split=row["split"],
                        state="complete",
                        elapsed_seconds=time.monotonic() - started,
                    )
                ),
                flush=True,
            )
        record["state"] = "completed"
    except Exception as error:
        record.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()
    return record


def load_cohort(root):
    root = Path(root)
    record = json.loads((root / "cohort.json").read_text())
    if record["state"] != "completed":
        raise ValueError("Cohort is incomplete")
    views = {}

    def arrays(item):
        path = safe_path(root, item["path"])
        if digest(path) != item["sha256"]:
            raise ValueError("Cohort checksum mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            return {k: loaded[k] for k in loaded.files}

    groups = {}
    for row in record["views"]:
        if row["split"] not in ("train", "val"):
            raise ValueError("Only training and validation groups are permitted")
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError("Cohort split leakage")
        groups[row["group"]] = row["split"]
        a, b = row["reference"], row["check"]
        if a["seed"] == b["seed"] or b["samples"] <= a["samples"]:
            raise ValueError("Invalid independent reference pairing")
        primary, check = arrays(a), arrays(b)["target"]
        validate_features(primary["features"])
        if not np.isfinite(primary["target"]).all() or np.any(primary["target"] < 0):
            raise ValueError("Invalid primary reference radiance")
        if (
            primary["target"].shape != check.shape
            or not np.isfinite(check).all()
            or np.any(check < 0)
        ):
            raise ValueError("Invalid reference check radiance")
        if (
            reference_errors(primary["target"], check, primary["features"])
            != row["reference_disagreement"]
        ):
            raise ValueError("Reference disagreement receipt mismatch")
        if row["id"] in views or identity(row["scene"]) != row["scene_sha256"]:
            raise ValueError("Duplicate view or scene identity mismatch")
        views[row["id"]] = dict(row=row, **primary, check=check)
    examples = []
    seen = set()
    for row in record["examples"]:
        view = views[row["view"]]
        if (row["split"], row["group"]) != (view["row"]["split"], view["row"]["group"]):
            raise ValueError("Example/reference split mismatch")
        if row["seed"] in (view["row"]["reference"]["seed"], view["row"]["check"]["seed"]):
            raise ValueError("Input/reference seed overlap")
        item = arrays(row)
        validate_features(item["features"])
        key = row["view"], row["samples"], row["noise"]
        if (
            key in seen
            or row["samples"] not in record["config"]["budgets"]
            or not 0 <= row["noise"] < record["config"]["noise_realizations"]
            or not np.all(item["features"][15] == row["samples"])
        ):
            raise ValueError("Duplicate example or invalid sample coverage")
        seen.add(key)
        if item["features"].shape[1:] != view["target"].shape[:2]:
            raise ValueError("Cohort input/target dimensions disagree")
        examples.append(dict(row=row, **item, reference=view))
    if (
        len(examples)
        != len(views) * len(record["config"]["budgets"]) * record["config"]["noise_realizations"]
    ):
        raise ValueError("Cohort example coverage mismatch")
    return record, examples


def score_regions(prediction, target, masks):
    error = np.mean((prediction.astype(np.float64) - target) ** 2, axis=-1)
    return {
        name: dict(
            pixels=int(mask.sum()), linear_mse=float(error[mask].mean()) if mask.any() else None
        )
        for name, mask in masks.items()
    }


def compare_cohort(root, output, cfg):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("Use a new comparison directory")
    for key in ("steps", "crop", "batch_size", "max_seconds"):
        if cfg[key] <= 0:
            raise ValueError("Comparison budgets must be positive")
    for trial in cfg["trials"]:
        validate_loss_config(trial["loss"])
        model = build_model(trial["model"])
        if model.scale != 1 or model.temporal or model.feature_schema != 2:
            raise ValueError("These bounded comparisons require spatial schema 2 at scale 1")
    cohort, examples = load_cohort(root)
    train = [r for r in examples if r["row"]["split"] == "train"]
    val = [r for r in examples if r["row"]["split"] == "val"]
    if not train or not val:
        raise ValueError("Missing training or validation examples")
    crop = cfg["crop"]
    if any(min(r["features"].shape[1:]) < crop for r in train):
        raise ValueError("Diagnostic crop exceeds training dimensions")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    device = device_for(cfg["device"])
    started = time.monotonic()
    deadline = started + cfg["max_seconds"]
    result = dict(
        state="running",
        scope="Bounded patch fitting and full-view validation; no release qualification",
        config=cfg,
        cohort_sha256=digest(root / "cohort.json"),
        code_commit=git_revision(),
        source_hashes=sources(),
        torch=str(torch.__version__),
        device=str(device),
        trials=[],
    )

    def persist():
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "comparison.json", result)

    def guard():
        if time.monotonic() >= deadline:
            raise TimeoutError("Comparison wall-time cap reached")

    persist()
    try:
        for seed in cfg["seeds"]:
            # Exactly the same input choices/crops for every recipe within a seed.
            rng = np.random.default_rng(seed)
            schedule = [
                [
                    (
                        int(i),
                        int(rng.integers(train[i]["features"].shape[1] - crop + 1)),
                        int(rng.integers(train[i]["features"].shape[2] - crop + 1)),
                    )
                    for i in rng.integers(len(train), size=cfg["batch_size"])
                ]
                for _ in range(cfg["steps"])
            ]
            for trial in cfg["trials"]:
                guard()
                torch.manual_seed(seed)
                model = build_model(trial["model"]).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=cfg["learning_rate"], weight_decay=0.0001
                )
                health = LearningHealth(
                    model,
                    dict(stall_after_epochs=1, warning_metric="model_region_log_mae"),
                    trial["loss"],
                )
                history = []
                trial_start = time.monotonic()
                try:
                    for step, batch in enumerate(schedule):
                        guard()
                        xs, ys = [], []
                        for i, top, left in batch:
                            xs.append(train[i]["features"][:, top : top + crop, left : left + crop])
                            ys.append(
                                train[i]["reference"]["target"][
                                    top : top + crop, left : left + crop
                                ].transpose(2, 0, 1)
                            )
                        x, y = (
                            torch.from_numpy(np.stack(xs)).to(device),
                            torch.from_numpy(np.stack(ys)).to(device),
                        )
                        optimizer.zero_grad(set_to_none=True)
                        health.reset()
                        prediction = model(x)
                        loss = reconstruction_loss(prediction, y, trial["loss"])
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Non-finite control loss")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 1, error_if_nonfinite=True
                        )
                        if step % 25 == 0 or step == cfg["steps"] - 1:
                            health.observe(x, y, prediction, loss, norm)
                            history.append(dict(step=step, **health.summary(0)))
                        optimizer.step()
                finally:
                    health.close()
                synchronize(device)
                rows = []
                model.eval()
                with torch.inference_mode():
                    for item in val:
                        guard()
                        features, ref = item["features"], item["reference"]
                        prediction = (
                            model(torch.from_numpy(features[None]).to(device))[0]
                            .cpu()
                            .numpy()
                            .transpose(1, 2, 0)
                        )
                        support = reconstruction_region(features, trial["model"])
                        masks = reference_regions(ref["target"], ref["features"])
                        masks.update(model_region=support, bypass_region=~support)
                        methods = dict(
                            raw=features[:3].transpose(1, 2, 0),
                            atrous=item["atrous"],
                            neural=prediction,
                        )
                        row = dict(
                            id=item["row"]["id"],
                            group=item["row"]["group"],
                            samples=item["row"]["samples"],
                            metrics={},
                        )
                        for ref_name in ("target", "check"):
                            row["metrics"][ref_name] = {
                                name: dict(
                                    **quality_scores(
                                        image, ref[ref_name], features, trial["model"]
                                    ),
                                    regions=score_regions(image, ref[ref_name], masks),
                                )
                                for name, image in methods.items()
                            }
                        row["reference_disagreement"] = score_regions(
                            ref["target"], ref["check"], masks
                        )
                        rows.append(row)
                        save_arrays(
                            output / f"{trial['name']}-s{seed}/predictions/{row['id']}.npz",
                            prediction=prediction,
                        )
                checkpoint = output / f"{trial['name']}-s{seed}/final.pt"
                save_checkpoint(
                    checkpoint,
                    dict(
                        config=dict(model=trial["model"], loss=trial["loss"]),
                        model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        manifest_sha256=result["cohort_sha256"],
                        seed=seed,
                        step=cfg["steps"],
                    ),
                )
                entry = dict(
                    name=trial["name"],
                    seed=seed,
                    model=trial["model"],
                    loss=trial["loss"],
                    parameters=sum(p.numel() for p in model.parameters()),
                    history=history,
                    seconds=time.monotonic() - trial_start,
                    checkpoint_sha256=digest(checkpoint),
                    rows=rows,
                )
                result["trials"].append(entry)
                persist()
                print(
                    json.dumps(
                        dict(
                            trial=trial["name"],
                            seed=seed,
                            state="complete",
                            seconds=entry["seconds"],
                        )
                    ),
                    flush=True,
                )
        result["state"] = "completed"
    except Exception as error:
        result.update(state="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        persist()
    return result
