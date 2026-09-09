"""Bounded training with validation selection and exact epoch-boundary resume state."""

import json
import math
import os
import platform
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock, Timeout
from torch.utils.data import DataLoader

from .data.dataset import RenderDataset
from .data.validate import validate
from .io import identity, write_json, git_revision, digest
from .training_checkpoints import commit_training_checkpoint, require_resume_state
from .losses import reconstruction_loss, validate_loss_config
from .models import build_model
from .preprocessing import model_schema
from .data.arrays import load_example
from .selection import (
    quality_scores,
    checkpoint_score,
    aggregate_scores,
    constraint_report,
    validate_selection_config,
)
from .data.preservation import PreservationValidation
from .temporal_metrics import TemporalComparison
from .data.sequences import SequenceDataset, collate_sequences
from .rollout import sequence_loss, validation_predictions
from .learning_health import LearningHealth, validate_health_config
from .checkpoint_selection import (
    update_selection,
    selected_checkpoint,
    publish_selection,
    restore_selection,
)


def device_for(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in ("cpu", "mps") or (name == "mps" and not torch.backends.mps.is_available()):
        raise ValueError(f"Unavailable training device: {name}")
    return torch.device(name)


def spatial_collator(cfg):
    if not cfg.get("crop_shapes"):
        return None
    from .augment import AlignedCropCollator

    if cfg.get("autoregressive_unroll") or not cfg.get("crop"):
        raise ValueError("Variable crops require cropped spatial training")
    return AlignedCropCollator(cfg["crop_shapes"], cfg["crop"], cfg["model"].get("scale", 1))


def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def save_checkpoint(path, state):
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def rng_state(generator, device):
    n = np.random.get_state()
    result = {
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
        "torch": torch.get_rng_state(),
        "loader": generator.get_state(),
    }
    if str(device) == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if str(device) == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def train(cfg, root, output, resume=False, max_new_epochs=None, resume_from=None):
    if resume and resume_from:
        raise ValueError("Choose --resume or --resume-from, not both")
    if max_new_epochs is not None and (type(max_new_epochs) is not int or max_new_epochs < 1):
        raise ValueError("max_new_epochs must be a positive integer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(output / ".training.lock", timeout=0):
            return _train(cfg, root, output, resume, max_new_epochs, resume_from)
    except Timeout as error:
        raise ValueError("Another process is using this training run") from error


def _train(cfg, root, output, resume, max_new_epochs, resume_from):
    validate_loss_config(cfg.get("loss"))
    root, output = Path(root), Path(output)
    checkpoint_path = output / "latest.pt"
    if not resume and (checkpoint_path.exists() or (output / "config.json").exists()):
        raise ValueError("Run already exists; use --resume or a new output directory")
    if resume and not checkpoint_path.is_file():
        raise ValueError("No latest checkpoint to resume")
    source_checkpoint = Path(resume_from).resolve() if resume_from else checkpoint_path
    state = load_checkpoint(source_checkpoint) if resume or resume_from else None
    if state is not None:
        require_resume_state(state)
    if type(cfg.get("checkpoint_history", True)) is not bool:
        raise ValueError("checkpoint_history must be boolean")
    validation = validate(root)
    data_info = json.loads((root / "dataset.json").read_text())
    if cfg["model"].get("scale", 1) != data_info["config"].get("scale", 1):
        raise ValueError("Model and dataset scales disagree")
    for key in ("epochs", "batch_size", "patience", "max_seconds"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    reserve = cfg.get("min_free_gib", 0)
    if (
        isinstance(reserve, bool)
        or not isinstance(reserve, (int, float))
        or not math.isfinite(reserve)
        or reserve < 0
    ):
        raise ValueError("min_free_gib must be finite and nonnegative")
    if shutil.disk_usage(output).free < reserve * 2**30:
        raise ValueError("Insufficient free disk space for the configured training reserve")
    device = device_for(cfg.get("device", "auto"))
    torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    feature_schema = model_schema(cfg["model"])
    unroll = int(cfg.get("autoregressive_unroll", 0))
    selection_config = cfg.get("selection") or {}
    validate_selection_config(selection_config)
    temporal_gate = "temporal_ratio" in selection_config
    if temporal_gate and (not unroll or cfg["model"].get("scale", 1) != 1):
        raise ValueError("Temporal selection requires autoregressive same-resolution validation")
    preservation = None
    if "preservation_ratio" in selection_config:
        if cfg["model"].get("temporal") or feature_schema != 2:
            raise ValueError("Preservation selection requires a spatial schema-2 model")
        preservation = PreservationValidation(root, cfg.get("near_clean_samples", 96))
    health_config = cfg.get("learning_diagnostics")
    if health_config is not None:
        validate_health_config(health_config)
        if unroll:
            raise ValueError("Learning diagnostics currently require spatial training")
    if unroll and (not cfg["model"].get("temporal") or feature_schema != 2 or not 2 <= unroll <= 8):
        raise ValueError(
            "Autoregressive training needs a schema-2 temporal model and 2..8 frame unroll"
        )
    if unroll and any(
        cfg.get(key, 0)
        for key in ("crop", "edge_sampling", "identity_probability", "fuse_probability")
    ):
        raise ValueError(
            "Autoregressive windows use full frames; crop/identity/fusion policies are spatial-only"
        )
    training = RenderDataset(
        root,
        "train",
        cfg.get("crop", 0),
        cfg["model"].get("temporal", False),
        feature_schema,
        cfg.get("edge_sampling", 0),
        cfg.get("identity_probability", 0),
        cfg.get("augmentation"),
        cfg.get("fuse_probability", 0),
        cfg.get("preservation_mode", "synthetic_identity"),
        cfg.get("near_clean_samples", 96),
        cfg.get("border_sampling", 0),
    )
    heldout = RenderDataset(
        root, "val", temporal=cfg["model"].get("temporal", False), feature_schema=feature_schema
    )
    if cfg.get("validation_budgets"):
        heldout.rows = [r for r in heldout.rows if r["samples"] in cfg["validation_budgets"]]
    if cfg.get("validation_first_noise_only"):
        heldout.rows = [r for r in heldout.rows if "-n0-s" in r["id"]]
    if not heldout.rows:
        raise ValueError("Validation selection is empty")
    if unroll:
        training = SequenceDataset(root, "train", unroll, cfg.get("training_budgets"))
        heldout = SequenceDataset(
            root,
            "val",
            budgets=cfg.get("validation_budgets"),
            first_noise_only=cfg.get("validation_first_noise_only", False),
        )
        if min(map(len, heldout.sequences)) < cfg.get("validation_min_frames", 2):
            raise ValueError(
                "Validation rollouts are shorter than the configured qualification length"
            )
    loader = DataLoader(
        training,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate_sequences if unroll else spatial_collator(cfg),
    )
    val_loader = DataLoader(
        heldout, batch_size=1, num_workers=0, collate_fn=collate_sequences if unroll else None
    )
    model = build_model(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    contract = identity({"config": cfg, "manifest": validation["manifest_sha256"]})
    output.mkdir(parents=True, exist_ok=True)
    first_epoch = 0
    best = math.inf
    best_loss = math.inf
    selection_state = {}
    baseline_scores = {}
    stale = 0
    total_seconds = 0.0
    step = 0
    best_resume_state = None
    schedule_extension = state.get("schedule_extension") if state else None
    if state is not None:
        if state["contract"] != contract:
            raise ValueError("Resume configuration or dataset differs from checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_epoch = state["epoch"] + 1
        best = state["best"]
        best_loss = state.get("best_loss", best)
        stale = state["stale"]
        step = state["step"]
        total_seconds = state["total_seconds"]
        selection_state = restore_selection(
            state, source_checkpoint.parent, load_checkpoint, bool(cfg.get("selection"))
        )
        state["selection_state"] = selection_state
        commit_training_checkpoint(output, state, save_checkpoint, load_checkpoint)
        best_resume_state = state.get("best_resume_state")
        publish_selection(selection_state, output, save_checkpoint)
        selected = selected_checkpoint(selection_state)
        best, best_loss = selected["selection_score"], selected["validation_loss"]
        restore_rng(state["rng"], generator, device)
        # Discard logs written after the last atomic checkpoint, if a process died there.
        log = output / "metrics.jsonl"
        if resume and log.exists():
            lines = [
                line
                for line in log.read_text().splitlines()
                if json.loads(line)["epoch"] < first_epoch
            ]
            log.write_text("".join(line + "\n" for line in lines))
        if resume_from:
            write_json(
                output / "lineage.json",
                {
                    "source_checkpoint": str(source_checkpoint),
                    "source_sha256": digest(source_checkpoint),
                    "parent_epoch_completed": first_epoch,
                    "parent_step": step,
                    "contract": contract,
                    "mode": "full-state continuation; original run is unchanged",
                },
            )
    write_json(output / "config.json", cfg)
    write_json(
        output / "environment.json",
        {
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": str(device),
            "code_commit": git_revision(),
            "dataset": validation,
            "contract": contract,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
    )
    start = time.perf_counter()
    reason = "epochs-complete"
    health = (
        LearningHealth(model, health_config, cfg.get("loss")) if health_config is not None else None
    )
    for epoch in range(first_epoch, cfg["epochs"]):
        if checkpoint_path.exists() and (output / "STOP_AFTER_EPOCH").exists():
            (output / "STOP_AFTER_EPOCH").unlink()
            reason = "pause-requested"
            break
        if stale >= cfg["patience"]:
            reason = "early-stopping"
            break
        if max_new_epochs is not None and epoch - first_epoch >= max_new_epochs:
            reason = "paused-at-epoch-boundary"
            break
        if time.perf_counter() - start >= cfg["max_seconds"]:
            reason = "time-cap"
            break
        epoch_start = time.perf_counter()
        model.train()
        losses = []
        if health is not None:
            health.reset()
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch in loader:
            if time.perf_counter() - start >= cfg["max_seconds"]:
                # Only epoch-boundary checkpoints are published; resume replays this partial epoch.
                reason = "time-cap"
                break
            optimizer.zero_grad(set_to_none=True)
            if unroll:
                loss = sequence_loss(model, batch, device, cfg)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                loss = reconstruction_loss(prediction, y, cfg.get("loss"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            if health is not None:
                health.observe(x, y, prediction, loss, gradient_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
        if reason == "time-cap":
            break
        model.eval()
        val_losses = []
        raw_losses = []
        measured_scores = []
        resolution_scores = {}
        slice_scores = {
            name: {}
            for name in ("budget", "family", "stratum", "transport", "domain", "domain_budget")
        }
        comparison = TemporalComparison() if temporal_gate else None
        with torch.inference_mode():
            for validation_index, (prediction, y, x, frame) in enumerate(
                validation_predictions(model, val_loader, device, bool(unroll))
            ):
                val_losses.append(float(reconstruction_loss(prediction, y, cfg.get("loss")).cpu()))
                raw = x[:, :3]
                if raw.shape[-2:] != y.shape[-2:]:
                    raw = torch.nn.functional.interpolate(
                        raw, size=y.shape[-2:], mode="bilinear", align_corners=False
                    )
                raw_losses.append(float(reconstruction_loss(raw, y, cfg.get("loss")).cpu()))
                if cfg.get("selection"):
                    target_image = y[0].cpu().numpy().transpose(1, 2, 0)
                    image = prediction[0].cpu().numpy().transpose(1, 2, 0)
                    features = x[0, : 27 if feature_schema == 2 else 17].cpu().numpy()
                    measured_score = quality_scores(image, target_image, features, cfg["model"])
                    if validation_index not in baseline_scores or temporal_gate:
                        baseline = (
                            frame["atrous"]
                            if frame is not None
                            else load_example(root, heldout.rows[validation_index])["atrous"]
                        )
                        if baseline.shape != target_image.shape:
                            baseline = (
                                torch.nn.functional.interpolate(
                                    torch.from_numpy(baseline.transpose(2, 0, 1)[None]),
                                    size=target_image.shape[:2],
                                    mode="bilinear",
                                    align_corners=False,
                                )[0]
                                .numpy()
                                .transpose(1, 2, 0)
                            )
                        baseline_scores[validation_index] = quality_scores(
                            baseline, target_image, features, cfg["model"]
                        )
                    if temporal_gate:
                        temporal_scores = comparison.measure(
                            frame["row"],
                            frame["features"],
                            frame["position"],
                            target_image,
                            {"neural": image, "atrous": baseline},
                        )
                        measured_score.update(temporal_scores["neural"])
                        baseline_scores[validation_index].update(temporal_scores["atrous"])
                    measured_scores.append(measured_score)
                    if not unroll:
                        h, w = x.shape[-2:]
                        oh, ow = y.shape[-2:]
                        key = f"{w}x{h}->{ow}x{oh}"
                        bucket = resolution_scores.setdefault(key, {"measured": [], "baseline": []})
                        bucket["measured"].append(measured_score)
                        bucket["baseline"].append(baseline_scores[validation_index])
                        row = heldout.rows[validation_index]
                        labels = {
                            "budget": str(row["samples"]),
                            "domain": row.get("cohort", "unspecified"),
                            "domain_budget": f"{row.get('cohort', 'unspecified')}:{row['samples']}",
                            "family": row.get("family", "unspecified"),
                            "stratum": row.get("stratum", "unspecified"),
                            "transport": (row.get("render_variant") or {}).get(
                                "transport", "unspecified"
                            ),
                        }
                        for name, label in labels.items():
                            bucket = slice_scores[name].setdefault(
                                label, {"measured": [], "baseline": []}
                            )
                            bucket["measured"].append(measured_score)
                            bucket["baseline"].append(baseline_scores[validation_index])
        value = float(np.mean(val_losses))
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite validation loss")
        measured = aggregate_scores(measured_scores)
        baseline = aggregate_scores(list(baseline_scores.values()))
        if preservation is not None:
            near_clean, near_clean_raw = preservation.measure(model, device)
            measured.update(near_clean)
            baseline.update(near_clean_raw)
        score, eligible = checkpoint_score(value, measured, baseline, cfg.get("selection"))
        constraints = constraint_report(measured, baseline, selection_config)
        resolution_constraints = {}
        budget_constraints = {}
        domain_constraints, domain_budget_constraints = {}, {}
        if any(
            cfg.get(f"selection_by_{name}")
            for name in ("resolution", "budget", "domain", "domain_budget")
        ):
            spatial_constraints = {
                k: v
                for k, v in selection_config.items()
                if k
                not in (
                    "preservation_ratio",
                    "min_preservation_views",
                    "temporal_ratio",
                    "min_temporal_transitions",
                )
            }
            if not spatial_constraints or not resolution_scores:
                raise ValueError("Resolution selection requires spatial quality constraints")
            for name, buckets, reports in (
                ("resolution", resolution_scores, resolution_constraints),
                ("budget", slice_scores["budget"], budget_constraints),
                ("domain", slice_scores["domain"], domain_constraints),
                ("domain_budget", slice_scores["domain_budget"], domain_budget_constraints),
            ):
                if not cfg.get(f"selection_by_{name}"):
                    continue
                if not buckets:
                    raise ValueError(f"No {name} slices available for checkpoint selection")
                for key, bucket in buckets.items():
                    reports[key] = constraint_report(
                        aggregate_scores(bucket["measured"]),
                        aggregate_scores(bucket["baseline"]),
                        spatial_constraints,
                    )
                passed = all(c["passed"] for report in reports.values() for c in report.values())
                constraints[f"{name}_slices"] = {"passed": passed, "slices": len(reports)}
                eligible = eligible and passed
            for name, reports in (
                ("domain", domain_constraints),
                ("domain_budget", domain_budget_constraints),
            ):
                if reports:
                    constraints[f"{name}_reports"] = {
                        "passed": all(
                            c["passed"] for report in reports.values() for c in report.values()
                        ),
                        "reports": reports,
                    }
        snapshot = {
            "selection_policy": "eligible-first-v1",
            "config": cfg,
            "contract": contract,
            "model": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "selection_score": score,
            "validation_loss": value,
            "validation_constraints_pass": eligible,
            "validation_structure": measured,
            "validation_constraints": constraints,
            "validation_resolution_constraints": resolution_constraints,
            "validation_budget_constraints": budget_constraints,
            "baseline_structure": baseline,
            "manifest_sha256": validation["manifest_sha256"],
            "training_scene_hashes": sorted({r["scene_sha256"] for r in training.rows}),
        }
        selection_state, improved = update_selection(
            selection_state, snapshot, bool(cfg.get("selection"))
        )
        selected = selected_checkpoint(selection_state)
        best, best_loss = selected["selection_score"], selected["validation_loss"]
        if improved:
            stale = 0
        else:
            stale += 1
        scheduler.step()
        synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        total_seconds += elapsed
        state = {
            "schema_version": 1,
            "config": cfg,
            "contract": contract,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(generator, device),
            "epoch": epoch,
            "step": step,
            "best": best,
            "best_loss": best_loss,
            "selection_policy": "eligible-first-v1",
            "selection_state": selection_state,
            "validation_constraints_pass": eligible,
            "validation_constraints": constraints,
            "validation_resolution_constraints": resolution_constraints,
            "validation_budget_constraints": budget_constraints,
            "validation_structure": measured,
            "baseline_structure": baseline,
            "stale": stale,
            "total_seconds": total_seconds,
            "manifest_sha256": validation["manifest_sha256"],
            "training_scene_hashes": sorted({r["scene_sha256"] for r in training.rows}),
            "best_resume_state": best_resume_state,
        }
        if schedule_extension is not None:
            state["schedule_extension"] = schedule_extension
        commit_training_checkpoint(output, state, save_checkpoint, load_checkpoint)
        best_resume_state = state.get("best_resume_state")
        publish_selection(selection_state, output, save_checkpoint, updated_epoch=epoch)
        metric = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": value,
            "selection_score": score,
            "validation_constraints_pass": eligible,
            "validation_constraints": constraints,
            "validation_resolution_constraints": resolution_constraints,
            "validation_budget_constraints": budget_constraints,
            "validation_structure": measured,
            "baseline_structure": baseline,
            "raw_validation_loss": float(np.mean(raw_losses)),
            "seconds": elapsed,
            "total_seconds": total_seconds,
            "best": best,
            "selected_checkpoint_epoch": selected["epoch"],
            "selected_checkpoint_eligible": bool(selection_state.get("best_eligible"))
            if selection_config
            else None,
            "learning_rate": learning_rate,
            "validation_by_resolution": {
                key: {
                    "images": len(bucket["measured"]),
                    "measured": aggregate_scores(bucket["measured"]),
                    "baseline": aggregate_scores(bucket["baseline"]),
                }
                for key, bucket in resolution_scores.items()
            },
            "validation_by_slice": {
                name: {
                    key: {
                        "images": len(bucket["measured"]),
                        "measured": aggregate_scores(bucket["measured"]),
                        "baseline": aggregate_scores(bucket["baseline"]),
                    }
                    for key, bucket in buckets.items()
                }
                for name, buckets in slice_scores.items()
            },
        }
        if health is not None:
            metric["learning_health"] = health.summary(epoch)
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if (output / "STOP_AFTER_EPOCH").exists():
            (output / "STOP_AFTER_EPOCH").unlink()
            reason = "pause-requested"
            break
        if stale >= cfg["patience"]:
            reason = "early-stopping"
            break
    result = {
        "reason": reason,
        "best_validation_loss": None if not math.isfinite(best_loss) else best_loss,
        "best_selection_score": None if not math.isfinite(best) else best,
        "total_seconds": total_seconds,
        "checkpoint": str(output / "best.pt"),
        "eligible_checkpoint": str(output / "best_eligible.pt")
        if selection_state.get("best_eligible")
        else None,
        "selection_policy": "eligible-first-v1",
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "latest_checkpoint": str(checkpoint_path),
        "best_resume_checkpoint": str(output / "best_resume.pt")
        if best_resume_state is not None
        else None,
        "epochs_complete": state["epoch"] + 1 if checkpoint_path.exists() else 0,
    }
    write_json(output / "summary.json", result)
    if health is not None:
        health.close()
    if not checkpoint_path.exists():
        raise RuntimeError(
            "No epoch completed within the budget; increase the cap or reduce workload"
        )
    return result
