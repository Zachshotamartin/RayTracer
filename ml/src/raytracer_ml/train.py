"""Bounded training with validation selection and exact epoch-boundary resume state."""

import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import RenderDataset
from .data.validate import validate
from .io import identity, write_json, git_revision
from .losses import reconstruction_loss
from .models import build_model
from .preprocessing import model_schema
from .data.arrays import load_example
from .selection import structural_scores, checkpoint_score
from .data.sequences import SequenceDataset, collate_sequences
from .rollout import sequence_loss, validation_predictions


def device_for(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name not in ("cpu", "mps") or (name == "mps" and not torch.backends.mps.is_available()):
        raise ValueError(f"Unavailable training device: {name}")
    return torch.device(name)


def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()


def save_checkpoint(path, state):
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
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


def train(cfg, root, output, resume=False, max_new_epochs=None):
    root, output = Path(root), Path(output)
    validation = validate(root)
    data_info = json.loads((root / "dataset.json").read_text())
    if cfg["model"].get("scale", 1) != data_info["config"].get("scale", 1):
        raise ValueError("Model and dataset scales disagree")
    for key in ("epochs", "batch_size", "patience", "max_seconds"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    device = device_for(cfg.get("device", "auto"))
    torch.set_num_threads(int(cfg.get("cpu_threads", 2)))
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    feature_schema = model_schema(cfg["model"])
    unroll = int(cfg.get("autoregressive_unroll", 0))
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
        collate_fn=collate_sequences if unroll else None,
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
    checkpoint_path = output / "latest.pt"
    first_epoch = 0
    best = math.inf
    best_loss = math.inf
    baseline_scores = {}
    stale = 0
    total_seconds = 0.0
    step = 0
    if resume:
        state = load_checkpoint(checkpoint_path)
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
        restore_rng(state["rng"], generator, device)
        # Discard logs written after the last atomic checkpoint, if a process died there.
        log = output / "metrics.jsonl"
        if log.exists():
            lines = [
                line
                for line in log.read_text().splitlines()
                if json.loads(line)["epoch"] < first_epoch
            ]
            log.write_text("".join(line + "\n" for line in lines))
    elif checkpoint_path.exists() or (output / "config.json").exists():
        raise ValueError("Run already exists; use --resume or a new output directory")
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
    for epoch in range(first_epoch, cfg["epochs"]):
        if max_new_epochs is not None and epoch - first_epoch >= max_new_epochs:
            reason = "paused-at-epoch-boundary"
            break
        if time.perf_counter() - start >= cfg["max_seconds"]:
            reason = "time-cap"
            break
        epoch_start = time.perf_counter()
        model.train()
        losses = []
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
                loss = reconstruction_loss(model(x), y, cfg.get("loss"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            step += 1
        if reason == "time-cap":
            break
        model.eval()
        val_losses = []
        raw_losses = []
        measured_scores = []
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
                    measured_scores.append(structural_scores(image, target_image))
                    if validation_index not in baseline_scores:
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
                        baseline_scores[validation_index] = structural_scores(
                            baseline, target_image
                        )
        value = float(np.mean(val_losses))
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite validation loss")
        measured = (
            {key: float(np.mean([s[key] for s in measured_scores])) for key in ("ssim", "edge")}
            if measured_scores
            else {}
        )
        baseline = (
            {
                key: float(np.mean([s[key] for s in baseline_scores.values()]))
                for key in ("ssim", "edge")
            }
            if baseline_scores
            else {}
        )
        score, eligible = checkpoint_score(value, measured, baseline, cfg.get("selection"))
        improved = score < best
        if improved:
            best = score
            best_loss = value
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
            "validation_constraints_pass": eligible,
            "validation_structure": measured,
            "baseline_structure": baseline,
            "stale": stale,
            "total_seconds": total_seconds,
            "manifest_sha256": validation["manifest_sha256"],
            "training_scene_hashes": sorted({r["scene_sha256"] for r in training.rows}),
        }
        if improved:
            save_checkpoint(output / "best.pt", state)
        save_checkpoint(checkpoint_path, state)
        metric = {
            "epoch": epoch,
            "step": step,
            "train_loss": float(np.mean(losses)),
            "validation_loss": value,
            "selection_score": score,
            "validation_constraints_pass": eligible,
            "validation_structure": measured,
            "baseline_structure": baseline,
            "raw_validation_loss": float(np.mean(raw_losses)),
            "seconds": elapsed,
            "total_seconds": total_seconds,
            "best": best,
        }
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(metric) + "\n")
        print(json.dumps(metric), flush=True)
        if stale >= cfg["patience"]:
            reason = "early-stopping"
            break
    result = {
        "reason": reason,
        "best_validation_loss": None if not math.isfinite(best_loss) else best_loss,
        "best_selection_score": None if not math.isfinite(best) else best,
        "total_seconds": total_seconds,
        "checkpoint": str(output / "best.pt"),
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    write_json(output / "summary.json", result)
    if not checkpoint_path.exists():
        raise RuntimeError(
            "No epoch completed within the budget; increase the cap or reduce workload"
        )
    return result
