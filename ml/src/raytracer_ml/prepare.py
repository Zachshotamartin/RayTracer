"""Dataset and Mac device preflight. No optimizer is constructed or advanced."""

import math
import platform
import shutil
import tempfile
import time
from pathlib import Path

import torch

from .data.dataset import RenderDataset
from .data.preservation import PreservationValidation
from .data.validate import validate
from .io import digest, git_revision, identity, write_json
from .learning_health import validate_health_config
from .losses import validate_loss_config
from .models import build_model
from .preprocessing import model_schema
from .selection import validate_selection_config
from .train import device_for, synchronize, spatial_collator


def prepare_training(cfg, root, output, report, *, validation=None):
    root, output, report = Path(root).resolve(), Path(output).resolve(), Path(report).resolve()
    if output.exists():
        raise ValueError("Preparation requires a new run directory; choose a fresh output")
    validate_loss_config(cfg.get("loss"))
    validate_selection_config(cfg.get("selection"))
    if cfg.get("learning_diagnostics") is not None:
        validate_health_config(cfg["learning_diagnostics"])
    for key in ("epochs", "batch_size", "patience", "cpu_threads", "crop"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(cfg["max_seconds"]) or cfg["max_seconds"] <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if cfg["model"].get("temporal"):
        raise ValueError("This preparation workflow is for spatial training")
    device = device_for(cfg.get("device", "auto"))
    collator = spatial_collator(cfg)
    torch.set_num_threads(cfg["cpu_threads"])
    validation = validate(root) if validation is None else validation
    if digest(root / "manifest.jsonl") != validation["manifest_sha256"]:
        raise ValueError("Manifest changed after integrity validation")
    schema = model_schema(cfg["model"])
    training = RenderDataset(
        root,
        "train",
        cfg["crop"],
        feature_schema=schema,
        edge_sampling=cfg.get("edge_sampling", 0),
        identity_probability=cfg.get("identity_probability", 0),
        augmentation=cfg.get("augmentation"),
        fuse_probability=cfg.get("fuse_probability", 0),
        preservation_mode=cfg.get("preservation_mode", "synthetic_identity"),
        near_clean_samples=cfg.get("near_clean_samples", 96),
        border_sampling=cfg.get("border_sampling", 0),
    )
    heldout = RenderDataset(root, "val", feature_schema=schema)
    if any(r["scale"] != cfg["model"].get("scale", 1) for r in training.rows + heldout.rows):
        raise ValueError("Model and dataset scales disagree")
    if cfg.get("validation_budgets"):
        heldout.rows = [r for r in heldout.rows if r["samples"] in cfg["validation_budgets"]]
    if cfg.get("validation_first_noise_only"):
        heldout.rows = [r for r in heldout.rows if "-n0-s" in r["id"]]
    if not heldout.rows:
        raise ValueError("Validation selection is empty")
    preservation_views = 0
    if "preservation_ratio" in (cfg.get("selection") or {}):
        preservation = PreservationValidation(root, cfg.get("near_clean_samples", 96))
        preservation_views = len(preservation.pairs)
        if preservation_views < cfg["selection"].get("min_preservation_views", 1):
            raise ValueError("Insufficient measured preservation validation views")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=output.parent) as probe:
        probe.write(b"training preparation write probe")
        probe.flush()
    free = shutil.disk_usage(output.parent).free
    if free < max(10, cfg.get("min_free_gib", 0)) * 2**30:
        raise ValueError("Training preparation requires at least 10 GiB free for checkpoints/logs")
    torch.manual_seed(cfg["seed"])
    model = build_model(cfg["model"]).to(device).eval()
    shapes = []
    with torch.inference_mode():
        for dataset, batch_size in ((training, cfg["batch_size"]), (heldout, 1)):
            seen = set()
            for index, row in enumerate(dataset.rows):
                size = (
                    (dataset.crop, dataset.crop)
                    if dataset.crop
                    else (row["stats"]["height"], row["stats"]["width"])
                )
                if size in seen:
                    continue
                seen.add(size)
                x, y = dataset[index]
                batch = x[None].repeat(batch_size, 1, 1, 1).to(device)
                prediction = model(batch)
                if (
                    prediction.shape != (batch_size, *y.shape)
                    or not torch.isfinite(prediction).all()
                ):
                    raise ValueError("Preflight inference returned invalid output")
                shapes.append(
                    {
                        "split": dataset.split,
                        "input": list(batch.shape),
                        "output": list(prediction.shape),
                    }
                )
        synchronize(device)
        if collator is not None:
            from .augment import AlignedCropCollator

            for shape in cfg["crop_shapes"]:
                batch, targets = AlignedCropCollator(
                    [shape], cfg["crop"], cfg["model"].get("scale", 1)
                )([training[0] for _ in range(cfg["batch_size"])])
                prediction = model(batch.to(device))
                if prediction.shape != targets.shape or not torch.isfinite(prediction).all():
                    raise ValueError("Variable crop preflight failed")
                shapes.append(
                    {
                        "split": "train-variable-crop",
                        "input": list(batch.shape),
                        "output": list(prediction.shape),
                    }
                )
    result = {
        "state": "prepared-awaiting-approval",
        "optimizer_updates": 0,
        "prepared_unix": time.time(),
        "source_commit": git_revision(),
        "source_files_sha256": {
            str(path.relative_to(Path(__file__).parent)): digest(path)
            for path in sorted(Path(__file__).parent.rglob("*.py"))
        },
        "data": str(root),
        "output": str(output),
        "config": cfg,
        "contract": identity({"config": cfg, "manifest": validation["manifest_sha256"]}),
        "validation": validation,
        "training_examples": len(training),
        "training_steps_per_epoch": math.ceil(len(training) / cfg["batch_size"]),
        "validation_examples_per_epoch": len(heldout),
        "preservation_views_per_epoch": preservation_views,
        "device": str(device),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "inference_shapes_checked": shapes,
        "ssd_free_gib": free / 2**30,
        "checkpoint_history": cfg.get("checkpoint_history", True),
        "limits": [
            "Inference-only preflight does not benchmark training or its peak memory",
            "Time cap excludes initial integrity validation and is checked between training batches",
            "Reference integrity is distinct from Monte Carlo convergence and model qualification",
        ],
    }
    write_json(report, result)
    return result
