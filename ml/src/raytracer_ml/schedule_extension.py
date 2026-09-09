"""Explicit, recorded schedule changes after a completed run; never starts training."""

import copy
import json
import math
import uuid
from pathlib import Path

import torch
from filelock import FileLock

from .io import digest, identity, write_json
from .models import build_model
from .training_checkpoints import require_resume_state


def validate_extension_config(parent, cfg):
    allowed = {"epochs", "learning_rate"}
    if {k: v for k, v in parent.items() if k not in allowed} != {
        k: v for k, v in cfg.items() if k not in allowed
    }:
        raise ValueError("Schedule extension may change only epochs and learning_rate")
    if type(cfg["epochs"]) is not int or cfg["epochs"] <= parent["epochs"]:
        raise ValueError("Extended epoch total must exceed the completed parent schedule")
    lr = cfg["learning_rate"]
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not math.isfinite(lr)
        or not 0 < lr < parent["learning_rate"]
    ):
        raise ValueError("Extension learning rate must be finite, positive and lower than parent")


def parent_best_compatible(state, best):
    """Keep an older parent's full best state honest about its original schedule."""
    extension = state.get("schedule_extension", {})
    parent_cfg = extension.get("parent_config")
    if not parent_cfg or best["config"] != parent_cfg:
        return False
    expected = identity({"config": parent_cfg, "manifest": state["manifest_sha256"]})
    if (
        best["contract"] != expected
        or extension.get("parent_contract") != expected
        or best["manifest_sha256"] != state["manifest_sha256"]
        or best["epoch"] >= extension["parent_epochs_completed"]
    ):
        return False
    validate_extension_config(parent_cfg, state["config"])
    return True


def prepare_extension(parent_run, cfg, output):
    """Produce a portable seed checkpoint in a NEW directory, with zero optimizer steps.

    The destination is a seed artifact directory, not the eventual training output.
    Ordinary strict resume can load the explicitly migrated seed using its new config.
    """
    # Local imports avoid a train/checkpoint compatibility import cycle.
    from .experiment import verify_checkpoint
    from .train import load_checkpoint, save_checkpoint

    parent_run, output = Path(parent_run).resolve(), Path(output).resolve()
    if output == parent_run or parent_run in output.parents or output in parent_run.parents:
        raise ValueError("Extension seed and parent run must be separate")
    if output.exists():
        raise ValueError("Extension seed requires a new directory")
    with FileLock(parent_run / ".training.lock", timeout=0):
        summary = json.loads((parent_run / "summary.json").read_text())
        parent = verify_checkpoint(parent_run)
        require_resume_state(parent)
        if parent.get("schedule_extension"):
            raise ValueError("Chained schedule extensions are not supported")
        completed = parent["epoch"] + 1
        if (
            summary.get("reason") != "epochs-complete"
            or summary.get("epochs_complete") != completed
            or completed != parent["config"]["epochs"]
            or parent["stale"] >= parent["config"]["patience"]
        ):
            raise ValueError("Parent must finish its full schedule without early stopping")
        validate_extension_config(parent["config"], cfg)
        expected = identity({"config": parent["config"], "manifest": parent["manifest_sha256"]})
        if parent["contract"] != expected:
            raise ValueError("Parent training contract is inconsistent")
        if parent["scheduler"]["last_epoch"] != completed:
            raise ValueError("Parent scheduler is not at its completed epoch boundary")
        archive = parent_run / "checkpoints" / f"epoch-{completed:06d}.pt"
        # A normal resume can serialize latest differently; compare complete state.
        if not states_equal(parent, load_checkpoint(archive)):
            raise ValueError("Latest full state differs from verified parent archive")
        migrated = copy.deepcopy(parent)
        provenance = {
            "mode": "explicit-cosine-extension-v1",
            "parent_checkpoint": str(archive),
            "parent_sha256": digest(archive),
            "parent_contract": parent["contract"],
            "parent_config": parent["config"],
            "parent_epochs_completed": completed,
            "parent_step": parent["step"],
            "target_total_epochs": cfg["epochs"],
            "additional_epochs": cfg["epochs"] - completed,
            "learning_rate": cfg["learning_rate"],
            "optimizer_updates": 0,
            "early_stopping_counter_preserved": True,
        }
        migrated.update(
            config=copy.deepcopy(cfg),
            contract=identity({"config": cfg, "manifest": parent["manifest_sha256"]}),
            checkpoint_id=str(uuid.uuid4()),
            schedule_extension=provenance,
        )
        # Construct on CPU only. Loading preserves moments/step; no forward/backward/update.
        model = build_model(cfg["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
        optimizer.load_state_dict(parent["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] = cfg["learning_rate"]
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["epochs"] - completed
        )
        migrated["optimizer"] = optimizer.state_dict()
        migrated["scheduler"] = scheduler.state_dict()
        assert states_equal(parent["optimizer"]["state"], migrated["optimizer"]["state"])
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = output / "seed.pt"
        save_checkpoint(checkpoint, migrated)
        receipt = {
            **provenance,
            "checkpoint": str(checkpoint),
            "sha256": digest(checkpoint),
            "contract": migrated["contract"],
            "config": cfg,
            "manifest_sha256": parent["manifest_sha256"],
        }
        write_json(output / "extension.json", receipt)
        return receipt


def states_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(states_equal(a[k], b[k]) for k in a)
        )
    if isinstance(a, (tuple, list)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(states_equal(x, y) for x, y in zip(a, b))
        )
    return a == b
