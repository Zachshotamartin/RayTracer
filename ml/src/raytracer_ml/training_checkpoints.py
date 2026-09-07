"""Portable full-state checkpoints and an immutable history of completed epochs."""

import os
import json
import shutil
import uuid
from pathlib import Path

import torch

from .checkpoint_selection import selected_checkpoint
from .io import digest, write_json


RESUME_FIELDS = {
    "config",
    "contract",
    "model",
    "optimizer",
    "scheduler",
    "rng",
    "epoch",
    "step",
    "best",
    "stale",
    "total_seconds",
    "manifest_sha256",
}


def require_resume_state(state):
    missing = RESUME_FIELDS - state.keys()
    if missing:
        raise ValueError(
            "Checkpoint is not a full training state; use latest.pt, best_resume.pt, or an "
            f"archived epoch (missing {', '.join(sorted(missing))})"
        )


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with Path(source).open("rb") as src, temporary.open("wb") as dst:
        shutil.copyfileobj(src, dst)
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(temporary, destination)


def checkpoint_record(path, state):
    selected = selected_checkpoint(state.get("selection_state", {}))
    return {
        "path": str(path),
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "checkpoint_id": state.get("checkpoint_id"),
        "epoch_completed": state["epoch"] + 1,
        "step": state["step"],
        "contract": state["contract"],
        "manifest_sha256": state["manifest_sha256"],
        "selected_epoch": selected["epoch"] + 1 if selected else None,
        "selected_eligible": bool(selected and selected.get("validation_constraints_pass")),
        "full_training_state": RESUME_FIELDS <= state.keys(),
    }


def commit_training_checkpoint(output, state, save, load):
    """latest.pt is authoritative; a resume repairs interrupted history/alias publication.

    The best full state is embedded without recursive nesting. A single copied latest,
    best_resume, or epoch file therefore suffices for a restart in a new directory.
    """
    output = Path(output)
    require_resume_state(state)
    if shutil.disk_usage(output).free < state["config"].get("min_free_gib", 0) * 2**30:
        raise ValueError("Checkpoint disk reserve reached; previous committed checkpoints retained")
    state.setdefault("checkpoint_id", str(uuid.uuid4()))
    selected = selected_checkpoint(state.get("selection_state", {}))
    if selected and selected["epoch"] == state["epoch"]:
        state["best_resume_state"] = cpu_snapshot(
            {key: value for key, value in state.items() if key != "best_resume_state"}
        )
    best = state.get("best_resume_state")
    if best is not None:
        require_resume_state(best)
        if (
            not selected
            or best["epoch"] != selected["epoch"]
            or best["contract"] != state["contract"]
        ):
            raise ValueError("Best resume state disagrees with checkpoint selection")
    latest = output / "latest.pt"
    path = None
    if state["config"].get("checkpoint_history", True):
        history = output / "checkpoints"
        history.mkdir(exist_ok=True)
        path = history / f"epoch-{state['epoch'] + 1:06d}.pt"
        if path.exists():
            receipt = path.with_suffix(".json")
            if receipt.exists() and json.loads(receipt.read_text())["sha256"] != digest(path):
                raise ValueError("Archived checkpoint checksum mismatch")
            archived = load(path)
            if archived.get("checkpoint_id") != state["checkpoint_id"]:
                raise ValueError("Refusing to overwrite a different archived epoch")
    save(latest, state)
    if path is not None:
        if not path.exists():
            atomic_copy(latest, path)
        record = checkpoint_record(path, state)
        record["path"] = path.name
        write_json(path.with_suffix(".json"), record)
    if best is not None:
        save(output / "best_resume.pt", best)
    else:
        # Legacy weight-only best files cannot reconstruct a lost optimizer/RNG state.
        (output / "best_resume.pt").unlink(missing_ok=True)
    write_json(
        output / "checkpoint-status.json",
        {
            "latest_epoch": state["epoch"] + 1,
            "best_epoch": selected["epoch"] + 1 if selected else None,
            "best_eligible": bool(selected and selected.get("validation_constraints_pass")),
            "best_resume_available": best is not None,
            "checkpoint_history": state["config"].get("checkpoint_history", True),
            "recovery_boundary": "completed epoch; an interrupted partial epoch is replayed",
        },
    )


def list_checkpoints(output, load):
    output = Path(output)
    if not (output / "latest.pt").is_file():
        raise ValueError("No completed training checkpoints at this output")
    files = [output / "latest.pt", output / "best_resume.pt"]
    files += sorted((output / "checkpoints").glob("epoch-*.pt"))
    return {
        "run": str(output.resolve()),
        "checkpoints": [checkpoint_record(path, load(path)) for path in files if path.is_file()],
        "note": "best.pt is for evaluation/export; best_resume.pt includes optimizer and RNG state",
    }
