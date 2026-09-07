"""Read-only progress from artifact receipts; completeness is distinct from quality."""

import json
import time
from pathlib import Path


def last_record(path):
    count, record, incomplete = 0, None, False
    if path.exists():
        # Never count the partial final line of a concurrent append.
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    incomplete = True
                    break
                if line.strip():
                    count += 1
                    record = line
    return count, json.loads(record) if record else None, incomplete


def status(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError("Artifact root does not exist (check that its volume is mounted)")
    datasets, runs = [], []
    for info_path in sorted((root / "datasets").glob("*/dataset.json")):
        info = json.loads(info_path.read_text())
        directory = info_path.parent
        count, last, partial = last_record(directory / "manifest.jsonl")
        total = info["estimate"]["examples"]
        reference_total = info["estimate"]["reference_images"]
        reference_count = sum(1 for _ in (directory / "references").glob("*.json"))
        check_count = sum(1 for _ in (directory / "reference_checks").glob("*.json"))
        datasets.append(
            dict(
                name=directory.name,
                path=str(directory),
                examples_complete=count,
                examples_target=total,
                examples_percent=100 * count / total,
                reference_receipts=reference_count,
                references_target=reference_total,
                independent_reference_checks=check_count,
                last_example=last["id"] if last else None,
                last_write_unix=(directory / "manifest.jsonl").stat().st_mtime
                if (directory / "manifest.jsonl").exists()
                else None,
                partial_append_observed=partial,
                counts_complete=count == total,
                integrity_rechecked=False,
            )
        )
    for config_path in sorted((root / "runs").rglob("config.json")):
        cfg = json.loads(config_path.read_text())
        directory = config_path.parent
        if "epochs" not in cfg:
            if not ("num_epochs" in cfg and isinstance(cfg.get("model"), str)):
                continue  # This artifact tree can also contain unrelated tool configurations.
            marker = directory / "checkpoints/latest"
            value = marker.read_text().strip() if marker.exists() else ""
            checkpoint = directory / "checkpoints" / f"checkpoint_{value}.pth"
            epoch = int(value) if value.isdecimal() and checkpoint.is_file() else 0
            runs.append(
                dict(
                    name=str(directory.relative_to(root / "runs")),
                    path=str(directory),
                    toolkit="oidn",
                    epochs_complete=epoch,
                    epoch_cap=cfg["num_epochs"],
                    epoch_cap_percent=100 * epoch / cfg["num_epochs"],
                    progress_basis="Last saved checkpoint receipt; may lag ongoing training",
                    checkpoint_integrity_rechecked=False,
                    stop_reason=None,
                    validation_constraints_pass=None,
                )
            )
            continue
        _, latest, _ = last_record(directory / "metrics.jsonl")
        summary_path = directory / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        environment_path = directory / "environment.json"
        environment = json.loads(environment_path.read_text()) if environment_path.exists() else {}
        epochs = latest["epoch"] + 1 if latest else 0
        baseline = latest.get("baseline_structure", {}) if latest else {}
        measured = latest.get("validation_structure", {}) if latest else {}
        edge_improvement = (
            (1 - measured["edge"] / baseline["edge"]) * 100 if baseline.get("edge", 0) > 0 else None
        )
        runs.append(
            dict(
                name=str(directory.relative_to(root / "runs")),
                path=str(directory),
                epochs_complete=epochs,
                epoch_cap=cfg["epochs"],
                epoch_cap_percent=100 * epochs / cfg["epochs"],
                stop_reason=summary["reason"] if summary else None,
                parameters=environment.get("parameters"),
                validation_structure=measured,
                baseline_structure=baseline,
                edge_error_improvement_percent=edge_improvement,
                validation_constraints_pass=latest.get("validation_constraints_pass")
                if latest
                else None,
                accumulated_training_seconds=latest.get("total_seconds", 0) if latest else 0,
                latest_epoch_seconds=latest.get("seconds") if latest else None,
                learning_health=latest.get("learning_health") if latest else None,
                selected_checkpoint_epoch=latest.get("selected_checkpoint_epoch")
                if latest
                else None,
                selected_checkpoint_eligible=latest.get("selected_checkpoint_eligible")
                if latest
                else None,
                validation_constraints=latest.get("validation_constraints") if latest else None,
            )
        )
    return dict(
        schema_version=1,
        observed_unix=time.time(),
        artifact_root=str(root),
        datasets=datasets,
        runs=runs,
        interpretation="Receipt counts are progress, not an integrity audit or a model quality qualification. "
        "Epoch percentage uses the configured cap; early stopping can finish sooner. "
        "Process liveness is not inferred from files.",
    )
