"""A pinned, resumable data-to-training job with explicit execution receipts.

Only clean resource time limits are continued automatically. A user pause,
unexplained interruption, bad checksum or runtime error requires inspection.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from filelock import FileLock

from .data.generate import generate
from .io import config, digest, write_json
from .metrics import image_metrics
from .prepare import prepare_training
from .train import load_checkpoint, train
from .training_checkpoints import require_resume_state


def verify_plan(plan):
    source = Path(plan["source"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != plan["source_commit"]
    ):
        raise ValueError("Pinned source revision changed")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise ValueError("Pinned source is dirty")
    package = Path(__file__).resolve().parent
    if package != (source / "ml/src/raytracer_ml").resolve():
        raise ValueError("Experiment imported a different source package")
    for name, checksum in plan["source_files_sha256"].items():
        if digest(package / name) != checksum:
            raise ValueError("Pinned source checksum changed")
    keys = (
        ("train_config",)
        if plan.get("data_mode") == "existing"
        else ("data_config", "train_config", "renderer")
    )
    for key in keys:
        if digest(plan[key]) != plan[f"{key}_sha256"]:
            raise ValueError(f"Pinned {key} checksum changed")
    if "extension_seed" in plan:
        if plan.get("data_mode") != "existing":
            raise ValueError("Schedule extensions require an existing-data plan")
        if digest(plan["extension_seed"]) != plan["extension_seed_sha256"]:
            raise ValueError("Pinned extension seed checksum changed")


def saved_examples(root):
    return sum(1 for _ in (Path(root) / "examples").glob("*.json"))


def verify_checkpoint(output):
    output = Path(output)
    state = load_checkpoint(output / "latest.pt")
    require_resume_state(state)
    archive = output / "checkpoints" / f"epoch-{state['epoch'] + 1:06d}.pt"
    receipt = json.loads(archive.with_suffix(".json").read_text())
    if digest(archive) != receipt["sha256"] or state["checkpoint_id"] != receipt["checkpoint_id"]:
        raise ValueError("Latest state and immutable checkpoint receipt disagree")
    return state


def audit_references(root, output):
    """Measure retained train/validation targets; sealed test is excluded from tuning."""
    root = Path(root)
    info_path = root / "dataset.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    if info.get("reuse"):
        reports = []
        for source in info["reuse"]["sources"]:
            path = root / "sources" / source["namespace"]
            report = audit_references(
                path, output.parent / f"reference-agreement-{source['namespace']}.json"
            )
            reports.append({"source": source["namespace"], **report})
        result = {
            "scope": "Retained original train/validation reference checks; no new renders",
            "images": sum(r["images"] for r in reports),
            "sources": reports,
            "below_pilot_agreement_floor": [
                f"{r['source']}/{name}"
                for r in reports
                for name in r["below_pilot_agreement_floor"]
            ],
        }
        write_json(output, result)
        return result
    checks = []
    for path in sorted((root / "reference_checks").glob("*.json")):
        record = json.loads(path.read_text())
        if record["split"] not in ("train", "val"):
            continue
        if (
            digest(root / record["path"]) != record["sha256"]
            or digest(root / record["reference"]) != record["reference_sha256"]
        ):
            raise ValueError("Reference audit checksum mismatch")
        with np.load(root / record["path"]) as a, np.load(root / record["reference"]) as b:
            metrics = image_metrics(b["target"], a["target"])
        checks.append(
            {"reference": record["reference"], "split": record["split"], "metrics": metrics}
        )
    report = {
        "scope": "Retained train/validation reference agreement, not model qualification or blanket convergence",
        "images": len(checks),
        "checks": checks,
        "below_pilot_agreement_floor": [
            c["reference"]
            for c in checks
            if c["metrics"]["psnr"] < 30 or c["metrics"]["ssim"] < 0.95
        ],
        "policy": "Report difficult/noisy targets for review; do not silently remove them, change reference budgets or relax model eligibility gates.",
    }
    write_json(output, report)
    return report


def run_experiment(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    prep, root, output = (Path(plan[k]).resolve() for k in ("preparation", "data", "output"))
    if len({prep, root, output}) != 3 or any(
        a in b.parents for a in (prep, root, output) for b in (prep, root, output) if a != b
    ):
        raise ValueError("Preparation, data and run directories must be separate")
    prep.mkdir(parents=True, exist_ok=True)
    with FileLock(prep / ".experiment.lock", timeout=0):
        previous = (
            json.loads((prep / "state.json").read_text())
            if (prep / "state.json").exists()
            else None
        )
        if previous and previous["phase"] not in ("generation-time-cap", "training-time-cap"):
            raise ValueError(
                "Existing experiment is active, stopped, complete or interrupted; inspect before restarting"
            )

        def record(phase, **values):
            import os

            value = {
                "phase": phase,
                "updated_unix": time.time(),
                "pid": os.getpid(),
                "source_commit": plan["source_commit"],
                **values,
            }
            write_json(prep / "state.json", value)
            with (prep / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(value) + "\n")
            print(json.dumps(value), flush=True)
            return value

        def guard():
            verify_plan(plan)
            if (prep / "STOP_EXPERIMENT").exists():
                record("user-stopped")
                return False
            if shutil.disk_usage(prep).free < plan.get("min_free_gib", 20) * 2**30:
                raise RuntimeError("Experiment free-space reserve reached")
            return True

        try:
            if not guard():
                return {"phase": "user-stopped"}
            train_cfg = config(plan["train_config"])
            data_complete = prep / "data-complete.json"
            if plan.get("data_mode") == "existing":
                if any(key in plan for key in ("renderer", "data_config")):
                    raise ValueError(
                        "Existing-data plans cannot contain renderer or generation config"
                    )
                data_info = json.loads((root / "dataset.json").read_text())
                if not data_info.get("reuse"):
                    raise ValueError(
                        "Existing-data plan requires a provenance-checked reuse dataset"
                    )
                if (
                    digest(root / "manifest.jsonl") != plan["manifest_sha256"]
                    or digest(root / "dataset.json") != plan["dataset_sha256"]
                ):
                    raise ValueError("Pinned existing dataset changed")
                expected = data_info["estimate"]
                write_json(
                    data_complete,
                    {
                        "completed": expected["examples"],
                        "manifest_sha256": plan["manifest_sha256"],
                        "new_rays": 0,
                    },
                )
            else:
                data_cfg = config(plan["data_config"])
                expected = generate(data_cfg, plan["renderer"], root, dry_run=True)
                write_json(prep / "estimate.json", expected)
                data_complete = prep / "data-complete.json"
                if not data_complete.exists():
                    for invocation in range(1, plan.get("max_generation_invocations", 200) + 1):
                        if not guard():
                            return {"phase": "user-stopped"}
                        before = saved_examples(root)
                        record(
                            "generating",
                            invocation=invocation,
                            examples_before=before,
                            expected_examples=expected["examples"],
                        )
                        start = time.monotonic()
                        try:
                            generated = generate(data_cfg, plan["renderer"], root)
                        except (TimeoutError, subprocess.TimeoutExpired):
                            count = saved_examples(root)
                            if (
                                time.monotonic() - start < data_cfg["max_seconds"] - 2
                                or count <= before
                            ):
                                raise RuntimeError(
                                    "Unexpected timeout or no completed-example progress in the generation budget"
                                )
                            record(
                                "generation-time-cap",
                                invocation=invocation,
                                examples_complete=count,
                                expected_examples=expected["examples"],
                            )
                            audit_references(root, prep / "reference-agreement.json")
                            continue
                        if generated["completed"] != expected["examples"]:
                            raise RuntimeError("Generation ended with incomplete data")
                        write_json(data_complete, generated)
                        break
                    else:
                        raise RuntimeError("Generation invocation limit reached")
            generated = json.loads(data_complete.read_text())
            if digest(root / "manifest.jsonl") != generated["manifest_sha256"]:
                raise ValueError("Completed dataset manifest changed")
            if not guard():
                return {"phase": "user-stopped"}
            report = prep / "preflight.json"
            if not report.exists():
                record("validating-data-and-device", examples_complete=expected["examples"])
                prepared = prepare_training(train_cfg, root, output, report)
                prepared.update(
                    state="validated-user-authorized", authorization=plan["authorization"]
                )
                write_json(report, prepared)
                audit_references(root, prep / "reference-agreement.json")
            prepared = json.loads(report.read_text())
            if (
                prepared["config"] != train_cfg
                or prepared["validation"]["manifest_sha256"] != generated["manifest_sha256"]
            ):
                raise ValueError("Prepared training contract changed")
            resume = (output / "latest.pt").is_file()
            if resume and (not previous or previous["phase"] != "training-time-cap"):
                raise ValueError("Existing training output was not a clean time-cap continuation")
            extension_seed = plan.get("extension_seed") if not resume else None
            if extension_seed:
                seed = load_checkpoint(extension_seed)
                require_resume_state(seed)
                extension = seed.get("schedule_extension", {})
                if (
                    extension.get("mode") != "explicit-cosine-extension-v1"
                    or seed["config"] != train_cfg
                    or seed["manifest_sha256"] != generated["manifest_sha256"]
                    or seed["epoch"] + 1 != extension.get("parent_epochs_completed")
                ):
                    raise ValueError("Extension seed does not match the prepared continuation")
            for invocation in range(1, plan.get("max_training_invocations", 200) + 1):
                if not guard():
                    return {"phase": "user-stopped"}
                before = verify_checkpoint(output)["epoch"] + 1 if resume else 0
                if extension_seed:
                    before = seed["epoch"] + 1
                record(
                    "training",
                    invocation=invocation,
                    epochs_before=before,
                    target_epochs=train_cfg["epochs"],
                    resume=resume,
                )
                if extension_seed:
                    result = train(train_cfg, root, output, resume_from=extension_seed)
                    extension_seed = None
                else:
                    result = train(train_cfg, root, output, resume=resume)
                state = verify_checkpoint(output)
                completed = state["epoch"] + 1
                if result["reason"] == "time-cap":
                    if completed <= before:
                        raise RuntimeError(
                            "Training budget completed no new epoch; inspect before extending limits"
                        )
                    record(
                        "training-time-cap",
                        epochs_complete=completed,
                        target_epochs=train_cfg["epochs"],
                    )
                    resume = True
                    continue
                if result["reason"] in ("epochs-complete", "early-stopping"):
                    return record(
                        "completed",
                        summary=result,
                        quality_eligible=bool(result.get("eligible_checkpoint")),
                    )
                return record("user-paused", summary=result)
            raise RuntimeError("Training invocation limit reached")
        except Exception as error:
            record("failed", error_type=type(error).__name__, error=str(error))
            raise
