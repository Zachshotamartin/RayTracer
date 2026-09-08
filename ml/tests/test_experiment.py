"""Exercise resource continuation and stop behavior without a large rendering job."""

import json
from types import SimpleNamespace

import pytest

from raytracer_ml import experiment
from raytracer_ml.io import digest, write_json


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    prep, data, run = (tmp_path / name for name in ("preparation", "data", "run"))
    prep.mkdir()
    data_cfg, train_cfg = prep / "data.json", prep / "train.json"
    write_json(data_cfg, {"max_seconds": 5})
    write_json(train_cfg, {"epochs": 2})
    plan = {
        "preparation": str(prep),
        "data": str(data),
        "output": str(run),
        "source_commit": "fixture",
        "data_config": str(data_cfg),
        "train_config": str(train_cfg),
        "renderer": "fixture-renderer",
        "min_free_gib": 0,
        "authorization": "Test fixture",
    }
    path = prep / "plan.json"
    write_json(path, plan)
    monkeypatch.setattr(experiment, "verify_plan", lambda p: None)
    calls = {"generation": 0, "train_resume": []}
    state = {"epoch": 0}

    def generate(cfg, binary, root, dry_run=False):
        if dry_run:
            return {"examples": 2}
        calls["generation"] += 1
        root.mkdir(exist_ok=True)
        (root / "manifest.jsonl").write_text("fixture\n")
        return {"completed": 2, "manifest_sha256": digest(root / "manifest.jsonl")}

    def prepare(cfg, root, output, report):
        value = {
            "optimizer_updates": 0,
            "config": cfg,
            "validation": {"manifest_sha256": digest(root / "manifest.jsonl")},
        }
        write_json(report, value)
        return value

    def train(cfg, root, output, resume=False):
        calls["train_resume"].append(resume)
        state["epoch"] = 1
        return {"reason": "epochs-complete", "epochs_complete": 2, "eligible_checkpoint": None}

    monkeypatch.setattr(experiment, "generate", generate)
    monkeypatch.setattr(experiment, "prepare_training", prepare)
    monkeypatch.setattr(experiment, "train", train)
    monkeypatch.setattr(experiment, "verify_checkpoint", lambda output: state)
    return path, prep, data, calls, state


def test_pipeline_reaches_training_only_after_preflight(pipeline):
    path, prep, _, calls, _ = pipeline
    result = experiment.run_experiment(path)
    assert result["phase"] == "completed" and not result["quality_eligible"]
    assert calls == {"generation": 1, "train_resume": [False]}
    events = [json.loads(s)["phase"] for s in (prep / "events.jsonl").read_text().splitlines()]
    assert events == ["generating", "validating-data-and-device", "training", "completed"]
    assert json.loads((prep / "preflight.json").read_text())["optimizer_updates"] == 0
    with pytest.raises(ValueError, match="inspect before restarting"):
        experiment.run_experiment(path)


def test_clean_training_time_cap_resumes_without_pausing_for_review(pipeline, monkeypatch):
    path, prep, _, calls, state = pipeline

    def train(cfg, root, output, resume=False):
        calls["train_resume"].append(resume)
        state["epoch"] = 1 if resume else 0
        return {
            "reason": "epochs-complete" if resume else "time-cap",
            "epochs_complete": state["epoch"] + 1,
        }

    monkeypatch.setattr(experiment, "train", train)
    assert experiment.run_experiment(path)["phase"] == "completed"
    assert calls["train_resume"] == [False, True]
    assert not (prep / "STOP_AFTER_EPOCH").exists()


def test_user_pause_and_runtime_errors_are_not_automatically_retried(pipeline, monkeypatch):
    path, prep, _, calls, _ = pipeline

    def pause(*args, **kwargs):
        calls["train_resume"].append(kwargs["resume"])
        return {"reason": "pause-requested", "epochs_complete": 1}

    monkeypatch.setattr(experiment, "train", pause)
    assert experiment.run_experiment(path)["phase"] == "user-paused"
    assert calls["train_resume"] == [False]
    with pytest.raises(ValueError, match="inspect before restarting"):
        experiment.run_experiment(path)


def test_generation_time_cap_needs_progress_and_real_elapsed_budget(pipeline, monkeypatch):
    path, prep, data, calls, _ = pipeline
    real_generate = experiment.generate
    clock = iter([0, 6, 6])
    monkeypatch.setattr(
        experiment, "time", SimpleNamespace(time=lambda: 100, monotonic=lambda: next(clock))
    )

    def generate(cfg, binary, root, dry_run=False):
        if not dry_run and not (data / "examples/first.json").exists():
            calls["generation"] += 1
            (data / "examples").mkdir(parents=True)
            write_json(data / "examples/first.json", {})
            raise TimeoutError("time cap")
        return real_generate(cfg, binary, root, dry_run)

    monkeypatch.setattr(experiment, "generate", generate)
    assert experiment.run_experiment(path)["phase"] == "completed"
    assert calls["generation"] == 2
    assert "generation-time-cap" in (prep / "events.jsonl").read_text()


def test_unexpected_generation_timeout_fails_without_launching_training(pipeline, monkeypatch):
    path, prep, _, calls, _ = pipeline
    real_generate = experiment.generate
    monkeypatch.setattr(experiment, "time", SimpleNamespace(time=lambda: 100, monotonic=lambda: 0))

    def generate(cfg, binary, root, dry_run=False):
        if dry_run:
            return real_generate(cfg, binary, root, dry_run)
        calls["generation"] += 1
        raise TimeoutError("unexpected renderer timeout")

    monkeypatch.setattr(experiment, "generate", generate)
    with pytest.raises(RuntimeError, match="Unexpected timeout"):
        experiment.run_experiment(path)
    assert calls["generation"] == 1 and calls["train_resume"] == []
    assert json.loads((prep / "state.json").read_text())["phase"] == "failed"


def test_controller_rejects_changed_pinned_source(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **kw: "different\n")
    with pytest.raises(ValueError, match="revision changed"):
        experiment.verify_plan({"source": str(tmp_path), "source_commit": "expected"})


def test_controller_does_not_retry_failed_integrity_preflight(pipeline, monkeypatch):
    path, prep, _, calls, _ = pipeline

    def invalid(*args):
        raise ValueError("Dataset file checksum mismatch")

    monkeypatch.setattr(experiment, "prepare_training", invalid)
    with pytest.raises(ValueError, match="checksum mismatch"):
        experiment.run_experiment(path)
    assert calls["train_resume"] == []
    assert json.loads((prep / "state.json").read_text())["phase"] == "failed"
