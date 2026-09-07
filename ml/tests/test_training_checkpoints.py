"""Checkpoint recovery uses tiny synthetic CPU fixtures, never the full render collection."""

import importlib
import json
import shutil

import pytest
import torch
from filelock import FileLock

from raytracer_ml.io import digest, write_json
from raytracer_ml.train import load_checkpoint, train
from raytracer_ml.training_checkpoints import list_checkpoints


@pytest.fixture
def synthetic_training(tmp_path, monkeypatch):
    module = importlib.import_module("raytracer_ml.train")
    root = tmp_path / "data"
    root.mkdir()
    write_json(root / "dataset.json", {"config": {"scale": 1}})
    monkeypatch.setattr(module, "validate", lambda root: {"manifest_sha256": "synthetic"})

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, root, split, *args, **kwargs):
            self.rows = [{"scene_sha256": split, "samples": 4, "id": "fixture-n0-s4"}] * 2

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            x = torch.rand(27, 8, 8)
            x[10:12], x[26], x[15] = 1, 1, 4
            return x, torch.full((3, 8, 8), 0.3)

    monkeypatch.setattr(module, "RenderDataset", TinyDataset)
    cfg = dict(
        seed=71,
        device="cpu",
        cpu_threads=1,
        model=dict(kind="guided", width=4, feature_schema=2),
        crop=8,
        batch_size=1,
        epochs=3,
        patience=8,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_seconds=120,
        checkpoint_history=True,
    )
    return root, cfg, module


def assert_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_state_equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    else:
        assert a == b


def assert_training_equal(a, b):
    for key in ("model", "optimizer", "scheduler", "rng", "step", "epoch", "best", "stale"):
        assert_state_equal(a[key], b[key])


def test_archived_epoch_and_best_can_restart_as_standalone_files(synthetic_training, tmp_path):
    root, cfg, _ = synthetic_training
    whole = tmp_path / "whole"
    train(cfg, root, whole)
    original = load_checkpoint(whole / "latest.pt")
    before = {p.name: digest(p) for p in whole.glob("*.pt")}
    for source in [whole / "checkpoints/epoch-000001.pt", whole / "best_resume.pt"]:
        copied = tmp_path / f"copied-{source.name}"
        shutil.copyfile(source, copied)
        destination = tmp_path / f"fork-{source.stem}"
        train(cfg, root, destination, resume_from=copied)
        assert_training_equal(original, load_checkpoint(destination / "latest.pt"))
        lineage = json.loads((destination / "lineage.json").read_text())
        assert lineage["source_sha256"] == digest(copied)
    assert before == {p.name: digest(p) for p in whole.glob("*.pt")}
    assert len(list(whole.glob("checkpoints/epoch-*.pt"))) == 3
    listing = list_checkpoints(whole, load_checkpoint)
    assert all(r["full_training_state"] for r in listing["checkpoints"])
    assert listing["checkpoints"][0]["epoch_completed"] == 3


def test_best_full_state_is_frozen_and_repaired_from_latest(synthetic_training, tmp_path):
    root, cfg, module = synthetic_training
    scores = iter(((0.03, True), (0.04, True), (0.05, True)))
    monkey = pytest.MonkeyPatch()
    monkey.setattr(module, "checkpoint_score", lambda *a: next(scores))
    out = tmp_path / "run"
    try:
        train(cfg, root, out, max_new_epochs=1)
        best = load_checkpoint(out / "best_resume.pt")
        train(cfg, root, out, resume=True, max_new_epochs=1)
        assert_training_equal(best, load_checkpoint(out / "best_resume.pt"))
        assert load_checkpoint(out / "latest.pt")["epoch"] == 1
        (out / "best_resume.pt").write_bytes(b"interrupted alias publication")
        (out / "checkpoints/epoch-000002.pt").unlink()
        (out / "STOP_AFTER_EPOCH").touch()
        result = train(cfg, root, out, resume=True)
        assert result["reason"] == "pause-requested"
        assert_training_equal(best, load_checkpoint(out / "best_resume.pt"))
        assert (out / "checkpoints/epoch-000002.pt").is_file()
    finally:
        monkey.undo()


def test_safe_pause_and_latest_resume_match_uninterrupted(
    synthetic_training, tmp_path, monkeypatch
):
    root, cfg, module = synthetic_training
    whole, paused = tmp_path / "whole", tmp_path / "paused"
    train(cfg, root, whole)
    original_publish = module.publish_selection

    def request_pause(selection, output, save, updated_epoch=None):
        original_publish(selection, output, save, updated_epoch)
        if output == paused and updated_epoch == 0:
            (output / "STOP_AFTER_EPOCH").touch()

    monkeypatch.setattr(module, "publish_selection", request_pause)
    assert train(cfg, root, paused)["reason"] == "pause-requested"
    assert not (paused / "STOP_AFTER_EPOCH").exists()
    train(cfg, root, paused, resume=True)
    assert_training_equal(
        load_checkpoint(whole / "latest.pt"), load_checkpoint(paused / "latest.pt")
    )


def test_rejects_unsafe_resume_and_concurrent_writers(synthetic_training, tmp_path):
    root, cfg, _ = synthetic_training
    out = tmp_path / "run"
    train(cfg, root, out, max_new_epochs=1)
    with pytest.raises(ValueError, match="full training state"):
        train(cfg, root, tmp_path / "bad", resume_from=out / "best.pt")
    with pytest.raises(ValueError, match="already exists"):
        train(cfg, root, out, resume_from=out / "latest.pt")
    with pytest.raises(ValueError, match="differs"):
        train({**cfg, "seed": 9}, root, tmp_path / "different", resume_from=out / "latest.pt")
    with pytest.raises(ValueError, match="positive integer"):
        train(cfg, root, out, resume=True, max_new_epochs=0)
    with FileLock(out / ".training.lock"):
        with pytest.raises(ValueError, match="Another process"):
            train(cfg, root, out, resume=True)
    archive = out / "checkpoints/epoch-000001.pt"
    archive.write_bytes(archive.read_bytes() + b"corruption")
    before = digest(out / "latest.pt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        train(cfg, root, out, resume=True)
    assert digest(out / "latest.pt") == before
