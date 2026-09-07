import pytest
import torch

from raytracer_ml.checkpoint_selection import (
    update_selection,
    selected_checkpoint,
    publish_selection,
    restore_selection,
)
from raytracer_ml.selection import checkpoint_score, constraint_report, validate_selection_config
from raytracer_ml.train import load_checkpoint, save_checkpoint


def snapshot(epoch, score, eligible, weight):
    return dict(
        epoch=epoch,
        selection_score=score,
        validation_loss=score,
        validation_constraints_pass=eligible,
        model={"weight": weight},
        contract="fixture",
    )


def test_eligible_checkpoint_survives_better_failing_scores_and_alias_corruption(tmp_path):
    weight = torch.tensor([1.0])
    state, _ = update_selection({}, snapshot(0, 0.015, False, weight), True)
    weight.fill_(2)
    state, improved = update_selection(state, snapshot(1, 0.03, True, weight), True)
    assert improved and selected_checkpoint(state)["epoch"] == 1
    weight.fill_(3)
    state, improved = update_selection(state, snapshot(2, 0.01, False, weight), True)
    assert not improved and state["best_score"]["epoch"] == 2
    torch.testing.assert_close(selected_checkpoint(state)["model"]["weight"], torch.tensor([2.0]))
    save_checkpoint(tmp_path / "latest.pt", {"selection_state": state})
    publish_selection(state, tmp_path, save_checkpoint)
    (tmp_path / "best.pt").write_bytes(b"interrupted alias publication")
    (tmp_path / "best_eligible.pt").unlink()
    recovered = restore_selection(
        load_checkpoint(tmp_path / "latest.pt"), tmp_path, load_checkpoint, True
    )
    publish_selection(recovered, tmp_path, save_checkpoint)
    assert load_checkpoint(tmp_path / "best.pt")["epoch"] == 1
    assert load_checkpoint(tmp_path / "best_score.pt")["epoch"] == 2
    assert load_checkpoint(tmp_path / "best_eligible.pt")["epoch"] == 1
    recovered, improved = update_selection(recovered, snapshot(3, 0.02, True, weight), True)
    assert improved and selected_checkpoint(recovered)["epoch"] == 3


def test_no_eligible_model_remains_explicitly_diagnostic_and_legacy_resume_is_checked(tmp_path):
    weight = torch.ones(1)
    state, _ = update_selection({}, snapshot(0, 0.01, False, weight), True)
    (tmp_path / "best_eligible.pt").write_bytes(b"stale alias from an uncommitted epoch")
    publish_selection(state, tmp_path, save_checkpoint)
    assert not (tmp_path / "best_eligible.pt").exists()
    assert not load_checkpoint(tmp_path / "best.pt")["validation_constraints_pass"]
    legacy = {**snapshot(1, 0.01, True, weight), "best": 0.01, "best_loss": 0.008}
    save_checkpoint(tmp_path / "best.pt", legacy)
    restored = restore_selection(dict(contract="fixture", epoch=2), tmp_path, load_checkpoint, True)
    assert selected_checkpoint(restored)["validation_loss"] == 0.008
    for current in (dict(contract="other", epoch=2), dict(contract="fixture", epoch=0)):
        with pytest.raises(ValueError, match="committed resume epoch"):
            restore_selection(current, tmp_path, load_checkpoint, True)


def test_all_requested_quality_gates_fail_closed_with_explicit_coverage():
    cfg = dict(
        hdr_ratio=1.01,
        model_hdr_ratio=1.01,
        preservation_ratio=1.02,
        temporal_ratio=1.02,
        min_preservation_views=4,
        min_temporal_transitions=8,
    )
    baseline = dict(
        ssim=0.96,
        edge=0.04,
        hdr_mse=0.2,
        model_hdr_mse=0.01,
        preservation_log_mae=0.02,
        temporal_log_mae=0.03,
    )
    good = {**baseline, "preservation_views": 4, "temporal_transitions": 8}
    assert checkpoint_score(0.03, good, baseline, cfg)[1]
    for metric in ("hdr_mse", "model_hdr_mse", "preservation_log_mae", "temporal_log_mae"):
        for bad in (
            {**good, metric: good[metric] * 2},
            {k: v for k, v in good.items() if k != metric},
        ):
            assert not checkpoint_score(0.001, bad, baseline, cfg)[1]
    for metric in ("preservation_views", "temporal_transitions"):
        assert not checkpoint_score(0.001, {**good, metric: good[metric] - 1}, baseline, cfg)[1]
    report = constraint_report(good, baseline, cfg)
    assert report["preservation_ratio"]["baseline"] == 0.02
    assert report["hdr_ratio"]["limit"] == pytest.approx(0.202)
    # A noisy HDR regression cannot be hidden behind a structural pass or lower loss.
    assert not checkpoint_score(0.001, {**good, "ssim": 0.94}, baseline, cfg)[1]
    for bad in (
        {"hdr_rato": 1},
        {"hdr_ratio": float("nan")},
        {"edge_ratio": -1},
        {"min_temporal_transitions": 8},
        {"temporal_ratio": 1, "min_temporal_transitions": 1.5},
    ):
        with pytest.raises(ValueError):
            validate_selection_config(bad)
