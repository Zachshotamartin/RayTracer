import json
from raytracer_ml.status import status


def test_progress_counts_ignore_incomplete_appends_and_do_not_claim_validation(tmp_path):
    data = tmp_path / "datasets/example"
    data.mkdir(parents=True)
    (data / "dataset.json").write_text(
        json.dumps(dict(estimate=dict(examples=4, reference_images=2)))
    )
    (data / "manifest.jsonl").write_text('{"id":"complete"}\n{"id":"partial')
    result = status(tmp_path)
    row = result["datasets"][0]
    assert row["examples_complete"] == 1 and row["examples_percent"] == 25
    assert row["partial_append_observed"] and not row["integrity_rechecked"]
    assert not row["counts_complete"]


def test_status_handles_upstream_checkpoints_without_claiming_run_completion(tmp_path):
    run = tmp_path / "runs/oidn/example"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(model="unet", num_epochs=200)))
    (checkpoints / "latest").write_text("15")
    (checkpoints / "checkpoint_15.pth").write_bytes(b"not loaded by progress reader")
    other = tmp_path / "runs/unrelated"
    other.mkdir()
    (other / "config.json").write_text('{"format":"tool-settings"}')
    result = status(tmp_path)
    assert len(result["runs"]) == 1
    row = result["runs"][0]
    assert row["toolkit"] == "oidn" and row["epochs_complete"] == 15
    assert row["epoch_cap_percent"] == 7.5 and row["stop_reason"] is None
    assert row["validation_constraints_pass"] is None
    assert not row["checkpoint_integrity_rechecked"]
    (checkpoints / "latest").write_text("20")
    assert status(tmp_path)["runs"][0]["epochs_complete"] == 0  # no corresponding saved file


def test_progress_exposes_ineffective_learning_without_inferring_completion(tmp_path):
    run = tmp_path / "runs/conditioning/control"
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps(dict(epochs=50)))
    health = dict(loss_reduction_vs_raw=0.001, warning="little-improvement-over-input")
    (run / "metrics.jsonl").write_text(json.dumps(dict(epoch=4, learning_health=health)) + "\n")
    row = status(tmp_path)["runs"][0]
    assert row["learning_health"] == health
    assert row["epochs_complete"] == 5
    assert row["stop_reason"] is None and row["validation_constraints_pass"] is None
