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
