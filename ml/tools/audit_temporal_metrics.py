"""Compare flicker of retained development predictions and baselines without retraining."""

import argparse
import json
from pathlib import Path

import numpy as np

from raytracer_ml.data.arrays import load_example
from raytracer_ml.data.validate import validate
from raytracer_ml.diagnostics import grouped_summary
from raytracer_ml.io import digest, manifest, safe_path, write_json
from raytracer_ml.temporal_metrics import TemporalComparison
from raytracer_ml.preprocessing import model_schema
from raytracer_ml.train import load_checkpoint


def audit(root, evaluation, checkpoint, output):
    root, evaluation, checkpoint, output = map(Path, (root, evaluation, checkpoint, output))
    if output.exists():
        raise ValueError("Use a new output file to preserve previous evidence")
    source = json.loads((evaluation / "summary.json").read_text())
    if source["split"] not in ("train", "val"):
        raise ValueError("Development audits must not inspect sealed test data")
    validation = validate(root)
    if source["manifest_sha256"] != validation["manifest_sha256"] or source[
        "checkpoint_sha256"
    ] != digest(checkpoint):
        raise ValueError("Evaluation, checkpoint and dataset identities must match")
    state = load_checkpoint(checkpoint)
    if state["config"]["model"].get("scale", 1) != 1 or not state["config"]["model"].get(
        "temporal"
    ):
        raise ValueError("Audit requires a same-resolution temporal checkpoint")
    rows = [r for r in manifest(root / "manifest.jsonl") if r["split"] == source["split"]]
    if len(rows) != source["images"]:
        raise ValueError("Audit requires the complete recorded development split")
    comparison, records = TemporalComparison(), []
    channels = 27 if model_schema(state["config"]["model"]) == 2 else 17
    for row in rows:
        arrays = load_example(root, row)
        features = arrays["features"][:channels]
        with np.load(safe_path(root, row["reference"]), allow_pickle=False) as ref:
            target = ref["target"]
        path = evaluation / "predictions" / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as saved:
            prediction = saved["prediction"]
        if (
            prediction.shape != target.shape
            or not np.isfinite(prediction).all()
            or np.any(prediction < 0)
        ):
            raise ValueError(f"Invalid retained prediction: {row['id']}")
        methods = dict(
            raw=features[:3].transpose(1, 2, 0),
            atrous=arrays["atrous"],
            neural=prediction,
        )
        records.append(
            dict(
                id=row["id"],
                group=row["group"],
                samples=row["samples"],
                prediction_sha256=digest(path),
                metrics=comparison.measure(row, features, arrays["position"], target, methods),
            )
        )
    names = ("raw", "atrous", "neural")
    metrics = ("temporal_linear_mae", "temporal_log_mae")
    package = Path(__import__("raytracer_ml").__file__).parent
    result = dict(
        temporal_metric_schema=1,
        split=source["split"],
        images=len(rows),
        groups=len({r["group"] for r in rows}),
        checkpoint_sha256=digest(checkpoint),
        manifest_sha256=validation["manifest_sha256"],
        reference_checks=validation["reference_checks"],
        source_evaluation_sha256=digest(evaluation / "summary.json"),
        audit_source_sha256=digest(Path(__file__)),
        package_sources={str(p.relative_to(package)): digest(p) for p in package.rglob("*.py")},
        methods={n: {key: grouped_summary(records, n, key) for key in metrics} for n in names},
        budgets={
            str(b): {
                n: {
                    key: grouped_summary([r for r in records if r["samples"] == b], n, key)
                    for key in metrics
                }
                for n in names
            }
            for b in sorted({r["samples"] for r in rows})
        },
        coverage={
            n: dict(
                measured_transitions=sum(
                    r["metrics"][n]["temporal_log_mae"] is not None for r in records
                ),
                confidence_sum=sum(r["metrics"][n]["temporal_confidence_sum"] for r in records),
                valid_pixels=sum(r["metrics"][n]["temporal_valid_pixels"] for r in records),
            )
            for n in names
        },
        per_image=records,
        interpretation="Confidence-weighted reference-corrected changes, identical input-geometry masks "
        "for all methods. Cuts and unmatched frames are missing measurements. Means/bootstrap intervals "
        "are grouped by scene. No inference timing, new training, reference-noise subtraction, dynamic-object "
        "qualification or release approval is implied.",
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "evaluation", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data, args.evaluation, args.checkpoint, args.output)
    print(json.dumps({k: result[k] for k in ("images", "groups", "methods", "coverage")}, indent=2))


if __name__ == "__main__":
    main()
