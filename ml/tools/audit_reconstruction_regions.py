"""Audit saved validation predictions by actual model/fallback policy, without inference."""

import argparse
import json
from pathlib import Path

import numpy as np

from raytracer_ml.data.arrays import load_example
from raytracer_ml.diagnostics import reconstruction_region_metrics
from raytracer_ml.io import digest, manifest, safe_path, write_json
from raytracer_ml.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "evaluation", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file to preserve previous evidence")
    source = json.loads((args.evaluation / "summary.json").read_text())
    if source["split"] not in ("train", "val"):
        parser.error("Development audits must not inspect a sealed test split")
    if source["manifest_sha256"] != digest(args.data / "manifest.jsonl") or source[
        "checkpoint_sha256"
    ] != digest(args.checkpoint):
        parser.error("Evaluation, checkpoint and dataset identities must match")
    state = load_checkpoint(args.checkpoint)
    if state["config"]["model"].get("scale", 1) != 1:
        parser.error("This saved-prediction audit currently requires scale 1")
    examples = [r for r in manifest(args.data / "manifest.jsonl") if r["split"] == source["split"]]
    if len(examples) != source["images"]:
        parser.error("Audit requires the complete recorded development split")
    records = []
    for row in examples:
        arrays = load_example(args.data, row)
        with np.load(safe_path(args.data, row["reference"]), allow_pickle=False) as data:
            target = data["target"]
        path = args.evaluation / "predictions" / f"{row['id']}.npz"
        with np.load(path, allow_pickle=False) as data:
            prediction = data["prediction"]
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid saved prediction: {row['id']}")
        methods = dict(
            raw=arrays["features"][:3].transpose(1, 2, 0),
            atrous=arrays["atrous"],
            neural=prediction,
        )
        records.append(
            dict(
                id=row["id"],
                group=row["group"],
                prediction_sha256=digest(path),
                methods={
                    name: reconstruction_region_metrics(
                        value, target, arrays["features"], state["config"]["model"]
                    )
                    for name, value in methods.items()
                },
            )
        )
    means = {}
    for method in ("raw", "atrous", "neural"):
        means[method] = {}
        for key in records[0]["methods"][method]:
            values = [r["methods"][method][key] for r in records]
            values = [v for v in values if v is not None]
            means[method][key] = float(np.mean(values)) if values else None
    write_json(
        args.output,
        dict(
            region_metric_schema=2,
            images=len(records),
            groups=len({r["group"] for r in examples}),
            split=source["split"],
            checkpoint_sha256=source["checkpoint_sha256"],
            manifest_sha256=source["manifest_sha256"],
            methods=means,
            per_image=records,
            interpretation="All baselines share the custom model's input-defined policy. Contributions "
            "add to whole-image MSE; conditional regional means do not. No reference-noise subtraction "
            "or quality qualification is implied. Existing independent checks saved only scalar "
            "disagreement, so their regional noise floor cannot be reconstructed from those receipts.",
        ),
    )
    print(json.dumps(dict(images=len(records), methods=means), indent=2))


if __name__ == "__main__":
    main()
