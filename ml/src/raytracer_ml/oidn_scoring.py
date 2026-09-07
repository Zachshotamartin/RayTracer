"""Score upstream linear EXRs under the same display/region metrics as custom models."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data.arrays import load_example
from .data.validate import validate
from .diagnostics import detail_metrics, grouped_summary
from .io import digest, manifest, safe_path, save_arrays, write_json, write_png
from .metrics import image_metrics


def score_outputs(data, exported, predictions, selection, output, split="val", suffix="trained"):
    import OpenImageIO as oiio

    data, exported, predictions, output = map(Path, (data, exported, predictions, output))
    audit = validate(data)
    receipt = json.loads((exported / "export.json").read_text())
    if receipt["descriptor"]["source_manifest_sha256"] != audit["manifest_sha256"]:
        raise ValueError("Export and source dataset identities differ")
    rows = {r["id"]: r for r in manifest(data / "manifest.jsonl") if r["split"] == split}
    entries = [r for r in receipt["records"] if r["split"] == split]
    if not rows or len(entries) != len(rows) or {r["id"] for r in entries} != set(rows):
        raise ValueError("Export does not contain exactly the requested split")
    # Check every expected output before computing any summary, so missing cases fail loudly.
    paths = {}
    for entry in entries:
        row = rows[entry["id"]]
        if (
            entry["source_sha256"] != row["sha256"]
            or entry["reference_sha256"] != row["reference_sha256"]
        ):
            raise ValueError("Source-to-export pairing changed")
        path = safe_path(predictions, entry["input"] + f".{suffix}.hdr.exr")
        if not path.is_file():
            raise ValueError(f"Missing prediction: {entry['id']}")
        paths[entry["id"]] = path
    output.mkdir(parents=True, exist_ok=False)
    results = []
    method = "oidn_trained"
    for index, row in enumerate(rows.values()):
        reader = oiio.ImageInput.open(str(paths[row["id"]]))
        if not reader:
            raise ValueError("Cannot read upstream prediction")
        try:
            prediction = reader.read_image(format=oiio.FLOAT)
        finally:
            reader.close()
        if prediction is None or not np.isfinite(prediction).all() or np.any(prediction < 0):
            raise ValueError("Invalid upstream radiance")
        with np.load(safe_path(data, row["reference"]), allow_pickle=False) as ref:
            target = ref["target"]
            reference_features = ref.get("features")
            annotations = ref.get("annotations")
        measured = load_example(data, row)
        x, atrous = measured["features"], measured["atrous"]
        raw = x[:3].transpose(1, 2, 0)
        methods = {"raw": raw, "atrous": atrous, method: prediction}
        metrics = {}
        for name, pixels in methods.items():
            metrics[name] = {
                **image_metrics(pixels, target),
                **detail_metrics(pixels, target, reference_features, annotations),
            }
            support = x[11] >= 0.999999
            highlights = target.max(2) > 1
            for region, mask in (("supported", support), ("highlight", highlights)):
                metrics[name][region + "_linear_mse"] = (
                    float(np.mean((pixels[mask] - target[mask]) ** 2)) if mask.any() else None
                )
        results.append(
            dict(
                id=row["id"],
                group=row["group"],
                samples=row["samples"],
                stratum=row.get("stratum", "room"),
                metrics=metrics,
                prediction_sha256=digest(paths[row["id"]]),
            )
        )
        save_arrays(output / "predictions" / (row["id"] + ".npz"), prediction=prediction)
        if index < 12:
            write_png(
                output / "comparisons" / (row["id"] + ".png"),
                np.concatenate([raw, atrous, prediction, target], axis=1),
            )
    keys = [
        "linear_mse",
        "log_mae",
        "psnr",
        "ssim",
        "edge_2px_gradient_mae",
        "halo_display_mae",
        "highlight_linear_mse",
        "dark_linear_mse",
        "thin_gradient_mae",
        "linear_energy_relative_bias",
        "corner_displacement_p95_px",
    ]
    summary = dict(
        split=split,
        images=len(results),
        groups=len({r["group"] for r in results}),
        manifest_sha256=audit["manifest_sha256"],
        export_sha256=digest(exported / "export.json"),
        selection=json.loads(Path(selection).read_text()),
        inference_timing_measured=False,
        display="ACES-fit + sRGB, exposure 0",
        methods={},
        distributions={},
        worst_cases={},
    )
    for name in ("raw", "atrous", method):
        summary["distributions"][name] = {key: grouped_summary(results, name, key) for key in keys}
        summary["methods"][name] = {
            key: summary["distributions"][name][key]["mean"]
            for key in ("linear_mse", "log_mae", "psnr", "ssim")
        }
    for key in ("linear_mse", "edge_2px_gradient_mae", "halo_display_mae"):
        cases = [r for r in results if r["metrics"][method].get(key) is not None]
        worst = sorted(cases, key=lambda r: r["metrics"][method][key], reverse=True)[:8]
        summary["worst_cases"][key] = [r["id"] for r in worst]
        for case in worst:
            row = rows[case["id"]]
            with np.load(output / "predictions" / (row["id"] + ".npz"), allow_pickle=False) as p:
                prediction = p["prediction"]
            with np.load(safe_path(data, row["reference"]), allow_pickle=False) as p:
                target = p["target"]
            write_png(
                output / "worst_cases" / (key + "-" + row["id"] + ".png"),
                np.concatenate([prediction, target, np.abs(prediction - target) * 4], axis=1),
            )
    (output / "per_image.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in results)
    )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "exported", "predictions", "selection", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--suffix", default="trained")
    args = parser.parse_args()
    summary = score_outputs(**vars(args))
    print(json.dumps(summary["methods"], indent=2))


if __name__ == "__main__":
    main()
