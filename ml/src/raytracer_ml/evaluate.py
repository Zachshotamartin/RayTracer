from .preprocessing import model_schema
import json
import time
from pathlib import Path
import numpy as np
import torch
from .io import manifest, safe_path, write_json, write_png, save_arrays, digest
from .metrics import image_metrics
from .models import build_model
from .train import load_checkpoint, device_for, synchronize
from .data.validate import validate
from .diagnostics import detail_metrics, grouped_summary, reconstruction_region_metrics
from .evaluation_contract import authorize
from .data.arrays import load_example


def evaluate(
    root,
    checkpoint,
    output,
    split="test",
    device_name="auto",
    repeats=3,
    oidn=None,
    oidn_device="cpu",
    registration=None,
):
    root, output = Path(root), Path(output)
    validation = validate(root)
    state = load_checkpoint(checkpoint)
    evaluation_scope = authorize(state, checkpoint, validation, registration)
    device = device_for(device_name)
    torch.set_num_threads(2)
    model = build_model(state["config"]["model"]).to(device).eval()
    model.load_state_dict(state["model"])
    rows = [r for r in manifest(root / "manifest.jsonl") if r["split"] == split]
    method_names = ["raw", "atrous", "neural"] + (["oidn"] if oidn else [])
    histories = {}
    history_group = None
    temporal = state["config"]["model"].get("temporal", False)
    base_channels = 27 if model_schema(state["config"]["model"]) == 2 else 17
    results = []
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index, r in enumerate(rows):
            data = load_example(root, r)
            x = data["features"].copy()
            if x.shape[0] < base_channels:
                raise ValueError("Evaluation data lacks model boundary features")
            x = x[:base_channels]
            atrous = data["atrous"].copy()
            position = data["position"].copy()
            with np.load(safe_path(root, r["reference"])) as data:
                target = data["target"].copy()
                reference_features = data["features"].copy() if "features" in data else None
                annotations = data["annotations"].copy() if "annotations" in data else None
            temporal_start = time.perf_counter()
            model_input = x
            history_fraction = 0.0
            temporal_residual = None
            key = (r["samples"], r["id"].split("-n")[1].split("-s")[0])
            if r["group"] != history_group:
                histories.clear()
                history_group = r["group"]
            previous = histories.get(key)
            if temporal:
                from .temporal import reproject, append_history

                history = np.zeros((*position.shape[:2], 3), dtype=np.float32)
                mask = np.zeros((*position.shape[:2], 1), dtype=np.float32)
                if previous and previous["frame"] + 1 == r["frame"]:
                    history, mask = reproject(
                        position,
                        x,
                        r["scene"],
                        previous["position"],
                        previous["features"],
                        previous["scene"],
                        previous["prediction"],
                    )
                model_input = append_history(x, history, mask)
                history_fraction = float(mask.mean())
            temporal_seconds = time.perf_counter() - temporal_start if temporal else 0.0
            timings = []
            for iteration in range(repeats + 1):
                synchronize(device)
                start = time.perf_counter()
                y = (
                    model(torch.from_numpy(model_input[None]).to(device))
                    .cpu()
                    .numpy()[0]
                    .transpose(1, 2, 0)
                    .copy()
                )
                synchronize(device)
                elapsed = time.perf_counter() - start
                if iteration == 0:
                    cold = elapsed
                else:
                    timings.append(elapsed)
            if temporal and previous and mask.any():
                reference_history, _ = reproject(
                    position,
                    x,
                    r["scene"],
                    previous["position"],
                    previous["features"],
                    previous["scene"],
                    previous["target"],
                )
                valid = mask[:, :, 0] > 0
                temporal_residual = float(
                    np.mean(np.abs((y - history)[valid] - (target - reference_history)[valid]))
                )
            histories[key] = {
                "frame": r["frame"],
                "position": position,
                "features": x,
                "scene": r["scene"],
                "prediction": y,
                "target": target,
            }
            raw = x[:3].transpose(1, 2, 0)
            if r["scale"] != 1:

                def resize(a):
                    return (
                        torch.nn.functional.interpolate(
                            torch.from_numpy(a.transpose(2, 0, 1)[None]),
                            size=target.shape[:2],
                            mode="bilinear",
                            align_corners=False,
                        )
                        .numpy()[0]
                        .transpose(1, 2, 0)
                    )

                raw, atrous = resize(raw), resize(atrous)
            methods = {"raw": raw, "atrous": atrous, "neural": y}
            oidn_seconds = None
            if oidn:
                from .oidn import denoise_oidn

                oidn_image, oidn_seconds, oidn_log = denoise_oidn(x, oidn, oidn_device)
                if r["scale"] != 1:
                    oidn_image = resize(oidn_image)
                methods["oidn"] = oidn_image

            metrics = {
                name: {
                    **image_metrics(image, target),
                    **detail_metrics(image, target, reference_features, annotations),
                    **reconstruction_region_metrics(image, target, x, state["config"]["model"]),
                }
                for name, image in methods.items()
            }
            for name, image in methods.items():
                metrics[name]["supported_linear_mse"] = metrics[name]["model_region_linear_mse"]
                metrics[name]["highlight_linear_mse"] = (
                    float(np.mean((image[target.max(2) > 1] - target[target.max(2) > 1]) ** 2))
                    if (target.max(2) > 1).any()
                    else None
                )
            result = {
                "id": r["id"],
                "samples": r["samples"],
                "cohort": r["cohort"],
                "group": r["group"],
                "stratum": r.get("stratum", "room"),
                "metrics": metrics,
                "first_call_seconds": cold,
                "temporal_preprocess_seconds": temporal_seconds,
                "history_valid_fraction": history_fraction,
                "temporal_residual_mae": temporal_residual,
                "oidn_process_seconds": oidn_seconds,
                "inference_median_seconds": float(np.median(timings)),
                "inference_p95_seconds": float(np.percentile(timings, 95)),
                "render_seconds": r["stats"]["render_seconds"],
                "bvh_seconds": r["stats"]["bvh_build_seconds"],
            }
            results.append(result)
            save_arrays(output / "predictions" / f"{r['id']}.npz", prediction=y)
            if index < 12:
                write_png(
                    output / "comparisons" / f"{r['id']}.png",
                    np.concatenate([raw, atrous, y, target], axis=1),
                )
                write_png(output / "errors" / f"{r['id']}.png", np.abs(y - target) * 4)
    temp = output / "per_image.jsonl"
    temp.write_text("".join(json.dumps(r) + "\n" for r in results))
    summary = {
        "region_metric_schema": 2,
        "region_metric_policy": "Current custom-model input support and sample bypass; shared across methods",
        "evaluation_scope": evaluation_scope,
        "split": split,
        "images": len(results),
        "device": str(device),
        "checkpoint_sha256": digest(checkpoint),
        "manifest_sha256": validation["manifest_sha256"],
        "methods": {
            name: {
                key: float(np.mean([r["metrics"][name][key] for r in results]))
                for key in ["linear_mse", "log_mae", "psnr", "ssim"]
            }
            for name in method_names
        },
        "cohorts": {
            cohort: {
                name: {
                    key: float(
                        np.mean([r["metrics"][name][key] for r in results if r["cohort"] == cohort])
                    )
                    for key in ["linear_mse", "log_mae", "psnr", "ssim"]
                }
                for name in method_names
            }
            for cohort in sorted({r["cohort"] for r in results})
        },
        "inference_median_seconds": float(
            np.median([r["inference_median_seconds"] for r in results])
        ),
        "timing_scope": "synchronized tensor transfer + model + output transfer; full renderer benchmark is separate",
    }
    keys = [
        "linear_mse",
        "psnr",
        "ssim",
        "edge_1px_gradient_mae",
        "edge_2px_gradient_mae",
        "edge_4px_gradient_mae",
        "image_edge_recall_1px",
        "image_edge_precision_1px",
        "halo_display_mae",
        "highlight_linear_mse",
        "dark_linear_mse",
        "flat_display_residual_variance",
        "linear_energy_relative_bias",
        "thin_linear_mse",
        "thin_gradient_mae",
        "thin_edge_recall_1px",
        "corner_displacement_p95_px",
    ]
    summary["distributions"] = {
        name: {key: grouped_summary(results, name, key) for key in keys} for name in method_names
    }
    summary["strata"] = {
        stratum: {
            name: {
                key: grouped_summary([r for r in results if r["stratum"] == stratum], name, key)
                for key in keys
            }
            for name in method_names
        }
        for stratum in sorted({r["stratum"] for r in results})
    }
    summary["worst_cases"] = {}
    lookup = {r["id"]: r for r in rows}
    for key in ("linear_mse", "edge_2px_gradient_mae", "halo_display_mae"):
        worst = sorted(
            [r for r in results if r["metrics"]["neural"].get(key) is not None],
            key=lambda r: r["metrics"]["neural"][key],
            reverse=True,
        )[:8]
        summary["worst_cases"][key] = [r["id"] for r in worst]
        for case in worst:
            row = lookup[case["id"]]
            with np.load(safe_path(root, row["reference"])) as data:
                reference = data["target"]
            with np.load(output / "predictions" / f"{row['id']}.npz") as data:
                prediction = data["prediction"]
            write_png(
                output / "worst_cases" / f"{key}-{row['id']}.png",
                np.concatenate([prediction, reference, np.abs(prediction - reference) * 4], axis=1),
            )
    write_json(output / "summary.json", summary)
    return summary
