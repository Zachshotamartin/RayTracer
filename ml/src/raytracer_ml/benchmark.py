"""Native end-to-end repeated-frame benchmark with predeclared quality thresholds."""

import json
import math
from html import escape
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from .io import manifest, safe_path, read_pfm, write_json, digest
from .metrics import image_metrics
from .data.validate import validate
from .evaluation_contract import authorize_digests


def json_stream(text):
    decoder = json.JSONDecoder()
    records = []
    while text.strip():
        value, end = decoder.raw_decode(text.lstrip())
        records.append(value)
        text = text.lstrip()[end:]
    return records


def benchmark(root, binary, model, output, cfg):
    root, binary, output = Path(root), Path(binary).resolve(), Path(output)
    if json.loads((root / "dataset.json").read_text()).get("reuse"):
        raise ValueError(
            "Reuse data includes synthetic input/crop measurements; native rendering benchmarks require original native pairs"
        )
    model = Path(model)
    metadata = json.loads(model.with_suffix(".json").read_text())
    if metadata["sha256"] != digest(model):
        raise ValueError("Model export metadata checksum mismatch")
    scale = metadata["model"].get("scale", 1)
    methods = (
        ["raw", "atrous", "neural"]
        if scale == 1
        else ["raw", "atrous", "raw_upscale", "atrous_upscale", "neural"]
    )
    threshold = cfg.get("quality", {})
    if "psnr" not in threshold or "ssim" not in threshold:
        raise ValueError("Set validation-selected PSNR and SSIM thresholds")
    if (
        not all(
            isinstance(threshold[k], (int, float)) and math.isfinite(threshold[k])
            for k in ("psnr", "ssim")
        )
        or threshold["psnr"] < 0
        or not 0 <= threshold["ssim"] <= 1
        or type(cfg.get("repeats", 3)) is not int
        or cfg.get("repeats", 3) < 1
        or not cfg.get("budgets")
        or any(type(n) is not int or n < 1 for n in cfg["budgets"])
    ):
        raise ValueError("Invalid benchmark thresholds, sample budgets or warm repeats")
    validation = validate(root)
    evaluation_scope = authorize_digests(
        metadata["manifest_sha256"],
        metadata["checkpoint_sha256"],
        validation,
        cfg.get("registration"),
    )
    records = manifest(root / "manifest.jsonl")
    config_ids = sorted(
        {r["configuration"] for r in records if r["split"] == cfg.get("split", "test")}
    )
    limit = cfg.get("max_configurations", 0)
    if limit:
        config_ids = (
            [
                config_ids[i]
                for i in np.linspace(0, len(config_ids) - 1, min(limit, len(config_ids)), dtype=int)
            ]
            if config_ids
            else []
        )
    results = []
    if not config_ids:
        raise ValueError("No benchmark configurations")
    with tempfile.TemporaryDirectory(prefix="rtml-benchmark-") as temporary:
        temp = Path(temporary)
        for ident in config_ids:
            record = next(r for r in records if r["configuration"] == ident)
            if record["scale"] != scale:
                raise ValueError("Benchmark model and dataset scales disagree")
            with np.load(safe_path(root, record["reference"])) as data:
                target = data["target"]
            write_json(temp / "scene.json", record["scene"])
            for samples in cfg["budgets"]:
                for method in methods:
                    low_resolution = method in ("neural", "raw_upscale", "atrous_upscale")
                    width = target.shape[1] // scale if low_resolution else target.shape[1]
                    command = [
                        str(binary),
                        "--headless",
                        "--quiet",
                        "--scene-file",
                        str(temp / "scene.json"),
                        "--width",
                        str(width),
                        "--samples",
                        str(samples),
                        "--seed",
                        str(record["input_seed"]),
                        "--depth",
                        str(record["stats"]["max_depth"]),
                        "--threads",
                        str(cfg.get("threads", 4)),
                        "--output",
                        str(temp / "image.pfm"),
                        "--benchmark-repeats",
                        str(cfg.get("repeats", 3) + 1),
                    ]
                    if method in ("atrous", "atrous_upscale"):
                        command += ["--denoise"]
                    if method.endswith("_upscale"):
                        command += ["--output-scale", str(scale)]
                    if method == "neural":
                        command += [
                            "--model",
                            str(Path(model).resolve()),
                            "--neural-provider",
                            cfg.get("provider", "cpu"),
                        ]
                    begin = time.perf_counter()
                    run = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        timeout=cfg.get("timeout_seconds", 300),
                    )
                    if run.returncode:
                        raise RuntimeError(run.stderr[-3000:])
                    stats = json_stream(run.stdout)
                    if method == "neural" and not all(s["reconstructed"] for s in stats):
                        raise RuntimeError("Native benchmark fell back instead of using the model")
                    prediction = read_pfm(temp / "image.pfm")
                    if prediction.shape != target.shape:
                        raise ValueError(
                            "Benchmark output differs from the native reference resolution"
                        )
                    metrics = image_metrics(prediction, target)
                    warm = [s["pipeline_seconds"] for s in stats[1:]]
                    result = {
                        "configuration": ident,
                        "cohort": record["cohort"],
                        "method": method,
                        "samples": samples,
                        "input_width": width,
                        "output_width": target.shape[1],
                        "output_height": target.shape[0],
                        "quality": metrics,
                        "cold_frame_seconds": stats[0]["pipeline_seconds"]
                        + stats[0]["model_load_seconds"],
                        "warm_median_seconds": float(np.median(warm)),
                        "warm_p95_seconds": float(np.percentile(warm, 95)),
                        "process_total_seconds": time.perf_counter() - begin,
                        "native_stats": stats,
                    }
                    results.append(result)
                    print(
                        f"{ident} {method} {samples} spp: {result['warm_median_seconds']:.4f}s, {metrics['psnr']:.2f} dB",
                        flush=True,
                    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_image.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    matched = []
    for ident in config_ids:
        item = {"configuration": ident}
        for method in methods:
            passing = [
                r
                for r in results
                if r["configuration"] == ident
                and r["method"] == method
                and r["quality"]["psnr"] >= threshold["psnr"]
                and r["quality"]["ssim"] >= threshold["ssim"]
            ]
            item[method] = min((r["warm_median_seconds"] for r in passing), default=None)
        matched.append(item)
    acceleration = []
    for item in matched:
        competitors = [item[m] for m in methods if m != "neural" and item[m] is not None]
        baseline = min(competitors) if competitors else None
        neural = item["neural"]
        acceleration.append(
            {
                "configuration": item["configuration"],
                "fastest_passing_baseline_seconds": baseline,
                "neural_seconds": neural,
                "speedup": baseline / neural
                if baseline is not None and neural is not None
                else None,
                "demonstrated_win": baseline is not None
                and neural is not None
                and neural < baseline,
            }
        )
    summary = {
        "config": cfg,
        "renderer_sha256": digest(binary),
        "model_sha256": digest(model),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "manifest_sha256": validation["manifest_sha256"],
        "evaluation_scope": evaluation_scope,
        "configurations": len(config_ids),
        "matched_quality": matched,
        "scale": scale,
        "matched_quality_acceleration": acceleration,
        "method_labels": {
            "raw": "native-resolution raw",
            "atrous": "native-resolution a-trous",
            "raw_upscale": "low-resolution raw + bilinear",
            "atrous_upscale": "low-resolution a-trous + bilinear",
            "neural": "learned reconstruction",
        },
        "timing_scope": "same-process repeated frames at identical output dimensions: BVH, tracing, aligned features, denoising/upscaling or inference, image write; excludes display and one-time JSON scene loading. Cold adds model load. Process total is reported separately.",
        "failures": {m: sum(r[m] is None for r in matched) for m in methods},
    }
    write_json(output / "summary.json", summary)
    # A compact, dependency-free HTML report is easy to open beside the viewer.
    rows = "".join(
        f"<tr><td>{escape(r['configuration'])}</td><td>{r['method']}</td><td>{r['samples']}</td><td>{r['warm_median_seconds'] * 1000:.2f}</td><td>{r['quality']['psnr']:.2f}</td><td>{r['quality']['ssim']:.4f}</td></tr>"
        for r in results
    )
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Ray reconstruction benchmark</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;color:#222}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}</style><h1>Time and image quality</h1><p>Every tested method and budget is shown. Lower latency and higher PSNR/SSIM are preferable.</p><table><tr><th>Scene</th><th>Method</th><th>Samples</th><th>Warm ms</th><th>PSNR dB</th><th>SSIM</th></tr>'
        + rows
        + "</table>"
    )
    return summary
