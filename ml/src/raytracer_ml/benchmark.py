"""Native end-to-end repeated-frame benchmark with predeclared quality thresholds."""

import json
from html import escape
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
from .io import manifest, safe_path, read_pfm, write_json, digest
from .metrics import image_metrics


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
    records = manifest(root / "manifest.jsonl")
    config_ids = sorted(
        {r["configuration"] for r in records if r["split"] == cfg.get("split", "test")}
    )
    limit = cfg.get("max_configurations", 0)
    if limit:
        config_ids = config_ids[:limit]
    results = []
    if not config_ids:
        raise ValueError("No benchmark configurations")
    with tempfile.TemporaryDirectory(prefix="rtml-benchmark-") as temporary:
        temp = Path(temporary)
        for ident in config_ids:
            record = next(r for r in records if r["configuration"] == ident)
            with np.load(safe_path(root, record["reference"])) as data:
                target = data["target"]
            write_json(temp / "scene.json", record["scene"])
            for samples in cfg["budgets"]:
                for method in ["raw", "atrous", "neural"]:
                    # All methods produce the reference resolution in the primary scale-1 study.
                    command = [
                        str(binary),
                        "--headless",
                        "--quiet",
                        "--scene-file",
                        str(temp / "scene.json"),
                        "--width",
                        str(target.shape[1]),
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
                    if method == "atrous":
                        command += ["--denoise"]
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
                            "Benchmark requires a scale-1 model; evaluate super resolution separately"
                        )
                    metrics = image_metrics(prediction, target)
                    warm = [s["pipeline_seconds"] for s in stats[1:]]
                    result = {
                        "configuration": ident,
                        "cohort": record["cohort"],
                        "method": method,
                        "samples": samples,
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
    threshold = cfg.get("quality", {})
    if "psnr" not in threshold or "ssim" not in threshold:
        raise ValueError("Set validation-selected PSNR and SSIM thresholds")
    matched = []
    for ident in config_ids:
        item = {"configuration": ident}
        for method in ["raw", "atrous", "neural"]:
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
    summary = {
        "config": cfg,
        "renderer_sha256": digest(binary),
        "model_sha256": digest(model),
        "configurations": len(config_ids),
        "matched_quality": matched,
        "timing_scope": "same-process repeated frames: BVH, tracing, aligned features, inference, image write; excludes display and one-time JSON scene loading. Cold adds model load. Process total is reported separately.",
        "failures": {m: sum(r[m] is None for r in matched) for m in ["raw", "atrous", "neural"]},
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
