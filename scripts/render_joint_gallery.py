#!/usr/bin/env python3
"""Render and assemble the HD joint-model gallery from committed scene configs.

Run with the ML environment: PYTHONPATH=ml/src ml/.venv/bin/python
scripts/render_joint_gallery.py --help. Checkpoints and working HDR buffers stay local.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from raytracer_ml.data.resample import atrous
from raytracer_ml.io import digest, display, read_pfm, write_png
from raytracer_ml.metrics import image_metrics
from raytracer_ml.models import build_model
from raytracer_ml.preprocessing import load_features, validate_features
from raytracer_ml.train import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("artifacts/gallery-hd"))
    parser.add_argument("--binary", type=Path, default=Path("build/raytracer"))
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Reuse completed HDR renders in --work",
    )
    parser.add_argument("--family", choices=["courtyard", "corridor", "shelves"])
    args = parser.parse_args()
    source = Path("ml/reports/gallery-scenes")
    figures = Path("ml/reports/figures")
    state = load_checkpoint(args.checkpoint)
    checkpoint_hash = digest(args.checkpoint)
    assert (
        checkpoint_hash
        == "750510e58d6ea0b61db7c967a42b20d52d9a5e79f4996faaa12376536c60b384"
    )
    torch.set_num_threads(1)
    model = build_model(state["config"]["model"]).cpu().eval()
    model.load_state_dict(state["model"])

    def infer(x):
        with torch.inference_mode():
            return model(torch.from_numpy(x[None])).numpy()[0].transpose(1, 2, 0)

    def fit(x, size, mode="area"):
        kwargs = {"align_corners": False} if mode == "bilinear" else {}
        return (
            torch.nn.functional.interpolate(
                torch.from_numpy(x.transpose(2, 0, 1)[None]),
                size=size,
                mode=mode,
                **kwargs,
            )
            .numpy()[0]
            .transpose(1, 2, 0)
        )

    def post_filter(x, prediction):
        guides = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        guides[:3] = prediction.transpose(2, 0, 1)
        return atrous(guides, iterations=3)

    families = [args.family] if args.family else ["courtyard", "corridor", "shelves"]
    for family in families:
        config = json.loads((source / f"{family}.json").read_text())
        folder = args.work / family
        folder.mkdir(parents=True, exist_ok=True)
        scene_path = folder / "scene.json"
        if args.assemble_only:
            assert json.loads(scene_path.read_text()) == config["scene"]
        else:
            scene_path.write_text(json.dumps(config["scene"], indent=2) + "\n")
        if not args.assemble_only:
            preflight = folder / "preflight"
            preflight.mkdir(exist_ok=True)
            subprocess.run(
                [
                    str(args.binary.resolve()),
                    "--headless",
                    "--quiet",
                    "--scene-file",
                    str(scene_path),
                    "--width",
                    "8",
                    "--samples",
                    "2",
                    "--features",
                    str(preflight / "features"),
                    "--output",
                    str(preflight / "image.pfm"),
                ],
                check=True,
                capture_output=True,
            )
            validate_features(load_features(preflight / "features", 2))
            for name, width, spp, seed in [
                ("input", 480, 64, config["input_seed"]),
                ("reference", 960, 2048, config["target_seed"]),
            ]:
                dest = folder / name
                dest.mkdir(exist_ok=True)
                command = [
                    str(args.binary.resolve()),
                    "--headless",
                    "--scene-file",
                    str(scene_path),
                    "--width",
                    str(width),
                    "--samples",
                    str(spp),
                    "--seed",
                    str(seed),
                    "--depth",
                    "16",
                    "--threads",
                    "8",
                    "--features",
                    str(dest / "features"),
                    "--output",
                    str(dest / "image.pfm"),
                ]
                with (
                    (dest / "stdout.json").open("w") as out,
                    (dest / "progress.log").open("w") as err,
                ):
                    subprocess.run(command, stdout=out, stderr=err, check=True)
        x = load_features(folder / "input/features", 2)
        full = load_features(folder / "reference/features", 2)
        validate_features(x)
        validate_features(full)
        reference = read_pfm(folder / "reference/image.pfm")
        assert x.shape[2] == 480 and full.shape[2] == 960
        assert np.all(x[15] == 64) and np.all(full[15] == 2048)
        assert np.array_equal(full[:3].transpose(1, 2, 0), reference)
        height, width = reference.shape[:2]
        filtered = atrous(x, iterations=3)
        ai = infer(x)
        pre = x.copy()
        pre[:3] = filtered.transpose(2, 0, 1)
        values = {
            "noisy": fit(x[:3].transpose(1, 2, 0), (height, width), "bilinear"),
            "atrous": fit(filtered, (height, width), "bilinear"),
            "reference": reference,
            "epoch59": ai,
            "ai_atrous": post_filter(x, ai),
            "atrous_ai": infer(pre),
        }
        metrics = {
            k: image_metrics(v, reference)
            for k, v in values.items()
            if k != "reference"
        }
        print(f"{family}: low-sample comparisons complete", flush=True)
        values["full_render_atrous"] = atrous(full, iterations=3)
        values["full_render_ai"] = infer(full)
        pre_full = full.copy()
        pre_full[:3] = values["full_render_atrous"].transpose(2, 0, 1)
        values["full_render_atrous_ai"] = infer(pre_full)
        values["full_render_ai_atrous"] = post_filter(full, values["full_render_ai"])
        prefix = f"joint-epoch59-{family}-hd"
        for name, value in values.items():
            assert np.isfinite(value).all()
            write_png(figures / f"{prefix}-{name.replace('_', '-')}.png", value)
        stats = {
            name: json.loads((folder / name / "stdout.json").read_text())
            for name in ["input", "reference"]
        }
        for name, expected_width, expected_spp, expected_seed in [
            ("input", 480, 64, config["input_seed"]),
            ("reference", 960, 2048, config["target_seed"]),
        ]:
            assert stats[name]["width"] == expected_width
            assert stats[name]["samples"] == expected_spp
            assert stats[name]["seed"] == expected_seed
        for item in stats.values():
            for key in ["scene_file", "output", "features", "feature_directory"]:
                item.pop(key, None)
        record = {
            "scene": family,
            "selection": "Same original metadata-selected validation scene configuration, newly rendered at gallery resolution; not chosen by score",
            "epoch": 59,
            "checkpoint_sha256": checkpoint_hash,
            "scene_config": f"gallery-scenes/{family}.json",
            "input_size": [480, x.shape[1]],
            "reference_size": [960, height],
            "input_samples": 64,
            "reference_samples": 2048,
            "reference_pfm_sha256": digest(folder / "reference/image.pfm"),
            "renderer_sha256": digest(args.binary),
            "render_stats": stats,
            "metrics": metrics,
            "display": "Common ACES/sRGB exposure zero. Reference-resolution PNGs are native pixels; 1920-pixel AI diagnostics are area-reduced only in the grid.",
            "full_render_processing": {
                "reference_fed": True,
                "quality_scores_reported": False,
                "ai_output_size": [1920, height * 2],
                "atrous_iterations": 3,
                "atrous_ai": "Filter full render, replace RGB only, retain measured guides/statistics, then epoch-59 AI",
                "ai_atrous": "Epoch-59 AI then filter at actual output resolution with nearest-neighbor 2x full-render guides",
                "limitations": "2048-spp and pre-denoised inputs are outside the training distribution; retained variance is not calibrated AI/filter error",
            },
            "evaluation_scope": "Three new higher-resolution illustrations, excluded from historical epoch metrics; no timing or generalization qualification",
        }
        (folder / "report.json").write_text(json.dumps(record, indent=2) + "\n")
        order = [
            "noisy",
            "atrous",
            "reference",
            "epoch59",
            "ai_atrous",
            "atrous_ai",
            "full_render_ai",
            "full_render_atrous",
            "full_render_atrous_ai",
            "full_render_ai_atrous",
        ]
        labels = [
            "64-spp input + bilinear 2x",
            "A-trous + bilinear 2x",
            "Full path-traced render | 2048 spp",
            "AI reconstruction | epoch 59",
            "AI then A-trous",
            "A-trous then AI",
            "Full render then AI",
            "Full render then A-trous",
            "Full render > A-trous > AI",
            "Full render > AI > A-trous",
        ]
        pw, ph = 992, height + 112
        canvas = Image.new("RGB", (pw * 2 + 24, ph * 5 + 132), (23, 26, 32))
        draw = ImageDraw.Draw(canvas)
        title, small = ImageFont.load_default(size=38), ImageFont.load_default(size=28)
        draw.text(
            (16, 10),
            f"{family.title()} | Real 960-pixel renders | Epoch 59",
            font=title,
            fill="white",
        )
        draw.text(
            (16, 58),
            f"Input 480 x {x.shape[1]} at 64 spp | Reference 960 x {height} at 2048 spp",
            font=small,
            fill="#c4cbd5",
        )
        for i, (name, label) in enumerate(zip(order, labels)):
            left, top = 12 + (i % 2) * pw, 110 + (i // 2) * ph
            draw.text((left + 12, top), label, font=title, fill="white")
            if name in metrics:
                detail = f"PSNR {metrics[name]['psnr']:.2f} dB | SSIM {metrics[name]['ssim']:.4f}"
            elif name == "reference":
                detail = f"960 x {height} native pixels | independent render"
            elif name == "full_render_atrous":
                detail = "Full-render diagnostic | native resolution | 3 passes"
            else:
                detail = f"1920 x {height * 2} output | fitted for comparison"
            draw.text((left + 12, top + 46), detail, font=small, fill="#c4cbd5")
            value = values[name]
            if value.shape[:2] != (height, width):
                value = fit(value, (height, width))
            pixels = Image.fromarray(np.rint(display(value) * 255).astype("uint8"))
            canvas.paste(pixels, (left + 16, top + 88))
        canvas.save(figures / f"{prefix}.png")
        print(f"{family}: gallery complete", flush=True)

    reports = [
        args.work / family / "report.json"
        for family in ["courtyard", "corridor", "shelves"]
    ]
    if all(path.exists() for path in reports):
        records = [json.loads(path.read_text()) for path in reports]
        assert all(record["checkpoint_sha256"] == checkpoint_hash for record in records)
        assert all(
            record["renderer_sha256"] == digest(args.binary) for record in records
        )
        Path("ml/reports/joint-examples.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
        report_path = Path("ml/reports/joint-reconstruction-results.md")
        text = report_path.read_text()
        for record in records:
            prefix = f"| {record['scene'].title()}, 64 spp |"
            cells = [f"{record['scene'].title()}, 64 spp"]
            for method in ["atrous", "epoch59", "ai_atrous", "atrous_ai"]:
                score = record["metrics"][method]
                cells.append(f"{score['psnr']:.2f} dB / {score['ssim']:.4f}")
            lines = text.splitlines()
            assert sum(line.startswith(prefix) for line in lines) == 1
            text = (
                "\n".join(
                    "| " + " | ".join(cells) + " |" if line.startswith(prefix) else line
                    for line in lines
                )
                + "\n"
            )
        report_path.write_text(text)


if __name__ == "__main__":
    main()
