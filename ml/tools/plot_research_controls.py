"""Labeled scientific comparison panel from verified render/model arrays, not generated artwork."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from raytracer_ml.io import digest, display
from raytracer_ml.models import build_model
from raytracer_ml.research_controls import load_cohort
from raytracer_ml.train import load_checkpoint


def panel(cohort, runs, output, seed=42):
    _, examples = load_cohort(cohort)
    result = json.loads((runs / "comparison.json").read_text())
    if result["state"] != "completed" or result["cohort_sha256"] != digest(cohort / "cohort.json"):
        raise ValueError("Incomplete or mismatched comparison")
    methods = ["guided-center-l1", "guided-sampled-l1", "refine-conditioned"]
    models = {}
    torch.set_num_threads(2)
    for name in methods:
        trial = next(t for t in result["trials"] if t["name"] == name and t["seed"] == seed)
        checkpoint = runs / f"{name}-s{seed}/final.pt"
        if digest(checkpoint) != trial["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        state = load_checkpoint(checkpoint)
        model = build_model(state["config"]["model"]).eval()
        model.load_state_dict(state["model"])
        models[name] = model
    rows = [
        e
        for e in examples
        if e["row"]["split"] == "val" and e["row"]["samples"] == 4 and e["row"]["noise"] == 0
    ]
    h, w = rows[0]["features"].shape[1:]
    factor, gutter, header, label = 2, 6, 42, 28
    width, height = (
        6 * (w * factor + gutter) + gutter,
        header + len(rows) * (h * factor + label + gutter) + gutter,
    )
    canvas = Image.new("RGB", (width, height), "#15191e")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    titles = [
        "Raw / 4 spp",
        "A-trous",
        "Guided / center L1",
        "Guided / sampled L1",
        "Refine / conditioned",
        "Reference / 8192 spp",
    ]
    for column, title in enumerate(titles):
        draw.text((gutter + column * (w * factor + gutter), 12), title, fill="white", font=font)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            x = row["features"]
            images = [x[:3].transpose(1, 2, 0), row["atrous"]]
            for name in methods:
                prediction = models[name](torch.from_numpy(x[None]))[0].numpy().transpose(1, 2, 0)
                path = runs / f"{name}-s{seed}/predictions/{row['row']['id']}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    np.testing.assert_allclose(
                        prediction, saved["prediction"], rtol=2e-4, atol=2e-5
                    )
                    images.append(saved["prediction"])
            images.append(row["reference"]["check"])
            top = header + index * (h * factor + label + gutter)
            draw.text(
                (gutter, top),
                f"{row['reference']['row']['stratum']} | {row['row']['view']} | seed {seed}; 200 updates | 128 x 72, shown at 2x",
                fill="#c1cbd6",
                font=font,
            )
            for column, image in enumerate(images):
                bitmap = Image.fromarray(
                    (display(image) * 255).round().clip(0, 255).astype(np.uint8)
                )
                bitmap = bitmap.resize((w * factor, h * factor), resample=Image.Resampling.NEAREST)
                canvas.paste(bitmap, (gutter + column * (w * factor + gutter), top + label))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new figure path")
    panel(args.cohort, args.runs, args.output)


if __name__ == "__main__":
    main()
