"""Metadata-selected, labeled comparisons; never select examples by model quality."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io import display


def comparison_ids(rows, limit=12):
    configurations = {}
    for row in rows:
        configurations.setdefault(row["configuration"], []).append(row)
    keys = sorted(configurations)
    if not keys:
        return set()
    indices = np.linspace(0, len(keys) - 1, min(limit, len(keys)), dtype=int)
    return {
        min(
            configurations[keys[i]],
            key=lambda r: (abs(r["samples"] - 4), "-n0-s" not in r["id"], r["id"]),
        )["id"]
        for i in indices
    }


def write_comparison(path, row, methods, target, metrics):
    """Keep output pixels at native size and print the method and reference budget."""
    h, w = target.shape[:2]
    panel_width = max(w, 245)
    names = {
        "raw": "Raw + bilinear 2x" if row["scale"] == 2 else "Raw",
        "atrous": "A-trous + bilinear 2x" if row["scale"] == 2 else "A-trous",
        "neural": "Learned reconstruction",
        "oidn": "OIDN + bilinear 2x" if row["scale"] == 2 else "OIDN",
    }
    entries = list(methods.items()) + [("reference", target)]
    canvas = Image.new("RGB", (panel_width * len(entries), h + 122), (24, 27, 33))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    small = ImageFont.load_default(size=13)
    draw.text((12, 10), f"{row['id']} | {row['split']} | final {w} x {h}", font=font, fill="white")
    draw.text(
        (12, 33),
        (
            "Synthetic reduced input; retained reference pixels"
            if row.get("reuse", {}).get("factor") == 2
            else "Native output pixels; common ACES/sRGB display at exposure 0"
        ),
        font=small,
        fill=(195, 201, 212),
    )
    for index, (name, pixels) in enumerate(entries):
        left = index * panel_width
        draw.text(
            (left + 12, 58), names.get(name, "Path-traced reference"), font=font, fill="white"
        )
        if name == "reference":
            detail = f"{row['reference_samples']} spp | independent seed"
        else:
            score = metrics[name]
            detail = f"{row['samples']} spp | {score['psnr']:.2f} dB | SSIM {score['ssim']:.4f}"
        draw.text((left + 12, 82), detail, font=small, fill=(195, 201, 212))
        image = Image.fromarray(np.rint(display(pixels) * 255).astype("uint8"))
        canvas.paste(image, (left + (panel_width - w) // 2, 110))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
