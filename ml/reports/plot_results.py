# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.7"]
# ///
"""Rebuild documentation plots from checked-in measurements; no training is run.

From the repository root: uv run --script ml/reports/plot_results.py
"""

import gzip
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "raytracer-pilot-v1",
    }
)


def save(figure, name):
    figure.savefig(FIGURES / f"{name}.png", dpi=150, facecolor="white")
    figure.savefig(FIGURES / f"{name}.svg", facecolor="white", metadata={"Date": None})
    svg = FIGURES / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(figure)


def training_curve():
    rows = [json.loads(line) for line in (ROOT / "pilot-training.jsonl").read_text().splitlines()]
    epochs = [r["epoch"] + 1 for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for key, label, color in [
        ("train_loss", "Training (crops)", "#177A6B"),
        ("validation_loss", "Validation (full frames)", "#2166AC"),
        ("raw_validation_loss", "Raw validation input", "#6E7378"),
    ]:
        ax.plot(epochs, [r[key] for r in rows], label=label, color=color, linewidth=2)
    best = min(rows, key=lambda r: r["validation_loss"])
    ax.scatter([best["epoch"] + 1], [best["validation_loss"]], color="#2166AC", s=45, zorder=3)
    ax.set(
        title="Custom U-Net training · pilot-v1",
        xlabel="Epoch (one-based)",
        ylabel="Reconstruction loss (lower is better)",
        yscale="log",
        xlim=(1, max(epochs)),
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "pilot-training")


def quality_curves():
    with gzip.open(ROOT / "pilot-test-per-image.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    budgets = sorted({r["samples"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for key, label, color in [
        ("raw", "Raw", "#6E7378"),
        ("atrous", "A-trous", "#C56C17"),
        ("neural", "Our U-Net", "#2166AC"),
        ("oidn", "OIDN 2.5.1", "#7955A0"),
    ]:
        for metric, ax in zip(["psnr", "ssim"], axes):
            values = [
                mean(r["metrics"][key][metric] for r in rows if r["samples"] == n) for n in budgets
            ]
            ax.plot(
                budgets, values, marker="o", label=label, color=color, linewidth=2, markersize=4
            )
    for ax, title, ylabel in zip(
        axes,
        ["Display PSNR", "Structural similarity"],
        ["PSNR (dB, higher is better)", "SSIM (higher is better)"],
    ):
        ax.set(title=title, xlabel="Input samples per pixel", ylabel=ylabel, xscale="log")
        ax.set_xticks(budgets)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.grid(alpha=0.18)
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(frameon=False, fontsize=10)
    save(fig, "pilot-quality")


if __name__ == "__main__":
    training_curve()
    quality_curves()
    print("Rebuilt training and test-quality figures from recorded JSON/JSONL.")
