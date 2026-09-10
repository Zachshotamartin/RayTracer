"""Rebuild committed validation charts: Python + matplotlib 3.10.6, no renderer."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "ml/reports"
ROWS = json.loads((REPORTS / "joint-training-history.json").read_text())["epochs"]
FIGURES = REPORTS / "figures"
BLUE, ORANGE, GREEN = "#2166ac", "#b65c12", "#16816b"


def finish(fig, filename):
    fig.savefig(FIGURES / filename, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    epochs = [r["epoch"] for r in ROWS]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle("Joint 2× reconstruction: 69 completed epochs", fontsize=18)
    ax = axes[0, 0]
    ax.plot(
        epochs,
        [r["validation_loss"] for r in ROWS],
        color=BLUE,
        label="Validation loss",
    )
    best = next(r for r in ROWS if r["epoch"] == 59)
    ax.scatter(
        [59],
        [best["validation_loss"]],
        color=GREEN,
        zorder=5,
        label="Selected best: epoch 59",
    )
    ax.set(title="Selection loss (lower is better)", ylabel="Validation loss")
    ax.legend(fontsize=9)
    for ax, key, target, title in [
        (axes[0, 1], "psnr", 30, "Native validation PSNR (higher is better)"),
        (axes[1, 0], "ssim", 0.95, "Native validation SSIM (higher is better)"),
    ]:
        ax.plot(
            epochs,
            [r["domains"]["native-2x"]["measured"][key] for r in ROWS],
            color=BLUE,
            label="AI: native 2×",
        )
        ax.axhline(
            ROWS[0]["domains"]["native-2x"]["baseline"][key],
            color=ORANGE,
            label="A-trous + bilinear 2×",
        )
        ax.axhline(target, color=GREEN, ls="--", label="Quality target")
        ax.set(title=title, ylabel="dB" if key == "psnr" else "SSIM")
        ax.legend(fontsize=9, loc="lower right")
    ax = axes[1, 1]
    for key, color, label in [
        ("native-2x", BLUE, "Native 2×"),
        ("synthetic-area2", "#8356a1", "Synthetic reduced input"),
    ]:
        ax.plot(
            epochs,
            [
                r["domains"][key]["measured"]["hdr_mse"]
                / r["domains"][key]["baseline"]["hdr_mse"]
                for r in ROWS
            ],
            color=color,
            label=label,
        )
    ax.axhline(1, color=ORANGE, ls="--", label="A-trous baseline")
    ax.set(title="HDR error / baseline (lower is better)", ylabel="Error ratio")
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.axvline(50.5, color="#777777", ls=":", lw=1)
        ax.set(xlabel="Completed epoch", xlim=(1, 69))
        ax.grid(alpha=0.18)
    fig.supxlabel(
        "Dotted line: lower-LR extension after epoch 50. Early stopping at 69; one seed, validation only.",
        fontsize=10,
    )
    finish(fig, "joint-training-history.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    fig.suptitle("Epoch 59: quality by native input sample count", fontsize=17)
    budgets = [1, 4, 16, 32, 64]
    for ax, key, target, title in [
        (axes[0], "psnr", 30, "PSNR ↑"),
        (axes[1], "ssim", 0.95, "SSIM ↑"),
    ]:
        for method, color, label in [
            ("measured", BLUE, "AI reconstruction"),
            ("baseline", ORANGE, "A-trous + bilinear 2×"),
        ]:
            ax.plot(
                budgets,
                [best["domain_budget"][f"native-2x:{n}"][method][key] for n in budgets],
                "o-",
                color=color,
                label=label,
            )
        ax.axhline(target, color=GREEN, ls="--", label="Quality target")
        ax.set(xscale="log", xlabel="Native input samples per pixel", ylabel=title)
        ax.set_xticks(budgets, [str(n) for n in budgets])
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9, loc="lower right")
    fig.supxlabel(
        "64 validation images per budget. Full qualification also requires resolution, HDR, edge and preservation checks.",
        fontsize=10,
    )
    finish(fig, "joint-native-quality.png")


if __name__ == "__main__":
    main()
