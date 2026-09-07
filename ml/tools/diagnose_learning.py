"""Bounded, paired fixed-patch fitting diagnostic; never a generalization benchmark.

Example: python ml/tools/diagnose_learning.py --config CONFIG --data DATA --output OUT
Use original training groups only. This deliberately disables random augmentation,
fusion, preservation sampling, scheduling and early stopping to isolate optimization.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from raytracer_ml.data.dataset import RenderDataset
from raytracer_ml.io import config, digest, git_revision, write_json
from raytracer_ml.losses import reconstruction_loss
from raytracer_ml.models import build_model
from raytracer_ml.train import device_for, synchronize


def fixed_patches(root, samples=4, count=4, crop=64):
    dataset = RenderDataset(root, "train", feature_schema=2)
    selected = {}
    for index, row in enumerate(dataset.rows):
        if row["samples"] == samples and row["scale"] == 1:
            selected.setdefault(row["group"], index)
    if len(selected) < count:
        raise ValueError("Insufficient distinct training groups at the requested sample budget")
    xs, ys, receipts = [], [], []
    for group in sorted(selected)[:count]:
        index = selected[group]
        x, y = dataset[index]
        if min(x.shape[-2:]) < crop:
            raise ValueError("Crop exceeds the measured image dimensions")
        top, left = (x.shape[-2] - crop) // 2, (x.shape[-1] - crop) // 2
        xs.append(x[:, top : top + crop, left : left + crop])
        ys.append(y[:, top : top + crop, left : left + crop])
        row = dataset.rows[index]
        receipts.append(dict(id=row["id"], group=group, top=top, left=left, crop=crop))
    return torch.stack(xs), torch.stack(ys), receipts


def mutable_mask(x):
    # Same-resolution schema-2 support policy, used only to measure reducible error.
    return (
        ((x[:, 10:11] - x[:, 11:12]).abs() < 1e-6)
        & (x[:, 10:11] > 0)
        & ((x[:, 26:27] > 0.5) | (x[:, 23:24] == 0))
        & (x[:, 15:16] < 128)
    )


def fit_trial(cfg, x, y, *, activation, rate, seed, steps, device, deadline):
    torch.manual_seed(seed)
    architecture = {**cfg["model"], "activation": activation}
    model = build_model(architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rate, weight_decay=cfg["weight_decay"])
    x, y = x.to(device), y.to(device)
    mask = mutable_mask(x).expand_as(y)
    if not mask.any():
        raise ValueError("Diagnostic patches contain no trainable pixels")

    def measure(prediction):
        return {
            "loss": float(reconstruction_loss(prediction, y, cfg.get("loss")).detach().cpu()),
            "mutable_log_mae": float(
                (torch.log1p(prediction) - torch.log1p(y)).abs()[mask].mean().detach().cpu()
            ),
        }

    raw = measure(x[:, :3])
    capture = {}

    def inspect_head(module, inputs, output):
        capture["decoder"] = inputs[0].detach()

    hook = model.head.register_forward_hook(inspect_head)
    history = []
    started = time.perf_counter()
    try:
        for step in range(steps + 1):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Diagnostic reached its wall-time cap")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = reconstruction_loss(prediction, y, cfg.get("loss"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite diagnostic loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            if step % 25 == 0 or step == steps:
                row = {
                    "step": step,
                    **measure(prediction),
                    "gradient_norm_before_clip": float(norm.detach().cpu()),
                    "encoder_gradient_norm": float(model.enc1[0].weight.grad.norm().cpu()),
                    "decoder_zero_fraction": float((capture["decoder"] == 0).float().mean().cpu()),
                }
                history.append(row)
                print(
                    json.dumps(dict(activation=activation, rate=rate, seed=seed, **row)), flush=True
                )
            if step < steps:
                optimizer.step()
    finally:
        hook.remove()
    synchronize(device)
    return {
        "model": architecture,
        "learning_rate": rate,
        "seed": seed,
        "optimizer_updates": steps,
        "training_pixels": steps * x.shape[0] * x.shape[-2] * x.shape[-1],
        "seconds": time.perf_counter() - started,
        "raw": raw,
        "history": history,
        "mutable_error_reduction": 1
        - history[-1]["mutable_log_mae"] / max(raw["mutable_log_mae"], 1e-12),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--patches", type=int, default=4)
    parser.add_argument("--crop", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0003, 0.001])
    parser.add_argument(
        "--activations", nargs="+", choices=["relu", "leaky_relu"], default=["relu", "leaky_relu"]
    )
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--max-seconds", type=float, default=900)
    args = parser.parse_args()
    if min(args.steps, args.patches, args.crop, args.samples, args.max_seconds, *args.rates) <= 0:
        parser.error("Budgets and learning rates must be positive")
    if args.output.exists():
        parser.error("Use a new output path to preserve previous diagnostics")
    cfg = config(args.config)
    if (
        cfg["model"].get("kind", "unet") not in ("unet", "conv")
        or cfg["model"].get("feature_schema") != 2
        or cfg["model"].get("temporal")
        or cfg["model"].get("scale", 1) != 1
    ):
        parser.error("This diagnostic requires a spatial schema-2 U-Net/conv at scale 1")
    torch.set_num_threads(2)
    device = device_for(args.device)
    started = time.perf_counter()
    deadline = started + args.max_seconds
    x, y, receipts = fixed_patches(args.data, args.samples, args.patches, args.crop)
    source = Path(__file__).resolve()
    model_source = source.parents[1] / "src/raytracer_ml/models/unet.py"
    record = {
        "scope": "Fixed training patches only; not validation or release qualification",
        "config": copy.deepcopy(cfg),
        "manifest_sha256": digest(args.data / "manifest.jsonl"),
        "code_commit": git_revision(),
        "source_hashes": {
            str(p.relative_to(source.parents[1])): digest(p) for p in [source, model_source]
        },
        "device": str(device),
        "torch": str(torch.__version__),
        "patches": receipts,
        "max_seconds": args.max_seconds,
        "state": "running",
        "trials": [],
    }
    write_json(args.output, record)
    try:
        for seed in args.seeds:
            for rate in args.rates:
                for activation in args.activations:
                    record["trials"].append(
                        fit_trial(
                            cfg,
                            x,
                            y,
                            activation=activation,
                            rate=rate,
                            seed=seed,
                            steps=args.steps,
                            device=device,
                            deadline=deadline,
                        )
                    )
                    write_json(args.output, record)
        record["state"] = "completed"
    except Exception as error:
        record["state"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        write_json(args.output, record)


if __name__ == "__main__":
    main()
