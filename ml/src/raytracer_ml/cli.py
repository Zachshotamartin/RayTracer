"""Command-line entry points; heavy ML imports are deferred until needed."""

import argparse
import json
import sys
from .io import config


def main():
    parser = argparse.ArgumentParser(
        description="RayTracer ML: paired data to native reconstruction"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "status", help="Read dataset and training progress from artifact receipts"
    )
    p.add_argument("--artifacts", required=True)
    p = commands.add_parser(
        "run-experiment", help="Run a pinned, user-authorized data-to-training plan"
    )
    p.add_argument("--plan", required=True)
    p = commands.add_parser("generate")
    p.add_argument("--config", required=True)
    p.add_argument("--renderer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser(
        "reuse-data", help="Reuse existing rendered arrays without tracing rays"
    )
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("validate-data")
    p.add_argument("--data", required=True)
    p = commands.add_parser(
        "export-oidn-data", help="Export linear EXRs for upstream OIDN training"
    )
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val"]
    )
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-from", help="Continue a full checkpoint into a NEW output directory")
    p.add_argument("--max-new-epochs", type=int)
    p = commands.add_parser(
        "extend-training", help="Prepare a recorded schedule extension without training"
    )
    p.add_argument("--parent-run", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True, help="New seed artifact directory, not the next run")
    p = commands.add_parser("checkpoints", help="List full-state restart points and SHA-256 hashes")
    p.add_argument("--output", required=True)
    p = commands.add_parser("pause-training", help="Request a safe pause after the current epoch")
    p.add_argument("--output", required=True)
    p = commands.add_parser("prepare-training", help="Validate a run without optimizer updates")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--oidn")
    p.add_argument("--oidn-device", default="cpu")
    p.add_argument("--registration")
    p = commands.add_parser("register-evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--training-data", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("export")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--precision", choices=["fp32", "mixed-fp16"], default="fp32")
    p = commands.add_parser("sequence")
    p.add_argument("--data", required=True)
    p.add_argument("--group", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("benchmark")
    p.add_argument("--data", required=True)
    p.add_argument("--renderer", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "status":
            from .status import status

            result = status(args.artifacts)
        elif args.command == "run-experiment":
            from .experiment import run_experiment

            result = run_experiment(args.plan)
        elif args.command == "export-oidn-data":
            from .oidn_dataset import export_dataset

            receipt = export_dataset(args.data, args.output, args.splits)
            result = {
                "examples": receipt["examples"],
                "descriptor": receipt["descriptor"],
                "receipt": str(args.output) + "/export.json",
            }
        elif args.command == "generate":
            from .data.generate import generate

            result = generate(config(args.config), args.renderer, args.output, args.dry_run)
        elif args.command == "reuse-data":
            from .data.reuse import build_reuse

            result = build_reuse(args.source, args.output)
        elif args.command == "validate-data":
            from .data.validate import validate

            result = validate(args.data)
        elif args.command == "train":
            from .train import train

            result = train(
                config(args.config),
                args.data,
                args.output,
                args.resume,
                args.max_new_epochs,
                args.resume_from,
            )
        elif args.command == "extend-training":
            from .schedule_extension import prepare_extension

            result = prepare_extension(args.parent_run, config(args.config), args.output)
        elif args.command == "checkpoints":
            from .train import load_checkpoint
            from .training_checkpoints import list_checkpoints

            result = list_checkpoints(args.output, load_checkpoint)
        elif args.command == "pause-training":
            from pathlib import Path

            output = Path(args.output)
            if not (output / "config.json").is_file():
                raise ValueError("No initialized training run at this output")
            (output / "STOP_AFTER_EPOCH").touch()
            result = {"state": "pause-requested", "boundary": "after the current epoch"}
        elif args.command == "prepare-training":
            from .prepare import prepare_training

            result = prepare_training(config(args.config), args.data, args.output, args.report)
        elif args.command == "evaluate":
            from .evaluate import evaluate

            result = evaluate(
                args.data,
                args.checkpoint,
                args.output,
                args.split,
                args.device,
                oidn=args.oidn,
                oidn_device=args.oidn_device,
                registration=args.registration,
            )
        elif args.command == "register-evaluation":
            from .evaluation_contract import register

            result = register(args.checkpoint, args.training_data, args.data, args.output)
        elif args.command == "sequence":
            from .sequence import export_sequence

            result = export_sequence(args.data, args.group, args.output)
        elif args.command == "benchmark":
            from .benchmark import benchmark

            result = benchmark(
                args.data, args.renderer, args.model, args.output, config(args.config)
            )
        else:
            from .export import export_model

            result = export_model(
                args.checkpoint, args.output, args.width, args.height, args.precision
            )
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"rtml: {error}", file=sys.stderr)
        return 1
    return 0
