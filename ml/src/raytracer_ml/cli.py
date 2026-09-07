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
    p = commands.add_parser("generate")
    p.add_argument("--config", required=True)
    p.add_argument("--renderer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = commands.add_parser("validate-data")
    p.add_argument("--data", required=True)
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-new-epochs", type=int)
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
        if args.command == "generate":
            from .data.generate import generate

            result = generate(config(args.config), args.renderer, args.output, args.dry_run)
        elif args.command == "validate-data":
            from .data.validate import validate

            result = validate(args.data)
        elif args.command == "train":
            from .train import train

            result = train(
                config(args.config), args.data, args.output, args.resume, args.max_new_epochs
            )
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

            result = export_model(args.checkpoint, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print(f"rtml: {error}", file=sys.stderr)
        return 1
    return 0
