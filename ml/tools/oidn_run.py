"""Bounded launch of pinned, unmodified upstream training with an experiment receipt.

The child initializes the LR finder's RNG too: upstream find_lr.py accepts --seed
but does not initialize it. It then executes the original script without patching it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time


UPSTREAM = "6602ee2ca38a1a2a02135beed8f6e68eed630180"


def write_receipt(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--stage", choices=["preprocess", "find_lr", "train", "infer", "export"], required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("upstream_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    source = args.source.resolve()
    forwarded = args.upstream_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.seconds <= 0 or args.threads < 1:
        parser.error("seconds and threads must be positive")
    if args.child:
        import torch

        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        if args.stage in {"train", "find_lr"}:
            forwarded += ["--seed", str(args.seed)]
        script = source / "training" / (args.stage + ".py")
        sys.path.insert(0, str(script.parent))
        sys.argv = [str(script), *forwarded]
        runpy.run_path(str(script), run_name="__main__")
        return 0

    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).strip()
    if revision != UPSTREAM or dirty:
        parser.error("Expected clean OIDN v2.5.1 source at the pinned commit")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", "-c", type=Path)
    config_args, _ = config_parser.parse_known_args(forwarded)
    configuration = None
    if config_args.config:
        configuration = config_args.config.read_bytes()
        configured_seed = json.loads(configuration).get("seed", args.seed)
        if configured_seed != args.seed:
            parser.error("Upstream JSON overrides --seed; its seed must match the launch receipt")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        str(args.python.absolute()),
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--source",
        str(source),
        "--stage",
        args.stage,
        "--seed",
        str(args.seed),
        "--threads",
        str(args.threads),
        "--run-dir",
        str(args.run_dir.resolve()),
        "--",
        *forwarded,
    ]
    receipt = {
        "upstream_commit": revision,
        "upstream_modified": False,
        "stage": args.stage,
        "seed": args.seed,
        "threads": args.threads,
        "command": command,
        "working_directory": str(source),
        "timeout_seconds": args.seconds,
        "started_unix": time.time(),
        "state": "running",
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(configuration).hexdigest()
        if configuration
        else None,
        "training_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((source / "training").glob("*.py"))
        },
        "resume_note": "Upstream training resume reseeds by epoch; it is not bitwise continuation.",
    }
    record = args.run_dir / "run.json"
    write_receipt(record, receipt)
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": str(args.threads),
        "VECLIB_MAXIMUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
    }
    with (args.run_dir / "output.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=args.seconds)
            receipt["state"] = "succeeded" if code == 0 else "failed"
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            # Only the process group created above is stopped, including its data loaders.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # It finished between timeout detection and signaling.
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            code = 124 if isinstance(exc, subprocess.TimeoutExpired) else 130
            receipt["state"] = "timed_out" if code == 124 else "interrupted"
    receipt.update(exit_code=code, finished_unix=time.time())
    write_receipt(record, receipt)
    print(json.dumps({"state": receipt["state"], "receipt": str(record)}, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
