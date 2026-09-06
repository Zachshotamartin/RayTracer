#!/usr/bin/env python3
"""Compare traversal/worker choices with identical seeds, settings, and PNG bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, default=Path("build/raytracer"))
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--depth", type=int, default=12)
parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=Path, default=Path("renders/benchmark.json"))
args = parser.parse_args()
if args.repeats < 1:
    parser.error("--repeats must be positive")
binary = str(args.binary.resolve())
cpu = platform.processor()
if platform.system() == "Darwin":
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
rows, expected_hash = [], None
variants = [("linear-1", ["--no-bvh", "--threads", "1"]),
            ("bvh-1", ["--threads", "1"]),
            ("bvh-multi", ["--threads", str(args.threads)])]
with tempfile.TemporaryDirectory(prefix="raytracer-benchmark-") as directory:
    # One warmup; recorded variants run sequentially to avoid competing for CPU.
    common = [binary, "--headless", "--quiet", "--scene", "field", "--width", str(args.width),
              "--samples", str(args.samples), "--depth", str(args.depth), "--seed", str(args.seed)]
    subprocess.run([*common, "--threads", "1", "--output", str(Path(directory) / "warmup.png")],
                   check=True, stdout=subprocess.DEVNULL)
    for name, extra in variants:
        runs = []
        for repeat in range(args.repeats):
            path = Path(directory) / (name + ".png")
            output = subprocess.check_output([*common, *extra, "--output", str(path)], text=True)
            run = json.loads(output)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected_hash is None:
                expected_hash = digest
            if digest != expected_hash:
                raise RuntimeError("Image mismatch; refusing to report a speedup for different output")
            runs.append(run)
            print(f"{name} {repeat + 1}/{args.repeats}: {run['render_seconds']:.4f} s", flush=True)
        median = statistics.median(run["render_seconds"] for run in runs)
        rows.append({"variant": name, "median_seconds": median, "workers": runs[0]["workers"],
                     "objects": runs[0]["objects"], "runs": runs})
baseline = rows[0]["median_seconds"]
for row in rows:
    row["speedup_vs_linear_1"] = baseline / row["median_seconds"]
report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
          "cpu": cpu, "logical_cpus": os.cpu_count(), "scene": "field", "width": args.width,
          "samples": args.samples, "max_depth": args.depth, "seed": args.seed, "repeats": args.repeats,
          "identical_png_sha256": expected_hash, "results": rows}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(f"Saved {args.output}")
