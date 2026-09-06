#!/usr/bin/env python3
"""Reproduce the README gallery and adjacent render metadata."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, default=Path("build/raytracer"))
parser.add_argument("--output-dir", type=Path, default=Path("docs/renders"))
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--threads", type=int, default=0)
args = parser.parse_args()
for scene, samples in [("studio", 256), ("field", 128), ("demo", 64)]:
    subprocess.run([str(args.binary.resolve()), "--headless", "--scene", scene,
                    "--width", str(args.width), "--samples", str(samples), "--depth", "16",
                    "--threads", str(args.threads), "--seed", "42",
                    "--output", str(args.output_dir / (scene + ".png"))], check=True)
