"""Render a bounded independent-reference cohort or fit explicitly paired controls."""

import argparse
from pathlib import Path
from raytracer_ml.io import config
from raytracer_ml.research_controls import render_cohort, compare_cohort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["render", "compare"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--renderer", type=Path)
    args = parser.parse_args()
    cfg = config(args.config)
    if args.action == "render":
        if not args.renderer:
            parser.error("--renderer is required for rendering")
        render_cohort(args.data, args.output, args.renderer, cfg["cohort"])
    else:
        compare_cohort(args.data, args.output, cfg["comparison"])


if __name__ == "__main__":
    main()
