"""Compact scene-mean evidence from a completed, retained-reference comparison."""

import argparse
import json
from pathlib import Path
import numpy as np
from raytracer_ml.io import digest, write_json
from raytracer_ml.research_controls import load_cohort


def scene_mean(rows, value):
    groups = {}
    for row in rows:
        measured = value(row)
        if measured is not None:
            groups.setdefault(row["group"], []).append(measured)
    return float(np.mean([np.mean(g) for g in groups.values()])) if groups else None


def summarize(cohort, comparison):
    record, _ = load_cohort(cohort)
    results = json.loads(comparison.read_text())
    if results["state"] != "completed" or results["cohort_sha256"] != digest(
        cohort / "cohort.json"
    ):
        raise ValueError("Incomplete or mismatched comparison")
    output = dict(
        scope=results["scope"],
        comparison_sha256=digest(comparison),
        cohort_sha256=results["cohort_sha256"],
        code_commit=results["code_commit"],
        config=results["config"],
        reference_budgets=[
            record["config"]["reference_samples"],
            record["config"]["check_samples"],
        ],
        validation_groups=len({v["group"] for v in record["views"] if v["split"] == "val"}),
        source_hashes=results["source_hashes"],
        trials=[],
    )
    for trial in results["trials"]:
        row = dict(
            name=trial["name"],
            seed=trial["seed"],
            parameters=trial["parameters"],
            checkpoint_sha256=trial["checkpoint_sha256"],
            metrics={},
            regions={},
        )
        for reference in ("target", "check"):
            row["metrics"][reference] = {
                method: {
                    metric: scene_mean(
                        trial["rows"], lambda r: r["metrics"][reference][method][metric]
                    )
                    for metric in ("ssim", "edge", "hdr_mse", "model_hdr_mse")
                }
                for method in ("raw", "atrous", "neural")
            }
        for region in ("edge_2px", "highlight", "model_region", "bypass_region"):
            row["regions"][region] = dict(
                reference_disagreement_mse=scene_mean(
                    trial["rows"], lambda r: r["reference_disagreement"][region]["linear_mse"]
                ),
                methods={
                    reference: {
                        method: scene_mean(
                            trial["rows"],
                            lambda r: r["metrics"][reference][method]["regions"][region][
                                "linear_mse"
                            ],
                        )
                        for method in ("raw", "atrous", "neural")
                    }
                    for reference in ("target", "check")
                },
            )
        output["trials"].append(row)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new report path")
    write_json(args.output, summarize(args.cohort, args.comparison))


if __name__ == "__main__":
    main()
