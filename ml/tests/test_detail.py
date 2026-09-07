import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from raytracer_ml.data.scenes import configurations
from raytracer_ml.preprocessing import load_features, validate_features
from raytracer_ml.diagnostics import detail_metrics, grouped_summary


def test_detail_metrics_detect_blur_and_hdr_energy():
    from scipy.ndimage import gaussian_filter

    target = np.zeros((32, 32, 3), np.float32)
    target[8:24, 8:24] = 2
    blurred = gaussian_filter(target, sigma=(1.5, 1.5, 0))
    exact, softened = detail_metrics(target, target), detail_metrics(blurred, target)
    assert exact["edge_2px_gradient_mae"] == 0
    assert softened["edge_2px_gradient_mae"] > 0.001
    assert detail_metrics(target * 0.5, target)["highlight_energy_relative_bias"] == -0.5
    rows = [
        {"id": str(i), "group": group, "metrics": {"neural": {"error": value}}}
        for i, (group, value) in enumerate([("a", 1), ("a", 1), ("b", 3)])
    ]
    summary = grouped_summary(rows, "neural", "error")
    assert summary["groups"] == 2 and summary["group_mean"] == 2


def test_challenge_strata_and_split_identity():
    cfg = dict(groups=16, split_counts=[8, 4, 4], views=2, seed=700, suite="challenge-v2")
    items = list(configurations(cfg))
    assert len(items) == 32
    for split in ("train", "val", "test"):
        assert {i["stratum"] for i in items if i["split"] == split} == {
            "edges",
            "textures",
            "lighting",
            "clutter",
        }
    for a, b in zip(items[::2], items[1::2]):
        assert a["scene"]["objects"] == b["scene"]["objects"]
        assert a["scene"]["camera"] != b["scene"]["camera"]
    cfg["split_counts"] = [8, 4, 5]
    with pytest.raises(ValueError, match="split_counts"):
        list(configurations(cfg))


def test_render_challenge_mesh_and_texture(tmp_path):
    binary = Path(os.environ.get("RAYTRACER_BINARY", "build-neural/raytracer")).resolve()
    if not binary.exists():
        pytest.skip("Renderer unavailable")
    items = list(configurations(dict(groups=4, views=1, seed=900, suite="challenge-v2")))
    for item in items:
        path = tmp_path / "scene.json"
        path.write_text(json.dumps(item["scene"]))
        common = [
            str(binary),
            "--headless",
            "--quiet",
            "--scene-file",
            str(path),
            "--width",
            "32",
            "--samples",
            "2",
            "--seed",
            "83",
            "--depth",
            "8",
        ]
        subprocess.run(
            common + ["--output", str(tmp_path / "raw.pfm")], check=True, capture_output=True
        )
        subprocess.run(
            common
            + ["--features", str(tmp_path / "features"), "--output", str(tmp_path / "with.pfm")],
            check=True,
            capture_output=True,
        )
        assert (tmp_path / "raw.pfm").read_bytes() == (tmp_path / "with.pfm").read_bytes()
        x = load_features(tmp_path / "features", 2)
        validate_features(x)
        assert np.isfinite(x).all()
    bad = items[0]["scene"]
    next(o for o in bad["objects"] if o["type"] == "mesh")["faces"][0][0] = -1
    path.write_text(json.dumps(bad))
    failed = subprocess.run(
        common + ["--output", str(tmp_path / "invalid.pfm")], capture_output=True, text=True
    )
    assert failed.returncode != 0 and "Mesh index" in failed.stderr


def test_diverse_families_are_balanced_and_reproducible():
    from raytracer_ml.data.families import FAMILIES
    from raytracer_ml.io import identity

    cfg = dict(groups=96, views=2, split_counts=[32, 32, 32], seed=5001, suite="families-v3")
    items = list(configurations(cfg))
    assert identity(items) == identity(list(configurations(cfg)))
    for split in ("train", "val", "test"):
        cohort = [r for r in items if r["split"] == split]
        assert {r["family"] for r in cohort} == set(FAMILIES)
        assert len({(r["family"], r["stratum"]) for r in cohort}) == 32
    geometries = {}
    for a, b in zip(items[::2], items[1::2]):
        assert a["scene"]["objects"] == b["scene"]["objects"]
        assert a["scene"]["lights"] == b["scene"]["lights"]
        assert a["scene"]["camera"] != b["scene"]["camera"]
        key = identity(a["scene"]["objects"])
        assert key not in geometries
        geometries[key] = a["split"]


def test_render_every_family_and_stratum(tmp_path):
    binary = Path(os.environ.get("RAYTRACER_BINARY", "build-detail/raytracer")).resolve()
    if not binary.is_file():
        pytest.skip("Renderer unavailable")
    for item in configurations(dict(groups=32, views=1, seed=9001, suite="families-v3")):
        path = tmp_path / "scene.json"
        path.write_text(json.dumps(item["scene"]))
        run = subprocess.run(
            [
                str(binary),
                "--headless",
                "--quiet",
                "--scene-file",
                str(path),
                "--width",
                "32",
                "--samples",
                "2",
                "--threads",
                "2",
                "--depth",
                "8",
                "--features",
                str(tmp_path / "features"),
                "--output",
                str(tmp_path / "out.pfm"),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, (item["family"], item["stratum"], run.stderr)
        x = load_features(tmp_path / "features", 2)
        validate_features(x)
        assert np.any(x[11] > 0.5) and np.any(x[:3] > 0.01)
