import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from raytracer_ml.data.generate import generate
from raytracer_ml.data.validate import validate
from raytracer_ml.io import read_pfm, write_pfm, manifest, write_json
from raytracer_ml.models import build_model
from raytracer_ml.preprocessing import validate_features, load_features
from raytracer_ml.train import train, load_checkpoint
from raytracer_ml.export import export_model


@pytest.fixture(scope="session")
def binary():
    path = Path(
        os.environ.get(
            "RAYTRACER_BINARY", str(Path(__file__).resolve().parents[2] / "build/raytracer")
        )
    ).resolve()
    if not path.is_file():
        pytest.skip("Build the renderer or set RAYTRACER_BINARY")
    return path


@pytest.fixture(scope="session")
def dataset(tmp_path_factory, binary):
    root = tmp_path_factory.mktemp("dataset")
    cfg = {
        "schema_version": 1,
        "seed": 73,
        "groups": 4,
        "views": 1,
        "width": 32,
        "budgets": [1, 4],
        "noise_realizations": 1,
        "reference_samples": 16,
        "reference_checks": 0,
        "scale": 1,
        "depth": 8,
        "threads": 2,
        "max_seconds": 60,
        "max_gib": 1,
    }
    generate(cfg, binary, root)
    return root, cfg


def test_pfm_hdr_row_order(tmp_path):
    image = np.arange(72, dtype=np.float32).reshape(4, 6, 3) / 2
    path = tmp_path / "test.pfm"
    write_pfm(path, image)
    np.testing.assert_array_equal(read_pfm(path), image)
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError):
        read_pfm(path)


def test_generate_validate_resume(dataset, binary):
    root, cfg = dataset
    before = (root / "manifest.jsonl").read_bytes()
    assert validate(root)["examples"] == 8
    result = generate(cfg, binary, root)
    assert result["completed"] == 8
    assert (root / "manifest.jsonl").read_bytes() == before
    changed = {**cfg, "seed": 74}
    with pytest.raises(ValueError, match="changed"):
        generate(changed, binary, root)


def test_split_and_checksum_rejection(dataset, tmp_path):
    root, _ = dataset
    copy_root = tmp_path / "copy"
    shutil.copytree(root, copy_root)
    rows = manifest(copy_root / "manifest.jsonl")
    rows[1]["split"] = "test"
    (copy_root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="leaks"):
        validate(copy_root)
    shutil.copy(root / "manifest.jsonl", copy_root / "manifest.jsonl")
    (copy_root / rows[0]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate(copy_root)


def test_features_leave_raw_unchanged(dataset, binary, tmp_path):
    root, _ = dataset
    scene = manifest(root / "manifest.jsonl")[0]["scene"]
    write_json(tmp_path / "scene.json", scene)
    base = [
        str(binary),
        "--headless",
        "--quiet",
        "--scene-file",
        str(tmp_path / "scene.json"),
        "--width",
        "32",
        "--samples",
        "4",
        "--seed",
        "51",
        "--threads",
        "2",
    ]
    subprocess.run(base + ["--output", str(tmp_path / "raw.pfm")], check=True, capture_output=True)
    subprocess.run(
        base
        + ["--output", str(tmp_path / "features.pfm"), "--features", str(tmp_path / "buffers")],
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "raw.pfm").read_bytes() == (tmp_path / "features.pfm").read_bytes()
    x = load_features(tmp_path / "buffers")
    validate_features(x)
    np.testing.assert_array_equal(x[:3].transpose(1, 2, 0), read_pfm(tmp_path / "raw.pfm"))
    assert (x[12:15] >= -1e-8).all()
    stats = json.loads((tmp_path / "features.json").read_text())
    assert stats["feature_rays"] == 32 * 18
    boundary = load_features(tmp_path / "buffers", schema=2)
    validate_features(boundary)
    assert boundary.shape == (27, 18, 32)
    np.testing.assert_array_equal(boundary[:17], x)
    assert np.any(boundary[23] > 0)
    assert np.any(boundary[24] > 0)


def test_unknown_variance_rejected(dataset):
    root, _ = dataset
    r = next(r for r in manifest(root / "manifest.jsonl") if r["samples"] == 1)
    with np.load(root / r["path"]) as data:
        x = data["features"].copy()
    x[16] = 1
    with pytest.raises(ValueError, match="variance"):
        validate_features(x)


@pytest.mark.parametrize("kind,scale", [("conv", 1), ("unet", 1), ("unet", 2)])
def test_model_shape_fallback_and_gradients(kind, scale):
    torch.set_num_threads(1)
    model = build_model({"kind": kind, "width": 8, "scale": scale})
    x = torch.rand(2, 17, 17, 23)
    x[:, 10:12] = 1
    x[:, 15] = 4
    x[:, 16] = 1
    x[:, :, 0, :] = 0
    y = model(x)
    assert y.shape == (2, 3, 17 * scale, 23 * scale)
    assert torch.isfinite(y).all() and (y >= 0).all()
    torch.mean((y - 0.2) ** 2).backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert torch.equal(y[:, :, 0, :], torch.zeros_like(y[:, :, 0, :]))


@pytest.fixture(scope="session")
def trained(dataset, tmp_path_factory):
    root, _ = dataset
    output = tmp_path_factory.mktemp("run")
    cfg = {
        "seed": 4,
        "device": "cpu",
        "cpu_threads": 1,
        "model": {"kind": "conv", "width": 8, "scale": 1, "inputs": "all"},
        "crop": 16,
        "batch_size": 2,
        "epochs": 3,
        "patience": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_seconds": 60,
    }
    train(cfg, root, output)
    return root, output, cfg


def test_checkpoint_resume_contract(trained):
    root, output, cfg = trained
    state = load_checkpoint(output / "latest.pt")
    assert state["epoch"] == 2 and "optimizer" in state and "rng" in state
    summary = train(cfg, root, output, resume=True)
    assert summary["reason"] == "epochs-complete"
    with pytest.raises(ValueError, match="differs"):
        train({**cfg, "seed": 5}, root, output, resume=True)


def test_export_parity(trained, tmp_path):
    _, output, _ = trained
    meta = export_model(output / "best.pt", tmp_path / "model.onnx")
    assert meta["parity_max_absolute_error"] < 2e-5
    assert len(meta["channels"]) == 17


def test_native_prediction_matches_python(trained, binary, tmp_path):
    neural = os.environ.get("RAYTRACER_NEURAL_BINARY")
    if not neural:
        pytest.skip("Set RAYTRACER_NEURAL_BINARY to test optional native runtime")
    root, output, _ = trained
    export_model(output / "best.pt", tmp_path / "model.onnx")
    row = manifest(root / "manifest.jsonl")[0]
    write_json(tmp_path / "scene.json", row["scene"])
    command = [
        neural,
        "--headless",
        "--quiet",
        "--scene-file",
        str(tmp_path / "scene.json"),
        "--width",
        "32",
        "--samples",
        "4",
        "--threads",
        "2",
        "--model",
        str(tmp_path / "model.onnx"),
        "--features",
        str(tmp_path / "features"),
        "--raw-output",
        str(tmp_path / "raw.pfm"),
        "--output",
        str(tmp_path / "predicted.pfm"),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)["reconstructed"]
    state = load_checkpoint(output / "best.pt")
    model = build_model(state["config"]["model"]).eval()
    model.load_state_dict(state["model"])
    with torch.inference_mode():
        expected = (
            model(torch.from_numpy(load_features(tmp_path / "features")[None]))
            .numpy()[0]
            .transpose(1, 2, 0)
        )
    np.testing.assert_allclose(read_pfm(tmp_path / "predicted.pfm"), expected, rtol=2e-4, atol=2e-5)
    np.testing.assert_array_equal(
        read_pfm(tmp_path / "raw.pfm"), read_pfm(tmp_path / "features/radiance.pfm")
    )


def test_epoch_resume_matches_uninterrupted(dataset, tmp_path):
    root, _ = dataset
    cfg = {
        "seed": 123,
        "device": "cpu",
        "cpu_threads": 1,
        "model": {"kind": "conv", "width": 8, "scale": 1, "inputs": "all"},
        "crop": 16,
        "batch_size": 2,
        "epochs": 3,
        "patience": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_seconds": 60,
    }
    train(cfg, root, tmp_path / "whole")
    partial = train(cfg, root, tmp_path / "resumed", max_new_epochs=1)
    assert partial["reason"] == "paused-at-epoch-boundary"
    train(cfg, root, tmp_path / "resumed", resume=True)
    a = load_checkpoint(tmp_path / "whole/latest.pt")
    b = load_checkpoint(tmp_path / "resumed/latest.pt")
    assert a["step"] == b["step"]
    for name, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)


def test_temporal_reprojection_rejects_changes():
    import copy
    from raytracer_ml.temporal import reproject

    scene = {
        "camera": {"lookfrom": [0, 0, 2], "lookat": [0, 0, 0], "up": [0, 1, 0], "vfov": 90},
        "objects": [],
    }
    position = np.array(
        [[[-0.5, 0.5, 1], [0.5, 0.5, 1]], [[-0.5, -0.5, 1], [0.5, -0.5, 1]]], dtype=np.float32
    )
    x = np.zeros((17, 2, 2), dtype=np.float32)
    x[8] = 1
    x[11] = 1
    rgb = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    history, mask = reproject(position, x, scene, position, x, scene, rgb)
    np.testing.assert_array_equal(history, rgb)
    assert mask.all()
    changed = copy.deepcopy(scene)
    changed["lights"] = ["changed"]
    assert not reproject(position, x, changed, position, x, scene, rgb)[1].any()
    moved = copy.deepcopy(scene)
    moved["camera"]["lookfrom"] = [8, 0, 2]
    assert not reproject(position, x, moved, position, x, scene, rgb)[1].any()
    assert not reproject(position, x, scene, position + 1, x, scene, rgb)[1].any()
    assert not reproject(position, x * 0, scene, position, x, scene, rgb)[1].any()


@pytest.mark.parametrize(
    "temporal,scale,schema,stationary",
    [(False, 2, 1, False), (True, 1, 1, False), (True, 1, 2, False), (True, 1, 2, True)],
)
def test_extensions_end_to_end(binary, tmp_path, temporal, scale, schema, stationary):
    from raytracer_ml.evaluate import evaluate
    from raytracer_ml.temporal import reproject, append_history
    from raytracer_ml.benchmark import json_stream

    root = tmp_path / "data"
    generate(
        {
            "schema_version": 1,
            "feature_schema": schema,
            "seed": 101,
            "groups": 4,
            "views": 2,
            "width": 32,
            "budgets": [1, 4],
            "noise_realizations": 1,
            "reference_samples": 16,
            "reference_checks": 0,
            "scale": scale,
            "depth": 8,
            "threads": 2,
            "max_seconds": 60,
            "max_gib": 1,
        },
        binary,
        root,
    )
    run = tmp_path / "run"
    cfg = {
        "seed": 4,
        "device": "cpu",
        "cpu_threads": 1,
        "model": {
            "kind": "guided" if schema == 2 else "conv",
            "width": 8,
            "scale": scale,
            "inputs": "all",
            "temporal": temporal,
            "feature_schema": schema,
        },
        "crop": 0 if schema == 2 else 16,
        "batch_size": 2,
        "epochs": 2,
        "patience": 8,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_seconds": 60,
    }
    if schema == 2:
        cfg["autoregressive_unroll"] = 2
        cfg["augmentation"] = dict(exposure_stops=1.5, lighting_color_stops=0.2)
    train(cfg, root, run)
    result = evaluate(root, run / "best.pt", tmp_path / "eval", device_name="cpu", repeats=1)
    assert result["images"] == 4
    meta = export_model(run / "best.pt", tmp_path / "model.onnx")
    assert len(meta["channels"]) == ((27 if schema == 2 else 17) + (4 if temporal else 0))
    neural = os.environ.get("RAYTRACER_NEURAL_BINARY")
    if not neural:
        return
    rows = [
        r
        for r in manifest(root / "manifest.jsonl")
        if r["group"] == "layout-0003" and r["samples"] == 4
    ]
    sequence = tmp_path / "sequence"
    sequence.mkdir()
    previous = None
    state = load_checkpoint(run / "best.pt")
    model = build_model(cfg["model"]).eval()
    model.load_state_dict(state["model"])
    expected_frames = []
    for i, row in enumerate(rows):
        if stationary:
            row["scene"] = rows[0]["scene"]
        scene_path = sequence / f"{i:03}.json"
        write_json(scene_path, row["scene"])
        buffers = tmp_path / f"features-{i}"
        subprocess.run(
            [
                neural,
                "--headless",
                "--quiet",
                "--scene-file",
                str(scene_path),
                "--width",
                "32",
                "--samples",
                "4",
                "--depth",
                "8",
                "--seed",
                str(42 + i),
                "--threads",
                "2",
                "--features",
                str(buffers),
                "--output",
                str(tmp_path / "raw.pfm"),
            ],
            check=True,
            capture_output=True,
        )
        x = load_features(buffers, schema)
        position = read_pfm(buffers / "position.pfm")
        inputs = x
        if temporal:
            history = np.zeros((18, 32, 3), dtype=np.float32)
            mask = np.zeros((18, 32, 1), dtype=np.float32)
            if previous:
                history, mask = reproject(position, x, row["scene"], *previous)
                assert mask.mean() > 0.05
            inputs = append_history(x, history, mask)
        with torch.inference_mode():
            predicted = model(torch.from_numpy(inputs[None])).numpy()[0].transpose(1, 2, 0)
        previous = (position, x, row["scene"], predicted)
        expected_frames.append(predicted)
    command = [
        neural,
        "--headless",
        "--quiet",
        "--sequence",
        str(sequence),
        "--width",
        "32",
        "--samples",
        "4",
        "--depth",
        "8",
        "--seed",
        "42",
        "--threads",
        "2",
        "--model",
        str(tmp_path / "model.onnx"),
        "--output",
        str(tmp_path / "sequence.pfm"),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stats = json_stream(result.stdout)
    assert len(stats) == 2 and all(r["reconstructed"] for r in stats)
    for i, expected in enumerate(expected_frames):
        np.testing.assert_allclose(
            read_pfm(tmp_path / f"sequence-{i:06}.pfm"), expected, rtol=3e-4, atol=3e-5
        )


def test_missing_model_preserves_raw(binary, tmp_path):
    cmd = [
        str(binary),
        "--headless",
        "--quiet",
        "--scene",
        "demo",
        "--width",
        "32",
        "--samples",
        "2",
        "--threads",
        "2",
        "--output",
        str(tmp_path / "raw.pfm"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    expected = (tmp_path / "raw.pfm").read_bytes()
    result = subprocess.run(
        cmd + ["--model", str(tmp_path / "missing.onnx")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not json.loads(result.stdout)["reconstructed"]
    assert "fallback" in result.stderr.lower()
    assert (tmp_path / "raw.pfm").read_bytes() == expected


def test_cohort_identity_and_public_schema():
    from raytracer_ml.data.scenes import configurations
    from raytracer_ml.contracts import EXAMPLE_SCHEMA

    items = list(configurations({"groups": 4, "views": 2, "seed": 10, "cohort_groups": 1}))
    base = items[0]
    camera = next(x for x in items if x["cohort"] == "heldout-camera")
    light = next(x for x in items if x["cohort"] == "new-light")
    assert camera["scene"]["objects"] == base["scene"]["objects"] == light["scene"]["objects"]
    assert camera["scene"]["camera"] != base["scene"]["camera"]
    assert light["scene"]["lights"] != base["scene"]["lights"]
    assert camera["group"] != base["group"] and camera["split"] == light["split"] == "test"
    schema = Path(__file__).parents[1] / "schemas/example.schema.json"
    assert json.loads(schema.read_text()) == EXAMPLE_SCHEMA


def test_scene_serialization_and_invalid_camera(binary, tmp_path):
    scene = tmp_path / "scene.json"
    command = [
        str(binary),
        "--headless",
        "--quiet",
        "--scene",
        "field",
        "--scene-seed",
        "7",
        "--seed",
        "42",
        "--width",
        "32",
        "--samples",
        "1",
        "--threads",
        "2",
        "--output",
        str(tmp_path / "a.pfm"),
        "--write-scene",
        str(scene),
    ]
    subprocess.run(command, check=True, capture_output=True)
    settings = json.loads(scene.read_text())
    assert settings["scene_seed"] == 7
    subprocess.run(
        [
            str(binary),
            "--headless",
            "--quiet",
            "--scene-file",
            str(scene),
            "--seed",
            "42",
            "--width",
            "32",
            "--samples",
            "1",
            "--threads",
            "2",
            "--output",
            str(tmp_path / "b.pfm"),
        ],
        check=True,
        capture_output=True,
    )
    assert (tmp_path / "a.pfm").read_bytes() == (tmp_path / "b.pfm").read_bytes()
    settings["camera"]["lookat"] = settings["camera"]["lookfrom"]
    write_json(scene, settings)
    result = subprocess.run(
        command[:-2] + ["--scene-file", str(scene)], capture_output=True, text=True
    )
    assert result.returncode != 0 and "camera" in result.stderr.lower()


def test_incompatible_model_falls_back(trained, tmp_path):
    import onnx

    neural = os.environ.get("RAYTRACER_NEURAL_BINARY")
    if not neural:
        pytest.skip("Optional runtime")
    _, run, _ = trained
    model = tmp_path / "bad.onnx"
    export_model(run / "best.pt", model)
    graph = onnx.load(model)
    for entry in graph.metadata_props:
        if entry.key == "rt_channels":
            entry.value = "[]"
    onnx.save(graph, model)
    result = subprocess.run(
        [
            neural,
            "--headless",
            "--quiet",
            "--width",
            "32",
            "--samples",
            "1",
            "--threads",
            "2",
            "--model",
            str(model),
            "--output",
            str(tmp_path / "fallback.pfm"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not json.loads(result.stdout)["reconstructed"]
    assert "metadata" in result.stderr.lower()


def test_export_sequence(dataset, tmp_path):
    from raytracer_ml.sequence import export_sequence

    root, _ = dataset
    out = tmp_path / "sequence"
    result = export_sequence(root, "layout-0003", out)
    assert result["frames"] == 1
    assert json.loads((out / "000000.json").read_text())["schema_version"] == 1
    with pytest.raises(ValueError, match="empty"):
        export_sequence(root, "layout-0003", out)
