"""Physical augmentation invariants, compact round trips, and native schema-2 parity."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from raytracer_ml.augment import (
    apply_transform,
    fuse_measurements,
    spatial_transform,
    undo_prediction,
)
from raytracer_ml.data.arrays import load_example
from raytracer_ml.data.generate import generate
from raytracer_ml.data.validate import validate
from raytracer_ml.export import export_model
from raytracer_ml.io import manifest, read_pfm, write_json
from raytracer_ml.models import build_model
from raytracer_ml.preprocessing import load_features
from raytracer_ml.train import train, load_checkpoint


def test_additive_head_can_reconstruct_missing_signal_without_changing_fallback():
    cfg = dict(kind="unet", width=4, feature_schema=2)
    bounded = build_model(cfg)
    additive = build_model({**cfg, "output_head": "additive_log"})
    with torch.no_grad():
        bounded.head.bias.fill_(3)
    additive.load_state_dict(bounded.state_dict())
    x = torch.zeros(1, 27, 8, 12)
    x[:, 10:12], x[:, 26], x[:, 15] = 1, 1, 4
    x[:, 11, 0] = 0
    x[:, 15, 1] = 128
    y = additive(x)
    assert bounded(x).max() < 0.064
    assert y[:, :, 2:].min() > 10
    torch.testing.assert_close(y[:, :, :2], x[:, :3, :2], atol=0, rtol=0)
    y[:, :, 2:].sum().backward()
    assert additive.head.bias.grad.min() > 0


def test_boundary_feature_control_keeps_policy_and_capacity():
    cfg = dict(kind="unet", width=4, feature_schema=2, output_head="additive_log")
    all_features = build_model(cfg)
    ablated = build_model({**cfg, "inputs": "no_boundary"})
    with torch.no_grad():
        all_features.head.weight.fill_(0.05)
        all_features.head.bias.fill_(0.2)
    ablated.load_state_dict(all_features.state_dict())
    assert sum(p.numel() for p in ablated.parameters()) == sum(
        p.numel() for p in all_features.parameters()
    )
    x = torch.rand(1, 27, 9, 13)
    x[:, 10:12], x[:, 26], x[:, 15] = 1, 1, 4
    other = x.clone()
    other[:, 17:26] = 1 - other[:, 17:26]
    torch.testing.assert_close(ablated(x), ablated(other), atol=0, rtol=0)
    x[:, 26, :2] = 0
    x[:, 23, :2] = 1
    torch.testing.assert_close(ablated(x)[:, :, :2], x[:, :3, :2], atol=0, rtol=0)


@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize(
    "flip_x,flip_y", [(False, False), (True, False), (False, True), (True, True)]
)
def test_paired_hdr_augmentation_and_inverse(turns, flip_x, flip_y):
    x = torch.arange(31 * 5 * 7, dtype=torch.float32).reshape(31, 5, 7) / 32
    target = x[:3] * 3
    transform = dict(turns=turns, flip_x=flip_x, flip_y=flip_y, gain=torch.tensor([0.5, 2.0, 4.0]))
    before = x.clone()
    transformed, truth = apply_transform(x, target, transform, 27)
    spatial = spatial_transform(x, transform)
    gain = transform["gain"][:, None, None]
    torch.testing.assert_close(transformed[:3], spatial[:3] * gain)
    torch.testing.assert_close(transformed[12:15], spatial[12:15] * gain.square())
    torch.testing.assert_close(transformed[27:30], spatial[27:30] * gain)
    # Albedo, world-space vector components, sample counts, and validity do not change values.
    for section in (slice(3, 12), slice(15, 27), slice(30, 31)):
        torch.testing.assert_close(transformed[section], spatial[section], rtol=0, atol=0)
    torch.testing.assert_close(undo_prediction(truth, transform), target, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    batch, batched_truth = apply_transform(x[None], target[None], transform, 27)
    torch.testing.assert_close(batch[0], transformed)
    torch.testing.assert_close(batched_truth[0], truth)


def test_noise_fusion_matches_combined_samples():
    rng = np.random.default_rng(34)
    radiance = rng.exponential(5, (7, 3, 4, 6))
    depths = rng.uniform(0, 8, (7, 4, 6))
    normals = rng.normal(size=(7, 3, 4, 6))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    def measurement(indices):
        n = len(indices)
        x = np.zeros((27, 4, 6), np.float32)
        x[:3] = radiance[indices].mean(0)
        x[6:9] = normals[indices].mean(0)
        x[9] = depths[indices].mean(0)
        x[10:12] = 1
        x[12:15] = radiance[indices].var(0, ddof=1) / n if n > 1 else 0
        x[15], x[16] = n, n > 1
        x[24] = depths[indices].var(0)
        x[25] = 1 - (x[6:9] ** 2).sum(0)
        return x

    for size in (1, 3):
        actual = fuse_measurements(
            measurement(list(range(size))), measurement(list(range(size, 7)))
        )
        np.testing.assert_allclose(actual, measurement(list(range(7))), rtol=5e-6, atol=2e-6)
    a, b = measurement([0]), measurement([1])
    b[23] = 2
    with pytest.raises(ValueError, match="center geometry"):
        fuse_measurements(a, b)


@pytest.mark.parametrize("kind", ["unet", "guided", "refine"])
def test_detail_models_preserve_unsupported_and_converged_pixels(kind):
    torch.set_num_threads(1)
    model = build_model(dict(kind=kind, width=8, feature_schema=2))
    x = torch.rand(1, 27, 17, 23)
    x[:, 10:12], x[:, 26], x[:, 15] = 1, 1, 4
    x[:, :3] *= 20
    x[:, 11, :4] = 0
    x[:, 15, 10:] = 128
    y = model(x)
    assert torch.isfinite(y).all() and (y >= 0).all()
    torch.testing.assert_close(y[:, :, :4], x[:, :3, :4], rtol=0, atol=0)
    torch.testing.assert_close(y[:, :, 10:], x[:, :3, 10:], rtol=0, atol=0)
    y.mean().backward()
    assert model.head.weight.grad.abs().sum() > 0


def test_guided_constant_hdr_including_borders():
    model = build_model(dict(kind="guided", width=8, feature_schema=2))
    x = torch.ones(1, 27, 17, 23)
    x[:, :3], x[:, 15] = 3000, 4
    torch.testing.assert_close(model(x), x[:, :3], rtol=2e-6, atol=0)


@pytest.fixture(scope="module")
def compact_dataset(tmp_path_factory):
    binary = Path(os.environ.get("RAYTRACER_BINARY", "build-detail/raytracer")).resolve()
    if not binary.is_file():
        pytest.skip("Renderer unavailable")
    root = tmp_path_factory.mktemp("compact")
    cfg = dict(
        schema_version=1,
        feature_schema=2,
        seed=973,
        suite="challenge-v2",
        groups=4,
        views=1,
        width=32,
        budgets=[1, 4],
        noise_realizations=2,
        reference_samples=16,
        scale=1,
        depth=8,
        threads=2,
        max_seconds=120,
        max_gib=1,
    )
    generate(cfg, binary, root / "full")
    generate({**cfg, "compact_storage": True}, binary, root / "compact")
    return root, cfg, binary


def test_compact_storage_is_lossless_and_resumes(compact_dataset):
    root, cfg, binary = compact_dataset
    full, compact = root / "full", root / "compact"
    assert validate(compact)["examples"] == 16
    for a, b in zip(manifest(full / "manifest.jsonl"), manifest(compact / "manifest.jsonl")):
        plain, packed = load_example(full, a), load_example(compact, b)
        for key in plain:
            np.testing.assert_array_equal(plain[key], packed[key])
    previous = (compact / "manifest.jsonl").read_bytes()
    generate({**cfg, "compact_storage": True}, binary, compact)
    assert previous == (compact / "manifest.jsonl").read_bytes()


def test_measured_preservation_uses_real_independent_noise(compact_dataset):
    from raytracer_ml.data.dataset import RenderDataset

    root = compact_dataset[0] / "compact"
    dataset = RenderDataset(
        root,
        "train",
        feature_schema=2,
        identity_probability=1,
        preservation_mode="measured_near_clean",
        near_clean_samples=5,
    )
    x, target = dataset[0]
    assert torch.all(x[15] == 5) and torch.all(x[16] == 1)
    assert (x[12:15] > 0).any()
    assert not torch.equal(x[:3], target)
    pairs = dataset.near_clean_pairs[dataset.view_key(dataset.rows[0])]
    expected = [
        fuse_measurements(load_example(root, a)["features"], load_example(root, b)["features"])
        for a, b in pairs
    ]
    assert any(np.array_equal(x.numpy(), candidate) for candidate in expected)
    model = build_model(dict(kind="unet", width=4, feature_schema=2, output_head="additive_log"))
    with torch.no_grad():
        model.head.bias.fill_(0.2)
    (model(x[None]) - target[None]).square().mean().backward()
    assert model.head.bias.grad.abs().sum() > 0
    heldout = RenderDataset(
        root,
        "val",
        feature_schema=2,
        identity_probability=1,
        preservation_mode="measured_near_clean",
        near_clean_samples=5,
    )
    actual, _ = heldout[0]
    np.testing.assert_array_equal(actual, load_example(root, heldout.rows[0])["features"])
    with pytest.raises(ValueError, match="independent measurements"):
        RenderDataset(
            root,
            "train",
            feature_schema=2,
            identity_probability=1,
            preservation_mode="measured_near_clean",
            near_clean_samples=7,
        )


def test_augmented_epoch_resume_is_exact(compact_dataset, tmp_path):
    root = compact_dataset[0] / "compact"
    cfg = dict(
        seed=103,
        device="cpu",
        cpu_threads=1,
        model=dict(kind="guided", width=8, feature_schema=2),
        crop=16,
        batch_size=2,
        epochs=3,
        patience=8,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_seconds=120,
        edge_sampling=0.5,
        identity_probability=0.2,
        fuse_probability=0.75,
        augmentation=dict(exposure_stops=1.5, lighting_color_stops=0.2),
        loss=dict(gradient=0.25, energy=0.02),
        selection=dict(ssim_tolerance=0.005),
        learning_diagnostics=dict(stall_after_epochs=2),
    )
    train(cfg, root, tmp_path / "whole")
    train(cfg, root, tmp_path / "resumed", max_new_epochs=1)
    train(cfg, root, tmp_path / "resumed", resume=True)
    a, b = (load_checkpoint(tmp_path / p / "latest.pt") for p in ("whole", "resumed"))
    assert a["step"] == b["step"] and a["best"] == b["best"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    reports = [
        json.loads(line) for line in (tmp_path / "whole/metrics.jsonl").read_text().splitlines()
    ]
    assert all("learning_health" in row and "learning_rate" in row for row in reports)


@pytest.mark.parametrize(
    "kind,scale,precision",
    [
        ("guided", 1, "fp32"),
        ("refine", 2, "fp32"),
        ("guided", 1, "mixed-fp16"),
        ("unet", 1, "fp32"),
    ],
)
def test_detail_export_and_native_parity(kind, scale, precision, compact_dataset, tmp_path):
    cfg = dict(kind=kind, width=8, feature_schema=2, scale=scale)
    if kind == "unet":
        cfg["output_head"] = "additive_log"
        cfg["activation"] = "leaky_relu"
        cfg["radiance_scale"] = 16
    torch.manual_seed(77)
    model = build_model(cfg).eval()
    # Nonzero learned weights exercise more than an untrained identity path.
    with torch.no_grad():
        model.head.weight.normal_(0, 0.01)
        if scale == 2:
            model.upscale[2].weight.normal_(0, 0.01)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        dict(config=dict(model=cfg), model=model.state_dict(), manifest_sha256="test"), checkpoint
    )
    meta = export_model(checkpoint, tmp_path / "model.onnx", precision=precision)
    assert len(meta["channels"]) == 27
    neural = os.environ.get("RAYTRACER_NEURAL_BINARY")
    if not neural or precision != "fp32":
        return
    row = manifest(compact_dataset[0] / "compact/manifest.jsonl")[0]
    write_json(tmp_path / "scene.json", row["scene"])
    result = subprocess.run(
        [
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
            "--output",
            str(tmp_path / "out.pfm"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["reconstructed"]
    with torch.inference_mode():
        expected = (
            model(torch.from_numpy(load_features(tmp_path / "features", 2)[None]))[0]
            .numpy()
            .transpose(1, 2, 0)
        )
    np.testing.assert_allclose(read_pfm(tmp_path / "out.pfm"), expected, rtol=2e-4, atol=2e-5)
    if kind == "unet":
        assert (
            model.output_head == "additive_log" and meta["model"]["output_head"] == "additive_log"
        )
        return  # The U-Net does not declare local tiling support.
    command = result.args.copy()
    command[command.index("--width") + 1] = "96"
    subprocess.run(command, check=True, capture_output=True)
    whole = read_pfm(tmp_path / "out.pfm")
    for tile in (16, 31):
        tiled = subprocess.run(
            command + ["--neural-tile", str(tile)], check=True, capture_output=True, text=True
        )
        stats = json.loads(tiled.stdout)
        assert stats["reconstructed"] and stats["neural_tiles"] > 1
        np.testing.assert_allclose(read_pfm(tmp_path / "out.pfm"), whole, rtol=2e-4, atol=2e-5)
