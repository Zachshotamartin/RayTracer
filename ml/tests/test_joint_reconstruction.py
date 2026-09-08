"""Native multi-size pairing, reconstruction gradients, resume, and runtime parity."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from raytracer_ml.augment import apply_transform, draw_transform
from raytracer_ml.data.dataset import RenderDataset
from raytracer_ml.data.diversity import rendering_variants
from raytracer_ml.data.generate import generate
from raytracer_ml.data.preservation import PreservationValidation
from raytracer_ml.data.scenes import configurations
from raytracer_ml.data.validate import validate
from raytracer_ml.diagnostics import reconstruction_region
from raytracer_ml.export import export_model
from raytracer_ml.io import digest, manifest, read_pfm, write_json
from raytracer_ml.learning_health import model_region_mask
from raytracer_ml.models import build_model
from raytracer_ml.preprocessing import load_features
from raytracer_ml.selection import constraint_report
from raytracer_ml.train import load_checkpoint, train


def data_config():
    return dict(
        schema_version=1,
        feature_schema=2,
        suite="families-v3",
        seed=9813,
        groups=4,
        views=1,
        resolutions=[[32, 24], [24, 32], [32, 32], [40, 24]],
        resolution_variants=2,
        camera_augmentation=dict(
            zoom=[0.7, 1.5],
            roll_degrees=30,
            lighting_filters=[[1, 1, 1], [1.2, 1, 0.8]],
            transport_probability=1,
        ),
        budgets=[1, 4],
        noise_realizations=2,
        reference_samples=32,
        reference_check_samples=64,
        reference_check_strategy="stratified",
        reference_check_limit=3,
        scale=2,
        depth=8,
        threads=2,
        max_seconds=120,
        max_gib=1,
        compact_storage=True,
    )


def train_config():
    return dict(
        seed=234,
        device="cpu",
        cpu_threads=1,
        model=dict(
            kind="joint",
            width=4,
            feature_schema=2,
            scale=2,
            activation="leaky_relu",
            refinement=True,
            radiance_scale=16,
        ),
        crop=16,
        batch_size=2,
        epochs=2,
        patience=5,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_seconds=120,
        checkpoint_history=True,
        edge_sampling=0.5,
        border_sampling=0.5,
        identity_probability=0.5,
        preservation_mode="measured_near_clean",
        near_clean_samples=5,
        fuse_probability=0.25,
        augmentation=dict(
            rotate90=True, exposure_stops=0.5, rgb_filters=[[1.1, 1, 0.9], [0.9, 1, 1.1]]
        ),
        loss=dict(gradient=0.25, gradient_scales=[1, 2, 4], energy=0.02),
        selection=dict(psnr_gain=0.1, hdr_ratio=1, preservation_ratio=1),
        selection_by_resolution=True,
        selection_by_budget=True,
        learning_diagnostics=dict(stall_after_epochs=2),
    )


def test_native_camera_variants_are_deterministic_and_inherit_splits():
    cfg = data_config()
    original = list(configurations(cfg))
    saved = deepcopy(original)
    a = list(rendering_variants(original, cfg))
    assert a == list(rendering_variants(original, cfg)) and original == saved
    assert len({r["id"] for r in a}) == 8
    parents = {r["id"]: r for r in original}
    for r in a:
        p = parents[r["parent_configuration"]]
        assert (r["group"], r["split"]) == (p["group"], p["split"])
        assert r["scene"]["camera"]["aspect_ratio"] == r["width"] / r["height"]
        assert r["scene"]["camera"]["vfov"] != p["scene"]["camera"]["vfov"]
        assert r["render_variant"]["transport"] in ("glass", "metal")
    assert {r["scene"]["camera"]["aspect_ratio"] for r in a} >= {1, 4 / 3, 3 / 4}


def test_joint_head_learns_subpixels_and_non_diffuse_near_clean_pixels():
    cfg = train_config()["model"]
    model = build_model(cfg)
    x = torch.zeros(1, 27, 13, 19)
    x[:, 15] = 128  # The legacy bypass must not prevent spatial reconstruction.
    with torch.no_grad():
        model.head.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4] * 3))
    y = model(x)
    assert y.shape == (1, 3, 26, 38) and torch.isfinite(y).all()
    assert torch.all(y > 0) and not torch.equal(y[:, :, 0::2, 0::2], y[:, :, 1::2, 1::2])
    y.mean().backward()
    assert model.head.bias.grad.abs().sum() > 0
    np.testing.assert_array_equal(
        reconstruction_region(x[0].numpy(), cfg), model_region_mask(x, model)[0, 0].numpy()
    )
    assert reconstruction_region(x[0].numpy(), cfg).all()


def test_filters_and_rotations_preserve_scaled_pair_and_variance():
    x = torch.rand(27, 8, 8)
    target = F.interpolate(x[None, :3], scale_factor=2, mode="nearest")[0]
    transform = draw_transform(dict(flip_x=1, flip_y=1, rgb_filters=[[2, 1, 0.5]]))
    transform["turns"] = 1
    a, b = apply_transform(x, target, transform, 27)
    torch.testing.assert_close(b, F.interpolate(a[None, :3], scale_factor=2, mode="nearest")[0])
    spatial_variance = torch.rot90(x.flip(-2).flip(-1)[12:15], 1, (-2, -1))
    torch.testing.assert_close(
        a[12:15], spatial_variance * torch.tensor([4, 1, 0.25])[:, None, None]
    )
    with pytest.raises(ValueError, match="RGB filters"):
        draw_transform(dict(rgb_filters=[[float("nan"), 1, 1]]))


@pytest.fixture(scope="module")
def joint_data(tmp_path_factory):
    binary = Path(os.environ.get("RAYTRACER_BINARY", "build-detail/raytracer")).resolve()
    if not binary.is_file():
        pytest.skip("Renderer unavailable")
    root = tmp_path_factory.mktemp("joint-data")
    generate(data_config(), binary, root)
    return root, binary


def test_native_pairs_and_preservation_cover_different_shapes(joint_data):
    root, binary = joint_data
    result = validate(root)
    assert result["examples"] == 32 and len(result["resolution_coverage"]["train"]) > 1
    before = (root / "manifest.jsonl").read_bytes()
    generate(data_config(), binary, root)
    assert before == (root / "manifest.jsonl").read_bytes()
    ds = RenderDataset(
        root,
        "train",
        crop=16,
        feature_schema=2,
        identity_probability=1,
        preservation_mode="measured_near_clean",
        near_clean_samples=5,
        border_sampling=1,
        augmentation=dict(rotate90=True),
    )
    for i in range(len(ds)):
        x, target = ds[i]
        assert x.shape == (27, 16, 16) and target.shape == (3, 32, 32)
        assert (x[15] == 5).all() and (x[12:15] > 0).any()

    class Bilinear(torch.nn.Module):
        def forward(self, x):
            return F.interpolate(x[:, :3], scale_factor=2, mode="bilinear", align_corners=False)

    a, b = PreservationValidation(root, samples=5).measure(Bilinear(), torch.device("cpu"))
    assert a == b and a["preservation_views"] == 2
    with pytest.raises(ValueError, match="synthetic identity"):
        RenderDataset(root, "train", feature_schema=2, identity_probability=1)
    with pytest.raises(ValueError, match="Crop exceeds"):
        RenderDataset(root, "train", crop=100)


def test_joint_training_resume_and_resolution_gates(joint_data, tmp_path):
    cfg = train_config()
    train(cfg, joint_data[0], tmp_path / "whole")
    train(cfg, joint_data[0], tmp_path / "resume", max_new_epochs=1)
    train(cfg, joint_data[0], tmp_path / "resume", resume=True)
    a = load_checkpoint(tmp_path / "whole/latest.pt")
    b = load_checkpoint(tmp_path / "resume/latest.pt")
    assert a["step"] == b["step"] and a["best"] == b["best"]
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
    rows = [
        json.loads(line) for line in (tmp_path / "whole/metrics.jsonl").read_text().splitlines()
    ]
    assert all(len(r["validation_by_resolution"]) == 2 for r in rows)
    assert all(r["learning_health"]["bypass_pixels"] == 0 for r in rows)
    assert all("resolution_slices" in r["validation_constraints"] for r in rows)
    assert all("budget_slices" in r["validation_constraints"] for r in rows)
    assert all(set(r["validation_by_slice"]["budget"]) == {"1", "4"} for r in rows)
    assert a["validation_budget_constraints"] == b["validation_budget_constraints"]
    assert all(
        set(r["validation_by_slice"]) == {"budget", "family", "stratum", "transport"} for r in rows
    )
    checks = constraint_report(
        dict(psnr=29, ssim=0.94, edge=0.1),
        dict(psnr=28, ssim=0.93, edge=0.2),
        dict(psnr_min=30, ssim_min=0.95, psnr_gain=0.1),
    )
    assert not checks["psnr_min"]["passed"] and checks["psnr_gain"]["passed"]


def test_joint_export_native_parity_and_equal_size_benchmark(joint_data, tmp_path):
    from raytracer_ml.benchmark import benchmark

    cfg = train_config()["model"]
    model = build_model(cfg).eval()
    with torch.no_grad():
        model.head.weight.normal_(0, 0.005)
    checkpoint = tmp_path / "functional-only.pt"
    torch.save(
        dict(
            config=dict(model=cfg),
            model=model.state_dict(),
            manifest_sha256=digest(joint_data[0] / "manifest.jsonl"),
        ),
        checkpoint,
    )
    exported = tmp_path / "functional-only.onnx"
    export_model(checkpoint, exported)
    meta_path = exported.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    write_json(meta_path, {**metadata, "manifest_sha256": "different-data"})
    with pytest.raises(ValueError, match="register an external evaluation"):
        benchmark(
            joint_data[0],
            joint_data[1],
            exported,
            tmp_path / "wrong-data",
            dict(budgets=[1], repeats=1, quality=dict(psnr=30, ssim=0.95)),
        )
    write_json(meta_path, metadata)
    with pytest.raises(ValueError, match="Invalid benchmark"):
        benchmark(
            joint_data[0],
            joint_data[1],
            exported,
            tmp_path / "no-repeats",
            dict(budgets=[1], repeats=0, quality=dict(psnr=30, ssim=0.95)),
        )
    binary = os.environ.get("RAYTRACER_NEURAL_BINARY")
    if not binary:
        return
    row = next(r for r in manifest(joint_data[0] / "manifest.jsonl") if r["split"] == "val")
    write_json(tmp_path / "scene.json", row["scene"])
    command = [
        binary,
        "--headless",
        "--quiet",
        "--scene-file",
        str(tmp_path / "scene.json"),
        "--width",
        str(row["stats"]["width"]),
        "--samples",
        "4",
        "--threads",
        "2",
        "--features",
        str(tmp_path / "features"),
        "--output",
        str(tmp_path / "out.pfm"),
    ]
    run = subprocess.run(
        command + ["--model", str(exported)], check=True, capture_output=True, text=True
    )
    assert json.loads(run.stdout)["reconstructed"]
    x = torch.from_numpy(load_features(tmp_path / "features", 2)[None])
    with torch.no_grad():
        expected = model(x)[0].numpy().transpose(1, 2, 0)
    np.testing.assert_allclose(read_pfm(tmp_path / "out.pfm"), expected, rtol=2e-4, atol=2e-5)
    run = subprocess.run(
        command + ["--output-scale", "2"], check=True, capture_output=True, text=True
    )
    meta = json.loads(run.stdout)
    assert meta["upscale_seconds"] > 0 and meta["pipeline_seconds"] >= meta["upscale_seconds"]
    expected_raw = (
        F.interpolate(x[:, :3], scale_factor=2, mode="bilinear", align_corners=False)[0]
        .numpy()
        .transpose(1, 2, 0)
    )
    np.testing.assert_allclose(read_pfm(tmp_path / "out.pfm"), expected_raw, rtol=2e-6, atol=2e-6)
    summary = benchmark(
        joint_data[0],
        binary,
        exported,
        tmp_path / "benchmark",
        dict(
            split="val",
            max_configurations=1,
            budgets=[1],
            threads=2,
            repeats=1,
            quality=dict(psnr=100, ssim=1),
        ),
    )
    assert set(summary["failures"]) == {"raw", "atrous", "raw_upscale", "atrous_upscale", "neural"}
    assert not summary["matched_quality_acceleration"][0]["demonstrated_win"]
    assert summary["model_sha256"] == digest(exported)
    assert summary["manifest_sha256"] == digest(joint_data[0] / "manifest.jsonl")
    assert summary["evaluation_scope"] == "original-dataset"


def test_gallery_selection_spans_configurations_and_labels_native_pixels(tmp_path):
    from PIL import Image
    from raytracer_ml.comparisons import comparison_ids, write_comparison

    rows = [
        dict(id=f"g{g:03d}-n{n}-s{s}", configuration=f"g{g:03d}", samples=s)
        for g in range(30)
        for n in (0, 1)
        for s in (1, 4, 16)
    ]
    selected = comparison_ids(rows)
    assert len(selected) == 12
    assert "g000-n0-s4" in selected and "g029-n0-s4" in selected
    assert all(name.endswith("-n0-s4") for name in selected)
    row = dict(id="portrait", split="val", samples=4, reference_samples=2048, scale=2)
    target = np.zeros((57, 31, 3), np.float32)
    path = tmp_path / "comparison.png"
    methods = {name: target for name in ("raw", "atrous", "neural", "oidn")}
    write_comparison(
        path, row, methods, target, {name: dict(psnr=31, ssim=0.97) for name in methods}
    )
    with Image.open(path) as image:
        assert image.size == (245 * 5, 57 + 122)
        pixels = np.asarray(image)
        # The native 31x57 image is preserved, not stretched to the label width.
        left = (245 - 31) // 2
        assert (pixels[110:167, left : left + 31] == 0).all()
        assert (pixels[10:100] != np.array([24, 27, 33])).any()
