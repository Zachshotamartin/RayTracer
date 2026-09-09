"""Existing-pixel reuse: no renderer subprocess is needed for these tests."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from raytracer_ml.augment import AlignedCropCollator
from raytracer_ml.data.arrays import load_example, load_reference
from raytracer_ml.data.dataset import RenderDataset
from raytracer_ml.data.reuse import build_reuse
from raytracer_ml.data.resample import reduce_features, area2, atrous
from raytracer_ml.data.validate import validate
from raytracer_ml.io import digest, identity, manifest, save_arrays, write_json
from raytracer_ml.preprocessing import validate_features
from raytracer_ml.train import train, load_checkpoint


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "existing"
    rows = []
    rng = np.random.default_rng(7)
    for index, split in enumerate(("train", "val", "test")):
        scene = {"schema_version": 1, "camera": {"index": index}}
        ref = f"references/g{index}.npz"
        target = rng.uniform(0.1, 1, (48, 96, 3)).astype(np.float32)
        save_arrays(root / ref, target=target, annotations=target)
        for noise in range(2):
            for samples in (1, 2, 4, 8, 16, 32, 64):
                x = np.zeros((27, 48, 96), np.float32)
                x[:3] = target.transpose(2, 0, 1) + rng.uniform(0, 0.2, (3, 48, 96))
                x[3:6] = x[17:20] = 0.5
                x[7] = x[21] = 1
                x[9:12] = x[23] = x[26] = 1
                x[15], x[16] = samples, int(samples > 1)
                x[12:15] = 0.1 / samples if samples > 1 else 0
                name = f"g{index}-v0-n{noise}-s{samples}"
                path = f"examples/{name}.npz"
                save_arrays(
                    root / path,
                    features=x,
                    position=np.zeros((48, 96, 3), np.float32),
                    atrous=x[:3].transpose(1, 2, 0),
                )
                rows.append(
                    dict(
                        schema_version=1,
                        feature_schema=2,
                        id=name,
                        configuration=f"g{index}-v0",
                        group=f"g{index}",
                        split=split,
                        cohort="original",
                        frame=0,
                        scene=scene,
                        scene_sha256=identity(scene),
                        path=path,
                        sha256=digest(root / path),
                        reference=ref,
                        reference_sha256=digest(root / ref),
                        input_seed=1 + samples + noise * 100,
                        target_seed=900,
                        samples=samples,
                        reference_samples=2048,
                        scale=1,
                        stats=dict(
                            height=48,
                            width=96,
                            samples=samples,
                            render_seconds=1,
                            bvh_build_seconds=0,
                        ),
                        reference_stats=dict(height=48, width=96, samples=2048),
                    )
                )
    (root / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    write_json(
        root / "dataset.json",
        dict(
            config=dict(scale=1, budgets=[1, 2, 4, 8, 16, 32, 64], noise_realizations=2),
            estimate=dict(examples=len(rows)),
        ),
    )
    return root


def test_reuse_preserves_files_splits_targets_and_independence(source, tmp_path):
    out = tmp_path / "reuse"
    before = digest(source / "manifest.jsonl")
    built = build_reuse([source], out)
    assert built["examples"] == 42 and built["new_rays"] == 0
    result = validate(out)
    assert result["splits"] == {"train": 14, "val": 14, "test": 14}
    originals = {r["id"]: r for r in manifest(source / "manifest.jsonl")}
    for row in manifest(out / "manifest.jsonl"):
        old = originals[row["reuse"]["source_id"]]
        assert row["split"] == old["split"] and row["group"] == old["group"]
        assert os.path.samefile(source / old["path"], out / row["path"])
        assert os.path.samefile(source / old["reference"], out / row["reference"])
        x = load_example(out, row, baseline=False)["features"]
        ref = load_reference(out, row)
        assert ref["target"].shape == (x.shape[1] * 2, x.shape[2] * 2, 3)
        assert ref["annotations"].shape == ref["target"].shape
        assert row["samples"] == old["samples"] * 4
        assert row["stats"]["render_seconds"] is None
    assert digest(source / "manifest.jsonl") == before
    # Changing only the clean target must not change model input pixels.
    row = manifest(out / "manifest.jsonl")[0]
    x = load_example(out, row, baseline=False)["features"]
    save_arrays(out / row["reference"], target=np.zeros((48, 96, 3), np.float32))
    np.testing.assert_array_equal(x, load_example(out, row, baseline=False)["features"])
    with pytest.raises(ValueError, match="checksum"):
        validate(out)


def test_modified_split_or_transform_is_rejected(source, tmp_path):
    out = tmp_path / "reuse"
    build_reuse([source], out)
    rows = manifest(out / "manifest.jsonl")
    rows[0]["split"] = "test"
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="inherited splits"):
        validate(out)


def test_area_variance_and_guides_are_not_ordinary_rgb_resizing():
    x = np.zeros((27, 4, 6), np.float32)
    x[15], x[16], x[12:15] = 8, 1, 0.4
    x[3:6], x[17:20] = 0.5, 0.75
    x[7], x[21], x[10:12], x[26] = 1, 1, 1, 1
    x[9], x[23] = 2, 2
    x[:3] = np.arange(24).reshape(1, 4, 6)
    a = reduce_features(x)
    validate_features(a)
    np.testing.assert_allclose(a[:3], area2(x[:3]))
    np.testing.assert_allclose(a[12:15], 0.1)
    assert np.all(a[15] == 32)
    x[15], x[16], x[12:15] = 1, 0, 0
    assert np.all(reduce_features(x)[12:15] > 0)


def test_variable_batch_crops_keep_alignment_and_rng():
    x = torch.arange(48 * 48).reshape(1, 48, 48).float().repeat(27, 1, 1)
    y = x[:3].repeat_interleave(2, -2).repeat_interleave(2, -1)
    collate = AlignedCropCollator([[32, 48], [48, 32], [32, 32]], 48, 2)
    state = torch.get_rng_state()
    a, b = collate([(x, y), (x, y)])
    torch.testing.assert_close(a[:, :3].repeat_interleave(2, -2).repeat_interleave(2, -1), b)
    torch.set_rng_state(state)
    aa, bb = collate([(x, y), (x, y)])
    torch.testing.assert_close(a, aa)
    torch.testing.assert_close(b, bb)


def test_reused_dataset_trains_and_resumes_with_domain_gates(source, tmp_path):
    out, run = tmp_path / "reuse", tmp_path / "run"
    build_reuse([source], out)
    cfg = dict(
        seed=7,
        device="cpu",
        cpu_threads=1,
        model=dict(kind="joint", width=4, scale=2, feature_schema=2),
        crop=16,
        crop_shapes=[[16, 16]],
        batch_size=4,
        epochs=2,
        patience=5,
        learning_rate=0.001,
        weight_decay=0.0001,
        max_seconds=120,
        checkpoint_history=True,
        identity_probability=0.5,
        preservation_mode="measured_near_clean",
        near_clean_samples=96,
        augmentation=dict(rotate90=True, exposure_stops=0.2),
        selection=dict(psnr_gain=0.1, preservation_ratio=1),
        selection_by_domain=True,
        validation_budgets=[4],
        validation_first_noise_only=True,
    )
    assert len(RenderDataset(out, "train", crop=16, feature_schema=2)) == 14
    train(cfg, out, run, max_new_epochs=1)
    first = load_checkpoint(run / "latest.pt")
    assert first["epoch"] == 0 and first["step"] == 4
    train(cfg, out, run, resume=True)
    assert load_checkpoint(run / "latest.pt")["epoch"] == 1
    metrics = manifest(run / "metrics.jsonl")
    assert "synthetic-area2" in metrics[-1]["validation_by_slice"]["domain"]
    assert "domain_slices" in metrics[-1]["validation_constraints"]


def test_atrous_matches_saved_native_baseline_without_rendering():
    root = os.environ.get("RTML_EXISTING_DATASET")
    if not root:
        pytest.skip("Optional existing artifact parity check")
    rows = manifest(Path(root) / "manifest.jsonl")
    row = next(r for r in rows if r["samples"] == 4)
    item = load_example(root, row)
    np.testing.assert_allclose(atrous(item["features"]), item["atrous"], rtol=2e-5, atol=2e-6)
