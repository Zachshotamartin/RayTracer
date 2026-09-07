"""Versioned head/blend invariants and research loss/telemetry controls."""

import pytest
import torch
from torch.nn import functional as F

from raytracer_ml.models import build_model
from raytracer_ml.models.detail import transform
from raytracer_ml.learning_health import LearningHealth, model_region_mask
from raytracer_ml.losses import reconstruction_loss, validate_loss_config


def features():
    torch.manual_seed(19)
    x = torch.rand(1, 27, 11, 17)
    x[:, 10:12], x[:, 26], x[:, 15] = 1, 1, 4
    return x


@pytest.mark.parametrize("kind,scale", [("refine", 1), ("refine", 2), ("guided", 2)])
def test_legacy_detail_config_is_unchanged(kind, scale):
    cfg = dict(kind=kind, width=4, scale=scale, feature_schema=2)
    legacy = build_model(cfg)
    explicit = build_model(
        {
            **cfg,
            "output_head": "bounded_multiplicative",
            "radiance_scale": 1,
            "blend_policy": "legacy",
            "guide_policy": "center",
        }
    )
    with torch.no_grad():
        legacy.head.weight.normal_(0, 0.1)
        if scale == 2:
            legacy.upscale[2].weight.normal_(0, 0.1)
    explicit.load_state_dict(legacy.state_dict())
    x = features()
    x[:, 15] = 64
    torch.testing.assert_close(legacy(x), explicit(x), atol=0, rtol=0)


@pytest.mark.parametrize("samples", [4, 32, 64, 96, 128])
def test_single_blend_matches_spatial_correction_at_uniform_spp(samples):
    cfg = dict(kind="guided", width=4, feature_schema=2, blend_policy="single")
    spatial, upscale = build_model(cfg), build_model({**cfg, "scale": 2})
    upscale.load_state_dict(spatial.state_dict(), strict=False)
    x = features()
    x[:, 15] = samples
    expected = F.interpolate(spatial(x), scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(upscale(x), expected, atol=2e-7, rtol=2e-6)


def test_single_blend_variable_spp_and_nonzero_upscale_head():
    cfg = dict(
        kind="refine",
        width=4,
        feature_schema=2,
        scale=2,
        blend_policy="single",
        output_head="additive_log",
        radiance_scale=16,
    )
    model = build_model(cfg)
    with torch.no_grad():
        model.head.bias.fill_(0.4)
        model.upscale[2].bias.copy_(torch.arange(12) / 10)
    x = features()
    x[:, :3] *= 8
    full = model(x)
    x[:, 15] = torch.tensor([4, 32, 64, 96, 128, 200, 64, 96, 4, 32, 4])[:, None]
    x[:, 11, :, :2] = 0
    raw = F.interpolate(x[:, :3], scale_factor=2, mode="bilinear", align_corners=False)
    strength = F.interpolate(((128 - x[:, 15:16]) / 96).clamp(0, 1), scale_factor=2, mode="nearest")
    expected = raw + strength * (full - raw)
    mask = model_region_mask(x, model)
    expected = torch.where(mask, expected, raw)
    actual = model(x)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        actual[~mask.expand_as(raw)], raw[~mask.expand_as(raw)], atol=0, rtol=0
    )


@pytest.mark.parametrize("scale", [1, 2])
def test_additive_detail_recovers_zero_input_with_gradients_and_physical_units(scale):
    cfg = dict(
        kind="refine",
        width=4,
        feature_schema=2,
        scale=scale,
        output_head="additive_log",
        radiance_scale=16,
        blend_policy="single",
    )
    model = build_model(cfg)
    x = features()
    x[:, :3] = 0
    x[:, 11, :2] = 0
    x[:, 15, 2:4] = 128
    with torch.no_grad():
        model.head.bias.fill_(4)
    output = model(x)
    mask = model_region_mask(x, model).expand_as(output)
    assert output[mask].min() > 3
    assert output[~mask].max() == 0
    output[mask].mean().backward()
    assert model.head.bias.grad.min() > 0
    with torch.no_grad():
        model.head.bias.fill_(100)
    assert float(model(x).max().detach()) == pytest.approx(
        float(torch.expm1(torch.tensor(12.0))), rel=3e-6
    )
    gain = x.clone()
    gain[:, :3] *= 16
    gain[:, 12:15] *= 256
    torch.testing.assert_close(transform(x, 2, 16), transform(gain, 2), atol=0, rtol=0)


def test_sampled_guide_control_changes_only_penalty_inputs():
    cfg = dict(kind="guided", width=4, feature_schema=2)
    center = build_model(cfg)
    sampled = build_model({**cfg, "guide_policy": "sampled"})
    sampled.load_state_dict(center.state_dict())
    x = features()
    x[:, 3:6], x[:, 6:9], x[:, 9] = 0.5, 0, 1
    changed = x.clone()
    changed[:, 20:24] *= 20
    # Zero kernel head isolates deterministic priors from encoder inputs.
    torch.testing.assert_close(sampled(x), sampled(changed), atol=0, rtol=0)
    assert (center(x) - center(changed)).abs().max() > 0.001
    assert sum(p.numel() for p in center.parameters()) == sum(
        p.numel() for p in sampled.parameters()
    )


def test_relative_l2_stops_denominator_gradient_and_preserves_linear_mean_stationarity():
    target = torch.tensor([0.0, 0.0, 0.0, 4.0])
    prediction = torch.tensor(1.0, requires_grad=True)
    loss = reconstruction_loss(prediction.expand_as(target), target, {"kind": "relative_l2"})
    loss.backward()
    assert prediction.grad.item() == pytest.approx(0, abs=1e-6)
    prediction = torch.tensor(0.5, requires_grad=True)
    reconstruction_loss(prediction.expand_as(target), target, {"kind": "relative_l2"}).backward()
    assert prediction.grad.item() == pytest.approx(2 * (0.5 - 1) / 0.51**2, rel=1e-6)
    prediction = torch.zeros(4, requires_grad=True)
    reconstruction_loss(prediction, target, {"kind": "relative_l2"}).backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad[-1] < 0
    torch.testing.assert_close(
        reconstruction_loss(target, target),
        reconstruction_loss(target, target, {"kind": "log_l1"}),
        atol=0,
        rtol=0,
    )
    with pytest.raises(ValueError):
        validate_loss_config({"kind": "typo"})


def test_supported_learning_not_hidden_by_bypass_error_and_observer_is_passive():
    model = build_model(dict(kind="refine", width=4, feature_schema=2))
    health = LearningHealth(
        model, {"stall_after_epochs": 1, "warning_metric": "model_region_log_mae"}, None
    )
    x = features()
    x[:, :3], x[:, 11] = 0, 0
    x[:, 11, 5, 5] = 1
    y = torch.ones_like(x[:, :3]) * 20
    y[:, :, 5, 5] = 0.1
    model(x).sum().backward()  # Populate noninvasive hook/gradient observations.
    pred = x[:, :3].clone()
    pred[:, :, 5, 5] = 0.1
    rng = torch.get_rng_state().clone()
    grads = [p.grad.clone() for p in model.parameters()]
    health.observe(x, y, pred, reconstruction_loss(pred, y), torch.tensor(1.0))
    report = health.summary(0)
    assert report["model_region_pixels"] == 1 and report["bypass_pixels"] == 186
    assert report["model_region_log_mae_reduction_vs_raw"] == 1
    assert report["loss_reduction_vs_raw"] < 0.02 and report["warning"] is None
    assert "encoder_gradient_norm_after_clip" in report
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(p.grad, g) for p, g in zip(model.parameters(), grads))
    health.reset()
    x[:, 11] = 0
    model(x)
    health.observe(x, y, pred, reconstruction_loss(pred, y), torch.tensor(1.0))
    assert health.summary(0)["warning"] == "no-model-region-pixels"
    health.close()


@pytest.mark.parametrize(
    "option",
    [
        {"blend_policy": "typo"},
        {"guide_policy": "typo"},
        {"radiance_scale": 0},
        {"radiance_scale": 16},
        {"output_head": "typo"},
        {"guide_policy": "sampled"},
    ],
)
def test_invalid_detail_controls_rejected(option):
    with pytest.raises(ValueError):
        build_model(dict(kind="refine", width=4, feature_schema=2, **option))
