import pytest
import torch
import numpy as np

from raytracer_ml.learning_health import LearningHealth, validate_health_config
from raytracer_ml.losses import reconstruction_loss
from raytracer_ml.models import build_model
from raytracer_ml.models.unet import block
from raytracer_ml.diagnostics import reconstruction_region, reconstruction_region_metrics


def inputs():
    torch.manual_seed(9)
    x = torch.rand(1, 27, 9, 13)
    x[:, :3] *= 20
    x[:, 10:12], x[:, 26], x[:, 15] = 1, 1, 4
    x[:, 11, :2] = 0
    x[:, 15, 2:4] = 128
    return x


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_defaults_preserve_weights_and_output(schema):
    cfg = dict(kind="unet", width=4, feature_schema=schema)
    torch.manual_seed(1)
    legacy = build_model(cfg)
    explicit = build_model({**cfg, "activation": "relu", "radiance_scale": 1})
    with torch.no_grad():
        legacy.head.weight.normal_(0, 0.1)
    explicit.load_state_dict(legacy.state_dict())
    x = inputs()[:, : 27 if schema == 2 else 17]
    torch.testing.assert_close(legacy(x), explicit(x), atol=0, rtol=0)


def test_negative_activations_retain_encoder_gradient():
    plain, leaky = block(1, 1), block(1, 1, "leaky_relu")
    with torch.no_grad():
        for layer in (plain[0], plain[2]):
            layer.weight.fill_(0.1)
            layer.bias.fill_(-1)
    leaky.load_state_dict(plain.state_dict())
    x = torch.ones(1, 1, 5, 5)
    plain(x).sum().backward()
    leaky(x).sum().backward()
    assert plain[0].weight.grad.abs().sum() == 0
    assert leaky[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("scale", [1, 2])
def test_radiance_conditioning_uses_correct_variance_and_restores_hdr(scale):
    cfg = dict(kind="unet", width=4, feature_schema=2, output_head="additive_log", scale=scale)
    ordinary = build_model(cfg)
    conditioned = build_model({**cfg, "radiance_scale": 16})
    with torch.no_grad():
        ordinary.head.weight.normal_(0, 0.01)
    conditioned.load_state_dict(ordinary.state_dict())
    x = inputs()
    gain = x.clone()
    gain[:, :3] *= 16
    gain[:, 12:15] *= 16**2
    torch.testing.assert_close(conditioned(x), ordinary(gain) / 16, rtol=2e-6, atol=2e-6)
    if scale == 1:
        torch.testing.assert_close(conditioned(x)[:, :, :4], x[:, :3, :4], rtol=0, atol=0)
    with torch.no_grad():
        ordinary.head.weight.zero_()
        ordinary.head.bias.fill_(100)
    conditioned.load_state_dict(ordinary.state_dict())
    torch.testing.assert_close(conditioned(x), ordinary(x), rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("value", [0, -1, 65, float("nan"), float("inf")])
def test_invalid_radiance_scale_rejected(value):
    with pytest.raises(ValueError, match="Radiance scale"):
        build_model(dict(width=4, output_head="additive_log", radiance_scale=value))


def test_observer_reports_identity_stall_without_changing_gradients_or_rng():
    model = build_model(dict(kind="unet", width=4, feature_schema=2, output_head="additive_log"))
    health = LearningHealth(model, dict(stall_after_epochs=2), None)
    x = inputs()
    target = x[:, :3] * 0.5
    prediction = model(x)
    loss = reconstruction_loss(prediction, target)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    grads = [p.grad.clone() for p in model.parameters()]
    rng = torch.get_rng_state()
    health.observe(x, target, prediction, loss, norm)
    assert health.summary(0)["warning"] is None
    report = health.summary(1)
    assert report["warning"] == "little-improvement-over-input"
    assert abs(report["loss_reduction_vs_raw"]) < 1e-5
    for parameter, original in zip(model.parameters(), grads):
        torch.testing.assert_close(parameter.grad, original, atol=0, rtol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, atol=0, rtol=0)
    health.close()
    assert not model.head._forward_pre_hooks


@pytest.mark.parametrize(
    "cfg", [True, {"stall_after_epochs": 0}, {"minimum_loss_reduction": float("nan")}]
)
def test_invalid_health_policy_rejected(cfg):
    with pytest.raises(ValueError):
        validate_health_config(cfg)


@pytest.mark.parametrize("kind", ["unet", "guided", "refine"])
def test_hdr_partition_matches_actual_model_fallback(kind):
    cfg = dict(kind=kind, width=4, feature_schema=2)
    model = build_model(cfg)
    x = inputs()
    x[:, 10:12, 4] = 0.5  # Diffuse plus background is eligible under schema 2.
    x[:, 26, 5], x[:, 23, 5] = 0, 1  # Center material is unsupported.
    with torch.no_grad():
        model.head.bias.fill_(0.5)
        prediction = model(x)[0].numpy().transpose(1, 2, 0)
    features = x[0].numpy()
    raw = features[:3].transpose(1, 2, 0)
    support = reconstruction_region(features, cfg)
    assert support[4].all() and not support[5].any()
    np.testing.assert_array_equal(prediction[~support], raw[~support])
    target = raw * 0.5
    result = reconstruction_region_metrics(prediction, target, features, cfg)
    assert (
        result["model_region_pixels"] + result["fallback_region_pixels"]
        == raw.shape[0] * raw.shape[1]
    )
    total = (
        result["model_region_linear_mse_contribution"]
        + result["fallback_region_linear_mse_contribution"]
    )
    assert total == pytest.approx(np.mean((prediction.astype(np.float64) - target) ** 2))
