"""Small synthetic probes for the literature audit; never train or change a model artifact."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from raytracer_ml.io import digest, git_revision, write_json
from raytracer_ml.losses import reconstruction_loss
from raytracer_ml.models import build_model


def features(size=25, samples=4):
    x = torch.zeros(1, 27, size, size)
    x[:, 3:6], x[:, 8], x[:, 10:12] = 0.5, 1, 1
    x[:, 15], x[:, 16], x[:, 17:20] = samples, 1, 0.5
    x[:, 22:24], x[:, 26] = 1, 1
    return x


def probe():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    package = Path(__import__("raytracer_ml").__file__).parent
    result = dict(
        code_commit=git_revision(),
        seed=19,
        torch_version=str(torch.__version__),
        scope="Synthetic implementation/math probes; no trained quality estimate, dataset change, or training",
        probe_source_sha256=digest(Path(__file__)),
        source_sha256={
            name: digest(package / name)
            for name in (
                "models/detail.py",
                "models/unet.py",
                "losses.py",
                "learning_health.py",
            )
        },
    )
    refinement = build_model(dict(kind="refine", width=8, feature_schema=2)).eval()
    with torch.no_grad():
        refinement.head.bias.fill_(100)
        output = refinement(features())
    result["refine_missing_signal"] = dict(
        raw=0.0,
        correction_bias=100,
        maximum_output=float(output.max()),
        theoretical_limit=0.01 * (torch.exp(torch.tensor(2.0)).item() - 1),
    )
    spatial = build_model(dict(kind="guided", width=8, stages=3, feature_schema=2)).eval()
    upscale = build_model(dict(kind="guided", width=8, stages=3, feature_schema=2, scale=2)).eval()
    upscale.load_state_dict(spatial.state_dict(), strict=False)
    x = features()
    x[:, :3] = torch.rand(1, 3, 25, 25)
    raw = F.interpolate(x[:, :3], scale_factor=2, mode="bilinear", align_corners=False)
    records = []
    with torch.no_grad():
        for samples in (4, 32, 64, 96, 128):
            x[:, 15] = samples
            single = F.interpolate(spatial(x), scale_factor=2, mode="bilinear", align_corners=False)
            double = upscale(x)
            a, b = (single - raw).abs().mean().item(), (double - raw).abs().mean().item()
            records.append(
                dict(
                    samples=samples,
                    spatial_correction_mae=a,
                    upscale_correction_mae=b,
                    remaining_correction_ratio=b / a if a else None,
                )
            )
    result["upscale_double_blend"] = records
    x = features()
    x[:, :3, 12, 20] = 10
    with torch.no_grad():
        spatial.head.weight.normal_(0, 1)
        far = spatial(x)[0, :, 12, 12].tolist()
        x[:, :3, 12, 20] = 0
        x[:, :3, 12, 19] = 10
        near = spatial(x)[0, :, 12, 12].tolist()
    result["guided_local_signal"] = dict(
        stages=3, radiance_radius=7, brightness=10, distance_8_output=far, distance_7_output=near
    )
    x = features(8)
    x[:, 11] = 0
    prediction = spatial(x)
    loss = (prediction - torch.ones_like(prediction)).square().mean()
    spatial.zero_grad(set_to_none=True)
    loss.backward()
    result["unsupported_training"] = dict(
        output_equals_raw=bool(torch.equal(prediction, x[:, :3])),
        sum_parameter_gradient=sum(
            float(p.grad.abs().sum()) for p in spatial.parameters() if p.grad is not None
        ),
    )
    target = torch.tensor([0.0, 0.0, 0.0, 4.0])[:, None, None, None].expand(4, 3, 2, 2)
    values = {}
    for label, config in (
        ("base", None),
        ("current_detail", dict(gradient=0.25, edge_weight=3, energy=0.02)),
    ):
        probes = []
        for value in (0.0, 0.5, 1.0, 2.0):
            z = torch.tensor(value, requires_grad=True)
            loss = reconstruction_loss(z.expand_as(target), target, config)
            loss.backward()
            probes.append(dict(prediction=value, loss=float(loss.detach()), gradient=float(z.grad)))
        values[label] = probes
    result["noisy_target_loss"] = dict(
        target_values=[0, 0, 0, 4],
        true_linear_mean=1.0,
        median=0.0,
        probes=values,
        interpretation="Skewed unbiased target example demonstrates a risk when references are unconverged; not a measurement of current dataset bias.",
    )
    raw = torch.full((1, 3, 8, 8), 100.0)
    target = torch.zeros_like(raw)
    raw[:, :, 0, 0], target[:, :, 0, 0] = 0, 1
    perfect = raw.clone()
    perfect[:, :, 0, 0] = 1
    raw_loss = reconstruction_loss(raw, target).item()
    model_loss = reconstruction_loss(perfect, target).item()
    result["global_health_masking"] = dict(
        supported_fraction=1 / 64,
        supported_error_reduction=1.0,
        full_image_loss_reduction=1 - model_loss / raw_loss,
        warning_threshold=0.02,
    )
    # Verify the recorded findings on the audited implementation. A corrected model
    # requires a new audit rather than silently replacing this historical evidence.
    assert result["unsupported_training"]["sum_parameter_gradient"] == 0
    assert max(far) == 0 and min(near) > 0
    assert abs(records[3]["remaining_correction_ratio"] - 1 / 3) < 1e-5
    assert result["refine_missing_signal"]["maximum_output"] < 0.064
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path to preserve earlier audit evidence")
    write_json(args.output, probe())
    print(f"Verified synthetic constraints; saved {args.output}")


if __name__ == "__main__":
    main()
