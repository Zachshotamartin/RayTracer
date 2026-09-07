"""Compact, geometry-conditioned reconstruction with an explicit full-resolution path."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def shifted(x, dy, dx):
    # Replicate padding keeps constant HDR signals invariant, including image borders.
    h, w = x.shape[-2:]
    radius = max(abs(dy), abs(dx))
    if radius == 0:
        return x
    padded = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    return padded[..., radius + dy : radius + dy + h, radius + dx : radius + dx + w]


def transform(x, schema=2, radiance_scale=1.0):
    base = torch.cat(
        [
            torch.log1p(x[:, :3].clamp_min(0) * radiance_scale),
            x[:, 3:9],
            torch.log1p(x[:, 9:10].clamp_min(0)) / 5,
            x[:, 10:12],
            torch.log1p(x[:, 12:15].clamp_min(0) * radiance_scale**2),
            torch.log2(x[:, 15:16].clamp_min(1)) / 7,
            x[:, 16:17],
        ],
        dim=1,
    )
    if schema == 2:
        base = torch.cat(
            [base, x[:, 17:23], torch.log1p(x[:, 23:25].clamp_min(0)) / 5, x[:, 25:27]], dim=1
        )
    return base


class DetailNet(nn.Module):
    def __init__(
        self,
        width=16,
        kind="guided",
        scale=1,
        inputs="all",
        temporal=False,
        feature_schema=2,
        stages=3,
        demodulate=False,
        output_head="bounded_multiplicative",
        radiance_scale=1.0,
        blend_policy="legacy",
        guide_policy="center",
    ):
        super().__init__()
        if kind not in ("guided", "refine") or feature_schema not in (1, 2):
            raise ValueError("Invalid detail model kind/schema")
        if not 4 <= width <= 128 or not 1 <= stages <= 4 or scale not in (1, 2):
            raise ValueError("Invalid detail model size")
        if temporal and scale != 1:
            raise ValueError("Qualify temporal and upscale models separately")
        if inputs not in ("all", "rgb", "no_variance"):
            raise ValueError("Invalid detail feature ablation")
        if output_head not in ("bounded_multiplicative", "additive_log"):
            raise ValueError("Invalid detail output head")
        if not math.isfinite(radiance_scale) or not 0 < radiance_scale <= 64:
            raise ValueError("Radiance scale must be finite and in (0, 64]")
        if radiance_scale != 1 and output_head != "additive_log":
            raise ValueError("Radiance conditioning requires an additive_log head")
        if kind == "guided" and scale == 1 and output_head != "bounded_multiplicative":
            raise ValueError("Spatial guided filtering has no radiance correction head")
        if blend_policy not in ("legacy", "single"):
            raise ValueError("Invalid sample blending policy")
        if guide_policy not in ("center", "sampled"):
            raise ValueError("Invalid filtering guide policy")
        if guide_policy != "center" and (kind != "guided" or inputs == "rgb"):
            raise ValueError("Sampled guide control requires geometry-guided filtering")
        self.kind, self.scale, self.temporal = kind, scale, temporal
        self.feature_schema, self.stages = feature_schema, stages
        self.inputs, self.demodulate = inputs, demodulate
        self.output_head, self.radiance_scale = output_head, float(radiance_scale)
        self.blend_policy, self.guide_policy = blend_policy, guide_policy
        self.base_channels = 27 if feature_schema == 2 else 17
        cin = 3 if inputs == "rgb" else self.base_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(cin + (4 if temporal else 0), width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Conv2d(width, stages * 9 if kind == "guided" else 3, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if temporal:
            self.history_gate = nn.Conv2d(width, 1, 1)
            nn.init.zeros_(self.history_gate.weight)
            nn.init.constant_(self.history_gate.bias, -2)
        if scale == 2:
            self.upscale = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(width, 12, 3, padding=1),
                nn.PixelShuffle(2),
            )
            nn.init.zeros_(self.upscale[2].weight)
            nn.init.zeros_(self.upscale[2].bias)

    def correct_radiance(self, base, correction):
        if self.output_head == "bounded_multiplicative":
            return ((base + 0.01) * torch.exp(2 * torch.tanh(correction)) - 0.01).clamp_min(0)
        # Match the U-Net's physical HDR ceiling, regardless of internal conditioning.
        logged = torch.log1p(base.clamp_min(0) * self.radiance_scale)
        ceiling = math.log1p(math.expm1(12) * self.radiance_scale)
        return (torch.exp((logged + correction).clamp(0, ceiling)) - 1) / self.radiance_scale

    def forward(self, x):
        raw = x[:, :3].clamp_min(0)
        z = transform(x, self.feature_schema, self.radiance_scale)
        if self.inputs == "rgb":
            z = z[:, :3]
        elif self.inputs == "no_variance":
            z = torch.cat(
                [
                    z[:, :12],
                    torch.zeros_like(z[:, 12:15]),
                    z[:, 15:16],
                    torch.zeros_like(z[:, 16:17]),
                    z[:, 17:],
                ],
                1,
            )
        if self.temporal:
            h = x[:, self.base_channels : self.base_channels + 4]
            z = torch.cat(
                [z, torch.log1p(h[:, :3].clamp_min(0) * self.radiance_scale), h[:, 3:4]], 1
            )
        encoding = self.encoder(z.to(self.encoder[0].weight.dtype))
        parameters = self.head(encoding).float()
        support = ((x[:, 10:11] - x[:, 11:12]).abs() < 1e-6) & (x[:, 10:11] > 0)
        if self.feature_schema == 2:
            support = support & ((x[:, 26:27] > 0.5) | (x[:, 23:24] == 0))
        # Guaranteed raw identity at >=128 real samples and asymptotic convergence.
        strength = ((128 - x[:, 15:16]) / 96).clamp(0, 1)
        if self.kind == "guided":
            albedo = x[:, 3:6]
            center_guides = self.feature_schema == 2 and self.guide_policy == "center"
            normal = x[:, 20:23] if center_guides else x[:, 6:9]
            depth = x[:, 23:24] if center_guides else x[:, 9:10]
            base = raw / (albedo + 0.1) if self.demodulate else raw
            for stage in range(self.stages):
                step = 2**stage
                weights, neighbors = [], []
                for k, (dy, dx) in enumerate(
                    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1))
                ):
                    dy, dx = dy * step, dx * step
                    penalty = torch.zeros_like(depth)
                    if self.inputs != "rgb":
                        penalty = (
                            8 * (albedo - shifted(albedo, dy, dx)).abs().sum(1, keepdim=True)
                            + 12 * (normal - shifted(normal, dy, dx)).square().sum(1, keepdim=True)
                            + 40 * (depth - shifted(depth, dy, dx)).abs() / depth.clamp_min(1)
                        )
                    log_weight = parameters[:, stage * 9 + k : stage * 9 + k + 1].clamp(-5, 5)
                    # The center is always available, preventing empty neighborhoods.
                    valid = shifted(support.float(), dy, dx) if k != 4 else torch.ones_like(depth)
                    weights.append(torch.exp(log_weight - penalty.clamp_max(60)) * valid)
                    neighbors.append(shifted(base, dy, dx))
                weights = torch.stack(weights, 2)
                base = (torch.stack(neighbors, 2) * weights).sum(2) / weights.sum(2).clamp_min(
                    1e-12
                )
            prediction = base * (albedo + 0.1) if self.demodulate else base
        else:
            prediction = self.correct_radiance(raw, parameters)
        if self.temporal:
            confidence = x[:, self.base_channels + 3 : self.base_channels + 4].clamp(0, 1)
            blend = 0.8 * torch.sigmoid(self.history_gate(encoding).float()) * confidence
            history = x[:, self.base_channels : self.base_channels + 3].clamp_min(0)
            prediction = prediction * (1 - blend) + history * blend
        # Legacy exports attenuated twice at scale 2. New experiments opt into a
        # single final blend after all reconstruction, including the upscale head.
        if self.scale == 1 or self.blend_policy == "legacy":
            prediction = raw + strength * (prediction - raw)
        if self.scale == 2:
            prediction = F.interpolate(
                prediction, scale_factor=2, mode="bilinear", align_corners=False
            )
            delta = self.upscale(encoding).float()
            prediction = self.correct_radiance(prediction, delta)
            raw = F.interpolate(raw, scale_factor=2, mode="bilinear", align_corners=False)
            support = F.interpolate(support.float(), scale_factor=2, mode="nearest") > 0.5
            strength = F.interpolate(strength, scale_factor=2, mode="nearest")
            prediction = raw + strength * (prediction - raw)
        return torch.where(support, prediction.clamp_min(0), raw)
