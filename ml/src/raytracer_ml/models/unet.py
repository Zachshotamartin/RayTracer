"""Small HDR reconstruction network; preprocessing and fallback travel with ONNX."""

import torch
from torch import nn
from torch.nn import functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.ReLU(),
    )


class ReconstructionNet(nn.Module):
    def __init__(
        self,
        width=24,
        kind="unet",
        scale=1,
        inputs="all",
        temporal=False,
        feature_schema=1,
        refinement=False,
    ):
        super().__init__()
        if (
            kind not in ("unet", "conv")
            or inputs not in ("all", "rgb", "guides")
            or scale not in (1, 2)
            or feature_schema not in (1, 2)
        ):
            raise ValueError("Invalid model architecture")
        if width < 4 or width > 128:
            raise ValueError("Model width must be 4..128")
        if temporal and scale != 1:
            raise ValueError("Temporal reconstruction currently requires scale=1")
        self.temporal = temporal
        self.feature_schema = feature_schema
        self.base_channels = 27 if feature_schema == 2 else 17
        self.feature_channels = {"all": self.base_channels, "rgb": 3, "guides": 12}[inputs]
        self.kind, self.scale, self.inputs = kind, scale, inputs
        channels = self.feature_channels + (4 if temporal else 0)
        self.enc1 = block(channels, width)
        if kind == "unet":
            self.enc2 = block(width, width * 2)
            self.middle = block(width * 2, width * 4)
            self.dec2 = block(width * 6, width * 2)
            self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
        self.refinement = block(width + channels, width) if refinement else None
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        raw = x[:, :3].clamp_min(0)
        logged = torch.log1p(raw)
        features = torch.cat(
            [
                logged,
                x[:, 3:9],
                torch.log1p(x[:, 9:10].clamp_min(0)) / 5,
                x[:, 10:12],
                torch.log1p(x[:, 12:15].clamp_min(0)),
                torch.log2(x[:, 15:16].clamp_min(1)) / 6,
                x[:, 16:17],
            ],
            dim=1,
        )
        if self.feature_schema == 2:
            features = torch.cat(
                [features, x[:, 17:23], torch.log1p(x[:, 23:25].clamp_min(0)) / 5, x[:, 25:27]], 1
            )
        features = features[:, : self.feature_channels]
        if self.temporal:
            features = torch.cat(
                [
                    features,
                    torch.log1p(x[:, self.base_channels : self.base_channels + 3].clamp_min(0)),
                    x[:, self.base_channels + 3 : self.base_channels + 4],
                ],
                dim=1,
            )
        a = self.enc1(features)
        if self.kind == "unet":
            b = self.enc2(F.avg_pool2d(a, 2, ceil_mode=True))
            c = self.middle(F.avg_pool2d(b, 2, ceil_mode=True))
            d = self.dec2(
                torch.cat(
                    [F.interpolate(c, size=b.shape[-2:], mode="bilinear", align_corners=False), b],
                    1,
                )
            )
            a = self.dec1(
                torch.cat(
                    [F.interpolate(d, size=a.shape[-2:], mode="bilinear", align_corners=False), a],
                    1,
                )
            )
        if self.refinement is not None:
            a = self.refinement(torch.cat([a, features], 1))
        correction = self.head(a)
        support = x[:, 11:12] >= 0.999999
        if self.feature_schema == 2:
            support = ((x[:, 10:11] - x[:, 11:12]).abs() < 1e-6) & (x[:, 10:11] > 0)
            support &= (x[:, 26:27] > 0.5) | (x[:, 23:24] == 0)
            correction = correction * ((128 - x[:, 15:16]) / 96).clamp(0, 1)
        if self.scale == 2:
            correction = F.interpolate(
                correction, scale_factor=2, mode="bilinear", align_corners=False
            )
            raw = F.interpolate(raw, scale_factor=2, mode="bilinear", align_corners=False)
            logged = torch.log1p(raw)
            support = F.interpolate(support.float(), scale_factor=2, mode="nearest") > 0.5
        if self.feature_schema == 2:
            prediction = ((raw + 0.01) * torch.exp(2 * torch.tanh(correction)) - 0.01).clamp_min(0)
            samples = x[:, 15:16]
            if self.scale == 2:
                samples = F.interpolate(samples, scale_factor=2, mode="nearest")
            prediction = torch.where(samples >= 128, raw, prediction)
        else:
            prediction = torch.exp((logged + correction).clamp(0, 12)) - 1
        return torch.where(support, prediction, raw)


def build_model(config):
    if config.get("kind") in ("guided", "refine"):
        from .detail import DetailNet

        return DetailNet(**config)
    return ReconstructionNet(**config)
