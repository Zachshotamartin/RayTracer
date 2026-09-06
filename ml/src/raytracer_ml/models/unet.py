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
    def __init__(self, width=24, kind="unet", scale=1, inputs="all", temporal=False):
        super().__init__()
        if (
            kind not in ("unet", "conv")
            or inputs not in ("all", "rgb", "guides")
            or scale not in (1, 2)
        ):
            raise ValueError("Invalid model architecture")
        if width < 4 or width > 128:
            raise ValueError("Model width must be 4..128")
        if temporal and scale != 1:
            raise ValueError("Temporal reconstruction currently requires scale=1")
        self.temporal = temporal
        self.kind, self.scale, self.inputs = kind, scale, inputs
        channels = {"all": 17, "rgb": 3, "guides": 12}[inputs] + (4 if temporal else 0)
        self.enc1 = block(channels, width)
        if kind == "unet":
            self.enc2 = block(width, width * 2)
            self.middle = block(width * 2, width * 4)
            self.dec2 = block(width * 6, width * 2)
            self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 3, 3, padding=1)
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
        features = features[:, : {"all": 17, "rgb": 3, "guides": 12}[self.inputs]]
        if self.temporal:
            features = torch.cat(
                [features, torch.log1p(x[:, 17:20].clamp_min(0)), x[:, 20:21]], dim=1
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
        correction = self.head(a)
        support = x[:, 11:12] >= 0.999999
        if self.scale == 2:
            correction = F.interpolate(
                correction, scale_factor=2, mode="bilinear", align_corners=False
            )
            raw = F.interpolate(raw, scale_factor=2, mode="bilinear", align_corners=False)
            logged = torch.log1p(raw)
            support = F.interpolate(support.float(), scale_factor=2, mode="nearest") > 0.5
        prediction = torch.exp((logged + correction).clamp(0, 12)) - 1
        return torch.where(support, prediction, raw)


def build_model(config):
    return ReconstructionNet(**config)
