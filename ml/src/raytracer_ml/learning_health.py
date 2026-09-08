"""Opt-in spatial training telemetry; observations never alter optimizer decisions."""

import math

import torch

from .losses import reconstruction_loss


def model_region_mask(x, model):
    """Mirror the forward policy, including sample bypass; not learned confidence."""
    if model.kind == "joint":
        return torch.ones_like(x[:, :1]).repeat_interleave(2, -2).repeat_interleave(2, -1).bool()
    detail = model.kind in ("guided", "refine")
    mask = x[:, 11:12] >= 0.999999
    if model.feature_schema == 2 or detail:
        mask = ((x[:, 10:11] - x[:, 11:12]).abs() < 1e-6) & (x[:, 10:11] > 0)
        mask = mask & (x[:, 15:16] < 128)
    if model.feature_schema == 2:
        mask = mask & ((x[:, 26:27] > 0.5) | (x[:, 23:24] == 0))
    if model.scale == 2:
        mask = torch.nn.functional.interpolate(mask.float(), scale_factor=2, mode="nearest") > 0.5
    return mask


class LearningHealth:
    def __init__(self, model, config, loss_config):
        self.config = config
        self.loss_config = loss_config
        self.model = model
        self.decoder = None
        self.hook = model.head.register_forward_pre_hook(self._capture)
        self.reset()

    def _capture(self, module, inputs):
        self.decoder = inputs[0].detach()

    def reset(self):
        self.rows = []
        self.decoder = None

    @torch.no_grad()
    def observe(self, x, y, prediction, loss, gradient_norm):
        raw = x[:, :3]
        if raw.shape[-2:] != y.shape[-2:]:
            raw = torch.nn.functional.interpolate(
                raw, size=y.shape[-2:], mode="bilinear", align_corners=False
            )
        encoder = getattr(self.model, "enc1", getattr(self.model, "encoder", None))
        encoder_gradient = encoder[0].weight.grad if encoder is not None else None
        row = {
            "raw_loss": reconstruction_loss(raw, y, self.loss_config),
            "prediction_loss": loss.detach(),
            "gradient_norm": gradient_norm.detach(),
            "prediction_change_log_mae": (torch.log1p(prediction) - torch.log1p(raw)).abs().mean(),
            "head_input_zero_fraction": (self.decoder == 0).float().mean(),
        }
        if encoder_gradient is not None:
            # Gradient after the existing clipping operation; named explicitly in the report.
            row["encoder_gradient_norm_after_clip"] = encoder_gradient.norm()
        mask = model_region_mask(x, self.model)
        row["model_region_pixels"] = mask.sum()
        row["bypass_pixels"] = (~mask).sum()
        for name, image in (("raw", raw), ("prediction", prediction)):
            for metric, error in (
                ("log_mae", (torch.log1p(image) - torch.log1p(y)).abs()),
                ("linear_mse", (image - y).square()),
            ):
                row[f"{name}_model_region_{metric}_sum"] = (
                    error.mean(1, keepdim=True) * mask
                ).sum()
        self.rows.append({k: float(v.cpu()) for k, v in row.items()})

    def summary(self, epoch):
        if not self.rows:
            return {}
        means = {key: sum(r[key] for r in self.rows) / len(self.rows) for key in self.rows[0]}
        raw = means["raw_loss"]
        reduction = 1 - means["prediction_loss"] / raw if raw > 1e-12 else None
        region = {}
        pixels = sum(row["model_region_pixels"] for row in self.rows)
        for name in ("raw", "prediction"):
            for metric in ("log_mae", "linear_mse"):
                key = f"{name}_model_region_{metric}"
                region[key] = (
                    sum(row[key + "_sum"] for row in self.rows) / pixels if pixels else None
                )
        raw_region = region["raw_model_region_log_mae"]
        region_reduction = (
            1 - region["prediction_model_region_log_mae"] / raw_region
            if raw_region is not None and raw_region > 1e-12
            else None
        )
        warning_metric = self.config.get("warning_metric", "whole_image_loss")
        observed = region_reduction if warning_metric == "model_region_log_mae" else reduction
        warning = None
        if epoch + 1 >= self.config.get("stall_after_epochs", 5):
            if warning_metric == "model_region_log_mae" and not pixels:
                warning = "no-model-region-pixels"
            elif observed is not None and observed < self.config.get(
                "minimum_loss_reduction", 0.02
            ):
                warning = "little-improvement-over-input"
        return {
            "schema_version": 2,
            "scope": "Paired current training batches, including configured augmentation",
            **{k: v for k, v in means.items() if not k.endswith("_sum")},
            **region,
            "model_region_pixels": int(pixels),
            "bypass_pixels": int(sum(row["bypass_pixels"] for row in self.rows)),
            "loss_reduction_vs_raw": reduction,
            "model_region_log_mae_reduction_vs_raw": region_reduction,
            "warning_metric": warning_metric,
            "warning": warning,
        }

    def close(self):
        self.hook.remove()


def validate_health_config(config):
    if not isinstance(config, dict):
        raise ValueError("learning_diagnostics must be a mapping")
    if config.get("warning_metric", "whole_image_loss") not in (
        "whole_image_loss",
        "model_region_log_mae",
    ):
        raise ValueError("Invalid learning diagnostics warning metric")
    after = config.get("stall_after_epochs", 5)
    reduction = config.get("minimum_loss_reduction", 0.02)
    if not isinstance(after, int) or isinstance(after, bool) or after < 1:
        raise ValueError("stall_after_epochs must be a positive integer")
    if (
        not isinstance(reduction, (int, float))
        or not math.isfinite(reduction)
        or not 0 <= reduction < 1
    ):
        raise ValueError("minimum_loss_reduction must be in [0, 1)")
