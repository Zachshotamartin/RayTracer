"""Opt-in spatial training telemetry; observations never alter optimizer decisions."""

import math

import torch

from .losses import reconstruction_loss


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
        encoder = getattr(self.model, "enc1", None)
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
        self.rows.append({k: float(v.cpu()) for k, v in row.items()})

    def summary(self, epoch):
        if not self.rows:
            return {}
        means = {key: sum(r[key] for r in self.rows) / len(self.rows) for key in self.rows[0]}
        raw = means["raw_loss"]
        reduction = 1 - means["prediction_loss"] / raw if raw > 1e-12 else None
        warning = None
        if epoch + 1 >= self.config.get("stall_after_epochs", 5) and reduction is not None:
            if reduction < self.config.get("minimum_loss_reduction", 0.02):
                warning = "little-improvement-over-input"
        return {
            "scope": "Paired current training batches, including configured augmentation",
            **means,
            "loss_reduction_vs_raw": reduction,
            "warning": warning,
        }

    def close(self):
        self.hook.remove()


def validate_health_config(config):
    if not isinstance(config, dict):
        raise ValueError("learning_diagnostics must be a mapping")
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
