import math

import torch


def validate_loss_config(config):
    if config is None:
        return
    if not isinstance(config, dict):
        raise ValueError("loss must be a mapping")
    if config.get("kind", "log_l1") not in ("log_l1", "relative_l2"):
        raise ValueError("Invalid reconstruction loss kind")
    for key in ("gradient", "edge_weight", "energy"):
        value = config.get(key, 0)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Loss {key} must be finite and nonnegative")
    scales = config.get("gradient_scales", [1])
    if (
        not isinstance(scales, list)
        or not scales
        or any(type(s) is not int or not 1 <= s <= 8 for s in scales)
        or len(set(scales)) != len(scales)
    ):
        raise ValueError("gradient_scales must be distinct integer offsets within 1..8")


def reconstruction_loss(prediction, target, config=None):
    kind = (config or {}).get("kind", "log_l1")
    if kind == "relative_l2":
        # Noise2Noise, section 3.3: linear targets and stopped prediction-dependent
        # normalization. Do not differentiate the denominator or log the target.
        loss = ((prediction - target).square() / (prediction.detach() + 0.01).square()).mean()
    elif kind == "log_l1":
        logarithmic = torch.mean(torch.abs(torch.log1p(prediction) - torch.log1p(target)))
        linear = torch.mean(torch.abs(prediction - target) / (1 + target))
        loss = logarithmic + 0.1 * linear
    else:
        raise ValueError("Invalid reconstruction loss kind")
    if not config:
        return loss
    p, t = torch.log1p(prediction), torch.log1p(target)
    if config.get("gradient", 0):
        terms = []
        for axis, step in (
            (axis, step) for axis in (-1, -2) for step in config.get("gradient_scales", [1])
        ):
            if target.shape[axis] > step:
                length = target.shape[axis] - step
                pg = (p.narrow(axis, step, length) - p.narrow(axis, 0, length)) / step
                tg = (t.narrow(axis, step, length) - t.narrow(axis, 0, length)) / step
                weight = 1 + (tg.abs().mean(1, keepdim=True) * 10).clamp_max(
                    float(config.get("edge_weight", 3))
                )
                terms.append(((pg - tg).abs() * weight).mean())
        if terms:
            loss = loss + float(config["gradient"]) * sum(terms) / len(terms)
    if config.get("energy", 0):
        # Per-channel patch energy and bright regions keep HDR/color bias visible.
        mean = target.mean(dim=(-2, -1))
        energy = ((prediction.mean(dim=(-2, -1)) - mean).abs() / (0.03 + mean)).mean()
        highlights = (target.amax(1, keepdim=True) > 1).to(target.dtype)
        denominator = highlights.sum(dim=(-2, -1)).clamp_min(1)
        actual = (prediction * highlights).sum(dim=(-2, -1)) / denominator
        expected = (target * highlights).sum(dim=(-2, -1)) / denominator
        energy = energy + ((actual - expected).abs() / (1 + expected)).mean()
        loss = loss + float(config["energy"]) * energy
    return loss
