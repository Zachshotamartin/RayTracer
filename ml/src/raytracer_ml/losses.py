import torch


def reconstruction_loss(prediction, target, config=None):
    # Relative linear term keeps bright emitters from overwhelming ordinary surfaces.
    logarithmic = torch.mean(torch.abs(torch.log1p(prediction) - torch.log1p(target)))
    linear = torch.mean(torch.abs(prediction - target) / (1 + target))
    loss = logarithmic + 0.1 * linear
    if not config:
        return loss
    p, t = torch.log1p(prediction), torch.log1p(target)
    if config.get("gradient", 0):
        terms = []
        for axis in (-1, -2):
            if target.shape[axis] > 1:
                pg, tg = torch.diff(p, dim=axis), torch.diff(t, dim=axis)
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
