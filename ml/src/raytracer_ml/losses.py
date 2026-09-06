import torch


def reconstruction_loss(prediction, target):
    # Relative linear term keeps bright emitters from overwhelming ordinary surfaces.
    logarithmic = torch.mean(torch.abs(torch.log1p(prediction) - torch.log1p(target)))
    linear = torch.mean(torch.abs(prediction - target) / (1 + target))
    return logarithmic + 0.1 * linear
