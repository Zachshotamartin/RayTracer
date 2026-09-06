import numpy as np
from skimage.metrics import structural_similarity
from .io import display


def image_metrics(prediction, target):
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid prediction")
    p, t = display(prediction), display(target)
    mse = float(np.mean((p - t) ** 2))
    size = min(7, p.shape[0], p.shape[1])
    size = size if size % 2 else size - 1
    ssim = (
        float(structural_similarity(p, t, channel_axis=2, data_range=1.0, win_size=size))
        if size >= 3
        else None
    )
    return {
        "linear_mse": float(np.mean((prediction.astype(np.float64) - target) ** 2)),
        "log_mae": float(np.mean(np.abs(np.log1p(prediction) - np.log1p(target)))),
        "display_mse": mse,
        "psnr": float(-10 * np.log10(max(mse, 1e-12))),
        "ssim": ssim,
    }
