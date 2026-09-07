"""Training-only transforms with HDR, world-guide and Monte Carlo moment semantics."""

import numpy as np
import torch


def fuse_measurements(a, b):
    """Merge independent sample streams using parallel Welford moments, losslessly in concept.

    Callers enforce same scene/camera, distinct input seeds and independent targets.
    Center guides are deterministic visibility probes and must agree exactly.
    """
    if a.shape != b.shape or a.shape[0] not in (17, 27):
        raise ValueError("Incompatible measurements")
    if a.shape[0] == 27 and not np.array_equal(
        a[[17, 18, 19, 20, 21, 22, 23, 26]], b[[17, 18, 19, 20, 21, 22, 23, 26]]
    ):
        raise ValueError("Cannot fuse different center geometry")
    na, nb = a[15:16].astype(np.float64), b[15:16].astype(np.float64)
    n = na + nb
    wa, wb = na / n, nb / n
    out = a.astype(np.float64).copy()
    out[:12] = a[:12] * wa + b[:12] * wb
    m2 = a[12:15] * na * (na - 1) + b[12:15] * nb * (nb - 1)
    m2 += (a[:3].astype(np.float64) - b[:3]) ** 2 * na * nb / n
    out[12:15] = m2 / (n * (n - 1))
    out[15:16], out[16:17] = n, 1
    if a.shape[0] == 27:
        out[24:25] = wa * a[24:25] + wb * b[24:25] + wa * wb * (a[9:10] - b[9:10]) ** 2
        out[25] = np.maximum(0, out[10] - np.sum(out[6:9] ** 2, axis=0))
    return out.astype(np.float32)


def draw_transform(config, *, square=True):
    if not config:
        return {"flip_x": False, "flip_y": False, "turns": 0, "gain": torch.ones(3)}
    for key in ("flip_x", "flip_y"):
        if not 0 <= config.get(key, 0.5) <= 1:
            raise ValueError("Flip probabilities must be in [0,1]")
    exposure = float(config.get("exposure_stops", 0))
    color = float(config.get("lighting_color_stops", 0))
    if not 0 <= exposure <= 4 or not 0 <= color <= 1:
        raise ValueError("Augmentation exposure/color range is too large")
    gain = 2 ** ((torch.rand(()) * 2 - 1) * exposure + (torch.rand(3) * 2 - 1) * color)
    return {
        "flip_x": bool(torch.rand(()) < config.get("flip_x", 0.5)),
        "flip_y": bool(torch.rand(()) < config.get("flip_y", 0.5)),
        "turns": int(torch.randint(4, ()).item()) if config.get("rotate90", True) and square else 0,
        "gain": gain,
    }


def spatial_transform(value, transform, inverse=False):
    if inverse:
        value = torch.rot90(value, -transform["turns"], (-2, -1))
    if transform["flip_y"]:
        value = value.flip(-2)
    if transform["flip_x"]:
        value = value.flip(-1)
    if not inverse:
        value = torch.rot90(value, transform["turns"], (-2, -1))
    return value


def apply_transform(x, target, transform, base_channels=17):
    """Accept CHW or NCHW tensors. World-coordinate vector components stay unchanged."""
    x, target = spatial_transform(x, transform).clone(), spatial_transform(target, transform)
    gain = transform["gain"].to(device=x.device, dtype=x.dtype)[:, None, None]
    channel_axis = x.ndim - 3
    x.narrow(channel_axis, 0, 3).mul_(gain)
    x.narrow(channel_axis, 12, 3).mul_(gain.square())
    if x.shape[channel_axis] == base_channels + 4:
        x.narrow(channel_axis, base_channels, 3).mul_(gain)
    return x.contiguous(), (target * gain).contiguous()


def undo_prediction(prediction, transform):
    gain = transform["gain"].to(device=prediction.device, dtype=prediction.dtype)[:, None, None]
    return spatial_transform(prediction / gain, transform, inverse=True)
