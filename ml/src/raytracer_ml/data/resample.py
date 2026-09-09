"""Reuse measured HDR buffers without tracing rays or reading clean targets as inputs.

Area reduction combines four equal-area strata. Sample counts record the actual
number of contributing paths. Its noise/center-guide distribution is synthetic,
so this is training data, not evidence of native low-resolution render speed.
"""

import numpy as np


def area2(value):
    """CHW non-overlapping box reduction; no gamma conversion or negative weights."""
    c, h, w = value.shape
    if h % 2 or w % 2:
        raise ValueError("Area reduction requires even dimensions")
    return value.reshape(c, h // 2, 2, w // 2, 2).mean(axis=(2, 4))


def reduce_features(x):
    if x.shape[0] != 27 or not np.all(x[15] == x[15, 0, 0]):
        raise ValueError("Reuse reduction requires uniform schema-2 measurements")
    n = float(x[15, 0, 0])
    a = area2(x.astype(np.float64))
    # Variance of the stratified mean: sum of four mean variances / 16.
    # At one path/stratum no within-stratum estimate exists. A conservative
    # between-stratum estimate includes genuine spatial variation; label it.
    a[12:15] = area2(x[12:15].astype(np.float64)) / 4
    if n == 1:
        a[12:15] = np.maximum(0, area2(x[:3].astype(np.float64) ** 2) - a[:3] ** 2) / 3
    a[15], a[16] = 4 * n, 1
    a[24] = np.maximum(0, area2(x[24:25] + x[9:10].astype(np.float64) ** 2)[0] - a[9] ** 2)
    a[25] = np.clip(a[10] - np.sum(a[6:9] ** 2, axis=0), 0, 1)
    # A deterministic existing visibility probe, never interpolated across a
    # silhouette. It is offset from the new pixel center; native pairs retain
    # true center probes and are reported separately.
    channels = [17, 18, 19, 20, 21, 22, 23, 26]
    a[channels] = x[channels, 1::2, 1::2]
    return a.astype(np.float32)


def crop_turn(value, spec, scale=1):
    top, left, height, width = (int(spec[k]) * scale for k in ("top", "left", "height", "width"))
    if (
        min(top, left) < 0
        or min(height, width) < 1
        or top + height > value.shape[-2]
        or left + width > value.shape[-1]
    ):
        raise ValueError("Reuse crop exceeds source pixels")
    return np.rot90(
        value[..., top : top + height, left : left + width], spec["turns"], (-2, -1)
    ).copy()


def atrous(x, iterations=3):
    """NumPy version of RayTracer/denoiser.cpp, applied AFTER input reduction."""
    if not 1 <= iterations <= 5:
        raise ValueError("Invalid a-trous iterations")
    image = x[:3].transpose(1, 2, 0).astype(np.float64).copy()
    normal = x[20:23].transpose(1, 2, 0).astype(np.float64)
    albedo = x[17:20].transpose(1, 2, 0).astype(np.float64)
    depth = x[23].astype(np.float64)
    valid = (depth > 0) & (x[26] > 0.5)
    h, w = depth.shape
    kernel = [1, 4, 6, 4, 1]
    for iteration in range(iterations):
        step = 1 << iteration
        light = image @ np.array([0.2126, 0.7152, 0.0722])
        sigma = 0.05 + 0.5 * np.maximum(0, light)
        total, weights = np.zeros_like(image), np.zeros((h, w))
        for ky in range(-2, 3):
            for kx in range(-2, 3):
                dy, dx = ky * step, kx * step
                if abs(dy) >= h or abs(dx) >= w:
                    continue
                center = (slice(max(0, -dy), min(h, h - dy)), slice(max(0, -dx), min(w, w - dx)))
                other = (slice(max(0, dy), min(h, h + dy)), slice(max(0, dx), min(w, w + dx)))
                dot = np.maximum(0, (normal[center] * normal[other]).sum(-1))
                delta = np.abs(depth[center] - depth[other]) / (
                    0.02 * np.maximum(0.1, depth[center]) * step
                )
                delta += ((albedo[center] - albedo[other]) ** 2).sum(-1) / 0.02
                delta += np.abs(light[center] - light[other]) / sigma[center]
                weight = kernel[kx + 2] * kernel[ky + 2] * dot**64 * np.exp(-delta)
                weight *= valid[center] & valid[other]
                total[center] += image[other] * weight[..., None]
                weights[center] += weight
        mask = weights > 0
        image[mask] = total[mask] / weights[mask, None]
    return image.astype(np.float32)
