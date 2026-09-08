"""Deterministic native-size and camera variants, assigned after scene splits.

Zoom and roll change the actual camera rays. Lighting filters change radiance in
the scene before both renders; they never blur or resample Monte Carlo buffers.
"""

from copy import deepcopy
import math

import numpy as np

from ..io import identity


def resolution_options(config):
    sizes = config.get("resolutions")
    if sizes is None:
        width = config["width"]
        if type(width) is not int or width < 16 or width % 16:
            raise ValueError("Legacy dataset width must be a multiple of 16 for 16:9 pairs")
        sizes = [[width, width * 9 // 16]]
    scale = config.get("scale", 1)
    if not isinstance(sizes, list) or not sizes:
        raise ValueError("resolutions must contain [width, height] pairs")
    result = []
    for size in sizes:
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("Each resolution must be [width, height]")
        w, h = size
        if (
            any(type(v) is not int or v < 8 for v in size)
            or w * scale > 8192
            or h * scale > 16384
            or w * h * scale**2 > 16777216
            or not 0.1 <= w / h <= 10
        ):
            raise ValueError("Resolution exceeds renderer dimensions/aspect limits")
        if (w, h) in result:
            raise ValueError("Duplicate resolution")
        result.append((w, h))
    return result


def rendering_variants(items, config):
    sizes = resolution_options(config)
    count = config.get("resolution_variants", 1)
    if type(count) is not int or not 1 <= count <= len(sizes):
        raise ValueError("resolution_variants must be between 1 and the number of resolutions")
    aug = config.get("camera_augmentation", {})
    if set(aug) - {"zoom", "roll_degrees", "lighting_filters", "transport_probability"}:
        raise ValueError("Unknown camera augmentation option")
    zoom = aug.get("zoom", [1, 1])
    if (
        not isinstance(zoom, list)
        or len(zoom) != 2
        or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in zoom)
        or not 0.5 <= zoom[0] <= zoom[1] <= 2
    ):
        raise ValueError("Camera zoom must be a [minimum, maximum] range within 0.5..2")
    roll = aug.get("roll_degrees", 0)
    probability = aug.get("transport_probability", 0)
    if not math.isfinite(roll) or not 0 <= roll <= 180:
        raise ValueError("Camera roll must be within 0..180 degrees")
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Transport probability must be within 0..1")
    filters = aug.get("lighting_filters", [[1, 1, 1]])
    if not filters or any(
        len(f) != 3 or any(not math.isfinite(v) or not 0.25 <= v <= 4 for v in f) for f in filters
    ):
        raise ValueError("Lighting filters must contain positive RGB gains within 0.25..4")
    if aug and config.get("suite") == "sequence-v2":
        raise ValueError("Independent camera augmentation is spatial-only")
    explicit = "resolutions" in config or bool(aug)
    for index, item in enumerate(items):
        # Reshuffle every balanced cycle, avoiding resolution/family correlation.
        size_rng = np.random.default_rng(
            int(identity([config["seed"], index // len(sizes), "size-cycle-v1"])[:15], 16)
        )
        order = size_rng.permutation(len(sizes)) if explicit else np.arange(len(sizes))
        for variant in range(count):
            w, h = sizes[order[(index + variant) % len(sizes)]]
            out = deepcopy(item)
            out["width"], out["height"] = w, h
            if not explicit:
                yield out
                continue
            out["parent_configuration"] = item["id"]
            out["id"] = f"{item['id']}-r{w}x{h}"
            # The split and layout group are inherited, never redrawn per variant.
            rng = np.random.default_rng(
                int(identity([config["seed"], out["id"], "camera-v1"])[:15], 16)
            )
            factor = float(np.exp(rng.uniform(np.log(zoom[0]), np.log(zoom[1]))))
            angle = float(rng.uniform(-roll, roll))
            gain = filters[int(rng.integers(len(filters)))]
            scene, camera = out["scene"], out["scene"]["camera"]
            camera["aspect_ratio"] = w / h
            camera["vfov"] = math.degrees(
                2 * math.atan(math.tan(math.radians(camera["vfov"]) / 2) / factor)
            )
            axis = np.asarray(camera["lookat"]) - np.asarray(camera["lookfrom"])
            axis /= np.linalg.norm(axis)
            up = np.asarray(camera["up"], dtype=float)
            theta = math.radians(angle)
            camera["up"] = (
                up * math.cos(theta)
                + np.cross(axis, up) * math.sin(theta)
                + axis * np.dot(axis, up) * (1 - math.cos(theta))
            ).tolist()
            for light in scene.get("lights", []):
                light["emission"] = (np.asarray(light["emission"]) * gain).tolist()
            env = scene.get("environment", {})
            if "background" in env:
                env["background"] = (np.asarray(env["background"]) * gain).tolist()
            transport = "diffuse"
            if rng.random() < probability:
                # Cover non-diffuse transport with real reference rays, including
                # reflected/refracted boundaries that legacy models bypassed.
                transport = "glass" if rng.random() < 0.5 else "metal"
                material = (
                    {"type": "glass", "ior": float(rng.uniform(1.2, 1.7))}
                    if transport == "glass"
                    else {
                        "type": "metal",
                        "albedo": [0.8, 0.7, 0.6],
                        "roughness": float(rng.uniform(0.02, 0.4)),
                    }
                )
                scene["objects"].append(
                    {
                        "type": "sphere",
                        "center": [float(rng.uniform(-1, 1)), 0.7, 1],
                        "radius": 0.7,
                        "material": material,
                    }
                )
            out["render_variant"] = dict(
                input_width=w,
                input_height=h,
                zoom=factor,
                roll_degrees=angle,
                lighting_filter=gain,
                transport=transport,
            )
            yield out
