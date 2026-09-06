"""Bounded diffuse rooms with independently reproducible layout, lighting and cameras."""

import math
import numpy as np


def room(layout_seed, lighting_seed, camera_seed, frame=0, views=1):
    rng = np.random.default_rng(layout_seed)
    light_rng = np.random.default_rng(lighting_seed)
    camera_rng = np.random.default_rng(camera_seed)

    def material():
        return {"type": "diffuse", "albedo": rng.uniform(0.08, 0.85, 3).tolist()}

    objects = [
        {
            "type": "quad",
            "origin": [-5, 0, 5],
            "u": [10, 0, 0],
            "v": [0, 0, -10],
            "material": material(),
        },
        {
            "type": "quad",
            "origin": [-5, 0, -4],
            "u": [10, 0, 0],
            "v": [0, 5, 0],
            "material": material(),
        },
        {
            "type": "quad",
            "origin": [-5, 0, 5],
            "u": [0, 0, -9],
            "v": [0, 5, 0],
            "material": material(),
        },
    ]
    for i in range(3):
        radius = float(rng.uniform(0.35, 0.85))
        center = [-2.2 + 2.0 * i, radius, float(rng.uniform(-1.8, 0.5))]
        if i % 2 == 0:
            objects.append(
                {"type": "sphere", "center": center, "radius": radius, "material": material()}
            )
        else:
            objects.append(
                {
                    "type": "box",
                    "center": center,
                    "size": [radius * 2] * 3,
                    "rotation_y": float(rng.uniform(-0.7, 0.7)),
                    "material": material(),
                }
            )
    origin = [float(light_rng.uniform(-2, 0)), 4.5, float(light_rng.uniform(-2, 0))]
    emission = light_rng.uniform(4, 12, 3).tolist()
    angle = float(camera_rng.uniform(-0.3, 0.3)) + (frame / max(views - 1, 1) - 0.5) * 0.35
    camera = {
        "lookfrom": [8 * math.sin(angle), float(camera_rng.uniform(2.1, 3.4)), 8 * math.cos(angle)],
        "lookat": [0, 1, -0.3],
        "up": [0, 1, 0],
        "vfov": 40,
        "aspect_ratio": 16 / 9,
        "defocus_angle": 0,
        "focus_dist": 8,
    }
    return {
        "schema_version": 1,
        "name": "ml-room",
        "camera": camera,
        "objects": objects,
        "lights": [
            {
                "type": "area",
                "origin": origin,
                "u": [2.5, 0, 0],
                "v": [0, 0, 2.5],
                "emission": emission,
            }
        ],
        "environment": {"sky": False, "background": [0.01, 0.01, 0.01]},
    }


def configurations(config):
    groups, views, seed = int(config["groups"]), int(config["views"]), int(config["seed"])
    if groups < 4 or views < 1 or groups * views > 100000:
        raise ValueError("Need at least four scene groups and positive bounded view count")
    train_end = max(1, int(groups * 0.7))
    val_end = min(groups - 1, max(train_end + 1, int(groups * 0.85)))
    for group in range(groups):
        split = "train" if group < train_end else "val" if group < val_end else "test"
        for frame in range(views):
            yield {
                "id": f"g{group:04d}-v{frame:03d}",
                "group": f"layout-{group:04d}",
                "split": split,
                "cohort": "unseen-layout" if split == "test" else split,
                "frame": frame,
                "scene": room(
                    seed + group * 31, seed + group * 31 + 1, seed + group * 31 + 2, frame, views
                ),
            }

    # These explicitly named cohorts reuse training geometry, never the same full scene/camera.
    probes = int(config.get("cohort_groups", 0))
    if probes < 0 or probes > train_end:
        raise ValueError("cohort_groups must fit within training layout groups")
    for cohort in ("heldout-camera", "new-light"):
        for group in range(probes):
            for frame in range(views):
                base = seed + group * 31
                yield {
                    "id": f"{cohort}-g{group:04d}-v{frame:03d}",
                    "group": f"{cohort}-{group:04d}",
                    "parent_layout": f"layout-{group:04d}",
                    "split": "test",
                    "cohort": cohort,
                    "frame": frame,
                    "scene": room(
                        base,
                        base + 1 + (200000 if cohort == "new-light" else 0),
                        base + 2 + (100000 if cohort == "heldout-camera" else 0),
                        frame,
                        views,
                    ),
                }
