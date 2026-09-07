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
    if "split_counts" in config:
        counts = config["split_counts"]
        if (
            len(counts) != 3
            or any(type(n) is not int or n < 1 for n in counts)
            or sum(counts) != groups
        ):
            raise ValueError("split_counts must contain three positive counts summing to groups")
        train_end, val_end = counts[0], counts[0] + counts[1]
    suite = config.get("suite", "room-v1")
    if suite not in ("room-v1", "challenge-v2", "transport-v2", "sequence-v2"):
        raise ValueError("Unknown scene suite")
    for group in range(groups):
        split = "train" if group < train_end else "val" if group < val_end else "test"
        for frame in range(views):
            yield {
                "id": f"g{group:04d}-v{frame:03d}",
                "group": f"layout-{group:04d}"
                if suite == "room-v1"
                else f"{suite}-{seed}-{group:04d}",
                "split": split,
                "cohort": "unseen-layout" if split == "test" else split,
                "frame": frame,
                "stratum": "room" if suite == "room-v1" else STRATA[group % len(STRATA)],
                "scene": room(
                    seed + group * 31, seed + group * 31 + 1, seed + group * 31 + 2, frame, views
                )
                if suite == "room-v1"
                else challenge(
                    seed + group * 31,
                    seed + group * 31 + 1,
                    seed + group * 31 + 2,
                    frame,
                    views,
                    stratum=STRATA[group % len(STRATA)],
                    suite=suite,
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
                    "parent_layout": f"layout-{group:04d}"
                    if suite == "room-v1"
                    else f"{suite}-{seed}-{group:04d}",
                    "split": "test",
                    "cohort": cohort,
                    "frame": frame,
                    "stratum": "room" if suite == "room-v1" else STRATA[group % len(STRATA)],
                    "scene": (room if suite == "room-v1" else challenge)(
                        base,
                        base + 1 + (200000 if cohort == "new-light" else 0),
                        base + 2 + (100000 if cohort == "heldout-camera" else 0),
                        frame,
                        views,
                        **(
                            {"stratum": STRATA[group % len(STRATA)], "suite": suite}
                            if suite != "room-v1"
                            else {}
                        ),
                    ),
                }


STRATA = ("edges", "textures", "lighting", "clutter")


def challenge(
    layout_seed, lighting_seed, camera_seed, frame=0, views=1, stratum="edges", suite="challenge-v2"
):
    """Self-contained geometry: no external assets can change a manifest's meaning."""
    scene = room(layout_seed, lighting_seed, camera_seed, frame, views)
    rng = np.random.default_rng(layout_seed + 900000)
    light_rng = np.random.default_rng(lighting_seed + 900000)
    scene["name"] = f"{suite}-{stratum}"
    scene["objects"] = scene["objects"][:3]
    camera = scene["camera"]
    phase = frame / max(1, views - 1)
    camera["vfov"] = float(rng.uniform(32, 55))
    camera["lookat"] = [0, float(rng.uniform(0.6, 1.5)), -0.5]
    camera["lookfrom"][1] = float(rng.uniform(1.4, 4))

    def matte():
        return {"type": "diffuse", "albedo": rng.uniform(0.04, 0.9, 3).tolist()}

    # Slanted boxes, a triangular wedge, and thin posts with reproducible dimensions.
    for i in range(3 if stratum != "clutter" else 7):
        x = float(rng.uniform(-3, 3))
        z = float(rng.uniform(-2.5, 1))
        h = float(rng.uniform(0.25, 2))
        width = float(rng.uniform(0.025, 0.11) if i == 0 else rng.uniform(0.25, 1.1))
        material = matte()
        if stratum == "textures":
            material["texture"] = {
                "type": "checker" if i % 2 == 0 else "bands",
                "scale": float(rng.uniform(0.025, 0.4)),
                "frequency": float(rng.uniform(3, 24)),
                "color": rng.uniform(0.03, 0.9, 3).tolist(),
            }
        scene["objects"].append(
            {
                "type": "box",
                "center": [x, h / 2, z],
                "size": [width, h, float(rng.uniform(0.08, 0.6))],
                "rotation_y": float(rng.uniform(-1.2, 1.2)),
                "material": material,
                "annotation": "thin" if i == 0 else "solid",
            }
        )
    x = float(rng.uniform(-2, 1))
    scene["objects"].append(
        {
            "type": "mesh",
            "vertices": [
                [x, 0, 0],
                [x + 1.2, 0, 0],
                [x + 0.15, 1.4, 0],
                [x, 0, -0.7],
                [x + 1.2, 0, -0.7],
                [x + 0.15, 1.4, -0.7],
            ],
            "faces": [
                [0, 1, 2],
                [3, 5, 4],
                [0, 3, 4],
                [0, 4, 1],
                [1, 4, 5],
                [1, 5, 2],
                [2, 5, 3],
                [2, 3, 0],
            ],
            "material": matte(),
        }
    )
    if stratum in ("textures", "clutter"):
        scene["objects"][0]["material"]["texture"] = {
            "type": "checker",
            "scale": float(rng.uniform(0.08, 0.8)),
            "color": rng.uniform(0.05, 0.8, 3).tolist(),
        }
    light = scene["lights"][0]
    size = float(light_rng.uniform(0.15, 1) if stratum == "lighting" else light_rng.uniform(1, 3))
    light["u"], light["v"] = [size, 0, 0], [0, 0, size]
    light["emission"] = (light_rng.uniform(3, 18, 3) * (2.5 / size) ** 1.5).tolist()
    scene["environment"]["background"] = [float(light_rng.uniform(0.002, 0.06))] * 3
    if suite == "transport-v2":
        material = (
            {"type": "glass", "ior": float(rng.uniform(1.2, 1.8))}
            if stratum in ("edges", "lighting")
            else {
                "type": "metal",
                "albedo": [0.85, 0.75, 0.65],
                "roughness": 0.02 if stratum == "textures" else 0.35,
            }
        )
        scene["objects"].append(
            {"type": "sphere", "center": [0, 0.8, 1.5], "radius": 0.8, "material": material}
        )
    if suite == "sequence-v2":
        camera["lookfrom"][0] += 1.5 * math.sin(phase * 2 * math.pi)
        camera["vfov"] += 4 * math.sin(phase * math.pi)
        if stratum == "clutter" and phase >= 0.5:
            camera["lookfrom"] = [5, 3, 4]  # A declared camera cut.
        if stratum == "lighting" and phase >= 0.5:
            light["emission"] = [c * 0.25 for c in light["emission"]]
    return scene
