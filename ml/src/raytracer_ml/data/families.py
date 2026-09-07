"""Eight procedural scene families with independent geometry, light, and camera seeds.

All meshes are embedded in the scene contract. These are diffuse training families;
glass and metal remain separately labelled transport stress tests.
"""

import math
import numpy as np

FAMILIES = ("studio", "courtyard", "corridor", "stairs", "shelves", "arches", "terrain", "pavilion")


def diverse_scene(layout_seed, lighting_seed, camera_seed, frame, views, *, family, stratum):
    if family not in FAMILIES:
        raise ValueError("Unknown scene family")
    rng = np.random.default_rng(layout_seed)
    light_rng = np.random.default_rng(lighting_seed)
    camera_rng = np.random.default_rng(camera_seed)
    objects = []

    def material(textured=False):
        result = {"type": "diffuse", "albedo": rng.uniform(0.04, 0.9, 3).tolist()}
        if textured:
            result["texture"] = {
                "type": "checker" if rng.random() < 0.5 else "bands",
                "scale": float(rng.uniform(0.025, 0.8)),
                "frequency": float(rng.uniform(2, 32)),
                "color": rng.uniform(0.02, 0.9, 3).tolist(),
            }
        return result

    def box(center, size, *, thin=False, rotation=0, textured=False):
        objects.append(
            dict(
                type="box",
                center=center,
                size=size,
                rotation_y=rotation,
                material=material(textured),
                annotation="thin" if thin else "solid",
            )
        )

    def quad(origin, u, v, textured=False):
        objects.append(dict(type="quad", origin=origin, u=u, v=v, material=material(textured)))

    textured = stratum == "textures"
    extent = float(rng.uniform(4.5, 6))
    height = float(rng.uniform(3.2, 5.5))
    target = [0, 1, -0.5]
    distance = 8.5
    if family != "terrain":
        quad([-extent, 0, extent], [2 * extent, 0, 0], [0, 0, -2 * extent], textured)
    if family == "studio":
        quad([-extent, 0, -extent], [2 * extent, 0, 0], [0, height, 0], textured)
        quad([-extent, 0, extent], [0, 0, -2 * extent], [0, height, 0])
        quad([extent, 0, -extent], [0, 0, 2 * extent], [0, height, 0])
        quad([-extent, height, -extent], [2 * extent, 0, 0], [0, 0, 2 * extent])
    elif family == "courtyard":
        for z in (-3.5, -0.5):
            for x in (-3.0, 3.0):
                box([x, height / 2, z], [0.45, height, 0.45], textured=textured)
        box([0, 0.45, -3.8], [6.5, 0.9, 0.35], textured=textured)
        box([-3.5, 0.3, -0.5], [0.35, 0.6, 6])
        box([3.5, 0.3, -0.5], [0.35, 0.6, 6])
    elif family == "corridor":
        width = float(rng.uniform(2.2, 3))
        for side in (-1, 1):
            box([side * width, height / 2, -1], [0.25, height, 9], textured=textured)
            for z in np.linspace(-4, 2, 5):
                box([side * (width - 0.2), height / 2, float(z)], [0.22, height, 0.18], thin=True)
        box([0, height, -1], [width * 2, 0.15, 9])
        target = [0, 1, -2]
    elif family == "stairs":
        rise, tread = float(rng.uniform(0.18, 0.32)), float(rng.uniform(0.4, 0.65))
        for step in range(9):
            h = rise * (step + 1)
            box([0, h / 2, 1.5 - step * tread], [3, h, tread], textured=textured)
            for side in (-1, 1):
                box([side * 1.6, h + 0.5, 1.5 - step * tread], [0.045, 1, 0.045], thin=True)
        target = [0, 1.2, -0.5]
    elif family == "shelves":
        for column in (-1, 1):
            x = column * 1.8
            for y in (0.35, 1.2, 2.05, 2.9):
                box([x, y, -1.5], [2.7, 0.085, 1.3], thin=True, textured=textured)
                for j in range(5):
                    h = float(rng.uniform(0.15, 0.6))
                    box(
                        [x - 1 + j * 0.48, y + 0.05 + h / 2, -1.5],
                        [0.25, h, 0.4],
                        textured=textured,
                    )
            for side in (-1, 1):
                box([x + side * 1.3, 1.5, -1.5], [0.08, 3, 0.08], thin=True)
        target = [0, 1.5, -1.5]
    elif family == "arches":
        radius, thickness = float(rng.uniform(1.3, 1.8)), float(rng.uniform(0.18, 0.4))
        for z in (-2.7, -0.5, 1.7):
            vertices = []
            segments = 16
            for depth in (z - 0.2, z + 0.2):
                for r in (radius, radius + thickness):
                    for theta in np.linspace(0, math.pi, segments + 1):
                        vertices.append(
                            [float(r * math.cos(theta)), float(1.2 + r * math.sin(theta)), depth]
                        )
            faces = []
            n = segments + 1
            for j in range(segments):
                for a, b in (
                    (j, n + j),
                    (2 * n + j, 3 * n + j),
                    (j, 2 * n + j),
                    (n + j, 3 * n + j),
                ):
                    faces.extend([[a, a + 1, b + 1], [a, b + 1, b]])
            for j in (0, segments):
                faces.extend([[j, n + j, 3 * n + j], [j, 3 * n + j, 2 * n + j]])
            objects.append(
                dict(type="mesh", vertices=vertices, faces=faces, material=material(textured))
            )
            for side in (-1, 1):
                box([side * (radius + thickness / 2), 0.6, z], [thickness, 1.2, 0.4])
        target = [0, 1.7, -0.5]
    elif family == "terrain":
        grid = 15
        axis = np.linspace(-extent, extent, grid)
        phase = float(rng.uniform(0, 2 * math.pi))
        vertices = [
            [
                float(x),
                float(0.35 * math.sin(x + phase) * math.cos(z * 0.7) + rng.uniform(-0.15, 0.15)),
                float(z),
            ]
            for z in axis
            for x in axis
        ]
        faces = []
        for y in range(grid - 1):
            for x in range(grid - 1):
                i = y * grid + x
                faces.extend([[i, i + grid, i + 1], [i + 1, i + grid, i + grid + 1]])
        objects.append(
            dict(type="mesh", vertices=vertices, faces=faces, material=material(textured))
        )
        for _ in range(9):
            radius = float(rng.uniform(0.15, 0.6))
            objects.append(
                dict(
                    type="sphere",
                    center=[float(rng.uniform(-3, 3)), radius + 0.35, float(rng.uniform(-3, 2))],
                    radius=radius,
                    material=material(textured),
                )
            )
    else:  # Open pavilion: silhouettes and roof shadows against a visible environment.
        for x in (-2.5, 2.5):
            for z in (-2.5, 1.5):
                box([x, height / 2, z], [0.15, height, 0.15], thin=True)
        box([0, height, -0.5], [6.2, 0.18, 5.2], textured=textured)
        for i in range(13):
            box([-2.7 + i * 0.45, height - 0.2, -0.5], [0.08, 0.15, 4.5], thin=True)

    # Every family also varies occlusion, curved silhouettes, and subpixel geometry.
    for i in range(9 if stratum == "clutter" else 3):
        x, z = float(rng.uniform(-2, 2)), float(rng.uniform(-2, 1.5))
        h = float(rng.uniform(0.25, 1.4))
        if i == 1:
            r = float(rng.uniform(0.2, 0.55))
            objects.append(
                dict(type="sphere", center=[x, r + 0.35, z], radius=r, material=material(textured))
            )
        else:
            width = float(rng.uniform(0.02, 0.07) if i == 0 else rng.uniform(0.2, 0.8))
            box(
                [x, 0.35 + h / 2, z],
                [width, h, float(rng.uniform(0.05, 0.6))],
                thin=i == 0,
                rotation=float(rng.uniform(-1.5, 1.5)),
                textured=textured,
            )

    lights = []
    for i in range(int(light_rng.integers(1, 4))):
        size = float(
            light_rng.uniform(0.12, 0.65) if stratum == "lighting" else light_rng.uniform(0.8, 2)
        )
        lights.append(
            dict(
                type="area",
                origin=[
                    float(light_rng.uniform(-1.8, 1)),
                    height - 0.35,
                    float(light_rng.uniform(-3, 0)),
                ],
                u=[size, 0, 0],
                v=[0, 0, size],
                emission=(light_rng.uniform(3, 16, 3) * (1.5 / size) ** 1.5).tolist(),
            )
        )
    phase = frame / max(views - 1, 1) - 0.5
    span = 0.16 if family == "corridor" else 0.8
    angle = float(camera_rng.uniform(-0.12, 0.12)) + span * phase
    camera = dict(
        lookfrom=[
            distance * math.sin(angle),
            float(camera_rng.uniform(1.8, 3.6)),
            distance * math.cos(angle),
        ],
        lookat=target,
        up=[0, 1, 0],
        vfov=float(camera_rng.uniform(35, 52)),
        aspect_ratio=16 / 9,
        defocus_angle=0,
        focus_dist=distance,
    )
    background = light_rng.uniform(0.005, 0.3, 3).tolist()
    return dict(
        schema_version=1,
        name=f"families-v3-{family}-{stratum}",
        camera=camera,
        objects=objects,
        lights=lights,
        environment=dict(sky=False, background=background),
    )
