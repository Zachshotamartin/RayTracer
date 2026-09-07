import copy
import numpy as np
import torch
from raytracer_ml.temporal import reproject, reprojection_map, warp_tensor


def plane():
    scene = dict(
        camera=dict(lookfrom=[0, 0, 2], lookat=[0, 0, 0], up=[0, 1, 0], vfov=90), objects=[]
    )
    axis = np.linspace(-0.75, 0.75, 4)
    x, y = np.meshgrid(axis, -axis)
    position = np.stack([x, y, np.ones_like(x)], axis=-1).astype(np.float32)
    features = np.zeros((27, 4, 4), np.float32)
    features[10:12], features[23], features[26], features[22] = 1, 1, 1, 1
    features[17:20] = 0.5
    return scene, position, features


def test_subpixel_interpolation_and_differentiable_warp():
    scene, previous, features = plane()
    current = previous.copy()
    current[..., 0] += 0.25
    rgb = np.repeat(np.arange(4)[None, :, None], 4, 0).repeat(3, 2).astype(np.float32)
    history, mask = reproject(current, features, scene, previous, features, scene, rgb)
    np.testing.assert_allclose(history[:, :3, 0], [[0.5, 1.5, 2.5]] * 4, atol=1e-6)
    assert (mask[:, :3] > 0.99).all()
    indices, weights, _ = reprojection_map(current, features, scene, previous, features, scene)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).requires_grad_()
    actual = warp_tensor(tensor, indices, weights)
    np.testing.assert_allclose(actual.detach().numpy().transpose(1, 2, 0), history, atol=1e-6)
    actual.sum().backward()
    assert tensor.grad is not None and tensor.grad.sum() > 0


def test_history_rejects_cuts_lighting_and_other_surfaces():
    scene, position, features = plane()
    rgb = np.ones((4, 4, 3), np.float32)
    for change in (
        dict(vfov=120),
        dict(lookat=[5, 0, 2]),
        dict(up=[1, 0, 0]),
        dict(lookfrom=[4, 0, 2]),
    ):
        changed = copy.deepcopy(scene)
        changed["camera"].update(change)
        assert not reproject(position, features, changed, position, features, scene, rgb)[1].any()
    changed = {**scene, "lights": ["changed"]}
    assert not reproject(position, features, changed, position, features, scene, rgb)[1].any()
    old = features.copy()
    old[17:20] = 0.8
    assert not reproject(position, features, scene, position, old, scene, rgb)[1].any()
    old = features.copy()
    old[22] = -1
    assert not reproject(position, features, scene, position, old, scene, rgb)[1].any()


def test_autoregressive_history_is_differentiable_and_never_uses_targets():
    from raytracer_ml.models import build_model
    from raytracer_ml.rollout import rollout

    scene, position, features = plane()
    features[:3], features[15], features[16] = 1, 4, 1
    frames = [
        dict(
            row=dict(scene=scene),
            features=features.copy(),
            position=position,
            target=np.full((3, 4, 4), 0.5, np.float32),
        )
        for _ in range(2)
    ]
    model = build_model(dict(kind="guided", width=8, feature_schema=2, temporal=True))
    stream = rollout(model, frames, torch.device("cpu"))
    first = next(stream)["prediction"]
    first.retain_grad()
    second = next(stream)["prediction"]
    second.mean().backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    original = [step["prediction"].detach() for step in rollout(model, frames, torch.device("cpu"))]
    changed = copy.deepcopy(frames)
    for frame in changed:
        frame["target"] *= 100
    altered = [step["prediction"].detach() for step in rollout(model, changed, torch.device("cpu"))]
    for a, b in zip(original, altered):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
