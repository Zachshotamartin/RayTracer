"""Differentiable short training rollouts and long validation from predicted history."""

import torch
from .augment import draw_transform, apply_transform, undo_prediction
from .losses import reconstruction_loss
from .temporal import reprojection_map, warp_tensor


def rollout(model, frames, device, augmentation=None, temporal_weight=0):
    transform = draw_transform(
        augmentation, square=frames[0]["features"].shape[1] == frames[0]["features"].shape[2]
    )
    previous = prediction = previous_target = None
    for frame in frames:
        x = torch.from_numpy(frame["features"].copy()).to(device)
        target = torch.from_numpy(frame["target"]).to(device)
        history = torch.zeros_like(x[:3])
        confidence = torch.zeros_like(x[:1])
        warped_target = None
        if previous is not None:
            indices, weights, mask = reprojection_map(
                frame["position"],
                frame["features"],
                frame["row"]["scene"],
                previous["position"],
                previous["features"],
                previous["row"]["scene"],
            )
            history = warp_tensor(prediction, indices, weights)
            confidence = torch.from_numpy(mask[None]).to(device)
            if temporal_weight:
                # Targets enter only the loss; inference history always uses predictions.
                warped_target = warp_tensor(previous_target, indices, weights)
        model_input = torch.cat([x, history, confidence])
        model_input, truth = apply_transform(model_input, target, transform, 27)
        estimate = model(model_input[None])[0]
        prediction = undo_prediction(estimate, transform)
        temporal_loss = estimate.sum() * 0
        if warped_target is not None:
            # Match temporal change, rather than forcing truly changing radiance to be constant.
            delta = (torch.log1p(prediction) - torch.log1p(history)) - (
                torch.log1p(target) - torch.log1p(warped_target)
            )
            temporal_loss = (delta.abs() * confidence).sum() / (3 * confidence.sum()).clamp_min(1)
        yield dict(
            prediction=prediction,
            target=target,
            augmented_prediction=estimate,
            augmented_target=truth,
            temporal_loss=temporal_loss,
            frame=frame,
            history_fraction=float(confidence.mean().detach().cpu()),
        )
        previous, previous_target = frame, target


def sequence_loss(model, sequences, device, cfg):
    values = []
    for sequence in sequences:
        for step in rollout(
            model, sequence, device, cfg.get("augmentation"), cfg.get("temporal_loss", 0.1)
        ):
            values.append(
                reconstruction_loss(
                    step["augmented_prediction"][None],
                    step["augmented_target"][None],
                    cfg.get("loss"),
                )
                + cfg.get("temporal_loss", 0.1) * step["temporal_loss"]
            )
    return torch.stack(values).mean()


def validation_predictions(model, loader, device, autoregressive=False):
    if autoregressive:
        for batch in loader:
            for frames in batch:
                for step in rollout(model, frames, device):
                    yield (
                        step["prediction"][None],
                        step["target"][None],
                        torch.from_numpy(step["frame"]["features"][None]).to(device),
                        step["frame"],
                    )
    else:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            yield model(x), y, x, None
