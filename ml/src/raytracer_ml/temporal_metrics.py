"""Reference-corrected temporal changes, with identical geometry masks for all methods."""

import numpy as np

from .temporal import reproject, reprojection_map


class TemporalComparison:
    def __init__(self):
        self.group = None
        self.histories = {}

    def measure(self, row, features, position, target, methods):
        if row["group"] != self.group:
            self.histories.clear()
            self.group = row["group"]
        key = (row["samples"], row["id"].split("-n")[1].split("-s")[0])
        previous = self.histories.get(key)
        metrics = {
            name: dict(
                temporal_linear_mae=None,
                temporal_log_mae=None,
                temporal_valid_pixels=0,
                temporal_confidence_sum=0.0,
            )
            for name in methods
        }
        if previous is not None and previous["row"]["frame"] + 1 == row["frame"]:
            if features.shape[0] >= 27:
                indices, weights, confidence = reprojection_map(
                    position,
                    features,
                    row["scene"],
                    previous["position"],
                    previous["features"],
                    previous["row"]["scene"],
                )

                def warp(image):
                    return np.sum(image.reshape(-1, 3)[indices] * weights[..., None], axis=0)
            else:

                def warp(image):
                    return reproject(
                        position,
                        features,
                        row["scene"],
                        previous["position"],
                        previous["features"],
                        previous["row"]["scene"],
                        image,
                    )[0]

                _, mask = reproject(
                    position,
                    features,
                    row["scene"],
                    previous["position"],
                    previous["features"],
                    previous["row"]["scene"],
                    previous["target"],
                )
                confidence = mask[..., 0]
            weight = confidence.astype(np.float64)[..., None]
            mass = float(weight.sum())
            if mass > 0:
                old_target = warp(previous["target"]).astype(np.float64)
                truth = target.astype(np.float64)
                delta = truth - old_target
                log_delta = np.log1p(truth) - np.log1p(old_target)
                for name, image in methods.items():
                    if name not in previous["methods"]:
                        continue
                    old = warp(previous["methods"][name]).astype(np.float64)
                    now = image.astype(np.float64)
                    metrics[name] = dict(
                        temporal_linear_mae=float(
                            (np.abs((now - old) - delta) * weight).sum() / (3 * mass)
                        ),
                        temporal_log_mae=float(
                            (np.abs((np.log1p(now) - np.log1p(old)) - log_delta) * weight).sum()
                            / (3 * mass)
                        ),
                        temporal_valid_pixels=int(np.count_nonzero(confidence)),
                        temporal_confidence_sum=mass,
                    )
        self.histories[key] = dict(
            row=row, features=features, position=position, target=target, methods=methods
        )
        return metrics
