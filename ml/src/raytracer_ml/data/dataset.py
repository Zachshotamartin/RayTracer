import numpy as np
import torch
from torch.utils.data import Dataset
from ..io import manifest, safe_path
from .arrays import load_example


class RenderDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        crop=0,
        temporal=False,
        feature_schema=1,
        edge_sampling=0,
        identity_probability=0,
        augmentation=None,
        fuse_probability=0,
        preservation_mode="synthetic_identity",
        near_clean_samples=96,
        border_sampling=0,
    ):
        from pathlib import Path

        self.root = Path(root)
        self.rows = [r for r in manifest(self.root / "manifest.jsonl") if r["split"] == split]
        self.crop = crop
        if type(crop) is not int or crop < 0 or not 0 <= border_sampling <= 1:
            raise ValueError("Invalid crop size/border sampling")
        if crop and any(min(r["stats"]["height"], r["stats"]["width"]) < crop for r in self.rows):
            raise ValueError("Crop exceeds a native input dimension; use a smaller fixed crop")
        self.border_sampling = border_sampling
        self.temporal = temporal
        self.split = split
        if (
            feature_schema not in (1, 2)
            or not 0 <= edge_sampling <= 1
            or not 0 <= identity_probability <= 1
        ):
            raise ValueError("Invalid dataset feature/crop policy")
        self.channels = 27 if feature_schema == 2 else 17
        self.edge_sampling = edge_sampling
        self.identity_probability = identity_probability if split == "train" else 0
        if preservation_mode not in ("synthetic_identity", "measured_near_clean"):
            raise ValueError("Invalid preservation mode")
        if not isinstance(near_clean_samples, int) or not 1 < near_clean_samples < 128:
            raise ValueError(
                "Near-clean samples must be an integer below the raw-identity threshold"
            )
        self.preservation_mode = preservation_mode
        if (
            self.identity_probability
            and preservation_mode == "synthetic_identity"
            and any(r["scale"] != 1 for r in self.rows)
        ):
            raise ValueError(
                "Upscaling requires measured preservation; synthetic identity is invalid"
            )
        self.near_clean_pairs = {}
        if self.identity_probability and preservation_mode == "measured_near_clean":
            if temporal:
                raise ValueError("Measured preservation pairs require spatial data")
            views = {}
            for row in self.rows:
                views.setdefault(self.view_key(row), []).append(row)
            for key, candidates in views.items():
                pairs = [
                    (a, b)
                    for i, a in enumerate(candidates)
                    for b in candidates[i + 1 :]
                    if a["samples"] + b["samples"] == near_clean_samples
                    and a["input_seed"] != b["input_seed"]
                    and a["input_seed"] != a["target_seed"]
                    and b["input_seed"] != b["target_seed"]
                    and a["reference_samples"] > near_clean_samples
                    and b["reference_samples"] > near_clean_samples
                ]
                if not pairs:
                    raise ValueError("Data lacks independent measurements for near-clean pairs")
                self.near_clean_pairs[key] = pairs
        if not 0 <= fuse_probability <= 1:
            raise ValueError("Invalid independent-noise fusion probability")
        self.augmentation = augmentation if split == "train" else None
        self.fuse_probability = fuse_probability if split == "train" and not temporal else 0
        self.noise_variants = {}
        for row in self.rows:
            self.noise_variants.setdefault((row["configuration"], row["samples"]), []).append(row)
        self.lookup = {
            (r["group"], r["frame"], r["id"].split("-n")[1].split("-s")[0], r["samples"]): r
            for r in self.rows
        }
        if not self.rows:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def view_key(row):
        return (
            row["group"],
            row["configuration"],
            row["scene_sha256"],
            row["target_seed"],
            row["reference_sha256"],
        )

    def __getitem__(self, index):
        r = self.rows[index]
        data = load_example(self.root, r)
        x = data["features"].copy()
        if x.shape[0] < self.channels:
            raise ValueError("Dataset lacks required feature channels")
        x = x[: self.channels]
        position = data["position"].copy()
        if self.fuse_probability and torch.rand(()) < self.fuse_probability:
            from ..augment import fuse_measurements

            choices = [
                other
                for other in self.noise_variants[(r["configuration"], r["samples"])]
                if other["input_seed"] != r["input_seed"]
                and other["scene_sha256"] == r["scene_sha256"]
                and other["target_seed"] == r["target_seed"]
            ]
            if choices:
                other = choices[int(torch.randint(len(choices), ()).item())]
                alternate = load_example(self.root, other)["features"][: self.channels]
                x = fuse_measurements(x, alternate)
        with np.load(safe_path(self.root, r["reference"]), allow_pickle=False) as data:
            y = data["target"].transpose(2, 0, 1).copy()
        if self.temporal:
            from ..temporal import reproject, append_history

            key = (r["group"], r["frame"] - 1, r["id"].split("-n")[1].split("-s")[0], r["samples"])
            previous = self.lookup.get(key)
            h = np.zeros((*position.shape[:2], 3), dtype=np.float32)
            mask = np.zeros((*position.shape[:2], 1), dtype=np.float32)
            if previous:
                old = load_example(self.root, previous)
                # History comes from a noisy measurement or ordinary filter, never a target.
                previous_rgb = (
                    old["atrous"]
                    if self.split != "train" or torch.rand(()) > 0.5
                    else old["features"][:3].transpose(1, 2, 0)
                )
                h, mask = reproject(
                    position,
                    x,
                    r["scene"],
                    old["position"],
                    old["features"],
                    previous["scene"],
                    previous_rgb,
                )
            x = append_history(x, h, mask)
        if self.crop:
            size = min(self.crop, x.shape[1], x.shape[2])
            top = int(torch.randint(x.shape[1] - size + 1, ()).item())
            left = int(torch.randint(x.shape[2] - size + 1, ()).item())
            s = r["scale"]
            if self.edge_sampling and torch.rand(()) < self.edge_sampling:
                target = np.log1p(y).mean(axis=0)
                edges = np.abs(np.diff(target, axis=0, prepend=target[:1])) + np.abs(
                    np.diff(target, axis=1, prepend=target[:, :1])
                )
                points = np.argwhere(edges > max(0.02, float(np.quantile(edges, 0.8))))
                if len(points):
                    cy, cx = points[int(torch.randint(len(points), ()).item())] // s
                    top = int(np.clip(cy - size // 2, 0, x.shape[1] - size))
                    left = int(np.clip(cx - size // 2, 0, x.shape[2] - size))
            if self.border_sampling and torch.rand(()) < self.border_sampling:
                side = int(torch.randint(4, ()).item())
                if side < 2:
                    top = 0 if side == 0 else x.shape[1] - size
                else:
                    left = 0 if side == 2 else x.shape[2] - size
            x = x[:, top : top + size, left : left + size]
            y = y[:, top * s : (top + size) * s, left * s : (left + size) * s]
        if (
            self.identity_probability
            and (r["scale"] == 1 or self.preservation_mode == "measured_near_clean")
            and torch.rand(()) < self.identity_probability
        ):
            if self.preservation_mode == "measured_near_clean":
                from ..augment import fuse_measurements

                pairs = self.near_clean_pairs[self.view_key(r)]
                a, b = pairs[int(torch.randint(len(pairs), ()).item())]
                x = fuse_measurements(
                    load_example(self.root, a)["features"][: self.channels],
                    load_example(self.root, b)["features"][: self.channels],
                )
                if self.crop:
                    x = x[:, top : top + size, left : left + size]
            else:
                x = x.copy()
                x[:3], x[12:15], x[15], x[16] = y, 0, 128, 1
        x, y = torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
        if self.augmentation:
            from ..augment import draw_transform, apply_transform

            transform = draw_transform(self.augmentation, square=x.shape[-1] == x.shape[-2])
            x, y = apply_transform(x, y, transform, self.channels)
        return x, y
