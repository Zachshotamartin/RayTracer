import numpy as np
import torch
from torch.utils.data import Dataset
from ..io import manifest, safe_path


class RenderDataset(Dataset):
    def __init__(self, root, split, crop=0, temporal=False):
        from pathlib import Path

        self.root = Path(root)
        self.rows = [r for r in manifest(self.root / "manifest.jsonl") if r["split"] == split]
        self.crop = crop
        self.temporal = temporal
        self.split = split
        self.lookup = {
            (r["group"], r["frame"], r["id"].split("-n")[1].split("-s")[0], r["samples"]): r
            for r in self.rows
        }
        if not self.rows:
            raise ValueError(f"Empty {split} split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        with np.load(safe_path(self.root, r["path"]), allow_pickle=False) as data:
            x = data["features"].copy()
            position = data["position"].copy()
        with np.load(safe_path(self.root, r["reference"]), allow_pickle=False) as data:
            y = data["target"].transpose(2, 0, 1).copy()
        if self.temporal:
            from ..temporal import reproject, append_history

            key = (r["group"], r["frame"] - 1, r["id"].split("-n")[1].split("-s")[0], r["samples"])
            previous = self.lookup.get(key)
            h = np.zeros((*position.shape[:2], 3), dtype=np.float32)
            mask = np.zeros((*position.shape[:2], 1), dtype=np.float32)
            if previous:
                with np.load(safe_path(self.root, previous["path"])) as old:
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
            x = x[:, top : top + size, left : left + size]
            y = y[:, top * s : (top + size) * s, left * s : (left + size) * s]
        return torch.from_numpy(x.copy()), torch.from_numpy(y.copy())
