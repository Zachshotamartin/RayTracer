"""Scene-isolated sequence windows; history is produced by the current model."""

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from ..io import manifest, safe_path
from .arrays import load_example


class SequenceDataset(Dataset):
    def __init__(self, root, split, length=0, budgets=None, first_noise_only=False):
        self.root = Path(root)
        self.rows = [
            r
            for r in manifest(self.root / "manifest.jsonl")
            if r["split"] == split
            and (not budgets or r["samples"] in budgets)
            and (not first_noise_only or "-n0-s" in r["id"])
        ]
        groups = {}
        for row in self.rows:
            if row.get("feature_schema") != 2 or row["scale"] != 1:
                raise ValueError("Autoregressive training needs schema-2 same-resolution data")
            noise = row["id"].split("-n")[1].split("-s")[0]
            groups.setdefault((row["group"], row["samples"], noise), []).append(row)
        self.sequences = []
        for rows in groups.values():
            rows.sort(key=lambda r: r["frame"])
            if len(rows) < 2 or any(b["frame"] != a["frame"] + 1 for a, b in zip(rows, rows[1:])):
                raise ValueError("Sequences need at least two contiguous frames")
            if length:
                if len(rows) < length:
                    raise ValueError("Sequence shorter than requested unroll")
                starts = list(range(0, len(rows) - length + 1, max(1, length // 2)))
                if starts[-1] != len(rows) - length:
                    starts.append(len(rows) - length)
                self.sequences.extend(rows[start : start + length] for start in starts)
            else:
                self.sequences.append(rows)
        if not self.sequences:
            raise ValueError("Empty sequence split")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        frames = []
        for row in self.sequences[index]:
            data = load_example(self.root, row)
            with np.load(safe_path(self.root, row["reference"]), allow_pickle=False) as target:
                y = target["target"].transpose(2, 0, 1).copy()
            frames.append(
                dict(
                    row=row,
                    features=data["features"],
                    position=data["position"],
                    target=y,
                    atrous=data["atrous"],
                )
            )
        return frames


def collate_sequences(batch):
    return batch
