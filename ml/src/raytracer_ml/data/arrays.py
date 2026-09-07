"""Lossless shared center guides; each noisy example retains its own sampled moments."""

from functools import lru_cache
import numpy as np
from ..io import safe_path


@lru_cache(maxsize=16)
def shared_arrays(path, checksum):
    # The checksum is part of the key, so a regenerated artifact cannot reuse stale data.
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_example(root, row):
    with np.load(safe_path(root, row["path"]), allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    if row.get("feature_layout") == "shared-center-v2":
        shared = shared_arrays(
            str(safe_path(root, row["shared_guides"])), row["shared_guides_sha256"]
        )
        core, center = result["features"], shared["center"]
        if core.shape[0] != 19 or center.shape[0] != 8 or core.shape[1:] != center.shape[1:]:
            raise ValueError("Invalid compact feature shape")
        result["features"] = np.concatenate([core[:17], center[:7], core[17:19], center[7:8]])
        result["position"] = shared["position"]
    elif row.get("feature_layout", "full") != "full":
        raise ValueError("Unknown feature layout")
    return result
