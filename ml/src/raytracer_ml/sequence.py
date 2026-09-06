"""Export complete, ordered camera-frame scenes for the native sequence player."""

from pathlib import Path
from .io import manifest, identity, write_json


def export_sequence(root, group, output):
    rows = [r for r in manifest(Path(root) / "manifest.jsonl") if r["group"] == group]
    if not rows:
        raise ValueError("No matching group in dataset")
    scenes = {}
    for row in rows:
        if identity(row["scene"]) != row["scene_sha256"]:
            raise ValueError("Scene hash mismatch")
        scenes[row["frame"]] = row["scene"]
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty sequence directory")
    output.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(sorted(scenes)):
        write_json(output / f"{index:06}.json", scenes[frame])
    return {"frames": len(scenes), "group": group, "output": str(output)}
