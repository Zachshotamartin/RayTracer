"""Exercise asynchronous final-save and sequence shutdown through SDL's dummy driver."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

binary, root = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
env = {**os.environ, "SDL_VIDEODRIVER": "dummy"}
with tempfile.TemporaryDirectory() as directory:
    tmp = pathlib.Path(directory)
    scene = json.loads((root / "assets/scenes/diffuse-room.json").read_text())
    frames = tmp / "frames"
    frames.mkdir()
    for i in range(3):
        scene["camera"]["lookfrom"][0] += .1
        (frames / f"{i:03}.json").write_text(json.dumps(scene))
    command = [str(binary), "--sequence", str(frames), "--width", "32", "--samples", "4", "--threads", "2", "--model", str(root / "assets/models/diffuse-pilot-v1.onnx"), "--output", str(tmp / "frame.pfm"), "--raw-output", str(tmp / "raw.pfm"), "--quit-after-render"]
    subprocess.run(command, check=True, capture_output=True, env=env, timeout=30)
    for i in range(3):
        data = json.loads((tmp / f"frame-{i:06}.json").read_text())
        assert data["samples"] == 4 and data["reconstructed"]
        assert (tmp / f"raw-{i:06}.pfm").is_file()
    command[command.index("--model") + 1] = str(tmp / "missing.onnx")
    subprocess.run(command, check=True, capture_output=True, env=env, timeout=30)
    assert not json.loads((tmp / "frame-000002.json").read_text())["reconstructed"]
print("PASS neural viewer final save, sequence generations, fallback and shutdown")
