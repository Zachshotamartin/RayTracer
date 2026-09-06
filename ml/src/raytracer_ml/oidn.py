"""Optional upstream OIDN CLI baseline; includes all temporary IO in its timing."""

import subprocess
import tempfile
import time
from pathlib import Path
from .io import write_pfm, read_pfm


def denoise_oidn(features, executable, device="cpu"):
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="rtml-oidn-") as temporary:
        root = Path(temporary)
        for name, value in [
            ("color", features[:3]),
            ("albedo", features[3:6]),
            ("normal", features[6:9]),
        ]:
            write_pfm(root / f"{name}.pfm", value.transpose(1, 2, 0))
        command = [
            str(Path(executable).resolve()),
            "--hdr",
            str(root / "color.pfm"),
            "--alb",
            str(root / "albedo.pfm"),
            "--nrm",
            str(root / "normal.pfm"),
            "--device",
            device,
            "--threads",
            "2",
            "--output",
            str(root / "output.pfm"),
        ]
        # Guides are sample-aligned but noisy: do not set clean_aux.
        run = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if run.returncode:
            raise RuntimeError(f"OIDN failed: {run.stderr} {run.stdout}")
        image = read_pfm(root / "output.pfm")
        return image, time.perf_counter() - start, run.stdout
