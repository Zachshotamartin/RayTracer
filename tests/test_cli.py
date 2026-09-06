"""End-to-end CLI, metadata, PNG CRC/DEFLATE, determinism, and cancellation checks."""
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import zlib

binary = str(Path(sys.argv[1]).resolve())

def decode_png(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset, compressed, width, height = 8, bytearray(), 0, 0
    ended = False
    while offset < len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
        if kind == b"IDAT":
            compressed.extend(payload)
        if kind == b"IEND":
            ended = True
        offset += length + 12
    assert ended and offset == len(data)
    raw = zlib.decompress(compressed)
    assert len(raw) == height * (1 + width * 3)
    assert all(raw[row * (1 + width * 3)] == 0 for row in range(height))
    return width, height, raw

with tempfile.TemporaryDirectory(prefix="raytracer tests ") as directory:
    root = Path(directory)
    baseline = None
    for suffix, extras in [("single", ["--threads", "1"]), ("multi", ["--threads", "4"]), ("linear", ["--threads", "4", "--no-bvh"])]:
        path = root / (suffix + ".png")
        result = subprocess.run([binary, "--headless", "--quiet", "--scene", "studio", "--width", "64",
                                 "--samples", "4", "--seed", "123", "--output", str(path), *extras],
                                capture_output=True, text=True, check=True, timeout=20)
        metadata = json.loads(result.stdout)
        assert metadata == json.loads(path.with_suffix(".json").read_text())
        assert (metadata["seed"], metadata["samples"], metadata["width"], metadata["height"]) == (123, 4, 64, 36)
        assert metadata["bvh"] == (suffix != "linear")
        decoded = decode_png(path)
        assert decoded[:2] == (64, 36)
        assert baseline is None or decoded == baseline
        baseline = decoded
    large_path = root / "multiple deflate blocks.png"
    subprocess.run([binary, "--headless", "--quiet", "--width", "256", "--samples", "1",
                    "--output", str(large_path)], check=True, stdout=subprocess.DEVNULL, timeout=20)
    assert decode_png(large_path)[:2] == (256, 144)
    if sys.platform == "darwin":
        subprocess.run([binary, "-NSTreatUnknownArgumentsAsOpen", "NO",
                        "-ApplePersistenceIgnoreState", "YES", "--headless", "--quiet",
                        "--width", "16", "--samples", "1", "--output", str(root / "cocoa.png")],
                       check=True, stdout=subprocess.DEVNULL, timeout=20)
    for args in [["--width", "0"], ["--samples", "0"], ["--samples", "abc"], ["--seed", "-1"],
                 ["--depth", "0"], ["--threads", "-2"], ["--scene", "missing"], ["--width"],
                 ["--unknown"], ["--output", "wrong.jpg"], ["--exposure", "nan"], ["--exposure", "4garbage"]]:
        result = subprocess.run([binary, "--headless", *args], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0, args
        assert "RayTracer:" in result.stderr, args
    assert subprocess.run([binary, "--help"], capture_output=True, timeout=10).returncode == 0
    names = subprocess.check_output([binary, "--list-scenes"], text=True).splitlines()
    assert names == ["demo", "field", "studio"]
    blocker = root / "not a directory"
    blocker.write_text("existing file")
    result = subprocess.run([binary, "--headless", "--quiet", "--width", "16", "--samples", "1",
                             "--output", str(blocker / "output.png")], capture_output=True, timeout=10)
    assert result.returncode != 0
    assert blocker.read_text() == "existing file"
    if os.name == "posix":
        process = subprocess.Popen([binary, "--headless", "--quiet", "--scene", "field", "--width", "128",
                                    "--samples", "100000", "--threads", "4", "--output", str(root / "partial.png")],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.3)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 130, (process.returncode, stderr)
        assert json.loads(stdout)["samples"] < 100000
print("PASS CLI, metadata, PNG decoding, reproducibility, validation, and cancellation")
