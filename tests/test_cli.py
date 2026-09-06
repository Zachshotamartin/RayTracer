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
import math

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

def decode_hdr(path):
    with path.open('rb') as stream:
        assert stream.readline().strip() == b'#?RADIANCE'
        header = []
        while True:
            line = stream.readline().strip()
            if not line:
                break
            header.append(line)
        assert b'FORMAT=32-bit_rle_rgbe' in header
        axis_y, height, axis_x, width = stream.readline().split()
        assert (axis_y, axis_x) == (b'-Y', b'+X')
        width, height = int(width), int(height)
        pixels = []
        for y in range(height):
            first = stream.read(4)
            if 8 <= width < 32768:
                assert first == bytes([2, 2, width >> 8, width & 255])
                channels = []
                for _ in range(4):
                    channel = bytearray()
                    while len(channel) < width:
                        code = stream.read(1)[0]
                        assert code != 0
                        if code > 128:
                            channel.extend(stream.read(1) * (code - 128))
                        else:
                            channel.extend(stream.read(code))
                    assert len(channel) == width
                    channels.append(channel)
                scanline = zip(*channels)
            else:
                raw = first + stream.read(4 * width - 4)
                scanline = [raw[i:i+4] for i in range(0,len(raw),4)]
            for red, green, blue, exponent in scanline:
                scale = math.ldexp(1.0, exponent - 136) if exponent else 0
                pixels.append((red * scale, green * scale, blue * scale))
        assert stream.read() == b''
        return width, height, pixels

def decode_pfm(path):
    with path.open('rb') as stream:
        assert stream.readline().strip() == b'PF'
        width, height = map(int, stream.readline().split())
        assert float(stream.readline()) == -1.0
        values = struct.unpack('<' + 'f' * (width * height * 3), stream.read())
        rows = [values[y*width*3:(y+1)*width*3] for y in range(height)]
        values = [c for row in reversed(rows) for c in row]
        return width, height, list(zip(values[::3], values[1::3], values[2::3]))

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
    subprocess.run([sys.argv[2], '--write-fixtures', str(root)], check=True, timeout=20)
    hdr, pfm = decode_hdr(root/'fixture.hdr'), decode_pfm(root/'fixture.pfm')
    assert hdr[:2] == pfm[:2] == (16, 2)
    assert pfm[2][0] == (0, 0, 0) and pfm[2][16] == (100, 25, 1)
    assert pfm[2][1] == (0.5, 2, 8)
    for actual, expected in zip(hdr[2], pfm[2]):
        for a, b in zip(actual, expected):
            assert abs(a-b) <= max(expected) / 128 + 1e-6
    assert (root/'compressed.png').stat().st_size < 4096
    decode_png(root/'compressed.png')
    for extension in ['hdr', 'pfm']:
        images = []
        for exposure in ['-4', '4']:
            path = root / f'linear-{exposure}.{extension}'
            subprocess.run([binary, '--headless', '--quiet', '--width', '16', '--samples', '2',
                            '--exposure', exposure, '--output', str(path)],
                           check=True, stdout=subprocess.DEVNULL, timeout=20)
            images.append(path.read_bytes())
        assert images[0] == images[1], 'Exposure altered linear HDR export'
    denoised = root/'denoised.png'
    metadata = json.loads(subprocess.check_output([binary, '--headless', '--quiet', '--denoise',
                         '--width', '32', '--samples', '4', '--output', str(denoised)], text=True))
    assert metadata['denoised'] and metadata['denoise_seconds'] > 0
    assert metadata == json.loads(denoised.with_suffix('.json').read_text())
    decode_png(denoised)
    mesh = Path(__file__).resolve().parents[1]/'assets/meshes/pedestal.obj'
    metadata = json.loads(subprocess.check_output([binary, '--headless', '--quiet', '--mesh', str(mesh),
                         '--width', '32', '--samples', '2', '--output', str(root/'mesh.png')], text=True))
    assert metadata['scene'] == 'mesh' and metadata['objects'] == 16
    if sys.platform == "darwin":
        subprocess.run([binary, "-NSTreatUnknownArgumentsAsOpen", "NO",
                        "-ApplePersistenceIgnoreState", "YES", "--headless", "--quiet",
                        "--width", "16", "--samples", "1", "--output", str(root / "cocoa.png")],
                       check=True, stdout=subprocess.DEVNULL, timeout=20)
    for args in [["--width", "0"], ["--samples", "0"], ["--samples", "abc"], ["--seed", "-1"],
                 ["--depth", "0"], ["--threads", "-2"], ["--scene", "missing"], ["--width"],
                 ["--unknown"], ["--output", "wrong.jpg"], ["--exposure", "nan"], ["--exposure", "4garbage"],
                 ['--caustics','-1'], ['--caustic-radius','0'], ['--caustic-radius','nan'],
                 ['--glass-shadows','wrong'], ['--caustics','100','--glass-shadows','transparent']]:
        result = subprocess.run([binary, "--headless", *args], capture_output=True, text=True, timeout=10)
        assert result.returncode != 0, args
        assert "RayTracer:" in result.stderr, args
    assert subprocess.run([binary, "--help"], capture_output=True, timeout=10).returncode == 0
    names = subprocess.check_output([binary, "--list-scenes"], text=True).splitlines()
    assert names == ["demo", "field", "studio", "caustics"]
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
print("PASS CLI, metadata, PNG compression, HDR/PFM decoding, denoising, OBJ, validation, and cancellation")
