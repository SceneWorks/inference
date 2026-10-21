"""Write one frame of a `veda_ab` render dir's raw `frames.rgb` as a PNG (sc-26214, `mlx-veda`).

Usage: veda_extract_frame_png.py <VEDA_AB_OUT dir> <frame index, -1 = last> <out.png>

`frames.rgb` is what `tests/veda_ab.rs` writes: rgb24, frame-major, each frame `width x height`
row-major, the geometry being `result.json`'s decoded `output.width` / `output.height`. Stdlib only
(zlib + struct), so the A/B's keyframes do not need ffmpeg on the runner.
"""

import json
import struct
import sys
import zlib
from pathlib import Path


def png(width: int, height: int, rgb: bytes) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride : (y + 1) * stride] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    render, index, out = Path(argv[1]), int(argv[2]), Path(argv[3])
    output = json.loads((render / "result.json").read_text(encoding="utf-8"))["output"]
    width, height, frames = int(output["width"]), int(output["height"]), int(output["frames"])
    size = width * height * 3
    data = (render / "frames.rgb").read_bytes()
    if len(data) != size * frames:
        raise SystemExit(f"frames.rgb is {len(data)} bytes, expected {frames} x {width}x{height}x3")
    if index < 0:
        index += frames
    if not 0 <= index < frames:
        raise SystemExit(f"frame {argv[2]} is outside 0..{frames - 1}")
    out.write_bytes(png(width, height, data[index * size : (index + 1) * size]))
    print(f"{out}: frame {index} of {frames}, {width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
