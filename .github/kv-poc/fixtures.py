#!/usr/bin/env python3.12
"""The SC-20686 VACE control/mask fixtures for kv-poc W2, generated and verified.

Four directories, one per geometry and role, exactly the dev-Mac W2 prep set:
  512x512-17/{control,mask}/0000.png..0016.png   (square-17f-control, both VACE routes)
  768x512-33/{control,mask}/0000.png..0032.png   (landscape-33f-reference, both VACE routes)
A control frame is `vace_smoke.rs` `synth_control` (x/y gradient, blue 128, a white square sliding
left to right) and every mask frame is `center_mask` (a white centred half-size rectangle); PNG RGB8,
one IDAT, zlib level 9, filter 0.

fixtures-w2.tsv pins every file's bytes, sha256 and the sha256 of its raw RGB pixels. The PIXELS are
the identity checked hard; the PNG bytes also depend on the zlib build, so a byte-only difference is
reported as a warning (the campaign adapter seals the actual file hashes into its evidence either
way).

  fixtures.py generate --out DIR --pins fixtures-w2.tsv   (re)materialize DIR atomically, verified
  fixtures.py verify   --out DIR --pins fixtures-w2.tsv   verify an existing DIR, never writes
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import sys
import tempfile
import zlib
from pathlib import Path

GEOMETRIES = ((512, 512, 17), (768, 512, 33))


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def chunk(kind: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def png(rgb: bytes, width: int, height: int) -> bytes:
    stride = width * 3
    scan = b"".join(b"\0" + rgb[y * stride:(y + 1) * stride] for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scan, 9)) + chunk(b"IEND", b""))


def rgb_of_png(data: bytes) -> bytes:
    """Raw RGB8 pixels of a filter-0, single-IDAT RGB8 PNG (the only shape this module writes)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    offset, idat, width, height = 8, b"", 0, 0
    while offset < len(data):
        (length,) = struct.unpack(">I", data[offset:offset + 4])
        kind, payload = data[offset + 4:offset + 8], data[offset + 8:offset + 8 + length]
        if kind == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", payload[:10])
            if (depth, colour) != (8, 2):
                raise ValueError("not an RGB8 PNG")
        elif kind == b"IDAT":
            idat += payload
        offset += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 3 + 1
    if len(raw) != stride * height or any(raw[y * stride] != 0 for y in range(height)):
        raise ValueError("unexpected PNG scanline layout")
    return b"".join(raw[y * stride + 1:(y + 1) * stride] for y in range(height))


def frames(width: int, height: int, count: int):
    base = bytearray(width * height * 3)
    for y in range(height):
        g = 255 * y // height
        for x in range(width):
            i = (y * width + x) * 3
            base[i:i + 3] = bytes((255 * x // width, g, 128))
    mask = bytearray(width * height * 3)
    bw, bh = width // 2, height // 2
    x0, y0 = (width - bw) // 2, (height - bh) // 2
    for y in range(y0, y0 + bh):
        start = (y * width + x0) * 3
        mask[start:start + bw * 3] = b"\xff" * (bw * 3)
    square = max(1, min(width, height) // 4)
    sy = (height - square) // 2
    for t in range(count):
        frac = f32(f32(t) / f32(count - 1)) if count > 1 else 0.0
        sx = int(f32(f32(width - square) * frac))
        image = bytearray(base)
        for y in range(sy, sy + square):
            start = (y * width + sx) * 3
            image[start:start + square * 3] = b"\xff" * (square * 3)
        yield t, bytes(image), bytes(mask)


def load_pins(path: Path) -> dict[str, tuple[int, str, str]]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, size, sha256, rgb_sha256 = line.split("\t")
            pins[name] = (int(size), sha256, rgb_sha256)
    return pins


def verify(root: Path, pins: dict[str, tuple[int, str, str]]) -> tuple[list[str], list[str]]:
    """(errors, byte-only differences) of root against the pins."""
    errors, byte_diffs = [], []
    actual = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
    if actual != sorted(pins):
        errors.append(f"file set differs: {len(actual)} present, {len(pins)} pinned")
        return errors, byte_diffs
    for name, (size, sha256, rgb_sha256) in sorted(pins.items()):
        data = (root / name).read_bytes()
        try:
            rgb_ok = hashlib.sha256(rgb_of_png(data)).hexdigest() == rgb_sha256
        except (ValueError, zlib.error, struct.error) as error:
            errors.append(f"{name}: unreadable ({error})")
            continue
        if not rgb_ok:
            errors.append(f"{name}: pixels differ from the pin")
        elif len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
            byte_diffs.append(name)
    return errors, byte_diffs


def report(root: Path, errors: list[str], byte_diffs: list[str]) -> int:
    if errors:
        print(f"::error title=W2 fixtures do not verify::{root}: {'; '.join(errors[:10])}")
        return 1
    if byte_diffs:
        print(f"::warning title=W2 fixture bytes differ (pixels identical)::{len(byte_diffs)} PNG(s) under {root} "
              f"encode differently from the dev-Mac set (zlib {zlib.ZLIB_RUNTIME_VERSION}); pixels verified")
    print(f"{root}: fixtures verified (pixels{'' if byte_diffs else ' + bytes'})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "verify"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pins", required=True, type=Path)
    args = parser.parse_args()
    pins = load_pins(args.pins)
    root = args.out
    errors, byte_diffs = verify(root, pins)
    if args.action == "verify" or not errors:
        return report(root, errors, byte_diffs)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        for width, height, count in GEOMETRIES:
            folder = staging / f"{width}x{height}-{count}"
            (folder / "control").mkdir(parents=True)
            (folder / "mask").mkdir(parents=True)
            for t, control, mask in frames(width, height, count):
                (folder / "control" / f"{t:04d}.png").write_bytes(png(control, width, height))
                (folder / "mask" / f"{t:04d}.png").write_bytes(png(mask, width, height))
        errors, byte_diffs = verify(staging, pins)
        if errors:
            return report(staging, errors, byte_diffs)
        if root.exists():
            shutil.rmtree(root)
        os.replace(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return report(root, errors, byte_diffs)


if __name__ == "__main__":
    sys.exit(main())
