#!/usr/bin/env python3
"""Emit a deterministic, structured binary-P6 PPM for the candle-gen-sdxl real-weight harnesses.

`edit_validate::real_weight_edit` (`EDIT_SRC`) and `ip_validate::real_weight_ip_adapter` (`IP_REF`)
each read a source/reference image as a `P6` PPM (see `candle-gen/src/testkit.rs::read_ppm`). The
inference repo never self-fetches, and no natural photo is committed, so the scheduled CUDA job stages
these inputs by generating them. The gates the images feed are *relative* (img2img strength ablation;
inpaint kept-vs-repaint; IP-Adapter CLIP-cosine with vs without conditioning), so any decodable,
structured image works — but a richly structured one (distinct saturated regions + a foreground shape
+ high-frequency texture) gives CLIP a strong signal, keeping the IP-Adapter cosine-delta gate
comfortably above its +0.02 threshold.

The output is byte-for-byte deterministic (pure integer math, no RNG), so a run is reproducible and
the same on every runner. Usage:

    python scripts/ci/make_test_image.py <out.ppm> [width] [height]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _clamp8(value: int) -> int:
    return 0 if value < 0 else 255 if value > 255 else value


def render(width: int, height: int) -> bytearray:
    """A synthetic 'landscape' composition: sky→ground vertical gradient, a sun disc, a few saturated
    blocks, and a checkerboard texture patch. All deterministic."""
    px = bytearray(width * height * 3)
    cx, cy = int(width * 0.30), int(height * 0.28)  # sun centre
    sun_r2 = (min(width, height) // 7) ** 2
    block_h0, block_h1 = int(height * 0.62), int(height * 0.80)  # foreground blocks band
    for y in range(height):
        t = y / max(height - 1, 1)  # 0 (top) → 1 (bottom)
        # Sky (cool blue) at the top blending into warm ground at the bottom.
        bg_r = _clamp8(int(70 + 150 * t))
        bg_g = _clamp8(int(120 + 60 * t))
        bg_b = _clamp8(int(210 - 150 * t))
        row = y * width * 3
        for x in range(width):
            r, g, b = bg_r, bg_g, bg_b
            # Sun disc.
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= sun_r2:
                r, g, b = 255, 224, 96
            # Three saturated foreground blocks with a high-frequency checkerboard overlay.
            elif block_h0 <= y < block_h1:
                third = x * 3 // width
                r, g, b = ((210, 40, 40), (40, 170, 90), (60, 90, 220))[third]
                if ((x >> 4) + (y >> 4)) & 1:
                    r, g, b = r * 3 // 4, g * 3 // 4, b * 3 // 4
            idx = row + x * 3
            px[idx] = r
            px[idx + 1] = g
            px[idx + 2] = b
    return px


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, help="output .ppm path")
    parser.add_argument("width", type=int, nargs="?", default=768)
    parser.add_argument("height", type=int, nargs="?", default=768)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("width and height must be positive")

    pixels = render(args.width, args.height)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = f"P6\n{args.width} {args.height}\n255\n".encode("ascii")
    args.out.write_bytes(header + pixels)
    print(f"wrote {args.out} ({args.width}x{args.height} P6 PPM, {len(pixels)} pixel bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
