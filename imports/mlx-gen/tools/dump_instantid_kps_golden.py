#!/usr/bin/env python
"""Golden dump for the InstantID kps control-image renderer (sc-3111).

`draw_kps` (vendored `_vendor/instantid/pipeline_stable_diffusion_xl_instantid.py`) rasterizes the
5-point facial-landmark control image IdentityNet consumes: 4 limb "sticks" (rotated filled ellipses,
dimmed ×0.6) then 5 filled circles (r=10), all via OpenCV. The Rust port must bit-match OpenCV's
integer rasterization, so this dumps several (canvas, kps) cases → the exact RGB8 output.

The `draw_kps` body below is copied **verbatim** from the vendored InstantID source (only cv2/numpy/
PIL/math), so the golden is exactly what production renders. Pinned OpenCV: 4.13.0.

Cases cover: square + view-angle kps, non-square + detected-style kps, an extreme profile (long
sticks), and a tiny canvas (easy pixel debugging).

Run from a torch/cv venv (has cv2 + numpy + Pillow):
    ~/repos/mflux/.venv-0312/bin/python ~/Repos/mlx-gen/tools/dump_instantid_kps_golden.py
"""
import math
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
from safetensors.torch import save_file

OUT = Path(__file__).resolve().parent / "golden" / "instantid_kps_golden.safetensors"


# ---- verbatim from _vendor/instantid/pipeline_stable_diffusion_xl_instantid.py ----
def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = PIL.Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


VIEW_ANGLE_KPS = {
    "front": [(0.4460, 0.5227), (0.5755, 0.5166), (0.5106, 0.5947), (0.4653, 0.6660), (0.5630, 0.6613)],
    "left_profile": [(0.4373, 0.3527), (0.4925, 0.3445), (0.3927, 0.4662), (0.4853, 0.5599), (0.5240, 0.5517)],
}


def case(w, h, kps):
    kps = np.asarray(kps, dtype=np.float32)
    img = draw_kps(PIL.Image.new("RGB", (w, h), (0, 0, 0)), kps)
    arr = np.asarray(img, dtype=np.uint8)  # [h, w, 3]
    assert arr.shape == (h, w, 3), arr.shape
    return kps, arr


# verbatim from instantid_adapter.py
def _letterbox(image, width, height):
    ratio = min(width / image.width, height / image.height)
    new_w, new_h = max(1, round(image.width * ratio)), max(1, round(image.height * ratio))
    resized = image.resize((new_w, new_h), PIL.Image.LANCZOS)
    canvas = PIL.Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def _gradient_image(w, h):
    """Deterministic RGB source so the letterbox golden needs no external asset."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = (xx * 255 // max(1, w - 1)).astype(np.uint8)
    g = (yy * 255 // max(1, h - 1)).astype(np.uint8)
    b = ((xx + yy) * 255 // max(1, w + h - 2)).astype(np.uint8)
    return PIL.Image.fromarray(np.stack([r, g, b], axis=2))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cases = {
        # square + "front" view-angle kps (scaled to the side)
        "a": case(512, 512, np.array(VIEW_ANGLE_KPS["front"], dtype=np.float32) * 512.0),
        # non-square + detected-style kps (eyes, nose, mouth corners)
        "b": case(640, 896, [(250.4, 360.2), (392.7, 351.9), (323.1, 455.6), (262.8, 560.3), (385.0, 553.1)]),
        # extreme left profile (long, steeply-angled sticks) on a square
        "c": case(256, 256, np.array(VIEW_ANGLE_KPS["left_profile"], dtype=np.float32) * 256.0),
        # tiny canvas (pixel-level debugging)
        "d": case(64, 64, [(18.0, 22.0), (40.0, 24.0), (30.0, 34.0), (20.0, 48.0), (42.0, 47.0)]),
    }

    out = {}
    for name, (kps, arr) in cases.items():
        h, wid = arr.shape[:2]
        out[f"{name}_kps"] = torch.from_numpy(kps.copy())
        out[f"{name}_img"] = torch.from_numpy(arr.copy())
        out[f"{name}_wh"] = torch.tensor([wid, h], dtype=torch.int32)
        nz = int((arr != 0).any(axis=2).sum())
        print(f"case {name}: {wid}x{h}  nonzero_px={nz}")

    # Letterbox case: odd source dims (137x91) → non-square target (256x320) exercises LANCZOS
    # downscale + odd center offsets + asymmetric padding.
    src = _gradient_image(137, 91)
    lb = _letterbox(src, 256, 320)
    src_arr = np.asarray(src, dtype=np.uint8)
    lb_arr = np.asarray(lb, dtype=np.uint8)
    out["lb_src_img"] = torch.from_numpy(src_arr.copy())
    out["lb_src_wh"] = torch.tensor([src.width, src.height], dtype=torch.int32)
    out["lb_out_img"] = torch.from_numpy(lb_arr.copy())
    out["lb_out_wh"] = torch.tensor([lb.width, lb.height], dtype=torch.int32)
    print(f"letterbox: {src.width}x{src.height} -> {lb.width}x{lb.height}")

    save_file(out, str(OUT))
    print(f"wrote {OUT}  (cv2 {cv2.__version__})")


if __name__ == "__main__":
    main()
