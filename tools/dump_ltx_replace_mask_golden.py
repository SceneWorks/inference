"""Dump a byte-exact golden for the replace_person mask op (epic 3040 / sc-3053): the worker's
`_apply_replacement_mask` (`apps/worker/scene_worker/video_adapters.py`) blends the person region
toward neutral gray 118 by `strength` via PIL `convert("L")` → `point(int(v*s))` → `composite`. The
Rust `apply_replacement_mask` port must match this Pillow output byte-for-byte.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_ltx_replace_mask_golden.py
Writes `mlx-gen-ltx/tests/fixtures/ltx_replace_mask_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.numpy import save_file

from _paths import fixture

rng = np.random.default_rng(3053)
H, W = 5, 7
frame_np = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
# A graded grayscale mask (R=G=B) so the L-convert + gate exercise non-binary values too.
mask_vals = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
mask_np = np.stack([mask_vals, mask_vals, mask_vals], axis=-1).astype(np.uint8)


def apply_replacement_mask(frame: Image.Image, mask: Image.Image, strength: float) -> Image.Image:
    strength = max(0.0, min(1.0, strength))
    neutral = Image.new("RGB", frame.size, (118, 118, 118))
    gate = mask.convert("L").resize(frame.size).point(lambda value: int(value * strength))
    return Image.composite(neutral, frame.convert("RGB"), gate)


tensors: dict[str, np.ndarray] = {
    "frame": frame_np,
    "mask": mask_np,
}
meta: dict[str, str] = {"h": str(H), "w": str(W)}
for tag, strength in [("s100", 1.0), ("s060", 0.6), ("s000", 0.0)]:
    out = apply_replacement_mask(Image.fromarray(frame_np), Image.fromarray(mask_np), strength)
    tensors[f"{tag}_out"] = np.asarray(out.convert("RGB"), dtype=np.uint8)
    meta[f"{tag}_strength"] = str(strength)

out_path = fixture("mlx-gen-ltx/tests/fixtures/ltx_replace_mask_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path, metadata=meta)
print(f"wrote {out_path}")
for k, v in tensors.items():
    print(f"  {k}: {v.shape} {v.dtype}")
