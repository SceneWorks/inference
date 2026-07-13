"""Dump a golden for the SVD image encoder (epic 3040 / sc-3373) from the real transformers
`CLIPVisionModelWithProjection` (the SVD `image_encoder`, ViT-H/14). Validates the Rust
`SvdImageEncoder::image_embeds` byte-close. Feeds a deterministic pre-normalized `pixel_values`
directly (isolates the encoder from the CLIP preprocessing, which is validated separately in S4).

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_svd_image_encoder_golden.py
Writes `mlx-gen-svd/tests/fixtures/svd_image_encoder_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import CLIPVisionModelWithProjection

from _paths import fixture, hf_hub_cache

SNAP = (
    hf_hub_cache()
    / "models--stabilityai--stable-video-diffusion-img2vid-xt"
    / "snapshots"
)
snap_dir = next(SNAP.iterdir())
enc_dir = snap_dir / "image_encoder"

model = CLIPVisionModelWithProjection.from_pretrained(enc_dir, torch_dtype=torch.float32)
model.eval()

rng = np.random.default_rng(3373)
# A deterministic CLIP-normalized pixel_values [1,3,224,224] (values ~ N(0,1), the post-normalize range).
pixel_values = rng.standard_normal((1, 3, 224, 224)).astype(np.float32)

with torch.no_grad():
    out = model(torch.from_numpy(pixel_values))
    image_embeds = out.image_embeds.cpu().numpy().astype(np.float32)  # [1, 1024]

tensors = {
    "pixel_values": pixel_values,  # NCHW
    "image_embeds": image_embeds,
}
out_path = fixture("mlx-gen-svd/tests/fixtures/svd_image_encoder_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  pixel_values:", pixel_values.shape, " image_embeds:", image_embeds.shape)
print("  embeds[:5]:", image_embeds[0, :5])
