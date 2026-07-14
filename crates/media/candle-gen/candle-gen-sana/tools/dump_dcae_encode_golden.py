#!/usr/bin/env python3
"""Dump a DC-AE **encoder** golden (image + reference latent) for the sc-11803 encode parity gate.

The mlx-gen port (sc-8486 / mlx-gen #612) only ported the DECODER, so there is no encoder reference to
parity against — sc-11803 sources one directly from diffusers. Loads diffusers `AutoencoderDC`
(dc-ae-f32c32-sana-1.0) in f32, runs a fixed-seed image through the RAW encoder module (no
scaling/tiling — matching the Rust `DcAeEncoder::encode`), and saves both the input image and the
reference latent to a safetensors the Rust test reads back.

A 256x256 input (→ 8x8 latent, 32x compression) exercises every DCDownBlock2d rung + both shortcuts
while keeping the committed fixture small (~0.8 MB).

Usage: python dump_dcae_encode_golden.py MODEL_DIR OUT.safetensors
"""
import sys
import torch
from diffusers import AutoencoderDC
from safetensors.torch import save_file

model_dir, out = sys.argv[1], sys.argv[2]
model = AutoencoderDC.from_pretrained(model_dir, torch_dtype=torch.float32).eval()

torch.manual_seed(0)
image = torch.randn(1, 3, 256, 256, dtype=torch.float32)
with torch.no_grad():
    latent = model.encoder(image)  # raw encoder forward → [1, 32, 8, 8]

print("image", tuple(image.shape), "-> latent", tuple(latent.shape),
      "min", float(latent.min()), "max", float(latent.max()))
save_file({"image": image.contiguous(), "latent": latent.contiguous()}, out)
print("wrote", out)
