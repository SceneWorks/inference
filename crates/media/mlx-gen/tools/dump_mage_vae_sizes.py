#!/usr/bin/env python3
"""Dump Mage-VAE encode/decode goldens at several geometries (sc-14039).

`dump_mage_flow_golden.py --stage vae` covers one geometry at a time in **bf16** — the dtype the
shipping pipeline uses (`load_from_repo` hard-codes it). That is the right production-fidelity
oracle, and the committed 256² bundle is exactly that. It is not, however, a practical way to reach
2048: torch's CPU bf16 kernels here are effectively single-threaded, so a 1024² `--stage vae` run
takes tens of minutes and 2048 several hours.

This script complements it rather than replacing it:

* it runs the codec in **f32**, which is multi-threaded on CPU and an order of magnitude faster;
* f32 is also the *tighter* oracle — it removes the reference's own bf16 rounding from the
  comparison, so what is left is the port's error rather than the reference's. The bf16 bundle then
  measures the other thing: how far apart the two dtypes are at all.

Output: `tools/golden/mage_flow_vae_f32_{size}.safetensors`, with the same tensor names as the
bf16 bundle (`enc_mean`, `enc_logvar`, `enc_latent`, `synth_latent`, `dec_from_latent`,
`dec_from_synth`, `pixels`, `image_u8`) so `tests/vae_decode_real_weights.rs` reads both through
one code path. Gitignored, like every other golden.

Run (from `crates/media/mlx-gen`):

    MAGE_VAE_SIZES=256,1024,2048 PYTHONPATH=_vendor \\
      python3 tools/dump_mage_vae_sizes.py

`MAGE_DEVICE` defaults to **cpu** here rather than auto-selecting: MPS dumps are silently corrupt
(sc-14250), and for a weights-light stage there is no reason to risk it.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "golden")
VENDOR = os.path.normpath(os.path.join(HERE, "..", "_vendor"))

DEVICE = os.environ.get("MAGE_DEVICE", "cpu")
SIZES = [int(s) for s in os.environ.get("MAGE_VAE_SIZES", "256,1024,2048").split(",")]
SEED = int(os.environ.get("MAGE_SEED", "42"))
REF_IMAGE = os.environ.get(
    "MAGE_EDIT_REF", os.path.join(VENDOR, "mage_flow", "assets", "dog.jpg")
)


def _snapshot() -> str:
    """The newest cached `microsoft/Mage-Flow*` snapshot that actually carries `vae/`."""
    if d := os.environ.get("MAGE_SNAPSHOT"):
        return d
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    found = []
    for repo in sorted(os.listdir(hub)):
        if not repo.startswith("models--microsoft--Mage-Flow"):
            continue
        snaps = os.path.join(hub, repo, "snapshots")
        if not os.path.isdir(snaps):
            continue
        for s in sorted(os.listdir(snaps)):
            p = os.path.join(snaps, s)
            if os.path.exists(os.path.join(p, "vae", "diffusion_pytorch_model.safetensors")):
                found.append(p)
    if not found:
        raise SystemExit(f"no Mage-Flow snapshot with vae/ under {hub}")
    return found[-1]


def main() -> int:
    from mage_flow.models.modules.mage_vae import MageVAE
    from mage_flow.pipeline import _preprocess_ref_image

    torch.set_num_threads(os.cpu_count() or 8)
    repo = _snapshot()
    ckpt = os.path.join(repo, "vae", "diffusion_pytorch_model.safetensors")
    print(f"snapshot: {repo}")
    print(f"device={DEVICE} dtype=float32 threads={torch.get_num_threads()}")

    vae = MageVAE(ckpt_path=ckpt, sample_posterior=False).eval().to(DEVICE).to(torch.float32)
    os.makedirs(GOLDEN, exist_ok=True)

    pil = Image.open(REF_IMAGE).convert("RGB")
    for size in SIZES:
        if size % 16:
            raise SystemExit(f"size {size} is not a multiple of 16")
        t0 = time.time()
        pixels = _preprocess_ref_image(pil, size, size, DEVICE)
        u8 = ((pixels.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        x = pixels.unsqueeze(0).to(DEVICE, torch.float32)

        with torch.no_grad():
            mean, logvar = vae._moments(x)
            latent = vae.encode(x)  # == mean (sample_posterior=False)
            decoded = vae.decode(latent)

            # Same construction as the bf16 harness: a seeded synthetic latent isolates the
            # decoder, so a broken encoder cannot mask a broken decoder.
            gen = torch.Generator(device="cpu").manual_seed(SEED + 1)
            synth = torch.randn(
                (1, MageVAE.latent_channels, size // 16, size // 16),
                generator=gen,
                dtype=torch.float32,
            ).to(DEVICE, torch.float32)
            decoded_synth = vae.decode(synth)

        f32 = lambda t: t.detach().to(torch.float32).cpu().numpy()  # noqa: E731
        tensors = {
            "image_u8": u8.permute(1, 2, 0).contiguous().numpy(),
            "pixels": f32(x),
            "enc_mean": f32(mean),
            "enc_logvar": f32(logvar),
            "enc_latent": f32(latent),
            "dec_from_latent": f32(decoded),
            "synth_latent": f32(synth),
            "dec_from_synth": f32(decoded_synth),
            "geometry": np.array([size, size], dtype=np.int32),
            "seed": np.array([SEED], dtype=np.int64),
        }
        out = os.path.join(GOLDEN, f"mage_flow_vae_f32_{size}.safetensors")
        save_file(
            tensors,
            out,
            metadata={
                "reference": "microsoft/Mage @ _vendor/mage_flow (see VENDORED.md)",
                "device": DEVICE,
                "dtype": "float32",
                "revision": os.path.basename(repo),
                "ref_image": os.path.basename(REF_IMAGE),
            },
        )
        dt = time.time() - t0
        print(
            f"{size}²: wrote {os.path.basename(out)} in {dt:.1f}s  "
            f"decode range [{decoded.min():.4f}, {decoded.max():.4f}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
