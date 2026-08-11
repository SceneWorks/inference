"""MiniMax-H3 video-VAE **real-weight** decode reference (sc-18740).

Decodes a seeded latent through the official diffusers ``AutoencoderKLMiniMaxH3`` loaded from the
real 9.7 GB ``vae/``, and writes `(latent, decoded)` so the Rust real-weight smoke can assert
numeric parity against the official implementation on the *published* tensors.

Why this exists — the committed tiny fixture is not enough
----------------------------------------------------------

sc-17140's real-weight smoke asserted **self-generated** numbers (`std 0.4586`, `max |px| 2.5625`,
`checksum -8.251e4`). Those were recorded from the port itself, so they re-derive whatever the port
does: the sc-18740 half-swap changed every one of them and the test would simply have been written
against the changed values. Only a reference computed by an *independent* implementation from the
*same published bytes* can gate the real-weight path.

The output is a few hundred KB and is deliberately **not** committed: point the Rust test at it with
``MINIMAX_H3_VIDEO_VAE_REFERENCE``. Keep it outside the repository.

    MINIMAX_H3_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_video_vae_real.py \
        --out ~/minimax-h3-real-vae-reference.safetensors

Then:

    MINIMAX_H3_SNAPSHOT=... MINIMAX_H3_VIDEO_VAE_REFERENCE=~/minimax-h3-real-vae-reference.safetensors \
        cargo test -p mlx-gen-minimax-h3 --test real_weights -- --ignored --nocapture
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.numpy import save_file

from _paths import hf_hub_cache

SEED = 18740
# Latent frames: 5 tokens -> exactly one `clip_length` chunk -> 17 pixel frames.
LATENT_FRAMES = 5
# 4x4 latent -> 64x64 pixels at the real 16x spatial ratio. Small enough that the decoder's
# 36-layer transformer runs quickly and the reference file stays under a megabyte, and — checked
# below — far enough under the 256x256 tile that the VAE's default spatial tiling stays inert, so
# the comparison is against the untiled path the Rust port implements.
LATENT_HW = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .safetensors path (keep it OUTSIDE the repo)")
    args = ap.parse_args()

    import diffusers
    from diffusers import AutoencoderKLMiniMaxH3

    snapshot = Path(
        os.environ.get("MINIMAX_H3_SNAPSHOT")
        or next(iter(sorted((hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3" / "snapshots").glob("*"))))
    )

    vae = AutoencoderKLMiniMaxH3.from_pretrained(snapshot / "vae", torch_dtype=torch.float32)
    vae.eval()

    pixels = LATENT_HW * vae.spatial_compression_ratio
    assert pixels < vae.tile_sample_min_height and pixels < vae.tile_sample_min_width, (
        f"{pixels}px would cross the {vae.tile_sample_min_height}px tile boundary; the reference "
        "would be a tiled decode and the Rust port does not tile"
    )
    vae.disable_tiling()

    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn(
        1, vae.config.latent_channels, LATENT_FRAMES, LATENT_HW, LATENT_HW, generator=generator
    )
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)

    with torch.no_grad():
        decoded = vae.decode(latent * latents_std + latents_mean, return_dict=False)[0]

    print(f"latent {tuple(latent.shape)} -> decoded {tuple(decoded.shape)}")
    print(
        f"  std {decoded.std().item():.6f}  max|px| {decoded.abs().max().item():.6f}  "
        f"L2 {decoded.flatten().norm().item():.6f}"
    )

    tensors = {
        "in.latent": latent.numpy().copy(),
        "out.video": decoded.to(torch.float32).numpy().copy(),
    }
    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.AutoencoderKLMiniMaxH3",
        "reference_version": diffusers.__version__,
        "snapshot": snapshot.name,
        "seed": str(SEED),
        "story": "sc-18740",
    }
    out = Path(args.out).expanduser()
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} from {meta['reference']} {meta['reference_version']} @ {snapshot.name}")


if __name__ == "__main__":
    main()
