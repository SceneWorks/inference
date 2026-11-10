"""MiniMax-H3-**Image** VAE real-weight decode reference (SW-7089).

Decodes seeded latents through the official diffusers ``AutoencoderKLMiniMaxH3Image`` loaded from
the real ``vae_diffusers/`` of ``MiniMaxAI/MiniMax-H3-Image``, and writes the inputs and outputs so
the Rust real-weight gates on BOTH backends (``mlx-gen-minimax-h3`` and ``candle-gen-minimax-h3``,
``real_weights::image_vae_decode_matches_the_official_diffusers_vae``) can assert numeric parity
against the official implementation on the published tensors.

Three decodes, each of the pipeline's own decode call ``vae.decode(latents * std + mean)``:

* ``small``  — a 4x4 latent (64x64 px). Below one 256 px tile, so tiling is inert; this pins the
  2-D decoder itself (2-axis rotary, registers, spatial-only unpatchify).
* ``tiled``  — a 24x40 latent (384x640 px) with tiling ON (the shipped default, 256 / 64). A 2x3
  tile grid whose row seam is widened to 128 px by the slack distribution, so it pins the split,
  the per-tile decode and the pixel-space blend.
* ``untiled`` — the same 24x40 latent with tiling OFF, so the Rust untiled branch is pinned too and
  the gate can show the two branches really differ at this canvas.

Hash-pinned: the output carries ``producer_sha256``, the SHA-256 of THIS file's bytes, and the Rust
gates compare it to the SHA-256 of the committed copy of this file. A reference made by any other
version of this producer is refused, so editing this file requires re-producing the reference.

The output is a few MB and is deliberately **not** committed (real-weight bytes). Produce it on a
box holding the snapshot, from a torch + diffusers (>= 0.42.0.dev0, with
``AutoencoderKLMiniMaxH3Image``) environment:

    MINIMAX_H3_IMAGE_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_image_vae_real.py \
        --out ~/minimax-h3-image-real-vae-reference.safetensors

and point ``MINIMAX_H3_IMAGE_VAE_REFERENCE`` at it.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
from safetensors.numpy import save_file

from _paths import hf_hub_cache

SEED = 7089
SMALL_HW = (4, 4)
TILED_HW = (24, 40)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .safetensors path (keep it OUTSIDE the repo)")
    args = ap.parse_args()

    import diffusers
    from diffusers import AutoencoderKLMiniMaxH3Image

    snapshot = Path(
        os.environ.get("MINIMAX_H3_IMAGE_SNAPSHOT")
        or next(
            iter(sorted((hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3-Image" / "snapshots").glob("*")))
        )
    )
    producer_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(snapshot / "vae_diffusers", torch_dtype=torch.float32)
    vae.eval()
    ratio = vae.spatial_compression_ratio
    assert vae.use_tiling, "the shipped VAE tiles by default; the reference must start from that"
    tile = (vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    assert tile == (256, 64), f"unexpected shipped tile geometry {tile}"
    assert SMALL_HW[0] * ratio < 256 and SMALL_HW[1] * ratio < 256, "small case must be sub-tile"

    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    generator = torch.Generator().manual_seed(SEED)
    channels = vae.config.latent_channels
    small = torch.randn(1, channels, *SMALL_HW, generator=generator)
    tiled = torch.randn(1, channels, *TILED_HW, generator=generator)

    with torch.no_grad():
        out_small = vae.decode(small * std + mean, return_dict=False)[0]
        out_tiled = vae.decode(tiled * std + mean, return_dict=False)[0]
        vae.disable_tiling()
        out_untiled = vae.decode(tiled * std + mean, return_dict=False)[0]

    for name, t in (("small", out_small), ("tiled", out_tiled), ("untiled", out_untiled)):
        print(
            f"{name}: {tuple(t.shape)} std {t.std().item():.6f} max|px| {t.abs().max().item():.6f} "
            f"L2 {t.flatten().norm().item():.6f}"
        )
    seam = (out_tiled - out_untiled).abs().max().item() / out_tiled.abs().max().item()
    print(f"tiled vs untiled peak-relative difference at {TILED_HW}: {seam:.3e}")

    tensors = {
        "in.small": small.numpy().copy(),
        "in.tiled": tiled.numpy().copy(),
        "out.small": out_small.to(torch.float32).numpy().copy(),
        "out.tiled": out_tiled.to(torch.float32).numpy().copy(),
        "out.untiled": out_untiled.to(torch.float32).numpy().copy(),
    }
    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.AutoencoderKLMiniMaxH3Image",
        "reference_version": diffusers.__version__,
        "snapshot": snapshot.name,
        "seed": str(SEED),
        "tile": f"{tile[0]}/{tile[1]}",
        "tiled_vs_untiled_rel": f"{seam:.6e}",
        "producer": "crates/media/mlx-gen/tools/dump_minimax_h3_image_vae_real.py",
        "producer_sha256": producer_sha256,
        "story": "SW-7089",
    }
    out = Path(args.out).expanduser()
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} from {meta['reference']} {meta['reference_version']} @ {snapshot.name}")
    print(f"producer_sha256 {producer_sha256}")


if __name__ == "__main__":
    main()
