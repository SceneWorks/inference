"""MiniMax-H3 video-VAE **spatial tiling** real-weight reference (sc-18786).

Why a second real-weight reference
----------------------------------

``tools/dump_minimax_h3_video_vae_real.py`` (sc-18740) decodes a 4x4 latent to 64x64 px and calls
``vae.disable_tiling()`` first, because at that canvas the tiled and untiled paths are bit-identical
anyway. That reference therefore **cannot** gate spatial tiling: it is inert at its own geometry, by
construction and by its own assertion.

``AutoencoderKLMiniMaxH3`` ships with ``use_tiling = True`` (256 px tiles, 64 px minimum overlap) for
**both** halves, and upstream states the consequence in the class docstring: MiniMax-H3 was released
with tiling enabled and *the released frames are the blended-tile ones, so disabling tiling changes
the output*. So the untiled path is not "the same answer, more memory" — it is a different answer.

This tool decodes a canvas that crosses a tile boundary on **both** axes and records the tiled and
untiled results side by side, so the Rust port can assert (a) that it matches the tiled reference and
(b) that the untiled path really does differ — the sc-18740 lesson being that new code which cannot
be distinguished from the old code is untested by construction.

Geometry
--------

``LATENT_HW = (32, 20)`` -> 512x320 px at the real 16x spatial ratio. Deliberately **not square**:
the height axis splits into 3 tiles and the width axis into 2, so a transposed height/width plan
produces the wrong answer instead of accidentally the right one. The height axis also has a genuine
*interior* tile (blended above and trimmed below), which a 2x2 grid never exercises.

The output carries two full videos (~33 MB each at f32) plus a sub-tile control, so it is a ~70 MB
file, deliberately **not committed**. Keep it outside the repository.

    MINIMAX_H3_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_video_vae_tiling.py \
        --out ~/minimax-h3-vae-tiling-reference.safetensors

Then:

    MINIMAX_H3_SNAPSHOT=... \
    MINIMAX_H3_VIDEO_VAE_TILING_REFERENCE=~/minimax-h3-vae-tiling-reference.safetensors \
        cargo test -p mlx-gen-minimax-h3 --test real_weights -- --ignored --nocapture tiling
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.numpy import save_file

from _paths import hf_hub_cache

SEED = 18786
# 5 tokens -> exactly one `clip_length` chunk -> 17 pixel frames. Tiling is applied per temporal
# clip (`_decode` calls `_decode_clip`), so one clip is enough to gate the spatial grid and keeps
# the artifact and the runtime down.
LATENT_FRAMES = 5
# 32x20 latent -> 512x320 px. Asymmetric on purpose; see the module docstring.
LATENT_H, LATENT_W = 32, 20
# The control: far below one tile, where tiling must be provably inert.
SUBTILE_HW = 4


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

    ratio = vae.spatial_compression_ratio

    # ---- what the shipped model actually does, read off the instance -----------------------------
    shipped = {
        "use_tiling": bool(vae.use_tiling),
        "tile_sample_min_height": int(vae.tile_sample_min_height),
        "tile_sample_min_width": int(vae.tile_sample_min_width),
        "tile_sample_min_overlap_height": int(vae.tile_sample_min_overlap_height),
        "tile_sample_min_overlap_width": int(vae.tile_sample_min_overlap_width),
        "spatial_compression_ratio": int(ratio),
    }
    print("shipped tiling defaults:", json.dumps(shipped))
    assert shipped["use_tiling"], (
        "this whole reference assumes the shipped VAE tiles by default; it does not, so the "
        "premise of sc-18786 is wrong and the port should NOT be changed"
    )

    height, width = LATENT_H * ratio, LATENT_W * ratio
    y_idx, y_len, y_ov = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    x_idx, x_len, x_ov = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    plan = {
        "height": height,
        "width": width,
        "rows": {"starts": y_idx, "lengths": y_len, "overlaps": y_ov},
        "cols": {"starts": x_idx, "lengths": x_len, "overlaps": x_ov},
    }
    print("tile plan:", json.dumps(plan))
    assert len(y_idx) > 1 and len(x_idx) > 1, (
        f"{height}x{width} does not cross a tile boundary on both axes "
        f"({len(y_idx)} rows x {len(x_idx)} cols); this reference could not gate tiling"
    )
    assert len(y_idx) != len(x_idx), "a square grid cannot catch a transposed height/width plan"
    assert y_idx[-1] + y_len[-1] == height and x_idx[-1] + x_len[-1] == width, "tiles must cover the canvas"

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    zc = vae.config.latent_channels

    generator = torch.Generator().manual_seed(SEED)
    latent = torch.randn(1, zc, LATENT_FRAMES, LATENT_H, LATENT_W, generator=generator)
    sub_latent = torch.randn(1, zc, LATENT_FRAMES, SUBTILE_HW, SUBTILE_HW, generator=generator)

    def decode(z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return vae.decode(z * latents_std + latents_mean, return_dict=False)[0].to(torch.float32)

    # ---- the released behaviour: tiling ON -------------------------------------------------------
    assert vae.use_tiling
    tiled = decode(latent)
    sub_tiled = decode(sub_latent)

    # ---- the counterfactual: tiling OFF ----------------------------------------------------------
    vae.disable_tiling()
    assert not vae.use_tiling
    untiled = decode(latent)
    sub_untiled = decode(sub_latent)
    vae.enable_tiling()
    assert vae.use_tiling, "enable_tiling() must restore the shipped default"

    tile_delta = (tiled - untiled).abs().max().item()
    peak = tiled.abs().max().item()
    sub_delta = (sub_tiled - sub_untiled).abs().max().item()

    print(f"latent {tuple(latent.shape)} -> decoded {tuple(tiled.shape)}")
    print(f"  tiled   std {tiled.std().item():.6f}  max|px| {tiled.abs().max().item():.6f}")
    print(f"  untiled std {untiled.std().item():.6f}  max|px| {untiled.abs().max().item():.6f}")
    print(f"  TILE DELTA max|tiled - untiled| = {tile_delta:.6e}  (rel {tile_delta / peak:.6e})")
    print(f"  sub-tile {SUBTILE_HW * ratio}px delta = {sub_delta:.6e}")

    # The whole point of this canvas. If tiling made no difference here the reference would be as
    # blind as sc-18740's was, and the port's new code would be untestable.
    assert tile_delta > 0.0, (
        f"{height}x{width} produced an identical tiled and untiled decode; this canvas cannot gate "
        "spatial tiling"
    )
    # And the mirror assertion: below one tile the two paths must agree exactly, which is what keeps
    # the sc-17140 / sc-18740 sub-tile fixtures valid.
    assert sub_delta == 0.0, (
        f"tiling is not inert at {SUBTILE_HW * ratio}px (delta {sub_delta}); the sub-tile fixtures "
        "would no longer be comparable"
    )

    tensors = {
        "in.latent": latent.numpy().copy(),
        "out.video.tiled": tiled.numpy().copy(),
        "out.video.untiled": untiled.numpy().copy(),
        "in.latent.subtile": sub_latent.numpy().copy(),
        "out.video.subtile.tiled": sub_tiled.numpy().copy(),
    }
    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.AutoencoderKLMiniMaxH3",
        "reference_version": diffusers.__version__,
        "snapshot": snapshot.name,
        "seed": str(SEED),
        "story": "sc-18786",
        "shipped_tiling": json.dumps(shipped),
        "tile_plan": json.dumps(plan),
        "tile_delta_max_abs": f"{tile_delta:.6e}",
        "tile_delta_rel_max_abs": f"{tile_delta / peak:.6e}",
        "subtile_delta_max_abs": f"{sub_delta:.6e}",
    }
    out = Path(args.out).expanduser()
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} from {meta['reference']} {meta['reference_version']} @ {snapshot.name}")


if __name__ == "__main__":
    main()
