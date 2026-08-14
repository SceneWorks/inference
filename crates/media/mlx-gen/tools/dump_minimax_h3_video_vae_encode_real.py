"""MiniMax-H3 video-VAE **real-weight encode** reference (sc-19008).

Encodes seeded pixels through the official diffusers ``AutoencoderKLMiniMaxH3`` loaded from the
real ``vae/``, and writes `(pixels, posterior mean, posterior std)` so the Rust real-weight smokes
can assert numeric parity against the official implementation on the *published* tensors.

Why this exists — the committed tiny fixture is not enough
----------------------------------------------------------

``tools/dump_minimax_h3_video_vae_encode.py`` builds a 32-channel, 4-level toy at
``norm_num_groups == block_out_channels``, which makes every GroupNorm an *instance* norm. The
shipped encoder runs 32 groups over 128…1024 channels, has SIX levels with two resnets each, and
its `conv_shortcut` sits at levels 1/3/5 rather than 3. A port can reproduce the toy exactly and
still index the real stack wrong. Only a reference computed by an independent implementation from
the *same published bytes* can gate that.

It is also the only way to exercise the shipped 256/64 **spatial tiling** on the real stack: the
committed fixture has to shrink the tile geometry to stay committable.

The output is a few MB and is deliberately **not** committed: point the Rust test at it with
``MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE``. Keep it outside the repository.

    MINIMAX_H3_SNAPSHOT=<snapshot-root> \
        python3 tools/dump_minimax_h3_video_vae_encode_real.py \
        --out ~/minimax-h3-real-vae-encode-reference.safetensors

Then:

    MINIMAX_H3_SNAPSHOT=... \
    MINIMAX_H3_VIDEO_VAE_ENCODE_REFERENCE=~/minimax-h3-real-vae-encode-reference.safetensors \
        cargo test -p candle-gen-minimax-h3 --test real_weights -- --ignored --nocapture

Memory: the whole ``vae/`` at float32 is ~19.4 GB resident. Run it on its own.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from safetensors.numpy import save_file

from _paths import hf_hub_cache

SEED = 19008

# The KEYFRAME probe — `fl2va`'s own path. 320 px is DELIBERATELY larger than the shipped 256 px
# tile so the reference runs its spatial tiling, which is on by default in `__init__` and which a
# port written from the class body alone would never see. Asserted below rather than assumed.
KEYFRAME_HW = 320

# The CLIP probe — exactly one `clip_length`, so the temporal strides, the frame-isolated
# GroupNorm (invisible at T = 1) and the `token_drop` tail trim are all live. Kept under one tile
# so it isolates the temporal path from the tiling one.
CLIP_FRAMES = 17
CLIP_HW = 128


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

    # Tiling is ON out of `__init__`, not through `enable_tiling()`. Assert it: a silent default
    # flip upstream would rewrite this reference and the Rust side would follow it blindly.
    assert vae.use_tiling, "diffusers ships tiling on; this assertion guards a silent default flip"
    assert KEYFRAME_HW > vae.tile_sample_min_height and KEYFRAME_HW > vae.tile_sample_min_width, (
        f"{KEYFRAME_HW}px does not cross the {vae.tile_sample_min_height}px tile boundary; the "
        "keyframe reference would not exercise the tiled encode"
    )
    assert CLIP_HW <= vae.tile_sample_min_height, (
        f"{CLIP_HW}px would tile; the clip reference is meant to isolate the temporal path"
    )
    ratio = vae.spatial_compression_ratio
    ratio_t = vae.temporal_compression_ratio
    print(
        f"vae: {ratio}x spatial / {ratio_t}x temporal, tiling {vae.tile_sample_min_height}px "
        f"/ {vae.tile_sample_min_overlap_height}px overlap"
    )

    generator = torch.Generator().manual_seed(SEED)
    tensors: dict = {}

    def encode(name: str, pixels: torch.Tensor) -> None:
        with torch.no_grad():
            posterior = vae.encode(pixels, return_dict=False)[0]
            mean, logvar = torch.chunk(posterior.parameters, 2, dim=1)
            std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        tensors[f"in.{name}.pixels"] = pixels.to(torch.float32).numpy().copy()
        tensors[f"out.{name}.mean"] = mean.to(torch.float32).numpy().copy()
        tensors[f"out.{name}.std"] = std.to(torch.float32).numpy().copy()
        print(
            f"  {name}: {tuple(pixels.shape)} -> mean {tuple(mean.shape)}  "
            f"std {float(mean.std()):.6f}  max|mean| {float(mean.abs().max()):.6f}"
        )
        assert torch.isfinite(mean).all() and torch.isfinite(std).all(), f"{name}: non-finite"
        assert float(mean.std()) > 1e-3, f"{name}: a ~constant golden gates nothing"

    # (a) the keyframe path — ONE frame, tiled at the shipped 256/64.
    keyframe = torch.randn(1, 3, 1, KEYFRAME_HW, KEYFRAME_HW, generator=generator)
    encode("keyframe", keyframe)
    assert tensors["out.keyframe.mean"].shape[2] == 1, "a keyframe must encode to ONE latent frame"
    assert tensors["out.keyframe.mean"].shape[-1] == KEYFRAME_HW // ratio, (
        "the tiled encode must still produce pixel/ratio latents — it is CROPPING"
    )

    # (b) the chunked path — one full clip, untiled.
    clip = torch.randn(1, 3, CLIP_FRAMES, CLIP_HW, CLIP_HW, generator=generator)
    encode("clip", clip)
    assert tensors["out.clip.mean"].shape[2] > 1, "the clip golden must keep a temporal axis"

    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.AutoencoderKLMiniMaxH3",
        "reference_version": diffusers.__version__,
        "half": "encode",
        "snapshot": snapshot.name,
        "seed": str(SEED),
        "tile_sample_min_size": str(vae.tile_sample_min_height),
        "tile_sample_min_overlap": str(vae.tile_sample_min_overlap_height),
        "spatial_compression_ratio": str(ratio),
        "temporal_compression_ratio": str(ratio_t),
        "story": "sc-19008",
    }
    out = Path(args.out).expanduser()
    save_file(tensors, str(out), metadata=meta)
    print(f"wrote {out} from {meta['reference']} {meta['reference_version']} @ {snapshot.name}")


if __name__ == "__main__":
    main()
