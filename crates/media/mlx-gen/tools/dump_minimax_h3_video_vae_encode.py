"""MiniMax-H3 video-VAE **encode** parity fixtures (sc-17148).

Runs the **official diffusers** ``AutoencoderKLMiniMaxH3`` at tiny dims and dumps the encode half's
inputs, weights and outputs. The Rust/MLX port is dimension-parametric, so it runs the same tiny
config and must reproduce these tensors.

Why this is a separate script and a separate fixture from ``dump_minimax_h3_video_vae.py``
------------------------------------------------------------------------------------------

Two independent reasons, and either alone would be sufficient:

1. **The geometries cannot be the same.** The encoder must not *crop*. A down-block gets a
   downsampler whenever ``temporal_factor * spatial_factor > 1``, and one with ``spatial_stride ==
   1`` still convolves a 3-wide kernel with **no** spatial padding — so it loses two columns
   instead of halving. The decode fixture's ``TIME_DOWN=(1,2,2,1)`` / ``SPATIAL_DOWN=(2,1,1,1)``
   does exactly that twice: 24 px encodes to 8 latent rather than 12, which breaks the tiled encode
   outright because the stitch assumes ``latent = pixel / ratio``. The shipped config never does
   this — all four of its downsamplers are spatial-2.

   Fixing it means every time-strided level must also be space-strided. The original MiniMax module
   asserts ``time_stride in [1, 2]``, so ``patch_size_t`` 4 needs **two** time-strided levels, and
   the spatial cumprod therefore becomes 4 rather than 2.

2. **The decode fixture's bytes are shared.** ``candle-gen-minimax-h3`` (sc-17154) holds a
   byte-identical copy and its ``cross_backend.rs`` digests it. Regenerating it re-randomizes every
   decoder weight, which measurably eroded that crate's
   ``a_gated_ffn_half_swap_is_loud_against_the_mlx_record`` mutation gate from ~1.2e-1 to 8.4e-2
   against its 1e-1 floor. Lowering another story's bound to accommodate an unrelated change is
   exactly the kind of quiet erosion the epic's fixture discipline exists to prevent.

Everything else — the converted-checkpoint provenance rule (layout rule 3), the non-degeneracy
checks, the metadata record — is the same discipline the decode script applies.

Usage::

    python3 crates/media/mlx-gen/tools/dump_minimax_h3_video_vae_encode.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import diffusers
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import fixture, hf_hub_cache  # noqa: E402

SEED = 17148
CLIP_LENGTH = 17
TOKEN_DROP = 3
Z_CHANNELS = 24

# The decoder knobs are irrelevant here but must still construct; keep them at the decode fixture's
# so the two files describe recognisably the same tiny model.
HEADS = 2
DIM_HEAD = 16
NUM_LAYERS = 2
FFN_MULT = 4
ROPE_THETA = 100.0
ROPE_DIM_RATIO = 0.75
DECODER_NORM_EPS = 1e-5

# The encoder geometry. Three constraints, all of them forced:
#
#  * every width must be a multiple of 32 — the ORIGINAL MiniMax module hardcodes ``num_groups=32``
#    in ``norm.py::get_group_norm_3d``, so a narrower fixture cannot construct the reference the
#    conversion is cross-checked against;
#  * every level carrying a downsampler must be spatial-stride-2, or it crops (see the module
#    docstring). With ``time_stride`` capped at 2, that puts the spatial cumprod at 4;
#  * the width must CHANGE somewhere, or ``conv_shortcut`` is never built and the residual
#    projection goes ungated.
CH = 32
NORM_NUM_GROUPS = CH
BLOCK_OUT_CHANNELS = (CH, CH, CH, 2 * CH)
LAYERS_PER_BLOCK = 1
SPATIAL_DOWN = (1, 2, 2, 1)
TIME_DOWN = (1, 2, 2, 1)

# The tiled-encode probe. ENC_TILE / ENC_OVERLAP are DELIBERATELY smaller than the shipped 256/64:
# a fixture canvas large enough to tile at production tile size would not be committable, and an
# inert tiling golden proves nothing. The Rust side takes the same two numbers as parameters
# (`encode_clip_tiled`) for exactly this.
ENC_TILE, ENC_OVERLAP = 16, 4
ENC_H = ENC_W = 24

# The production per-channel statistics, verbatim from `vae/config.json`.
LATENTS_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
]
LATENTS_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
]


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).cpu().numpy().copy()


def build_diffusers(token_drop: int):
    from diffusers import AutoencoderKLMiniMaxH3

    model = AutoencoderKLMiniMaxH3(
        in_channels=3,
        out_channels=3,
        latent_channels=Z_CHANNELS,
        block_out_channels=BLOCK_OUT_CHANNELS,
        layers_per_block=LAYERS_PER_BLOCK,
        spatial_downsample_factors=SPATIAL_DOWN,
        temporal_downsample_factors=TIME_DOWN,
        norm_num_groups=NORM_NUM_GROUPS,
        spatial_padding_mode="reflect",
        decoder_num_layers=NUM_LAYERS,
        decoder_num_attention_heads=HEADS,
        decoder_attention_head_dim=DIM_HEAD,
        decoder_num_register_tokens=4,
        decoder_ffn_mult=FFN_MULT,
        decoder_rope_theta=ROPE_THETA,
        decoder_rope_dim_ratio=ROPE_DIM_RATIO,
        decoder_norm_eps=DECODER_NORM_EPS,
        clip_length=CLIP_LENGTH,
        token_drop=token_drop,
        latents_mean=tuple(LATENTS_MEAN),
        latents_std=tuple(LATENTS_STD),
    )
    model.eval()
    return model


def randomize(model, generator):
    """Re-randomize every parameter.

    Default init leaves the conv stack close enough to a scaled identity in places that a golden
    taken from it would pass against a port with the wrong padding or the wrong norm. The decode
    half is randomized too — this fixture carries **no decode goldens**, but it does carry the
    decoder weights so `MiniMaxH3VideoVae::from_weights` can build a complete model from it, and a
    half-initialized model in a committed artifact is a trap for whoever reads it next.
    """
    with torch.no_grad():
        for _, param in model.named_parameters():
            param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.35)


def main() -> None:
    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    out: dict[str, np.ndarray] = {}

    model = build_diffusers(TOKEN_DROP)
    randomize(model, generator)

    # Tiling is ON by default in `AutoencoderKLMiniMaxH3` — set in `__init__`, not through
    # `enable_tiling()`. Assert it rather than assume it: a port written from the class body alone
    # would never see it, and a silent default flip upstream would rewrite these goldens.
    assert model.use_tiling, "diffusers ships tiling on; this assertion guards a silent default flip"
    assert model.spatial_compression_ratio == int(np.prod(SPATIAL_DOWN))
    assert model.temporal_compression_ratio == int(np.prod(TIME_DOWN))

    # The published weights, straight out of the official class — this IS the converted layout.
    #
    # BOTH halves are dumped. The encode half is what this fixture gates; the decode half is here
    # only so `MiniMaxH3VideoVae::from_weights` can construct a complete model from this file
    # alone, since `encode` lives on that type. No decode golden is taken from it — the decode
    # parity suite has its own fixture, whose bytes are shared with `candle-gen-minimax-h3`.
    for key, tensor in model.state_dict().items():
        out[key] = np32(tensor)
    n_weights = len(out)
    n_encode = sum(1 for k in out if k.startswith(("encoder.", "quant_conv.")))

    # ---- (a) the bare encoder stack, untiled ------------------------------------------------
    # T = 5 > 1 so the frame-isolated GroupNorm and the temporal strides are both live; at T = 1 a
    # plain 3-D GroupNorm is indistinguishable from the isolated one.
    with torch.no_grad():
        enc_in = torch.randn(1, 3, 5, ENC_H, ENC_W, generator=generator)
        enc_out = model.encoder(enc_in)
    out["in.encoder.pixels"] = np32(enc_in)
    out["out.encoder.params"] = np32(enc_out)
    assert enc_out.shape[2] > 1, "the encoder golden must keep a temporal axis"
    ratio = model.spatial_compression_ratio
    assert enc_out.shape[-1] == ENC_W // ratio and enc_out.shape[-2] == ENC_H // ratio, (
        f"the encoder is CROPPING ({list(enc_out.shape)} from {ENC_H}x{ENC_W} at ratio {ratio}); "
        "the tiled stitch assumes latent = pixel / ratio"
    )

    # ---- (b) `_encode_clip` untiled: encoder then quant_conv ---------------------------------
    with torch.no_grad():
        model.disable_tiling()
        clip_in = torch.randn(1, 3, 5, ENC_H, ENC_W, generator=generator)
        clip_out = model._encode_clip(clip_in)
    out["in.encode_clip.pixels"] = np32(clip_in)
    out["out.encode_clip.params"] = np32(clip_out)

    # ---- (c) `_encode_clip` TILED, at the shrunk tile geometry --------------------------------
    with torch.no_grad():
        model.enable_tiling(
            tile_sample_min_height=ENC_TILE,
            tile_sample_min_width=ENC_TILE,
            tile_sample_min_overlap_height=ENC_OVERLAP,
            tile_sample_min_overlap_width=ENC_OVERLAP,
        )
        assert model.use_tiling
        tiled_out = model._encode_clip(clip_in)
    out["out.encode_clip_tiled.params"] = np32(tiled_out)
    assert tiled_out.shape == clip_out.shape, (
        f"tiled {list(tiled_out.shape)} != untiled {list(clip_out.shape)}"
    )
    tile_delta = float((tiled_out - clip_out).abs().max() / clip_out.abs().max())
    # The point of the tiled golden: it must DIFFER from the untiled one, or it gates nothing.
    assert tile_delta > 1e-2, (
        f"tiled and untiled encode agree to {tile_delta:.3e}; this fixture cannot gate the blend"
    )
    y_idx, _, y_ovl = model._split_tiles(ENC_H, ENC_TILE, ENC_OVERLAP)
    assert len(y_idx) > 1, "the tiled golden must actually span more than one tile"
    out["const.encode_tile"] = np.array([ENC_TILE, ENC_OVERLAP], dtype=np.int32)
    print(
        f"  tiling: {len(y_idx)} tiles starts={y_idx} overlaps={y_ovl}; "
        f"tiled vs untiled rel {tile_delta:.3e}"
    )

    # ---- (d) THE KEYFRAME PATH: a single frame ------------------------------------------------
    # `_encode` short-circuits on num_frames == 1 — no clip padding, no chunking, no token_drop —
    # and returns exactly ONE latent frame.
    with torch.no_grad():
        model.disable_tiling()
        kf_in = torch.randn(1, 3, 1, ENC_H, ENC_W, generator=generator)
        kf_posterior = model.encode(kf_in, return_dict=False)[0]
        kf_mean, kf_logvar = torch.chunk(kf_posterior.parameters, 2, dim=1)
    assert kf_mean.shape[2] == 1, f"a keyframe must encode to ONE latent frame, got {kf_mean.shape}"
    out["in.encode_single.pixels"] = np32(kf_in)
    out["out.encode_single.mean"] = np32(kf_mean)
    out["out.encode_single.std"] = np32(torch.exp(0.5 * kf_logvar.clamp(-30.0, 20.0)))

    # ---- (e) the CHUNKED video path -----------------------------------------------------------
    # 17 frames is exactly one clip, then token_drop trims the tail. This is the path the keyframe
    # short circuit deliberately avoids; kept so the two gate each other.
    with torch.no_grad():
        vid_in = torch.randn(1, 3, CLIP_LENGTH, ENC_H, ENC_W, generator=generator)
        vid_posterior = model.encode(vid_in, return_dict=False)[0]
        vid_mean = torch.chunk(vid_posterior.parameters, 2, dim=1)[0]
    out["in.encode_chunked.pixels"] = np32(vid_in)
    out["out.encode_chunked.mean"] = np32(vid_mean)
    assert vid_mean.shape[2] > 1
    print(
        f"  encode: 1 frame -> {list(kf_mean.shape)}; "
        f"{CLIP_LENGTH} frames -> {list(vid_mean.shape)}"
    )

    # Non-degeneracy: a golden of constant / all-zero tensors is a false green.
    for key, value in out.items():
        assert np.isfinite(value).all(), f"{key} has non-finite entries"
        if key.startswith("out."):
            assert float(value.std()) > 1e-4, f"{key} is ~constant ({value.std()})"

    for key in sorted(out):
        if key.startswith(("in.", "out.")):
            print(f"  {key}: {list(out[key].shape)}")

    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.AutoencoderKLMiniMaxH3",
        "reference_version": diffusers.__version__,
        "half": "encode",
        "encode_tile_blend_rel": f"{tile_delta:.6e}",
        "spatial_compression_ratio": str(model.spatial_compression_ratio),
        "temporal_compression_ratio": str(model.temporal_compression_ratio),
        "story": "sc-17148",
    }

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/video_vae_encode.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path, metadata=meta)
    print(
        f"wrote {path} ({len(out)} tensors: {n_weights} weights, {n_encode} of them the encode "
        f"half) from {meta['reference']} {meta['reference_version']}"
    )


if __name__ == "__main__":
    main()
