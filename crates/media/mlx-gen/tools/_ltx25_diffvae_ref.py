"""Shared loader for the LTX-2.5 DiffVAE reference decoder (sc-18766).

Builds the **upstream** `DiffusionVideoDecoder` (Lightricks/LTX-2 v1.2.0, commit
`d151147788a9284cca791edc6ce898007e727fe6`) straight off the real
`vae/ltx-2.5-video-vae-bf16.safetensors`, using upstream's own configurator, state-dict ops and
`CombinedDiffusionNABlock` pathway. Neighborhood attention runs on upstream's vendored
`fallback_na.eager` backend — NATTEN is CUDA-only, and that fallback is upstream's own statement of
`natten.na3d` semantics (windows shift inward at the grid border rather than clamp-and-mask), which
is exactly what the MLX port targets.

`apply_diffvae_config` is deliberately NOT used: it refuses the COMBINED pathway without natten and
would otherwise install the CHUNKED deferred-stage-4 variant. The combined pathway is the one the
port implements, so the block class / attention backend are installed directly here.

Requires:
  * `LTX2_SRC`      — `packages/ltx-core/src` of a v1.2.0 LTX-2 checkout;
  * `LTX25_VAE_DIR` — a directory holding the LTX-2.5 `vae/` component checkpoints.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

from _paths import require_env

DIFF_VAE = "ltx-2.5-video-vae-bf16.safetensors"
CONV_VAE = "ltx-2.5-video-vae-conv-bf16.safetensors"

REFERENCE_COMMIT = "d151147788a9284cca791edc6ce898007e727fe6"


def ltx_core_on_path() -> None:
    """Put a v1.2.0 `ltx_core` on `sys.path` (idempotent)."""
    src = require_env(
        "LTX2_SRC",
        f"path to packages/ltx-core/src of a Lightricks/LTX-2 checkout at {REFERENCE_COMMIT}",
    )
    src = str(Path(src).expanduser())
    if src not in sys.path:
        sys.path.insert(0, src)


def vae_dir() -> Path:
    return Path(
        require_env(
            "LTX25_VAE_DIR",
            f"directory holding {DIFF_VAE} / {CONV_VAE} from the gated Lightricks/LTX-2.5 repo",
        )
    ).expanduser()


def probe(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """A deterministic, band-limited probe tensor, exactly representable in float16.

    Smooth rather than white: the decoder is a resampling stack, so a low-frequency field exercises
    the neighborhood windows instead of being averaged into nothing. The result is rounded through
    float16 because that is how the golden stores its inputs — the port must consume the very bits
    the reference consumed, not a re-derivation of them.
    """
    n = 1
    for dim in shape:
        n *= dim
    idx = torch.arange(n, dtype=torch.float64)
    values = (
        torch.sin(idx * 0.013_1 + seed * 1.7) * torch.cos(idx * 0.007_3 - seed * 0.31) * 0.9
        + 0.1 * torch.sin(idx * 0.000_37 + seed)
    )
    return values.reshape(shape).to(torch.float16).to(torch.float32)


def load_reference_decoder(path: Path, *, dtype: torch.dtype = torch.float32):
    """Upstream `DiffusionVideoDecoder` with the real weights, combined pathway, eager NA."""
    ltx_core_on_path()

    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.model.video_vae.model_configurator import (
        VideoDecoderConfigurator,
        video_decoder_sd_ops_for_checkpoint,
    )
    from ltx_core.model.video_vae.transformer.attention import NeighborhoodAttention3D
    from ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
    from ltx_core.model.video_vae.transformer.combined.block import CombinedDiffusionNABlock
    from ltx_core.model.video_vae.transformer.fallback_na import EagerSdpaAttention

    loader = SafetensorsModelStateDictLoader()
    metadata = loader.metadata(str(path))
    decoder = VideoDecoderConfigurator.from_metadata(metadata)
    state = loader.load(str(path), video_decoder_sd_ops_for_checkpoint(str(path), diffusion_vae=True))
    missing, unexpected = decoder.load_state_dict(state.sd, strict=False)
    # `type_emb` is carried by the checkpoint but consumed by no reference module; everything else
    # must land, or the golden would be dumped from a partially-initialised decoder.
    unexpected = [k for k in unexpected if k != "type_emb"]
    missing = [k for k in missing if "rope_inv" not in k and "default_inference_timesteps" not in k]
    assert not missing, f"decoder weights missing: {missing[:8]}"
    assert not unexpected, f"checkpoint keys with no home: {unexpected[:8]}"

    for block in decoder.diff_blocks:
        if isinstance(block, DiffusionNABlock):
            block.__class__ = CombinedDiffusionNABlock
    decoder.deferred_stage4_upsample = False
    decoder.mark_dynamic_shapes = False
    eager = EagerSdpaAttention()
    for module in decoder.modules():
        if isinstance(module, NeighborhoodAttention3D):
            module.attention_function = eager
            module.natten_backend = None
            module.w_chunks = 1
    return decoder.to(dtype=dtype).eval()


def stage5_geometry(decoder, latent_t: int, latent_h: int, latent_w: int) -> tuple[int, int, int]:
    """Stage-5 pixel `(F, H, W)` for an untiled decode of a `(T, H, W)` latent."""
    from ltx_core.model.video_vae import diffusion_tiling as dt

    strides = [tuple(u.stride) for u in decoder.upsamples]
    s4 = dt.stage4_thw_from_latent(strides, latent_t, latent_h, latent_w, drop_leading_frame=True)
    return dt.stage5_pixel_shape_from_stage4(
        *s4,
        upsample_stride=tuple(decoder.upsamples[3].stride),
        patch_size=decoder.patch_size,
        stage5_kernel_t=decoder.stage5_kernel[0],
        drop_leading_frame=True,
        pad_trailing=True,
    )


def untiled_decode(decoder, latent: torch.Tensor, noise: torch.Tensor):
    """The reference untiled single-step-x0 decode, with an EXPLICIT `x_t`.

    Same composition as `DiffusionVideoDecoder._decode_pixels` for `tiling_config=None` — the only
    change is that the stage-5 noise is supplied instead of drawn from a `torch.Generator`, so the
    result is reproducible by a port that has no torch RNG. Returns
    `(pixels, stage123, context)`.
    """
    from ltx_core.model.video_vae import diffusion_tiling as dt
    from ltx_core.types import VideoLatentShape

    assert decoder.model_output_type == "x0", "this composition assumes single-step x0"
    content = VideoLatentShape.from_torch_shape(latent.shape)
    content_pixel = content.upscale(decoder.video_downscale_factors)._replace(channels=decoder.out_channels)

    latent, (_t_pad, h_pad, w_pad) = dt.ensure_min_latent_shape(latent, decoder.stage_min_tile_sizes)
    padded = dt.pad_trailing_latent_for_natten_border(latent, decoder._natten_trailing_pad_latent_frames)

    stage123 = decoder.forward_stages_1_to_3(padded, drop_leading_frame=True)
    context = decoder.forward_stage_4(stage123, drop_leading_frame=True, pad_trailing=True)

    timestep = decoder.default_inference_timesteps.to(latent.device)
    assert timestep.numel() == 1, f"expected a single-step schedule, got {timestep.tolist()}"
    t_now = timestep[:1].expand(latent.shape[0])

    context_and_x = decoder._context_and_x_for_diff_step(context, noise)
    model_out = decoder.forward_diff_step(context_and_x, t_now)

    pixels = dt.crop_pixels_to_content(
        model_out,
        content_pixel.frames,
        content_pixel.height,
        content_pixel.width,
        h_pad=h_pad,
        w_pad=w_pad,
        spatial_scale=(decoder.video_downscale_factors.height, decoder.video_downscale_factors.width),
    )
    return pixels, stage123, context


def report(name: str, tensor: torch.Tensor) -> None:
    flat = tensor.detach().to(torch.float32).flatten()
    print(
        f"  {name:<14} {tuple(tensor.shape)!s:<24} "
        f"mean {flat.mean():+.6f} std {flat.std():.6f} max|v| {flat.abs().max():.6f}"
    )


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise SystemExit(f"{name} is not finite — refusing to commit a poisoned golden")


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


__all__ = [
    "CONV_VAE",
    "DIFF_VAE",
    "REFERENCE_COMMIT",
    "ceil_div",
    "env_int",
    "load_reference_decoder",
    "ltx_core_on_path",
    "math",
    "probe",
    "report",
    "require_finite",
    "stage5_geometry",
    "untiled_decode",
    "vae_dir",
]
