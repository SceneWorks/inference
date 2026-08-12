"""MiniMax-H3 video-VAE decode parity fixtures (sc-17140, corrected by sc-18740).

Runs the **official diffusers** ``AutoencoderKLMiniMaxH3`` at tiny dims and dumps inputs, weights
and outputs. The Rust/MLX port is dimension-parametric, so it runs the same tiny config and must
reproduce these tensors.

Why the official diffusers class and not the MiniMax reference modules — read this before changing
it back
-------------------------------------------------------------------------------------------------

The first version of this script ran the reference bundle shipped inside the snapshot
(``FL2VA/video_vae/*.py``) and emitted its parameters under the *published* key names through a pure
rename table (``"ff.w1": "ff.net.0.proj"``). That was a **false green**, and it shipped a
functionally wrong 36-layer decoder (sc-18740).

``scripts/convert_minimax_h3_to_diffusers.py`` does not merely rename. For every gated FFN it
**swaps the two row halves** of the fused projection, because diffusers' ``SwiGLU`` reads
``[value | gate]`` where the MiniMax modules store ``[gate | value]``. Both forms have identical
shapes. A fixture dumped from the reference modules therefore carried the *source* layout under
*published* names, the loader read it the source way, the two agreed, parity passed at 1e-3 — and
production, which loads the genuinely converted ``vae/``, was wrong by a half-swap in every block
(measured relative max-abs-diff 0.86-0.99 per block on real weights, with output norms essentially
unchanged, which is why no magnitude or checksum gate could see it).

    A golden and a loader that share a layout prove only that they agree with each other.

So: the golden now comes from ``AutoencoderKLMiniMaxH3`` — the converted layout production actually
reads — and the fixture records that provenance in its safetensors metadata. ``diffusers`` must be
installed from ``main``; ``MiniMaxH3`` landed in PR #14355 (merged 2026-08-05) and is in no tagged
release, so a released ``pip install diffusers`` has none of these classes.

The conversion is *also* checked here rather than trusted
---------------------------------------------------------

The script additionally builds the MiniMax reference ``AutoencoderKLLegacy`` at the same tiny dims,
loads the diffusers weights into it through the **inverse** of the official conversion (un-swap the
FFN halves, re-fuse QKV per-head-interleaved, rename back), and asserts the two models decode to the
same tensor. That is what makes the ``src.``-prefixed pre-conversion tensors in the fixture real
evidence rather than a restatement of the published ones: if the un-swap were wrong, the reference
would compute something different and this script would refuse to write.

Why tiny-but-real: the shipped decoder is a 36-layer / 2048-dim transformer (~5.2 B params,
10.4 GB) — far too large to commit. Every *structural* knob is preserved though, and the temporal
knobs are the REAL ones:

    clip_length 17, token_drop 3, vae_ratio_t 4
      -> frame_pre_padding  = (-17) % 4 = 3
      -> tokens_chunk_size  = ceil(17/4) = 5
      -> token_overlap      = (-3) % 5   = 2
      -> frame_overlap      = max(2*4 - 3, 0) = 5

which are identical to the production model. Only the *width* is shrunk (heads 32->2,
dim_head 64->16, layers 36->2, vae_ratio 16->2). ``rope_dim_ratio`` stays 0.75, so the tiny model
has rot_dim = int(16 * 0.75) = 12 < 16 and exercises the same PARTIAL-rotary path the real one does
(48 < 64). ``latent_channels`` stays 24 so the real 24-entry ``latents_mean``/``latents_std``
de-normalization is exercised verbatim.

The reference's ``_init_weights`` leaves ``scale1``/``scale2`` at ZERO, which makes every
transformer block an exact identity passthrough — a golden dumped that way would pass against a
model that never implements attention or the feed-forward at all. diffusers initializes them the
same way, so this script re-randomizes every decoder parameter from a seeded generator and asserts
non-degeneracy before writing.

Requires torch + diffusers@main + torchvision + einops. Run:

    MINIMAX_H3_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_video_vae.py
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

SEED = 17140
CLIP_LENGTH = 17
TOKEN_DROP = 3

# Tiny geometry. The downsample factor cumprods give vae_ratio 2 / vae_ratio_t 4; the latter is the
# production value, which is what makes the chunk arithmetic identical to the real model.
# The encoder geometry is chosen (sc-17148), not inherited. Three constraints:
#
#  * `SPATIAL_DOWN` / `TIME_DOWN` must still cumprod to vae_ratio 2 / vae_ratio_t 4, the latter
#    being the production value that makes the chunk arithmetic identical to the real model.
#  * EVERY level that carries a downsampler must have `spatial_stride == 2`. A downsampler is
#    built whenever `temporal · spatial > 1`, and one with `spatial_stride == 1` still convolves a
#    3-wide kernel with NO spatial padding — so it CROPS two columns instead of halving. The
#    shipped config never does this (all four of its downsamplers are spatial-2), but the previous
#    fixture geometry `TIME_DOWN=(1,2,2,1)` did, twice: 24px encoded to 8 latent rather than 12.
#    That breaks spatial tiling outright, because the stitch assumes latent = pixel / ratio.
#    Concentrating all the temporal reduction on the one spatial-2 level is the only 4-level
#    arrangement satisfying both cumprods with no cropping.
#  * The widths must CHANGE somewhere, or `conv_shortcut` is never built and the residual
#    projection goes ungated. (16, 16, 32, 32) puts one at level 2.
#  * Every width must stay a multiple of 32: the ORIGINAL MiniMax module hardcodes
#    `num_groups=32` in `norm.py::get_group_norm_3d`, so a narrower fixture cannot even
#    construct the reference the conversion is cross-checked against. It also asserts
#    `time_stride in [1, 2]`, so the temporal reduction CANNOT be concentrated on one level.
#
# Those two together force the arrangement below: vae_ratio_t 4 needs two time-stride-2 levels,
# and each of them must also be spatial-stride-2, so the spatial cumprod is 4 rather than the
# previous 2. `fixture_config` in tests/common tracks it.
CH = 32
NORM_NUM_GROUPS = CH
BLOCK_OUT_CHANNELS = (CH, CH, CH, 2 * CH)
LAYERS_PER_BLOCK = 1
SPATIAL_DOWN = (1, 2, 2, 1)
TIME_DOWN = (1, 2, 2, 1)
Z_CHANNELS = 24
HEADS = 2
DIM_HEAD = 16
NUM_LAYERS = 2
FFN_MULT = 4
ROPE_THETA = 100.0
ROPE_DIM_RATIO = 0.75
DECODER_NORM_EPS = 1e-5

# The production per-channel de-normalization statistics, verbatim from `vae/config.json`.
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


# ── Reference (pre-conversion) model, used only to CHECK the conversion ──────────────────────────


def load_reference_class(snapshot: Path):
    """Import the snapshot's ``FL2VA/video_vae`` bundle as a package and return ``AutoencoderKLLegacy``.

    The bundle has no ``__init__.py`` but uses relative imports, so synthesize a package whose
    ``__path__`` points at it rather than copying files around.
    """
    video_vae = snapshot / "FL2VA" / "video_vae"
    if not (video_vae / "klvae.py").is_file():
        raise SystemExit(f"reference bundle not found under {video_vae}")
    pkg = types.ModuleType("mmh3_video_vae")
    pkg.__path__ = [str(video_vae)]
    sys.modules["mmh3_video_vae"] = pkg
    from mmh3_video_vae.klvae import AutoencoderKLLegacy  # noqa: E402
    from mmh3_video_vae.parallel import get_parallel_state  # noqa: E402

    state = get_parallel_state()
    if not state:
        state.update(
            {
                "group_size": 1, "group_rank": 0, "local_process_group": None,
                "sp_size": 1, "sp_rank": 0, "sp_enabled": False,
                "sp_process_group": None, "tp_size": 1, "tp_rank": 0,
            }
        )
    return AutoencoderKLLegacy


def build_reference(cls, token_drop: int):
    model = cls(
        in_channels=3,
        out_ch=3,
        ch=CH,
        embed_dim=Z_CHANNELS,
        z_channels=Z_CHANNELS,
        use_3d_conv=True,
        num_res_blocks=LAYERS_PER_BLOCK,
        ch_mult=[1, 1, 1, 1],
        space_down=list(SPATIAL_DOWN),
        time_down=list(TIME_DOWN),
        padding_mode="reflect",
        use_t_isolated_gn=True,
        causal_encoder=True,
        causal_decoder=False,
        use_vit_decoder=True,
        vit_decoder_kwargs={
            "heads": HEADS,
            "dim_head": DIM_HEAD,
            "num_layers": NUM_LAYERS,
            "norm_type": "rms_norm",
            "norm_affine": True,
            "qk_norm_type": "rms_norm",
            "qk_norm_affine": False,
            "ffn_activation_fn": "silu",
            "ffn_use_gated": True,
            "rope_theta": ROPE_THETA,
            "rope_dim_ratio": ROPE_DIM_RATIO,
        },
        pixel_norm_type="imagenet",
        clip_length=CLIP_LENGTH,
        token_drop=token_drop,
    )
    model.eval()
    return model


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
    """Re-randomize every decoder AND encoder parameter (sc-17148 added the encode half).

    Both implementations initialize `scale1`/`scale2` to zeros, which would collapse each block to
    an identity map and make the golden pass against a port with no attention and no feed-forward.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.startswith(
                ("decoder.", "post_quant_conv.", "encoder.", "quant_conv.")
            ):
                continue
            param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.35)


# ── The conversion, and its inverse ──────────────────────────────────────────────────────────────


def fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Inverse of the published QKV split: interleave `to_q`/`to_k`/`to_v` back per head.

    The conversion composes `reorder_interleaved_qkv` (raw per-head-interleaved rows ->
    `[q_all; k_all; v_all]`) with a contiguous-thirds `split_fused_qkv`. Net, published row
    `j`-projection row `h*dim_head + d` comes from fused row `h*(3*dim_head) + j*dim_head + d`, so
    the inverse is a per-head interleave. `crate::vae::split_fused_qkv` implements the forward
    direction and `fused_qkv_split_reproduces_the_published_split` pins it against these tensors.
    """
    rest = q.shape[1:]
    parts = [t.reshape(HEADS, DIM_HEAD, *rest) for t in (q, k, v)]
    return torch.stack(parts, dim=1).reshape(HEADS * 3 * DIM_HEAD, *rest).contiguous()


def unswap_gated_halves(published: torch.Tensor) -> torch.Tensor:
    """Inverse of the conversion's FFN half-swap: published `[value | gate]` -> source `[gate | value]`."""
    value, gate = published.chunk(2, dim=0)
    return torch.cat([gate, value], dim=0).contiguous()


def diffusers_to_reference_state(diffusers_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a diffusers decoder state dict back onto the MiniMax reference's parameter names/layouts."""
    out: dict[str, torch.Tensor] = {}
    pending_qkv: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in diffusers_sd.items():
        if not (key.startswith("decoder.") or key.startswith("post_quant_conv.")):
            continue
        if ".attn.to_q." in key or ".attn.to_k." in key or ".attn.to_v." in key:
            head, tag_suffix = key.split(".attn.to_")
            tag, suffix = tag_suffix.split(".", 1)
            pending_qkv.setdefault(f"{head}.attn.to_qkv.{suffix}", {})[tag] = tensor
            continue
        if key.endswith(".ff.net.0.proj.weight") or key.endswith(".ff.net.0.proj.bias"):
            suffix = key.rsplit(".", 1)[1]
            stem = key[: -len(f".net.0.proj.{suffix}")]
            out[f"{stem}.w1.{suffix}"] = unswap_gated_halves(tensor)
            continue
        name = key
        name = name.replace("decoder.proj_in.", "decoder.x_embedder.")
        name = name.replace(".attn.to_out.0.", ".attn.to_out.")
        name = name.replace(".ff.net.2.", ".ff.w2.")
        out[name] = tensor
    for name, parts in pending_qkv.items():
        out[name] = fuse_qkv(parts["q"], parts["k"], parts["v"])
    return out


def np32(t: torch.Tensor):
    return t.detach().to(torch.float32).numpy().copy()


def main() -> None:
    import diffusers

    snapshot = Path(
        os.environ.get("MINIMAX_H3_SNAPSHOT")
        or next(
            iter(
                sorted(
                    (hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3" / "snapshots").glob("*")
                )
            )
        )
    )

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    out: dict[str, np.ndarray] = {}

    # ---- the AUTHORITATIVE model: the official converted-checkpoint class ------------------
    model = build_diffusers(TOKEN_DROP)
    randomize(model, generator)

    # Spatial tiling is ON by default in `AutoencoderKLMiniMaxH3` (256x256 tiles). At fixture
    # dims the canvas is far smaller than one tile, so it is inert — assert that rather than
    # assume it, then decode through the untiled path the Rust port implements.
    assert model.use_tiling, "diffusers ships tiling on; this assertion guards a silent default flip"
    scale1 = model.decoder.transformer_blocks[0].scale1.detach()
    assert float(scale1.abs().max()) > 1e-3, "scale1 left at zero => blocks are identity"

    latents_mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, Z_CHANNELS, 1, 1, 1)
    latents_std = torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, Z_CHANNELS, 1, 1, 1)
    out["const.latents_mean"] = np32(latents_mean.flatten())
    out["const.latents_std"] = np32(latents_std.flatten())

    lat_h, lat_w = 3, 4

    with torch.no_grad():
        probe = torch.randn(1, Z_CHANNELS, 5, lat_h, lat_w, generator=generator)
        tiled = model.decode(probe, return_dict=False)[0]
        model.disable_tiling()
        untiled = model.decode(probe, return_dict=False)[0]
    tile_delta = float((tiled - untiled).abs().max())
    assert tile_delta == 0.0, (
        f"spatial tiling is NOT inert at fixture dims (max |delta| {tile_delta:.3e}); the fixture "
        "would encode a tiled decode the Rust port does not implement"
    )

    # The published weights, straight out of the official class — this IS the converted layout.
    diffusers_sd = {k: v.detach().to(torch.float32) for k, v in model.state_dict().items()}
    for key, tensor in diffusers_sd.items():
        if key.startswith(("decoder.", "post_quant_conv.", "encoder.", "quant_conv.")):
            out[key] = tensor.numpy().copy()

    # ---- CHECK the conversion by running the reference on inverse-converted weights ---------
    reference = build_reference(load_reference_class(snapshot), TOKEN_DROP)
    ref_state = diffusers_to_reference_state(diffusers_sd)
    missing, unexpected = reference.load_state_dict(ref_state, strict=False)
    # `decoder.mask_token` is the one legitimately-absent decoder entry: the reference registers it
    # as a zeros BUFFER and only replaces it with a learned parameter when `mask_enabled`, which is
    # false for this checkpoint. The official conversion drops it for that reason. Assert it really
    # is inert rather than waving it through.
    assert float(reference.decoder.mask_token.abs().max()) == 0.0, (
        "decoder.mask_token is not the inert zeros buffer; the inverse conversion would be dropping "
        "a real parameter"
    )
    unfilled = [
        k
        for k in missing
        if k.startswith(("decoder.", "post_quant_conv.")) and k != "decoder.mask_token"
    ]
    assert not unfilled, f"reference decoder parameters left unset by the inverse conversion: {unfilled}"
    assert not unexpected, f"inverse conversion produced keys the reference does not have: {unexpected}"
    reference.eval()

    # `AutoencoderKLLegacy.decode_temporal` is the reference's chunked entry point and the exact
    # counterpart of diffusers' `_decode`; `AutoencoderKLLegacy.decode` is the single-clip one.
    with torch.no_grad():
        ref_decoded = reference.decode_temporal(probe)
        dif_decoded = model.decode(probe, return_dict=False)[0]
    cross = float((ref_decoded - dif_decoded).abs().max() / dif_decoded.abs().max())
    assert cross < 1e-5, (
        f"the inverse conversion does not reproduce the reference (rel {cross:.3e}); the `src.` "
        "tensors below would not be the genuine pre-conversion form"
    )

    # A negative control on the assertion above: skipping the FFN un-swap must break it. This is
    # what proves the swap is real rather than a transform that happens to be inert.
    bad_state = dict(ref_state)
    for key in list(bad_state):
        if ".ff.w1." in key:
            bad_state[key] = unswap_gated_halves(bad_state[key])  # un-swap twice == no swap
    reference.load_state_dict(bad_state, strict=False)
    with torch.no_grad():
        bad_decoded = reference.decode_temporal(probe)
    swap_rel = float((bad_decoded - dif_decoded).abs().max() / dif_decoded.abs().max())
    assert swap_rel > 1e-2, (
        f"reading the FFN halves the source way changed the decode by only {swap_rel:.3e}; this "
        "fixture cannot gate the sc-18740 half-swap"
    )
    reference.load_state_dict(ref_state, strict=False)
    print(f"  conversion cross-check: rel {cross:.3e}; FFN half-swap moves the decode by {swap_rel:.3e}")

    # The pre-conversion tensors, so the Rust side can assert the fixture is in the CONVERTED
    # layout rather than the source one — the assertion that makes this whole file honest.
    for name, tensor in ref_state.items():
        if ".attn.to_qkv." in name or ".ff.w1." in name:
            out[f"src.{name}"] = np32(tensor)

    # ---- activations, from the official model ---------------------------------------------
    # (a) a single transformer block, in isolation.
    with torch.no_grad():
        seq = 9
        blk_in = torch.randn(1, seq, HEADS * DIM_HEAD, generator=generator)
        ids = torch.randn(1, seq, 3, generator=generator)
        rope = model.decoder.rope(ids)
        blk_out = model.decoder.transformer_blocks[0](blk_in, rope)
    out["in.block.hidden"] = np32(blk_in)
    out["in.block.ids"] = np32(ids)
    out["out.block.rope_cos"] = np32(rope[0])
    out["out.block.rope_sin"] = np32(rope[1])
    out["out.block.hidden"] = np32(blk_out)

    # (b) the bare ViT decoder forward (register tokens, cls zero token, pack/unpack).
    with torch.no_grad():
        vit_in = torch.randn(1, Z_CHANNELS, 5, lat_h, lat_w, generator=generator)
        vit_out = model.decoder(vit_in)
    out["in.vit.latent"] = np32(vit_in)
    out["out.vit.video"] = np32(vit_out)

    # (c) `decode_clip` = post_quant_conv -> ViT decoder.
    with torch.no_grad():
        dec_in = torch.randn(1, Z_CHANNELS, 5, lat_h, lat_w, generator=generator)
        dec_out = model.decoder(model.post_quant_conv(dec_in))
    out["in.decode.latent"] = np32(dec_in)
    out["out.decode.video"] = np32(dec_out)

    # (d) the chunked decode at token counts that straddle the clip_length-17 chunk boundary.
    #     tokens_chunk_size is 5, so 5 / 7 / 9 / 12 / 17 cover: exact-chunk, one full chunk with
    #     overlap, a padded remainder, two chunks (first blend), and three chunks.
    for n_tokens in (5, 7, 9, 12, 17):
        with torch.no_grad():
            z = torch.randn(1, Z_CHANNELS, n_tokens, lat_h, lat_w, generator=generator)
            # Per-channel de-normalization, exactly as the pipeline applies it before decode.
            dec = model.decode(z * latents_std + latents_mean, return_dict=False)[0]
        out[f"in.temporal{n_tokens}.latent"] = np32(z)
        out[f"out.temporal{n_tokens}.video"] = np32(dec)

    # ---- (e) the CNN ENCODER half (sc-17148) ------------------------------------------------
    # `fl2va` conditions a keyframe through the VAE as well as the vision tower, so the encode half
    # needs its own goldens. Four separate paths, because they diverge on the two axes that matter:
    # tiled vs untiled, and the T == 1 keyframe short circuit vs the chunked video path.
    #
    # ENC_TILE / ENC_OVERLAP are DELIBERATELY smaller than the shipped 256/64. A fixture canvas
    # large enough to tile at production tile size would not be committable, and an inert tiling
    # golden proves nothing — so the tile geometry is shrunk to exercise the SAME code path at
    # fixture scale. The Rust side takes the same two numbers as parameters for exactly this.
    ENC_TILE, ENC_OVERLAP = 16, 4
    ENC_H = ENC_W = 24

    with torch.no_grad():
        # (e1) the bare encoder stack, untiled: conv_in -> down_blocks -> norm_out -> conv_out.
        #      T = 5 > 1 so the frame-isolated GroupNorm and the temporal strides are both live;
        #      at T = 1 a plain 3-D GroupNorm is indistinguishable from the isolated one.
        enc_in = torch.randn(1, 3, 5, ENC_H, ENC_W, generator=generator)
        enc_out = model.encoder(enc_in)
    out["in.encoder.pixels"] = np32(enc_in)
    out["out.encoder.params"] = np32(enc_out)
    assert enc_out.shape[2] > 1, "the encoder golden must keep a temporal axis"

    with torch.no_grad():
        # (e2) `_encode_clip` UNTILED — encoder then quant_conv.
        model.disable_tiling()
        clip_in = torch.randn(1, 3, 5, ENC_H, ENC_W, generator=generator)
        clip_out = model._encode_clip(clip_in)
    out["in.encode_clip.pixels"] = np32(clip_in)
    out["out.encode_clip.params"] = np32(clip_out)

    with torch.no_grad():
        # (e3) `_encode_clip` TILED, at the shrunk tile geometry.
        model.enable_tiling(
            tile_sample_min_height=ENC_TILE,
            tile_sample_min_width=ENC_TILE,
            tile_sample_min_overlap_height=ENC_OVERLAP,
            tile_sample_min_overlap_width=ENC_OVERLAP,
        )
        assert model.use_tiling
        tiled_out = model._encode_clip(clip_in)
    out["out.encode_clip_tiled.params"] = np32(tiled_out)
    tile_enc_delta = float((tiled_out - clip_out).abs().max() / clip_out.abs().max())
    assert tiled_out.shape == clip_out.shape, (
        f"tiled {list(tiled_out.shape)} != untiled {list(clip_out.shape)}; the encoder is CROPPING "
        "rather than halving, so the stitch's latent = pixel / ratio assumption does not hold"
    )
    # The whole point of the tiled golden: it must DIFFER from the untiled one, or it is not
    # gating the blend at all. (The decode fixture asserts the opposite — that tiling is inert
    # there — which is why the two cannot share one probe.)
    assert tile_enc_delta > 1e-2, (
        f"tiled and untiled encode agree to {tile_enc_delta:.3e}; this fixture cannot gate the "
        "tile blend"
    )
    y_idx, y_len, y_ovl = model._split_tiles(ENC_H, ENC_TILE, ENC_OVERLAP)
    assert len(y_idx) > 1, "the tiled encode golden must actually span more than one tile"
    out["const.encode_tile"] = np.array([ENC_TILE, ENC_OVERLAP], dtype=np.int32)
    print(
        f"  encode tiling: {len(y_idx)} tiles starts={y_idx} overlaps={y_ovl}; "
        f"tiled vs untiled rel {tile_enc_delta:.3e}"
    )

    with torch.no_grad():
        # (e4) THE KEYFRAME PATH: a single frame. `_encode` short-circuits on num_frames == 1 —
        #      no clip padding, no chunking, no token_drop — and returns exactly ONE latent frame.
        model.disable_tiling()
        kf_in = torch.randn(1, 3, 1, ENC_H, ENC_W, generator=generator)
        kf_posterior = model.encode(kf_in, return_dict=False)[0]
        kf_mean, kf_logvar = torch.chunk(kf_posterior.parameters, 2, dim=1)
    assert kf_mean.shape[2] == 1, f"a keyframe must encode to ONE latent frame, got {kf_mean.shape}"
    out["in.encode_single.pixels"] = np32(kf_in)
    out["out.encode_single.mean"] = np32(kf_mean)
    out["out.encode_single.std"] = np32(torch.exp(0.5 * kf_logvar.clamp(-30.0, 20.0)))

    with torch.no_grad():
        # (e5) the CHUNKED video path: 17 frames = exactly one clip, then token_drop trims the
        #      tail. This is the path the keyframe short circuit deliberately avoids, kept so the
        #      two are gated against each other rather than only one being exercised.
        vid_in = torch.randn(1, 3, CLIP_LENGTH, ENC_H, ENC_W, generator=generator)
        vid_posterior = model.encode(vid_in, return_dict=False)[0]
        vid_mean = torch.chunk(vid_posterior.parameters, 2, dim=1)[0]
    out["in.encode_chunked.pixels"] = np32(vid_in)
    out["out.encode_chunked.mean"] = np32(vid_mean)
    print(
        f"  encode: 1 frame -> {list(kf_mean.shape)}; "
        f"{CLIP_LENGTH} frames -> {list(vid_mean.shape)}"
    )
    model.enable_tiling()

    # ---- token_drop = 0 (the two-pass alignment path: no overlap, single split) ------------
    model0 = build_diffusers(0)
    model0.disable_tiling()
    with torch.no_grad():
        model0.load_state_dict(model.state_dict())
    assert model0.token_overlap == 0 and model0.frame_overlap == 0
    for n_tokens in (5, 10):
        with torch.no_grad():
            z = torch.randn(1, Z_CHANNELS, n_tokens, lat_h, lat_w, generator=generator)
            dec = model0.decode(z * latents_std + latents_mean, return_dict=False)[0]
        out[f"in.drop0_temporal{n_tokens}.latent"] = np32(z)
        out[f"out.drop0_temporal{n_tokens}.video"] = np32(dec)

    # Non-degeneracy: a golden of constant / all-zero tensors is a false green.
    for key, value in out.items():
        assert np.isfinite(value).all(), f"{key} has non-finite entries"
        if key.startswith("out."):
            assert float(value.std()) > 1e-4, f"{key} is ~constant ({value.std()})"

    for name, model_ref in (("drop3", model), ("drop0", model0)):
        print(
            f"  {name}: tokens_chunk_size={model_ref.tokens_chunk_size} "
            f"token_overlap={model_ref.token_overlap} "
            f"frame_pre_padding={model_ref.frame_pre_padding} "
            f"frame_overlap={model_ref.frame_overlap}"
        )
    for key in sorted(out):
        if key.startswith(("in.", "out.")):
            print(f"  {key}: {list(out[key].shape)}")

    # Provenance. `tests/video_vae_parity.rs::fixture_provenance_records_the_converted_path`
    # asserts this, so a regeneration that silently reverts to the reference-module path fails
    # rather than passing — that regression is exactly sc-18740.
    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.AutoencoderKLMiniMaxH3",
        "reference_version": diffusers.__version__,
        "gated_ffn_layout": "value_first",
        "conversion_cross_check_rel": f"{cross:.6e}",
        "ffn_half_swap_rel": f"{swap_rel:.6e}",
        "encode_tile_blend_rel": f"{tile_enc_delta:.6e}",
        "halves": "decode+encode",
        "story": "sc-17140, corrected by sc-18740, encode half added by sc-17148",
    }

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/video_vae_decode.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path, metadata=meta)
    print(f"wrote {path} ({len(out)} tensors) from {meta['reference']} {meta['reference_version']}")


if __name__ == "__main__":
    main()
