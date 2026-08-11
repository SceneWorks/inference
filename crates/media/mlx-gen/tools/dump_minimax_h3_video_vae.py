"""MiniMax-H3 video-VAE decode parity fixtures (sc-17140).

Runs the **Apache-2.0 reference implementation shipped inside the MiniMax-H3 snapshot**
(``FL2VA/video_vae/*.py`` — ``klvae.py``'s ``AutoencoderKLLegacy`` and ``vae_vit.py``'s
``ViT3DDecoder``) at tiny dims and dumps inputs, weights and outputs. The Rust/MLX port is
dimension-parametric, so it runs the same tiny config and must reproduce these tensors.

Why tiny-but-real: the shipped decoder is a 36-layer / 2048-dim transformer (~5.2 B params,
10.4 GB) — far too large to commit. Every *structural* knob is preserved though, and the
temporal knobs are the REAL ones:

    clip_length 17, token_drop 3, vae_ratio_t 4
      -> frame_pre_padding  = (-17) % 4 = 3
      -> tokens_chunk_size  = ceil(17/4) = 5
      -> token_overlap      = (-3) % 5   = 2
      -> frame_overlap      = max(2*4 - 3, 0) = 5

which are identical to the production model. Only the *width* is shrunk (heads 32->2,
dim_head 64->16, layers 36->2, vae_ratio 16->2). ``rope_dim_ratio`` stays 0.75, so the tiny
model has rot_dim = int(16 * 0.75) = 12 < 16 and exercises the same PARTIAL-rotary path the
real one does (48 < 64). ``latent_channels`` stays 24 so the real 24-entry
``latents_mean``/``latents_std`` de-normalization is exercised verbatim.

The reference module's own ``_init_weights`` leaves ``scale1``/``scale2`` at ZERO, which makes
every transformer block an exact identity passthrough — a golden dumped that way would pass
against a model that never implements attention or the feed-forward at all. This script
therefore re-randomizes every parameter (including the scales and the norm gains) from a seeded
generator and asserts non-degeneracy before writing.

Weights are emitted in the **root / diffusers** naming of ``vae/`` in the published snapshot
(``proj_in``, ``attn.to_q|to_k|to_v``, ``attn.to_out.0``, ``ff.net.0.proj``, ``ff.net.2``),
because that is the form the Rust loader consumes for real weights. The reference module itself
uses the original fused form, so the fused ``attn.to_qkv`` tensors are dumped alongside under a
``src.`` prefix to pin the fused->split rule (see ``split_qkv``).

Requires torch + diffusers + torchvision. Run:

    MINIMAX_H3_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_video_vae.py
"""

from __future__ import annotations

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

# Tiny geometry. `space_down`/`time_down` cumprods give vae_ratio 2 / vae_ratio_t 4; the latter
# is the production value, which is what makes the chunk arithmetic identical to the real model.
CH = 32  # the CNN encoder's group norms are hardcoded to 32 groups
CH_MULT = [1, 1, 1, 1]
SPACE_DOWN = [2, 1, 1, 1]
TIME_DOWN = [1, 2, 2, 1]
Z_CHANNELS = 24
HEADS = 2
DIM_HEAD = 16
NUM_LAYERS = 2

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


def load_reference_package(snapshot: Path):
    """Import the snapshot's ``FL2VA/video_vae`` bundle as a package.

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


def build(cls, token_drop: int):
    model = cls(
        in_channels=3,
        out_ch=3,
        ch=CH,
        embed_dim=Z_CHANNELS,
        z_channels=Z_CHANNELS,
        use_3d_conv=True,
        num_res_blocks=1,
        ch_mult=CH_MULT,
        space_down=SPACE_DOWN,
        time_down=TIME_DOWN,
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
            "rope_theta": 100.0,
            "rope_dim_ratio": 0.75,
        },
        pixel_norm_type="imagenet",
        clip_length=CLIP_LENGTH,
        token_drop=token_drop,
    )
    model.eval()
    return model


def randomize(model, generator):
    """Re-randomize every decoder parameter.

    The reference initializes `scale1`/`scale2` to zeros, which would collapse each block to an
    identity map and make the golden pass against a port with no attention and no feed-forward.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not (name.startswith("decoder.") or name.startswith("post_quant_conv.")):
                continue
            param.copy_(torch.randn(param.shape, generator=generator, dtype=torch.float32) * 0.35)


def split_qkv(fused: torch.Tensor, heads: int, dim_head: int):
    """Split a fused ``to_qkv`` tensor into the root form's ``to_q``/``to_k``/``to_v``.

    The reference reads the fused projection as ``qkv.view(B, S, -1, 3*dim_head)`` followed by
    ``chunk(3, dim=-1)``, i.e. the output features are ordered PER HEAD as
    ``[q_h, k_h, v_h]`` — NOT as three contiguous ``heads*dim_head`` blocks. Output row
    ``h*(3*dim_head) + j*dim_head + d`` therefore belongs to projection ``j`` of head ``h``.
    """
    rest = fused.shape[1:]
    view = fused.reshape(heads, 3, dim_head, *rest)
    return tuple(view[:, j].reshape(heads * dim_head, *rest).contiguous() for j in range(3))


SRC_TO_ROOT = {
    "x_embedder": "proj_in",
    "attn.to_out": "attn.to_out.0",
    "ff.w1": "ff.net.0.proj",
    "ff.w2": "ff.net.2",
}


def to_root_name(name: str) -> str:
    for src, root in SRC_TO_ROOT.items():
        if name.startswith(src + "."):
            return root + name[len(src):]
        marker = "." + src + "."
        if marker in name:
            return name.replace(marker, "." + root + ".", 1)
    return name


def decoder_state_root(model):
    """Decoder + post_quant_conv parameters in the published root/diffusers naming."""
    out, fused = {}, {}
    for name, param in model.named_parameters():
        if name.startswith("post_quant_conv."):
            out[name] = param.detach().to(torch.float32).numpy().copy()
            continue
        if not name.startswith("decoder."):
            continue
        body = name[len("decoder."):]
        if ".attn.to_qkv." in "." + body:
            suffix = body.rsplit(".", 1)[1]  # weight | bias
            stem = body[: -len(".to_qkv." + suffix)]
            q, k, v = split_qkv(param.detach().to(torch.float32), HEADS, DIM_HEAD)
            for tag, tensor in (("to_q", q), ("to_k", k), ("to_v", v)):
                out[f"decoder.{stem}.{tag}.{suffix}"] = tensor.numpy().copy()
            fused[f"src.decoder.{body}"] = param.detach().to(torch.float32).numpy().copy()
            continue
        out["decoder." + to_root_name(body)] = param.detach().to(torch.float32).numpy().copy()
    out.update(fused)
    return out


def np32(t: torch.Tensor):
    return t.detach().to(torch.float32).numpy().copy()


def main() -> None:
    snapshot = Path(
        __import__("os").environ.get("MINIMAX_H3_SNAPSHOT")
        or next(
            iter(
                sorted(
                    (hf_hub_cache() / "models--MiniMaxAI--MiniMax-H3" / "snapshots").glob("*")
                )
            )
        )
    )
    cls = load_reference_package(snapshot)

    generator = torch.Generator().manual_seed(SEED)
    torch.manual_seed(SEED)

    out: dict[str, np.ndarray] = {}

    # ---- token_drop = 3 (the production value) --------------------------------------------
    model = build(cls, TOKEN_DROP)
    randomize(model, generator)
    out.update(decoder_state_root(model))

    scale1 = model.decoder.transformer_blocks[0].scale1
    assert float(scale1.abs().max()) > 1e-3, "scale1 left at zero => blocks are identity"

    latents_mean = torch.tensor(LATENTS_MEAN, dtype=torch.float32).view(1, Z_CHANNELS, 1, 1, 1)
    latents_std = torch.tensor(LATENTS_STD, dtype=torch.float32).view(1, Z_CHANNELS, 1, 1, 1)
    out["const.latents_mean"] = np32(latents_mean.flatten())
    out["const.latents_std"] = np32(latents_std.flatten())

    lat_h, lat_w = 3, 4

    # (a) a single transformer block, in isolation.
    with torch.no_grad():
        seq = 9
        blk_in = torch.randn(1, seq, HEADS * DIM_HEAD, generator=generator)
        ids = torch.randn(1, seq, 3, generator=generator)
        rope = model.decoder.pos_embed(ids)
        blk_out = model.decoder.transformer_blocks[0](blk_in, rope)
    out["in.block.hidden"] = np32(blk_in)
    out["in.block.ids"] = np32(ids)
    out["out.block.rope_cos"] = np32(rope[0])
    out["out.block.rope_sin"] = np32(rope[1])
    out["out.block.hidden"] = np32(blk_out)

    # (b) the bare ViT3DDecoder forward (register tokens, cls zero token, pack/unpack).
    with torch.no_grad():
        vit_in = torch.randn(1, Z_CHANNELS, 5, lat_h, lat_w, generator=generator)
        vit_out = model.decoder(vit_in)
    out["in.vit.latent"] = np32(vit_in)
    out["out.vit.video"] = np32(vit_out)

    # (c) `decode` = post_quant_conv -> ViT decoder.
    with torch.no_grad():
        dec_in = torch.randn(1, Z_CHANNELS, 5, lat_h, lat_w, generator=generator)
        dec_out = model.decode(dec_in)
    out["in.decode.latent"] = np32(dec_in)
    out["out.decode.video"] = np32(dec_out)

    # (d) `decode_temporal` at token counts that straddle the clip_length-17 chunk boundary.
    #     tokens_chunk_size is 5, so 5 / 7 / 9 / 12 / 17 cover: exact-chunk, one full chunk with
    #     overlap, a padded remainder, two chunks (first blend), and three chunks.
    for n_tokens in (5, 7, 9, 12, 17):
        with torch.no_grad():
            z = torch.randn(1, Z_CHANNELS, n_tokens, lat_h, lat_w, generator=generator)
            # Per-channel de-normalization, exactly as the pipeline applies it before decode.
            z_denorm = z * latents_std + latents_mean
            dec = model.decode_temporal(z_denorm)
        out[f"in.temporal{n_tokens}.latent"] = np32(z)
        out[f"out.temporal{n_tokens}.video"] = np32(dec)

    # ---- token_drop = 0 (the two-pass alignment path: no overlap, single split) ------------
    model0 = build(cls, 0)
    with torch.no_grad():
        for (_, dst), (_, src) in zip(
            model0.named_parameters(), model.named_parameters()
        ):
            dst.copy_(src)
    assert model0.token_overlap == 0 and model0.frame_overlap == 0
    for n_tokens in (5, 10):
        with torch.no_grad():
            z = torch.randn(1, Z_CHANNELS, n_tokens, lat_h, lat_w, generator=generator)
            dec = model0.decode_temporal(z * latents_std + latents_mean)
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

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/video_vae_decode.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(out, path)
    print(f"wrote {path} ({len(out)} tensors)")


if __name__ == "__main__":
    main()
