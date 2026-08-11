"""MiniMax-H3 DiT block + MM-RoPE parity fixtures (sc-17144).

Runs the **official diffusers** ``MiniMaxH3Transformer3DModel`` — i.e. the converted-checkpoint
layout production loads — at tiny dims and dumps inputs, weights and outputs. The Rust/MLX port is
dimension-parametric, so it runs the same tiny config and must reproduce these tensors.

Read ``mlx-gen-minimax-h3/src/layout.rs`` before changing anything here
---------------------------------------------------------------------

``scripts/convert_minimax_h3_to_diffusers.py`` is not a rename table. For the DiT it applies **two**
transforms whose results are shape-identical to their inputs, so nothing structural can see either
one:

1. the gated FFN's fused ``mlp.fc1`` becomes ``ff.net.0.proj`` with its two row halves **swapped**
   (``[gate | value]`` -> ``[value | gate]``), because diffusers' ``SwiGLU`` reads value-first;
2. the fused ``attn.qkv_proj`` is de-interleaved per head into ``[q_all; k_all; v_all]`` and then
   split into contiguous thirds.

sc-18740 is what happens when a fixture is dumped through a pure rename instead: the golden and the
loader share the *source* layout, agree with each other, and production — which loads the genuinely
converted checkpoint — is wrong by a half-swap. That shipped a functionally wrong 36-layer video-VAE
decoder past a fully green parity suite.

    A golden and a loader that share a layout prove only that they agree with each other.

So the goldens here come from ``MiniMaxH3Transformer3DModel``, and the ``src.``-prefixed tensors are
built by inverting the conversion and then **re-running the official conversion functions forward**
(imported from the pinned diffusers checkout, not reimplemented) to check they reproduce the
published tensors byte-for-byte. That is what makes them evidence rather than a restatement.

Why tiny-but-real
-----------------

The shipped DiT is 50 layers x 5376 hidden x 56 heads x 128 (~33 B, 62 GB) — uncommittable. Every
*structural* property is preserved:

* ``heads * head_dim`` (96) stays **wider** than ``hidden_size`` (64), as 7168 > 5376 in the real
  model, so a ``hidden_size / heads`` derivation still fails;
* ``2 * 3 * rope_freq_dim`` (12) stays **smaller** than ``attention_head_dim`` (24), as 96 < 128, so
  the PARTIAL-rotary path is the one exercised;
* the packed sequence carries all three modalities at more than one timestep, so the
  ``timestep_index * 3 + token_tag`` AdaLN row addressing is load-bearing;
* the position grid comes from the reference's own ``build_packed_sequence``, so the audio
  convention (no height; channel-major; pinned to the width grid's extremes) is dumped rather than
  described.

Requires ``diffusers`` from ``main`` — ``MiniMaxH3`` landed in PR #14355 (merged 2026-08-05) and is
in no tagged release, so a released ``pip install diffusers`` has none of these classes.

    ~/.minimax-h3-spike/venv/bin/python tools/dump_minimax_h3_dit.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from _paths import fixture

# --- the tiny geometry -------------------------------------------------------------------------
HIDDEN = 64
HEADS = 4
HEAD_DIM = 24  # inner = 96 > hidden 64, mirroring 7168 > 5376
LAYERS = 2
REFINER_LAYERS = 2
FFN_DIM = 32
ROPE_FREQ_DIM = 2  # rotary = 12 of 24, mirroring 96 of 128
ROPE_THETA = 10_000.0
TIME_EMBED_DIM = 48
TIME_EMBED_HIDDEN = 64
FREQ_DIM = 16
TEXT_DIM = 40
IN_CHANNELS = 24
AUDIO_IN_CHANNELS = 32
PATCH_SIZE = (1, 2, 2)
NORM_EPS = 1e-5

# The packed layout the goldens are computed over.
NUM_TEXT_TOKENS = 5
NUM_LATENT_FRAMES = 3
LATENT_HEIGHT = 4
LATENT_WIDTH = 6
NUM_AUDIO_LATENTS = 3
AUDIO_CHANNELS = 2
# Modality tags: video 0, text 1, audio 2.
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


def np32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).contiguous().numpy()


def conversion_module():
    """Import ``convert_minimax_h3_to_diffusers.py`` from the diffusers source checkout.

    The conversion transforms are the *contract* this fixture pins, so they are imported from
    upstream rather than reimplemented here — a reimplementation could agree with a wrong port.
    ``MINIMAX_H3_CONVERT_SCRIPT`` overrides the search; otherwise the uv/pip git checkout that
    provided the installed ``diffusers`` is used.
    """
    if explicit := os.environ.get("MINIMAX_H3_CONVERT_SCRIPT"):
        candidates = [Path(explicit)]
    else:
        import diffusers

        root = Path(diffusers.__file__).resolve().parents[1]
        candidates = [root / "scripts" / "convert_minimax_h3_to_diffusers.py"]
        candidates += sorted(
            (Path.home() / ".cache" / "uv" / "git-v0" / "checkouts").glob(
                "*/*/scripts/convert_minimax_h3_to_diffusers.py"
            )
        )
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("mmh3_convert", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"  conversion transforms imported from {path}")
            return module
    raise SystemExit(
        "convert_minimax_h3_to_diffusers.py not found; set MINIMAX_H3_CONVERT_SCRIPT to it. "
        f"Looked in: {[str(c) for c in candidates]}"
    )


def build_model():
    from diffusers import MiniMaxH3Transformer3DModel

    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=HEADS,
        attention_head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        num_layers=LAYERS,
        num_refiner_layers=REFINER_LAYERS,
        ffn_dim=FFN_DIM,
        in_channels=IN_CHANNELS,
        audio_in_channels=AUDIO_IN_CHANNELS,
        patch_size=PATCH_SIZE,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        time_embed_hidden_dim=TIME_EMBED_HIDDEN,
        time_embed_dim=TIME_EMBED_DIM,
        rope_freq_dim=ROPE_FREQ_DIM,
        rope_theta=ROPE_THETA,
        norm_eps=NORM_EPS,
        qk_norm_eps=NORM_EPS,
        final_norm_eps=NORM_EPS,
    )
    model.eval()
    return model


def randomize(model, generator):
    """Re-randomize every parameter.

    An RMSNorm initializes to all-ones and a `nn.Linear` bias to zero; both would make the tensor
    unobservable in a mutation check and let a port that ignored it pass. Weights are drawn small so
    a 2-layer stack stays numerically tame.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("norm_q.weight") or name.endswith("norm_k.weight"):
                # qk-norm weights sit around 1; a scatter about 1 keeps the norm meaningful while
                # still making the tensor observable.
                param.copy_(1.0 + 0.25 * torch.randn(param.shape, generator=generator))
            elif ".norm" in name or name.endswith("final_norm.weight"):
                param.copy_(1.0 + 0.25 * torch.randn(param.shape, generator=generator))
            else:
                param.copy_(0.2 * torch.randn(param.shape, generator=generator))


def packed_layout():
    """The reference's own packed layout, so the position conventions are dumped, not described."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
    )

    # Text rows are tagged 1 here (no keyframe vision block in a t2va request).
    text_token_tags = torch.full((NUM_TEXT_TOKENS,), TEXT_TAG, dtype=torch.long)
    (
        position_ids,
        token_tags,
        video_indices,
        audio_indices,
        text_indices,
        num_condition_rows,
        num_condition_audio_rows,
    ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=NUM_LATENT_FRAMES,
        latent_height=LATENT_HEIGHT,
        latent_width=LATENT_WIDTH,
        num_audio_latents=NUM_AUDIO_LATENTS,
        patch_size=PATCH_SIZE,
        audio_channels=AUDIO_CHANNELS,
        audio_tag=AUDIO_TAG,
        video_tag=VIDEO_TAG,
        keyframe_anchors=(),
    )
    assert num_condition_rows == 0 and num_condition_audio_rows == 0
    return position_ids, token_tags, video_indices, audio_indices, text_indices


def main() -> None:
    import diffusers

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(17144)

    convert = conversion_module()
    model = build_model()
    randomize(model, generator)

    out: dict[str, np.ndarray] = {}

    # ---- the packed layout, straight from the reference ---------------------------------------
    position_ids, token_tags, video_indices, audio_indices, text_indices = packed_layout()
    seq_len = position_ids.shape[0]
    out["layout.position_ids"] = np32(position_ids)
    out["layout.token_tags"] = token_tags.to(torch.float32).numpy()
    out["layout.video_indices"] = video_indices.to(torch.float32).numpy()
    out["layout.audio_indices"] = audio_indices.to(torch.float32).numpy()
    out["layout.text_indices"] = text_indices.to(torch.float32).numpy()

    # ---- MM-RoPE ------------------------------------------------------------------------------
    with torch.no_grad():
        cos, sin = model.rope(position_ids.to(torch.float32))
    assert cos.shape == (seq_len, 2 * 3 * ROPE_FREQ_DIM), cos.shape
    out["out.rope_cos"] = np32(cos)
    out["out.rope_sin"] = np32(sin)

    # ---- the AdaLN row addressing --------------------------------------------------------------
    # Two distinct timesteps so `timestep_indices * 3 + token_tags` addresses more than one block of
    # the modulation table; a single-timestep golden would pass against a port that ignored the
    # timestep axis entirely.
    timestep = torch.tensor([0.25, 0.8], dtype=torch.float32)
    timestep_indices = torch.zeros(seq_len, dtype=torch.long)
    timestep_indices[audio_indices] = 1
    adaln_indices = timestep_indices * 3 + token_tags
    out["in.timestep"] = np32(timestep)
    out["layout.timestep_indices"] = timestep_indices.to(torch.float32).numpy()
    out["layout.adaln_indices"] = adaln_indices.to(torch.float32).numpy()

    with torch.no_grad():
        temb = model.time_embedder(model.time_proj(timestep))
    assert temb.shape == (2, TIME_EMBED_DIM), temb.shape
    out["in.temb"] = np32(temb)

    block = model.transformer_blocks[0]
    with torch.no_grad():
        modulation = block.adaln_proj(temb)
    for name, tensor in zip(
        ("shift_msa", "scale_msa", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp"), modulation
    ):
        assert tensor.shape == (2 * 3, HIDDEN), tensor.shape
        out[f"out.modulation.{name}"] = np32(tensor)

    # ---- one transformer block -----------------------------------------------------------------
    hidden = 0.5 * torch.randn(1, seq_len, HIDDEN, generator=generator)
    out["in.block.hidden"] = np32(hidden)
    with torch.no_grad():
        block_out = block(hidden, temb, adaln_indices, (cos, sin))
    out["out.block.hidden"] = np32(block_out)

    # The bare attention, so a qk-norm / rotary error localizes instead of only showing up folded
    # into the block's two residuals.
    attn_in = 0.5 * torch.randn(1, seq_len, HIDDEN, generator=generator)
    out["in.attn.hidden"] = np32(attn_in)
    with torch.no_grad():
        out["out.attn.hidden"] = np32(block.attn(attn_in, (cos, sin)))
        # ...and the same input with NO rotary, which is the refiner's configuration.
        out["out.attn.norope"] = np32(block.attn(attn_in, None))

    # ---- the token refiner ----------------------------------------------------------------------
    text_context = 0.5 * torch.randn(1, NUM_TEXT_TOKENS, TEXT_DIM, generator=generator)
    out["in.refiner.context"] = np32(text_context)
    with torch.no_grad():
        embedded = model.context_embedder(text_context)
        out["in.refiner.hidden"] = np32(embedded)
        out["out.refiner.block0"] = np32(model.token_refiner.refiner_blocks[0](embedded))
        out["out.refiner.hidden"] = np32(model.token_refiner(embedded))

    # ---- weights, in the published (converted) naming ------------------------------------------
    state = model.state_dict()
    for key, tensor in state.items():
        out[key] = np32(tensor)

    # ---- the conversion, checked rather than trusted ---------------------------------------------
    # Invert the official conversion to recover the pre-conversion (original MiniMax) tensors, then
    # re-run the OFFICIAL forward transforms on them and require the published tensors back exactly.
    config = {
        "num_attention_heads": HEADS,
        "attention_head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "ffn_dim": FFN_DIM,
        "time_embed_dim": TIME_EMBED_DIM,
        "in_channels": IN_CHANNELS,
        "patch_size": list(PATCH_SIZE),
        "num_layers": LAYERS,
        "num_refiner_layers": REFINER_LAYERS,
    }
    inner = HEADS * HEAD_DIM
    prefixes = [f"transformer_blocks.{i}" for i in range(LAYERS)]
    prefixes += [f"token_refiner.refiner_blocks.{i}" for i in range(REFINER_LAYERS)]

    for prefix in prefixes:
        # (a) fused QKV. The reference's in-memory layout is `[q_all; k_all; v_all]`; the raw shard
        # layout is per-head interleaved. Build both, and check the official transforms agree.
        q = state[f"{prefix}.attn.to_q.weight"]
        k = state[f"{prefix}.attn.to_k.weight"]
        v = state[f"{prefix}.attn.to_v.weight"]
        reference_layout = torch.cat([q, k, v], dim=0)
        raw = (
            reference_layout.reshape(3, HEADS, HEAD_DIM, HIDDEN)
            .permute(1, 0, 2, 3)
            .reshape(3 * inner, HIDDEN)
            .contiguous()
        )
        assert torch.equal(
            convert.reorder_interleaved_qkv(raw, HEADS, HEAD_DIM), reference_layout
        ), f"{prefix}: the synthesized raw interleave does not round-trip through the official reorder"
        split = convert.split_fused_qkv(reference_layout, HEADS, HEAD_DIM)
        for part, want in zip(split, (q, k, v)):
            assert torch.equal(part, want), f"{prefix}: official split_fused_qkv disagrees"
        out[f"src.{prefix}.attn.qkv_proj.weight"] = np32(raw)
        out[f"src.{prefix}.attn.qkv_proj.reordered"] = np32(reference_layout)

        # (b) the gated FFN half-swap. `mlp.fc1` is `[gate | value]`; the published
        # `ff.net.0.proj` is `[value | gate]`.
        published = state[f"{prefix}.ff.net.0.proj.weight"]
        value, gate = published.chunk(2, dim=0)
        source = torch.cat([gate, value], dim=0).contiguous()
        converted = convert.convert_transformer_key(
            f"{prefix.replace('token_refiner.refiner_blocks', 'token_refiner.blocks')}"
            ".mlp.fc1.weight",
            source,
            config,
        )
        assert len(converted) == 1, converted
        target_key, tensor = converted[0]
        assert target_key == f"{prefix}.ff.net.0.proj.weight", target_key
        assert torch.equal(tensor, published), f"{prefix}: official fc1 conversion disagrees"
        assert not torch.equal(source, published), f"{prefix}: the half-swap is a no-op here"
        out[f"src.{prefix}.mlp.fc1.weight"] = np32(source)

    # A measured negative control for the half-swap, so the Rust suite can assert the mutation is
    # gateable rather than assuming it.
    with torch.no_grad():
        swapped = model.state_dict()
        for prefix in [f"transformer_blocks.{i}" for i in range(LAYERS)]:
            key = f"{prefix}.ff.net.0.proj.weight"
            a, b = swapped[key].chunk(2, dim=0)
            swapped[key] = torch.cat([b, a], dim=0).contiguous()
        mutant = build_model()
        mutant.load_state_dict(swapped)
        mutant.eval()
        swap_out = mutant.transformer_blocks[0](hidden, temb, adaln_indices, (cos, sin))
    swap_rel = float(
        (swap_out - block_out).abs().max() / block_out.abs().max().clamp_min(1e-12)
    )
    assert swap_rel > 1e-2, f"the FFN half-swap moves the block by only {swap_rel:.3e}"

    # ---- non-degeneracy -------------------------------------------------------------------------
    for key, value in out.items():
        assert np.isfinite(value).all(), f"{key} has non-finite entries"
        if key.startswith("out.") and value.size > 1:
            assert float(value.std()) > 1e-4, f"{key} is ~constant ({value.std()})"

    for key in sorted(out):
        if key.startswith(("in.", "out.", "layout.")):
            print(f"  {key}: {list(out[key].shape)}")
    print(f"  seq_len={seq_len} (text {NUM_TEXT_TOKENS}, audio {NUM_AUDIO_LATENTS * AUDIO_CHANNELS})")
    print(f"  ffn half-swap negative control: {swap_rel:.6e}")

    meta = {
        "provenance": "converted-checkpoint",
        "reference": "diffusers.MiniMaxH3Transformer3DModel",
        "reference_version": diffusers.__version__,
        "gated_ffn_layout": "value_first",
        "qkv_transform": "contiguous_thirds_of_reordered",
        "ffn_half_swap_rel": f"{swap_rel:.6e}",
        "story": "sc-17144",
    }

    path = fixture("mlx-gen-minimax-h3/tests/fixtures/dit_block.safetensors")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: torch.from_numpy(v) for k, v in out.items()}, path, metadata=meta)
    print(f"wrote {path} ({len(out)} tensors) from {meta['reference']} {meta['reference_version']}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
