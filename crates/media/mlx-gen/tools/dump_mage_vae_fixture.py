#!/usr/bin/env python3
"""Dump the committed, weights-free Mage-VAE decode fixture (sc-14039).

Unlike `dump_mage_flow_golden.py --stage vae`, this needs **no model weights and no HF cache**: it
instantiates a *tiny* randomly-initialised `_Decoder` + `_DConvDenoiser` straight from the vendored
reference (`_vendor/mage_flow/models/modules/mage_vae.py`) and records both its parameters and its
decode output. The Rust port replays the same tensors through
`MageVae::from_weights_with_shape`, so the entire decode path — CoD decoder, the 21-block DiCo
stack's shape, the two *different* 8192-channel orderings, the DCT position code, `SimpleMLPAdaLN`
and the unfold/fold round trip — is gated by `cargo test` on a fresh clone.

Output: `mlx-gen-mage/tests/fixtures/mage_vae_tiny.safetensors` (committed, ~2 MB).

Run:

    PYTHONPATH=crates/media/mlx-gen/_vendor \\
      python3 crates/media/mlx-gen/tools/dump_mage_vae_fixture.py

The tiny shape is deliberately *not* a scaled-down copy of the published one:

* `attn_tile = 4` against a `6 × 6` latent forces the `AttnBlock`'s **replicate** padding and the
  crop back off — the published 1024²/2048² geometries divide evenly and never exercise it, while
  256² does, so the fixture keeps that path covered without weights.
* `patch = 4` keeps the unfold/fold layout small enough to reason about by hand.
* `max_freqs` stays 8 because `_DConvDenoiser.__init__` hard-codes it (`mage_vae.py:475`); the
  production `patch = 16` table is dumped alongside so the real geometry is pinned too.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(
    HERE, "..", "mlx-gen-mage", "tests", "fixtures", "mage_vae_tiny.safetensors"
)

# Tiny shape — mirrored by `MageVaeShape` in `tests/vae_decode_fixture.rs`.
PATCH = 4
HIDDEN = 64
HIDDEN_X = 8
IN_CHANNELS = 3
BOTTLENECK = 16
NUM_COND_BLOCKS = 2
NUM_BLOCKS = 3  # dec_net res blocks = NUM_BLOCKS - NUM_COND_BLOCKS = 1
ATTN_TILE = 4
LATENT_HW = 6  # 6 % 4 != 0 -> replicate padding in AttnBlock
SEED = 20260724


def _randomise(module: torch.nn.Module, gen: torch.Generator) -> None:
    """Give every parameter a non-degenerate value.

    PyTorch's defaults leave `LayerNorm`/`GroupNorm`/`RMSNorm` at `weight = 1, bias = 0`, which
    would make a dropped affine term or a swapped weight/bias invisible in the fixture. Every
    parameter is overwritten so each one is load-bearing.
    """
    for _, p in module.named_parameters():
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=gen, dtype=torch.float32) * 0.15)


def main() -> int:
    from mage_flow.models.modules import mage_vae as mv

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    torch.manual_seed(SEED)

    denoiser = mv._DConvDenoiser(
        patch_size=PATCH,
        in_channels=IN_CHANNELS,
        hidden_size=HIDDEN,
        hidden_size_x=HIDDEN_X,
        mlp_ratio=4.0,
        num_blocks=NUM_BLOCKS,
        num_cond_blocks=NUM_COND_BLOCKS,
        bottleneck_dim=BOTTLENECK,
    ).eval()
    _randomise(denoiser, gen)

    # `_Decoder` hard-codes `patch_size=32` for its two AttnBlocks (`mage_vae.py:384,386`).
    # Overriding the attribute (not the vendored source) is what lets the fixture cover the
    # replicate-padding branch at a tiny latent size.
    for m in denoiser.y_embedder.decoder.block:
        if isinstance(m, mv.AttnBlock):
            m.patch_size = ATTN_TILE

    latent = torch.randn(
        (1, BOTTLENECK, LATENT_HW, LATENT_HW), generator=gen, dtype=torch.float32
    )

    with torch.no_grad():
        # `MageVAE.decode` verbatim (`mage_vae.py:625-633`).
        cond = denoiser.y_embedder.decoder(latent)
        h = latent.shape[2] * PATCH
        w = latent.shape[3] * PATCH
        noise = torch.zeros(1, IN_CHANNELS, h, w, dtype=torch.float32)
        t = torch.zeros(1, dtype=torch.float32)
        decoded = denoiser.forward(noise, t, cond)

        # Sub-boundary probes so a failure localises instead of just "the image is wrong".
        c_t0 = denoiser.t_embedder(t)
        adaln = torch.stack(
            [blk.adaLN_modulation(c_t0)[0] for blk in denoiser.blocks], dim=0
        )
        dct_tiny = denoiser.x_embedder.fetch_pos(PATCH, torch.device("cpu"), torch.float32)
        dct_published = denoiser.x_embedder.fetch_pos(16, torch.device("cpu"), torch.float32)

    tensors: dict[str, np.ndarray] = {}
    for name, p in denoiser.state_dict().items():
        # Production naming: the published checkpoint stores the denoiser under `pipeline.*`
        # (`mage_vae.py:579`), and the Rust loader takes that prefix.
        tensors[f"pipeline.{name}"] = p.detach().cpu().numpy().astype(np.float32)

    tensors["fixture.latent"] = latent.numpy().astype(np.float32)
    tensors["fixture.cond"] = cond.numpy().astype(np.float32)
    tensors["fixture.decoded"] = decoded.numpy().astype(np.float32)
    tensors["fixture.t_embed_zero"] = c_t0.numpy().astype(np.float32)
    tensors["fixture.adaln_zero"] = adaln.numpy().astype(np.float32)
    tensors["fixture.dct_tiny"] = dct_tiny.numpy().astype(np.float32)
    tensors["fixture.dct_published"] = dct_published.numpy().astype(np.float32)
    tensors["fixture.shape"] = np.array(
        [
            PATCH,
            HIDDEN,
            HIDDEN_X,
            IN_CHANNELS,
            BOTTLENECK,
            NUM_COND_BLOCKS,
            NUM_BLOCKS - NUM_COND_BLOCKS,
            8,  # max_freqs, hard-coded at mage_vae.py:475
            ATTN_TILE,
            LATENT_HW,
        ],
        dtype=np.int32,
    )

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    save_file(
        tensors,
        os.path.normpath(OUT),
        metadata={
            "source": "_vendor/mage_flow/models/modules/mage_vae.py (microsoft/Mage, MIT)",
            "seed": str(SEED),
            "torch": torch.__version__,
            "note": "tiny randomly-initialised decode path; no model weights involved",
        },
    )
    print(f"wrote {os.path.normpath(OUT)} ({len(tensors)} tensors)")
    print(f"  decoded: {tuple(decoded.shape)} "
          f"min={decoded.min():.4f} max={decoded.max():.4f} std={decoded.std():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
