"""FLUX 3 Image Klein autoencoder golden (sc-26234, delegated AE leaf of epic sc-26233).

Builds the miniature autoencoder through the FROZEN upstream `flux3.autoencoder.AutoEncoder`
(module constants shrunk by `_flux3_common.patch_autoencoder_constants`, weights seeded by
`_flux3_common.tiny_autoencoder`), writes it in the released layout as
`tests/fixtures/tiny-snapshot/ae.safetensors`, then records one encode/decode round trip plus one
decode of seeded random tokens as `tests/fixtures/flux3_ae_golden.safetensors`.

Every tensor keeps upstream's own layout (the Rust test transposes to NHWC with
`common::nchw_to_nhwc` / `nhwc_to_nchw`):

    image         [3, H, W]        f32   seeded, in [-1, 1]            (upstream `encode` input)
    tokens        [h*w, 4*EMBED]   f32   `AutoEncoder.encode(image)[0]`, row-major id order
    ids           [h*w, 4]         i32   `AutoEncoder.encode(image)[1]` = `latent_ids(h, w)`
    decoded       [3, H, W]        f32   `AutoEncoder.decode(tokens, ids)` (unclamped)
    tokens_rand   [h*w, 4*EMBED]   f32   seeded N(0, 1) tokens
    decoded_rand  [3, H, W]        f32   `AutoEncoder.decode(tokens_rand, ids)` (unclamped)

H x W is 32 x 48 (h x w = 2 x 3 tokens, 16 pixels per token side): non-square so a row/column
transposition anywhere in the port is caught, and small enough to run on the hosted macOS lane
with no weights. The oracle runs on CPU in f32 (upstream loads the autoencoder `float32` and
decodes outside autocast), so the Rust gate is a native f32 comparison on the MLX CPU stream.

Run from the repository (never from the source tree), inside the isolated oracle venv:

    FLUX3_SRC=/path/to/flux-3-image-klein python -I crates/media/mlx-gen/tools/dump_flux3_ae_golden.py

Oracle used for the committed fixtures: torch 2.14.1+cpu, safetensors 0.8.0, Python 3.12.11.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# `python -I` keeps the script directory off `sys.path`; the shared `_flux3_common` / `_paths`
# helpers live beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from _flux3_common import (  # noqa: E402
    AE_LATENT_CHANNELS,
    FIXTURE_DIR,
    FLUX3_COMMIT,
    import_upstream,
    save_safetensors,
    tiny_autoencoder,
    write_autoencoder_fixture,
)

GOLDEN = FIXTURE_DIR / "flux3_ae_golden.safetensors"
# Pixels: multiples of DOWNSAMPLING_RATIO (16). Deliberately non-square.
IMAGE_H = 32
IMAGE_W = 48
SEED = 26234


def main() -> None:
    import_upstream()
    autoencoder = importlib.import_module("flux3.autoencoder")
    model = tiny_autoencoder(autoencoder)
    write_autoencoder_fixture(model)

    ratio = autoencoder.DOWNSAMPLING_RATIO
    h, w = autoencoder.latent_shape(IMAGE_H, IMAGE_W)
    assert (h, w) == (IMAGE_H // ratio, IMAGE_W // ratio)

    g = torch.Generator().manual_seed(SEED)
    image = torch.rand((3, IMAGE_H, IMAGE_W), generator=g) * 2.0 - 1.0
    tokens, ids = model.encode(image)
    assert tuple(tokens.shape) == (h * w, AE_LATENT_CHANNELS), tokens.shape
    assert tuple(ids.shape) == (h * w, 4), ids.shape
    expected_ids = autoencoder.latent_ids(h, w)
    assert torch.equal(ids, expected_ids)
    decoded = model.decode(tokens, ids)
    assert tuple(decoded.shape) == (3, IMAGE_H, IMAGE_W), decoded.shape

    tokens_rand = torch.randn((h * w, AE_LATENT_CHANNELS), generator=g)
    decoded_rand = model.decode(tokens_rand, ids)

    save_safetensors(
        GOLDEN,
        {
            "image": image,
            "tokens": tokens,
            "ids": ids.to(torch.int32),
            "decoded": decoded,
            "tokens_rand": tokens_rand,
            "decoded_rand": decoded_rand,
        },
        metadata={
            "upstream_commit": FLUX3_COMMIT,
            "image_hw": f"{IMAGE_H}x{IMAGE_W}",
            "latent_hw": f"{h}x{w}",
            "latent_channels": AE_LATENT_CHANNELS,
            "seed": SEED,
            "torch": torch.__version__,
        },
    )
    print(
        f"  image {tuple(image.shape)} -> tokens {tuple(tokens.shape)} (h, w)=({h}, {w}) "
        f"-> decoded {tuple(decoded.shape)}; |decoded| max {decoded.abs().max().item():.4f}, "
        f"|decoded_rand| max {decoded_rand.abs().max().item():.4f}"
    )


if __name__ == "__main__":
    main()
