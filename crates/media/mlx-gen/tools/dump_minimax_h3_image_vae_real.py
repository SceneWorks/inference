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

The output is a few MB and is deliberately **not** committed (real-weight bytes). The
``minimax-h3-image`` real-weights profile produces it in the job itself, in the reviewed
hash-locked Mage reference environment (torch + diffusers 0.38.0); run it the same way by hand:

    MINIMAX_H3_IMAGE_SNAPSHOT=<snapshot-root> python3 tools/dump_minimax_h3_image_vae_real.py \
        --out <dir>/minimax-h3-image-vae-reference.safetensors

The reference class
-------------------

``AutoencoderKLMiniMaxH3Image`` landed on diffusers ``main`` at ``REFERENCE_COMMIT`` and is in no
released wheel yet. When the installed diffusers does not ship the module, this producer loads that
exact upstream file — downloaded from the pinned commit and refused unless its SHA-256 is
``REFERENCE_MODULE_SHA256`` — as ``diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image``
on top of the installed package. Everything it imports (``AutoencoderMixin``, ``FeedForward``'s
``swiglu``, ``dispatch_attention_fn``, ``AttentionModuleMixin``) exists unchanged in diffusers
0.38.0. The module's identity is recorded in the output metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch
from safetensors.numpy import save_file

SEED = 7089
MODULE = "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image"
REFERENCE_COMMIT = "875373ac7f5856071e345a3e394c74efccc6b9e6"
REFERENCE_MODULE_SHA256 = "82da645ee07b0c0443aacd64f160c14c3fe928d61d64023e0102ecbe99557ce1"
REFERENCE_MODULE_URL = (
    f"https://raw.githubusercontent.com/huggingface/diffusers/{REFERENCE_COMMIT}/"
    "src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_image.py"
)
SMALL_HW = (4, 4)
TILED_HW = (24, 40)


def reference_class():
    """Return ``(AutoencoderKLMiniMaxH3Image, source)`` — installed, or the pinned upstream file."""
    try:
        module = importlib.import_module(MODULE)
        return module.AutoencoderKLMiniMaxH3Image, "installed"
    except ModuleNotFoundError:
        pass
    with urllib.request.urlopen(REFERENCE_MODULE_URL, timeout=60) as response:
        body = response.read()
    digest = hashlib.sha256(body).hexdigest()
    if digest != REFERENCE_MODULE_SHA256:
        raise SystemExit(f"{REFERENCE_MODULE_URL} has sha256 {digest}, expected {REFERENCE_MODULE_SHA256}")
    path = Path(tempfile.mkdtemp()) / "autoencoder_kl_minimax_h3_image.py"
    path.write_bytes(body)
    import diffusers.models.autoencoders  # noqa: F401 — the parent package the module's relative imports resolve against

    spec = importlib.util.spec_from_file_location(MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    return module.AutoencoderKLMiniMaxH3Image, f"diffusers@{REFERENCE_COMMIT}:{digest}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .safetensors path (keep it OUTSIDE the repo)")
    args = ap.parse_args()

    import diffusers

    AutoencoderKLMiniMaxH3Image, module_source = reference_class()

    snapshot = Path(os.environ["MINIMAX_H3_IMAGE_SNAPSHOT"])
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
        "reference_module": module_source,
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
