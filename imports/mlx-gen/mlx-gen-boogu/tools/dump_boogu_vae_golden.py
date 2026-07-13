"""E4 (sc-6392) — golden dump for the Boogu VAE decode parity test.

Boogu ships the FLUX.1 16-channel ``AutoencoderKL`` (``vae/``). This loads it into the diffusers
reference (f32), de-normalizes a seeded latent exactly as ``mlx_gen_z_image::vae::Vae::decode`` does
(``z / scaling_factor + shift_factor``, the values straight from the VAE config), decodes, and saves
the raw latent + decoded image (both NCHW) so the Rust test can match.

The Rust ``Vae::decode`` applies the same de-normalize internally, so the test feeds it the **raw**
latent ``z`` and compares against this ``golden`` (which is the reference decode of the de-normalized
``z``). NCHW throughout to match the z-image VAE I/O (avoids any layout-order skew in the cosine).

Run (paths default to the cached Base snapshot + the committed golden location):
  ~/mlx-flux-venv/bin/python mlx-gen-boogu/tools/dump_boogu_vae_golden.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import mlx.core as mx
import torch
from safetensors.torch import load_file


def default_snapshot() -> Path:
    hits = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--Boogu--Boogu-Image-0.1-Base/snapshots/*/"
            )
        )
    )
    if not hits:
        sys.exit("Boogu-Image-0.1-Base snapshot not found in the HF cache")
    return Path(hits[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "reference/goldens/boogu_vae.safetensors",
    )
    args = ap.parse_args()

    snapshot = args.snapshot or default_snapshot()
    vae_dir = snapshot / "vae"
    if not (vae_dir / "diffusion_pytorch_model.safetensors").exists():
        sys.exit(f"VAE not found: {vae_dir}")

    from diffusers import AutoencoderKL

    print(f"loading AutoencoderKL (f32) from {vae_dir} …")
    cfg = json.loads((vae_dir / "config.json").read_text())
    vae = AutoencoderKL.from_config(cfg)
    state = load_file(str(vae_dir / "diffusion_pytorch_model.safetensors"))
    missing, unexpected = vae.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"VAE state_dict mismatch  missing={missing}  unexpected={unexpected}")
    vae = vae.eval()

    scaling = float(vae.config.scaling_factor)  # 0.3611
    shift = float(vae.config.shift_factor)  # 0.1159
    print(f"scaling_factor={scaling}  shift_factor={shift}")

    torch.manual_seed(0)
    z = torch.randn(1, vae.config.latent_channels, 32, 32)  # 256² image latent
    img_in = torch.randn(1, 3, 256, 256)  # img2img encode probe (deterministic math)
    with torch.no_grad():
        denorm = z / scaling + shift
        img = vae.decode(denorm, return_dict=False)[0]  # [1, 3, 256, 256]
        # Encode = (posterior mean − shift) · scaling, matching `Vae::encode`.
        mean = vae.encode(img_in).latent_dist.mean  # [1, 16, 32, 32]
        enc = (mean - shift) * scaling
    print(f"decoded: {tuple(img.shape)}  encoded: {tuple(enc.shape)}")

    # Match the Rust `Vae::decode` output layout: [1, 3, 1, 256, 256] (frame axis restored).
    golden = img.unsqueeze(2).contiguous()

    def cpu(t):
        return mx.array(t.detach().to(torch.float32).cpu().numpy())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.out),
        {
            "z": cpu(z),
            "golden": cpu(golden),
            "img_in": cpu(img_in),
            "enc_golden": cpu(enc),
        },
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
