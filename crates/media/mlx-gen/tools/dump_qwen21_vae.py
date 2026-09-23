"""Dump Qwen-Image 2.1 RGBA VAE goldens for the Rust port (sc-24108).

Runs the tiny snapshot's `AutoencoderKLQwenImage21` (the frozen upstream class) on fixed RGBA
inputs and records the encoder moments (`quant_conv` output: mean | logvar), the posterior mode
(the latent the pipeline's `_encode_vae_image` uses), and the decoded RGBA image for a fixed latent.
Two geometries: 64x64 (4x4 latent) and 96x64 (6x4 latent, non-square). Single frame throughout —
the 2.1 convolutions are the image specialisation of Wan's causal Conv3d and reject a temporal cache.

Run with the pinned venv: `python3.12 tools/dump_qwen21_vae.py`
Output: `tests/fixtures/qwen21_vae.safetensors`.
"""

from __future__ import annotations

import torch

from _qwen21_common import FIXTURE_DIR, Z_DIM, load_tiny_pipeline, save_safetensors


def run_case(vae, name, height, width, out, meta, seed):
    g = torch.Generator("cpu").manual_seed(seed)
    image = torch.rand((1, 4, 1, height, width), generator=g) * 2 - 1  # RGBA in [-1, 1]
    z = torch.randn((1, Z_DIM, 1, height // 16, width // 16), generator=g)
    trace = {}
    handles = []

    def tap(stage):
        def hook(m, a, o):
            trace[stage] = (o[0] if isinstance(o, tuple) else o).detach().clone()

        return hook

    taps = [
        ("encoder/conv_in", vae.encoder.conv_in),
        ("encoder/mid_block", vae.encoder.mid_block),
        ("encoder/conv_out", vae.encoder.conv_out),
        ("quant_conv", vae.quant_conv),
        ("post_quant_conv", vae.post_quant_conv),
        ("decoder/conv_in", vae.decoder.conv_in),
        ("decoder/mid_block", vae.decoder.mid_block),
        ("decoder/conv_out", vae.decoder.conv_out),
    ]
    taps += [(f"encoder/down_block_{i}", b) for i, b in enumerate(vae.encoder.down_blocks)]
    taps += [(f"decoder/up_block_{i}", b) for i, b in enumerate(vae.decoder.up_blocks)]
    for stage, module in taps:
        handles.append(module.register_forward_hook(tap(stage)))
    try:
        with torch.no_grad():
            moments = vae._encode(image)
            posterior = vae.encode(image).latent_dist
            mode = posterior.mode()
            decoded = vae.decode(z, return_dict=False)[0]
    finally:
        for h in handles:
            h.remove()
    for stage, value in trace.items():
        out[f"{name}/trace/{stage}"] = value
    out[f"{name}/image"] = image
    out[f"{name}/moments"] = moments
    out[f"{name}/mode"] = mode
    out[f"{name}/z"] = z
    out[f"{name}/decoded"] = decoded
    meta[f"{name}/height"] = height
    meta[f"{name}/width"] = width
    print(
        f"{name}: moments {tuple(moments.shape)} mode {tuple(mode.shape)} decoded {tuple(decoded.shape)} "
        f"|dec|max={decoded.abs().max():.4f}"
    )


def main() -> None:
    pipe = load_tiny_pipeline()
    vae = pipe.vae.eval()
    out, meta = {}, {}
    run_case(vae, "square", 64, 64, out, meta, seed=21)
    run_case(vae, "tall", 96, 64, out, meta, seed=22)
    save_safetensors(FIXTURE_DIR / "qwen21_vae.safetensors", out, meta)


if __name__ == "__main__":
    main()
