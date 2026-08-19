"""LTX-2.5 **conv** video-VAE decode golden vs the v1.2.0 reference — sc-18766 (added scope).

sc-18765 established that the LTX-2.5 conv VAE loads through the shipped `LtxVideoVae` and
round-trips at 53-58 dB. That is a self-consistency claim: encode then decode with the same port.
This golden is the missing external one — the reference `ConvVideoDecoder` from Lightricks/LTX-2
v1.2.0, on the real 2.5 conv checkpoint, decoding a fixed latent, compared **absolutely**.

Deliberately backend-neutral: the fixture holds only a latent and the reference pixels, so
`candle-gen-ltx` (sc-18767) asserts against the same file without a second dump.

Run:
    LTX2_SRC=~/src/LTX-2/packages/ltx-core/src \\
    LTX25_VAE_DIR=/path/to/Lightricks--LTX-2.5/vae \\
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx25_conv_vae_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx25_conv_vae_golden.safetensors
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from _ltx25_diffvae_ref import (
    CONV_VAE,
    REFERENCE_COMMIT,
    env_int,
    ltx_core_on_path,
    probe,
    report,
    require_finite,
    vae_dir,
)
from _paths import fixture

torch.manual_seed(0)

LATENT_T = env_int("LTX25_GOLDEN_T", 3)
LATENT_H = env_int("LTX25_GOLDEN_H", 4)
LATENT_W = env_int("LTX25_GOLDEN_W", 5)

ltx_core_on_path()

from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader  # noqa: E402
from ltx_core.model.video_vae.model_configurator import (  # noqa: E402
    VideoDecoderConfigurator,
    video_decoder_sd_ops_for_checkpoint,
)

path = vae_dir() / CONV_VAE
print(f"[ref] {path}")

loader = SafetensorsModelStateDictLoader()
metadata = loader.metadata(str(path))
vae_config = metadata.get("config", {}).get("vae", {})
assert vae_config.get("_class_name") == "CausalVideoAutoencoder", vae_config.get("_class_name")
assert vae_config.get("timestep_conditioning") is False, "the port implements the non-ts path only"
assert vae_config.get("causal_decoder") is False, "the port implements the non-causal path only"

decoder = VideoDecoderConfigurator.from_metadata(metadata)
state = loader.load(str(path), video_decoder_sd_ops_for_checkpoint(str(path), diffusion_vae=False))
missing, unexpected = decoder.load_state_dict(state.sd, strict=False)
assert not missing, f"decoder weights missing: {missing[:8]}"
assert not unexpected, f"checkpoint keys with no home: {unexpected[:8]}"
decoder = decoder.to(dtype=torch.float32).eval()

latent = probe((1, 128, LATENT_T, LATENT_H, LATENT_W), seed=11)
with torch.no_grad():
    pixels = decoder(latent)
print(f"[decode] {tuple(latent.shape)} -> {tuple(pixels.shape)}")
report("dec_in", latent)
report("dec_out", pixels)

tensors = {
    "dec_in": latent.to(torch.float16).contiguous(),
    "dec_out": pixels.to(torch.float32).contiguous(),
}
for name, tensor in tensors.items():
    require_finite(name, tensor)

meta = {
    "story": "sc-18766",
    "reference": f"Lightricks/LTX-2 @ {REFERENCE_COMMIT} (v1.2.0) ConvVideoDecoder",
    "checkpoint": CONV_VAE,
    "dtype": "float32 compute (weights upcast from bf16); dec_in stored float16",
    "dec_in": "x".join(map(str, latent.shape)),
    "dec_out": "x".join(map(str, pixels.shape)),
    "backend_neutral": "consumed by mlx-gen-ltx (sc-18766) and candle-gen-ltx (sc-18767)",
}

out = fixture("mlx-gen-ltx/tests/fixtures/ltx25_conv_vae_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out, metadata=meta)
print(f"wrote {out}")
