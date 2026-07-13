"""LTX-2.3 video-VAE golden — reference decode + encode I/O (sc-2679 S2).

Loads the **real** `vae_decoder.safetensors` / `vae_encoder.safetensors` from the on-disk
`ltx_2_3_base_q8` snapshot via the reference `load_vae_decoder` / `load_vae_encoder`, casts the
modules + inputs to **f32** (the VAE's quality target; isolates correctness from bf16 rounding),
runs a deterministic small decode (latent -> video) and encode (video -> latent), and dumps the
f32 I/O. The Rust `LtxVideoVae` (mlx-gen-ltx/tests/vae_parity.rs) loads the SAME bf16 weights,
upcasts to f32, and must reproduce both.

Small shapes keep the committed fixture ~0.9 MB while exercising every path: 2 latent frames ->
9 video frames (temporal up/down + first/last-frame replication), 3 spatial up/downsamples
(DepthToSpace / SpaceToDepth + group-mean skip), pixel-norm, denorm/norm, patchify/unpatchify.

Run (mflux venv + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_vae_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_vae_golden.safetensors
"""

import glob
import os
import sys
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())

import mlx.core as mx  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder  # noqa: E402
from mlx_video.models.ltx.video_vae.encoder import load_vae_encoder  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"


def to_f32_module(mod):
    """Cast every parameter (incl. the latents stats attrs) to f32 — the Rust port upcasts the
    same bf16 weights losslessly, so this keeps the gate a pure correctness check."""
    mod.update(tree_map(lambda p: p.astype(mx.float32), mod.parameters()))
    for attr in ("latents_mean", "latents_std"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr).astype(mx.float32))
    if hasattr(mod, "per_channel_statistics"):
        st = mod.per_channel_statistics
        for attr in ("mean", "std", "_mean_of_means", "_std_of_means"):
            if hasattr(st, attr):
                setattr(st, attr, getattr(st, attr).astype(mx.float32))
    mx.eval(mod.parameters())
    return mod


# --- Decoder ---
decoder = to_f32_module(load_vae_decoder(str(MODEL), use_unified=True))

mx.random.seed(1)
dec_in = mx.random.normal((1, 128, 2, 2, 2)).astype(mx.float32)
dec_out = decoder(dec_in)  # causal=False default
mx.eval(dec_out)
print(f"decode: {dec_in.shape} -> {dec_out.shape}")

# --- Encoder ---
encoder = to_f32_module(load_vae_encoder(str(MODEL), use_unified=True))

mx.random.seed(2)
# Video in [-1, 1], F = 1 + 8*1 = 9, spatial 64 (= 32 * 2 latent).
enc_in = (mx.random.uniform(shape=(1, 3, 9, 64, 64)) * 2.0 - 1.0).astype(mx.float32)
enc_out = encoder(enc_in)
mx.eval(enc_out)
print(f"encode: {enc_in.shape} -> {enc_out.shape}")

tensors = {
    "dec_in": dec_in,
    "dec_out": dec_out.astype(mx.float32),
    "enc_in": enc_in,
    "enc_out": enc_out.astype(mx.float32),
}
meta = {
    "dec_in": "1x128x2x2x2",
    "dec_out": "x".join(map(str, dec_out.shape)),
    "enc_in": "1x3x9x64x64",
    "enc_out": "x".join(map(str, enc_out.shape)),
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_vae_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}")
