"""LTX-2.3 video-VAE **tiling** golden — reference `decode_with_tiling` I/O (sc-2679 S2b).

Loads the real `ltx_2_3_base_q8` `vae_decoder.safetensors` via the reference `load_vae_decoder`,
casts to f32, and runs the reference `decode_with_tiling` (chunked_conv=False, causal=False — the
exact config the Rust `LtxVideoVae::decode_tiled` reproduces) on two small tiling-triggering cases:

  - spatial-tiled: latent (1,128,1,4,4) -> (1,3,1,128,128), 64px tile / 32px overlap (3x3 tiles)
  - temporal-tiled: latent (1,128,3,2,2) -> (1,3,17,64,64), 16f tile / 8f overlap (2 tiles, causal)

Small tile/overlap (the validators floor at 64px/32px and 16f/8f) make tiling fire at sizes that
keep the committed golden ~1 MB while exercising the trapezoidal blend on each axis (the blend is a
separable product of 1-D masks, so per-axis coverage is sufficient).

Run (mflux venv + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_vae_tiling_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_vae_tiling_golden.safetensors
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
from mlx_video.models.ltx.video_vae.tiling import (  # noqa: E402
    SpatialTilingConfig,
    TemporalTilingConfig,
    TilingConfig,
    decode_with_tiling,
)

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"


def to_f32(mod):
    mod.update(tree_map(lambda p: p.astype(mx.float32), mod.parameters()))
    for attr in ("latents_mean", "latents_std"):
        if hasattr(mod, attr):
            setattr(mod, attr, getattr(mod, attr).astype(mx.float32))
    mx.eval(mod.parameters())
    return mod


decoder = to_f32(load_vae_decoder(str(MODEL), use_unified=True))


def tiled(latent, cfg):
    return decode_with_tiling(
        decoder,
        latent,
        cfg,
        spatial_scale=32,
        temporal_scale=8,
        causal=False,
        timestep=None,
        chunked_conv=False,
    )


# --- Spatial-tiled (3x3 tiles) ---
mx.random.seed(11)
sp_in = mx.random.normal((1, 128, 1, 4, 4)).astype(mx.float32)
sp_cfg = TilingConfig(spatial_config=SpatialTilingConfig(64, 32), temporal_config=None)
sp_out = tiled(sp_in, sp_cfg)
mx.eval(sp_out)
print(f"spatial: {sp_in.shape} -> {sp_out.shape}")

# --- Temporal-tiled (2 tiles, causal) ---
mx.random.seed(12)
tp_in = mx.random.normal((1, 128, 3, 2, 2)).astype(mx.float32)
tp_cfg = TilingConfig(spatial_config=None, temporal_config=TemporalTilingConfig(16, 8))
tp_out = tiled(tp_in, tp_cfg)
mx.eval(tp_out)
print(f"temporal: {tp_in.shape} -> {tp_out.shape}")

tensors = {
    "sp_in": sp_in,
    "sp_out": sp_out.astype(mx.float32),
    "tp_in": tp_in,
    "tp_out": tp_out.astype(mx.float32),
}
meta = {
    "sp": "spatial 64px/32 overlap",
    "tp": "temporal 16f/8 overlap",
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_vae_tiling_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}")
