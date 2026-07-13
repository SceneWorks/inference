"""LTX-2.3 S0 golden — reference RoPE / position-grid / sigma fixtures (sc-2679 S0).

Weight-free: dumps the *actual* `mlx-video-with-audio` reference functions (no re-implementation),
so the Rust S0 parity test (`mlx-gen-ltx/tests/s0_parity.rs`) compares against ground truth:

  - `create_position_grid` (generate.py / generate_av.py) — pixel-space grid, causal first-frame
    fix, fps division.
  - `precompute_freqs_cis(..., rope_type=SPLIT, double_precision=True)` — the LTX-2.3 video-stream
    cos/sin (dim=inner_dim=4096, 32 heads → (1,32,T,64)).
  - `apply_split_rotary_emb` — the half-rotation applied to a (1,32,T,128) q tensor.
  - the distilled STAGE_1 / STAGE_2 sigma schedules.

The reference package isn't pip-installed anywhere; its source ships in the uv archive cache. This
script finds it and adds it to sys.path, then runs under any env with mlx + numpy (e.g. the mflux
fork venv):

    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      ~/Repos/mflux/.venv/bin/python tools/dump_ltx_s0_golden.py

If MLX_VIDEO_SRC is unset, the archive is auto-discovered under ~/.cache/uv/archive-v0/*.
Output (committed — small, synthetic, weight-free): mlx-gen-ltx/tests/fixtures/ltx_s0_golden.safetensors
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
    raise SystemExit(
        "Could not find the mlx_video source. Set MLX_VIDEO_SRC to the dir containing `mlx_video/`."
    )


sys.path.insert(0, _find_mlx_video_src())

import mlx.core as mx  # noqa: E402

from mlx_video.generate import (  # noqa: E402
    STAGE_1_SIGMAS,
    STAGE_2_SIGMAS,
    create_position_grid,
)
from mlx_video.models.ltx.config import LTXRopeType  # noqa: E402
from mlx_video.models.ltx.rope import (  # noqa: E402
    apply_split_rotary_emb,
    precompute_freqs_cis,
)

# --- Fixed S0 test config (small but non-trivial: frames>1 exercises the causal fix). ---
BATCH = 1
LATENT_FRAMES = 3
LATENT_H = 4
LATENT_W = 6  # 3*4*6 = 72 patches
DIM = 4096  # inner_dim = heads*head_dim (the video stream)
HEADS = 32
HEAD_DIM = 128
THETA = 10000.0
MAX_POS = [20, 2048, 2048]

mx.random.seed(0)

# Position grid (pixel-space, causal fix, fps division) — float32.
positions = create_position_grid(BATCH, LATENT_FRAMES, LATENT_H, LATENT_W)

# SPLIT RoPE, double precision (the LTX-2.3 video path).
cos, sin = precompute_freqs_cis(
    positions,
    dim=DIM,
    theta=THETA,
    max_pos=MAX_POS,
    use_middle_indices_grid=True,
    num_attention_heads=HEADS,
    rope_type=LTXRopeType.SPLIT,
    double_precision=True,
)

# apply_split_rotary_emb on a random q tensor (B, H, T, head_dim).
seq = LATENT_FRAMES * LATENT_H * LATENT_W
apply_in = mx.random.normal((BATCH, HEADS, seq, HEAD_DIM)).astype(mx.float32)
apply_out = apply_split_rotary_emb(apply_in, cos, sin)

tensors = {
    "positions": positions.astype(mx.float32),
    "rope_cos": cos.astype(mx.float32),
    "rope_sin": sin.astype(mx.float32),
    "apply_in": apply_in.astype(mx.float32),
    "apply_out": apply_out.astype(mx.float32),
    "stage1_sigmas": mx.array(STAGE_1_SIGMAS, dtype=mx.float32),
    "stage2_sigmas": mx.array(STAGE_2_SIGMAS, dtype=mx.float32),
}
mx.eval(list(tensors.values()))

meta = {
    "batch": str(BATCH),
    "latent_frames": str(LATENT_FRAMES),
    "latent_h": str(LATENT_H),
    "latent_w": str(LATENT_W),
    "dim": str(DIM),
    "heads": str(HEADS),
    "head_dim": str(HEAD_DIM),
    "theta": repr(THETA),
    "max_pos": ",".join(str(x) for x in MAX_POS),
}

out = fixture("mlx-gen-ltx/tests/fixtures/ltx_s0_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata=meta)
print(f"wrote {out}")
print(f"  positions {positions.shape}  cos {cos.shape}  apply_out {apply_out.shape}")
print(f"  stage1 {STAGE_1_SIGMAS}")
print(f"  stage2 {STAGE_2_SIGMAS}")
