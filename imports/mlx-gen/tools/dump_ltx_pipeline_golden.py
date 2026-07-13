"""LTX-2.3 two-stage T2V pipeline golden — reference `generate_av.py` video path (sc-2679 S5).

Runs the **real** `ltx_2_3_base_q8` transformer (Q8) + upsampler + VAE decoder through the reference
2-stage distilled denoise — stage-1 denoise (8 steps) → 2× spatial upsample → re-noise → stage-2
denoise (3 steps) → VAE decode → uint8 frames — over deterministic **injected** inputs (initial
noise, re-noise sample, synthetic text embeddings, position grids). The Rust `pipeline::generate_t2v`
(mlx-gen-ltx/tests/pipeline_parity.rs, `Precision::F32Q8`) reproduces it.

**Precision: f32** — every module is upcast to f32 activations (the Q8 packed weights stay U32) and
the latents/noise/embeddings are f32, so this gates the pipeline *math* (the legacy dtype-preserving
Euler, the re-noise, the 2-stage orchestration, the flatten/unflatten, the uint8 conversion) isolated
from bf16 rounding — consistent with the S3b DiT gate. The bf16-**production** px>8 verdict is S6.
The legacy Euler + fixed distilled sigmas are the reference's `use_unified` branch (base_q8 is a
split-weight unified checkpoint → `is_unified_mlx_model` True).

**MUST run with mlx 0.31.2** (the Rust build): `quantized_matmul` changed 0.31.0→0.31.2; a 0.31.0
golden mismatches the Rust quant path by ~5e-4/op. Use `/tmp/mlx312/bin/python`.

Run (mlx 0.31.2 env + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      /tmp/mlx312/bin/python tools/dump_ltx_pipeline_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_pipeline_golden.safetensors
"""

import glob
import os
import sys
import types
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())

# Stub the mlx_vlm Gemma import so transformer/text_encoder import without the mlx_lm tree.
for _name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.gemma3"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_lang = types.ModuleType("mlx_vlm.models.gemma3.language")
_lang.Gemma3Model = object
sys.modules["mlx_vlm.models.gemma3.language"] = _lang
_cfg = types.ModuleType("mlx_vlm.models.gemma3.config")
_cfg.TextConfig = object
sys.modules["mlx_vlm.models.gemma3.config"] = _cfg

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_video.generate_av import (  # noqa: E402
    DEFAULT_STAGE_1_SIGMAS,
    DEFAULT_STAGE_2_SIGMAS,
    create_video_position_grid,
)
from mlx_video.models.ltx.config import (  # noqa: E402
    LTXModelConfig,
    LTXModelType,
    LTXRopeType,
)
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.transformer import Modality  # noqa: E402
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents  # noqa: E402
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder  # noqa: E402
from mlx_video.utils import to_denoised  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
DIM, CTX = 4096, 16
LF, H1, W1 = 2, 2, 2  # stage-1 latent frames/height/width → stage-2 is 2× spatial (4×4)


def _f32_acts(model):
    """Upcast every non-packed param (incl. Q8 scales/biases) to f32; keep the U32 packed weight."""
    model.update(
        tree_map(
            lambda p: p.astype(mx.float32) if p.dtype != mx.uint32 else p, model.parameters()
        )
    )
    mx.eval(model.parameters())


# --- transformer (Q8, video-only AudioVideo config; mirrors dump_ltx_dit_golden.py) ---
config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=32,
    attention_head_dim=128,
    in_channels=128,
    out_channels=128,
    num_layers=48,
    cross_attention_dim=4096,
    caption_channels=4096,
    caption_projection_first_linear=False,
    caption_projection_second_linear=False,
    adaln_embedding_coefficient=9,
    apply_gated_attention=True,
    audio_num_attention_heads=32,
    audio_attention_head_dim=64,
    audio_in_channels=128,
    audio_out_channels=128,
    audio_cross_attention_dim=2048,
    audio_caption_channels=2048,
    rope_type=LTXRopeType.SPLIT,
    double_precision_rope=True,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20],
    use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
model = LTXModel(config)
raw = mx.load(str(MODEL / "transformer.safetensors"))
video = {k: v for k, v in raw.items() if "audio" not in k and "av_ca" not in k and "a2v" not in k}
quantized_paths = {k.rsplit(".", 1)[0] for k in video if k.endswith(".scales")}
nn.quantize(
    model,
    group_size=64,
    bits=8,
    class_predicate=lambda p, m: isinstance(m, nn.Linear) and p in quantized_paths,
)
model.load_weights(list(video.items()), strict=False)
_f32_acts(model)


def forward_velocity(video_flat, timesteps, positions, context):
    """The reference video velocity forward (prepare → 48 blocks → process_output), matching the
    Rust `LtxDiT::forward` (S3b bit-exact). `video_flat` is (1, S, 128); returns (1, S, 128)."""
    modality = Modality(
        latent=video_flat,
        timesteps=timesteps,
        positions=positions,
        context=context,
        context_mask=None,
        enabled=True,
    )
    args = model.video_args_preprocessor.prepare(modality, None)
    emb_ts = args.embedded_timestep
    v = args
    for block in model.transformer_blocks.values():
        v, _ = block(video=v, audio=None)
    return model._process_output(
        model.scale_shift_table, model.norm_out, model.proj_out, v.x, emb_ts
    )


def denoise_video_only(latents, positions, context, sigmas):
    """Reference T2V video denoise (legacy dtype-preserving Euler, no CFG, no state)."""
    dtype = latents.dtype
    lat = latents
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        b, c, f, h, w = lat.shape
        num_tokens = f * h * w
        flat = mx.transpose(mx.reshape(lat, (b, c, -1)), (0, 2, 1))
        ts = mx.full((b, num_tokens), sigma, dtype=dtype)
        vel = forward_velocity(flat, ts, positions, context)
        vel = mx.reshape(mx.transpose(vel, (0, 2, 1)), (b, c, f, h, w))
        denoised = to_denoised(lat, vel, sigma)
        if sigma_next > 0:
            sn = mx.array(sigma_next, dtype=dtype)
            sg = mx.array(sigma, dtype=dtype)
            lat = denoised + sn * (lat - denoised) / sg
        else:
            lat = denoised
        mx.eval(lat)
    return lat


# --- upsampler + VAE decoder (real weights, f32) ---
upsampler = load_upsampler(str(MODEL / "upsampler.safetensors"))
upsampler.update(tree_map(lambda p: p.astype(mx.float32), upsampler.parameters()))
mx.eval(upsampler.parameters())

vae = load_vae_decoder(str(MODEL), timestep_conditioning=None, use_unified=True)
vae.update(tree_map(lambda p: p.astype(mx.float32), vae.parameters()))
vae.latents_mean = vae.latents_mean.astype(mx.float32)
vae.latents_std = vae.latents_std.astype(mx.float32)
mx.eval(vae.parameters())

# --- deterministic injected inputs (f32) ---
mx.random.seed(7)
stage1_noise = (mx.random.normal((1, 128, LF, H1, W1)) * 0.5).astype(mx.float32)
stage2_noise = (mx.random.normal((1, 128, LF, H1 * 2, W1 * 2)) * 0.5).astype(mx.float32)
context = (mx.random.normal((1, CTX, DIM)) * 0.5).astype(mx.float32)
stage1_positions = create_video_position_grid(1, LF, H1, W1)  # (1,3,S1,2) f32
stage2_positions = create_video_position_grid(1, LF, H1 * 2, W1 * 2)  # (1,3,S2,2) f32

stage1_sigmas = list(DEFAULT_STAGE_1_SIGMAS)
stage2_sigmas = list(DEFAULT_STAGE_2_SIGMAS)

# --- 2-stage pipeline ---
stage1_out = denoise_video_only(stage1_noise, stage1_positions, context, stage1_sigmas)
upsampled = upsample_latents(stage1_out, upsampler, vae.latents_mean, vae.latents_std)
noise_scale = mx.array(stage2_sigmas[0], dtype=upsampled.dtype)
renoised = stage2_noise * noise_scale + upsampled * (mx.array(1.0, dtype=upsampled.dtype) - noise_scale)
final_latents = denoise_video_only(renoised, stage2_positions, context, stage2_sigmas)

# --- decode → uint8 frames (the reference output conversion) ---
vid = vae(final_latents)
vid = mx.squeeze(vid, axis=0)
vid = mx.transpose(vid, (1, 2, 3, 0))
vid = mx.clip((vid + 1.0) / 2.0, 0.0, 1.0)
frames = (vid * 255).astype(mx.uint8)
mx.eval(final_latents, frames)
print(f"pipeline: stage1{stage1_out.shape} -> final{final_latents.shape} -> frames{frames.shape}")

tensors = {
    "stage1_noise": stage1_noise,
    "stage2_noise": stage2_noise,
    "context": context,
    "stage1_positions": stage1_positions,
    "stage2_positions": stage2_positions,
    "latent_mean": vae.latents_mean,
    "latent_std": vae.latents_std,
    "stage1_out": stage1_out,
    "upsampled": upsampled,
    "renoised": renoised,
    "final_latents": final_latents,
    "frames": frames,
}
out_path = fixture("mlx-gen-ltx/tests/fixtures/ltx_pipeline_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out_path, tensors, metadata={"S1": str(LF * H1 * W1), "ctx": str(CTX)})
print(f"wrote {out_path}")
