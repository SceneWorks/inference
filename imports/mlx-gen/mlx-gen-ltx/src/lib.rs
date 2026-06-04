//! # mlx-gen-ltx
//!
//! LTX-2.3 **video** (text-to-video) provider crate for [`mlx-gen`]. Port of the
//! `mlx-video-with-audio` package's LTX video path (`generate_av.py`, `models/ltx/*`,
//! `models/ltx/video_vae/*`) onto Rust + `mlx-rs`.
//!
//! **Scope:** the **video-only** T2V core (sc-2679) + **single-image I2V** (sc-2685: VAE-encode the
//! conditioning image at both stage resolutions, inject it as a clean latent at frame 0 with a
//! per-frame denoise mask). The audio half (`generate_av.py`'s AudioVideo path — whose *video* I2V
//! conditioning is the same primitives reused), Q4/Q8-of-everything, LoRA, and LoKr are sibling
//! stories.
//!
//! This crate self-registers `ltx_2_3` into the `mlx-gen` model registry; load it with
//! `mlx_gen::load("ltx_2_3", spec)`.
//!
//! ## Status (S0–S6 complete)
//! The full text-to-video path is wired and pixel-parity vs the reference `generate_av.py`: Gemma-3
//! tokenizer (byte-exact) → [`LtxTextEncoder`] (Gemma backbone + connector) → seeded noise → the
//! 2-stage distilled denoise ([`pipeline`]: stage-1 half-res → 2× [`upsampler::LatentUpsampler`] →
//! re-noise → stage-2 full-res over the 48-layer [`transformer::LtxDiT`]) → [`vae::LtxVideoVae`]
//! decode → uint8 frames. Built on SPLIT 3-D RoPE (double-precision), an f32 position grid, the
//! distilled sigma schedules, and the legacy dtype-preserving Euler step.
//!
//! The distilled stage-1 sampler is chaos-sensitive, so e2e pixel-parity requires a **bit-exact
//! per-forward DiT** (sc-2842 — the adaLN timestep table must be built in MLX f32, not host f64). Two
//! shipped precisions, both gated bit-exact vs their reference golden: [`transformer::Precision::quant_f32`]
//! (f32 activations × quantized weights — the quality target) and [`transformer::Precision::quant_bf16`]
//! (the reference's native bf16 activations — the production-speed path). The quant geometry (**Q4**/Q8)
//! rides on the checkpoint's `split_model.json` (sc-2686). **I2V** single-image conditioning (sc-2685)
//! is wired into the same 2-stage path ([`conditioning`] + [`pipeline::generate_i2v_latents`], gated
//! bit-exact by `tests/i2v_parity.rs`). LoRA, LoKr, and audio are siblings.

pub mod conditioning;
pub mod config;
pub mod connector;
pub mod gemma;
pub mod model;
pub mod pipeline;
pub mod positions;
pub mod rope;
pub mod schedule;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod upsampler;
pub mod vae;

pub use conditioning::{apply_conditioning, apply_denoise_mask, I2vConditioning};
pub use config::{LtxConfig, LtxVaeConfig, RopeType, VaeBlock};
pub use connector::Connector;
pub use model::{descriptor, load, Ltx, MODEL_ID};
pub use pipeline::{
    decode_to_frames, denoise, generate_i2v_latents, generate_t2v, generate_t2v_latents,
    preprocess_conditioning_image, renoise, to_uint8_frames, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
pub use text_encoder::LtxTextEncoder;
// Tiling moved to `mlx_gen` core (shared with the Wan VAE — sc-2808). Re-export the module + config
// so `mlx_gen_ltx::tiling::*` / `mlx_gen_ltx::TilingConfig` keep resolving for existing callers.
pub use mlx_gen::tiling::{self, TilingConfig};
pub use tokenizer::LtxTokenizer;
pub use transformer::{to_denoised, LtxDiT, Precision, VideoBlock};
pub use upsampler::{upsample_latents, LatentUpsampler};
pub use vae::LtxVideoVae;
