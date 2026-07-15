//! # candle-gen-mochi
//!
//! The **Mochi 1** (`genmo/mochi-1-preview`, Apache-2.0) text-to-video provider for
//! [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of `mlx-gen-mochi`. Mochi is a
//! T5-XXL-conditioned dual-stream MMDiT (the **AsymmDiT**) with an asymmetric 3-D causal-conv VAE
//! (6× temporal, 8× spatial). It has **no** `candle-transformers` reference: the masked T5 encode
//! (run through the reused `candle_gen_flux::PackedT5Encoder` + the tokenizer key-padding mask Mochi's
//! `_get_t5_prompt_embeds` applies), the learned 3-D RoPE, the linear-quadratic flow-match scheduler,
//! the dual-stream AsymmDiT denoiser, and the AsymmVAE decoder (on a from-scratch conv2d-tap conv3d)
//! are all ported here, preserving the exact `mlx-gen-mochi` math.
//!
//! **txt2video, true CFG:** Mochi is **not** distilled, so it exposes negative-prompt + `guidance`
//! true classifier-free guidance over the `[neg, pos]` batch. The denoise runs through the unified
//! `candle_gen::run_curated_sampler` (the CFG recombine `uncond + g·(cond − uncond)` lives inside the
//! predict closure) over the inverted linear-quadratic sigma schedule.
//!
//! **Dtypes:** the AsymmDiT + T5 run **bf16** (the checkpoint's native dtype; the 10B DiT does not fit
//! f32 on a single consumer GPU), the AsymmVAE runs **f32**; attention and every weightless/weighted
//! RMS-norm upcast to f32. `backend = "candle"`, `mac_only = false`. Quant tiers ship as pre-quantized
//! per-tier checkpoints (epic 1788 / A6), *not* on-the-fly requant, so `supported_quants` is empty.

pub mod config;
pub mod conv3d;
pub mod nn;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod vae;

use candle_gen::candle_core::DType;

pub use config::{MochiConfig, MochiVaeConfig};
pub use rope::{get_positions, MochiRope};
pub use scheduler::{cfg_combine, linear_quadratic_schedule, MochiScheduler};
pub use text_encoder::{encode_prompt, load_indexed_var_builder, MochiT5, MochiTextConditioning};
pub use tokenizer::{load_tokenizer, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID};
pub use vae::MochiVaeDecoder;

/// Public provider id: `"mochi_1"`.
pub const MODEL_ID: &str = "mochi_1";

/// The AsymmDiT + T5 compute dtype (the checkpoint's native bf16; attention/norms upcast to f32).
pub const DIT_DTYPE: DType = DType::BF16;
/// The AsymmVAE compute dtype (the decoder is numerically f32-only — bf16 intermediates reach O(100)).
pub const VAE_DTYPE: DType = DType::F32;
