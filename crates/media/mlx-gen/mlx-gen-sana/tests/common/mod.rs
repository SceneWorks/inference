#![allow(dead_code)]
//! Cross-backend fixture geometry for the SANA parity goldens (sc-19496).
//!
//! `tests/fixtures/sana_transformer_golden.safetensors` and
//! `tests/fixtures/sana_sprint_trunk_golden.safetensors` are committed **byte-identical** to
//! `candle-gen-sana`'s copies under the same names. Both lanes therefore load the same tensors through
//! their own hand-typed `SanaTransformerConfig`, so a drift in either config leaves both lanes
//! internally consistent and both parity suites green while the two backends compare a tensor dumped
//! at one geometry against a model built at another. Nothing could see that: the two crates cannot
//! import each other, because `mlx-gen-*` builds on macOS only.
//!
//! The numbers the two lanes must agree on are declared here as `SHARED_FIXTURE_*` constants and the
//! constructors below build from them. `check_cross_backend_geometry` in `scripts/check-workspace.py`
//! compares every `SHARED_FIXTURE_*` declaration under this crate's `tests/` against the candle crate's,
//! by name set and by value.
//!
//! `mod common;` is compiled into every including test binary, so only a subset is used by any one of
//! them.

use mlx_gen_sana::SanaTransformerConfig;

/// Latent channels the tiny golden was dumped at.
pub const SHARED_FIXTURE_TINY_IN_CHANNELS: i32 = 4;
/// Predicted-noise channels.
pub const SHARED_FIXTURE_TINY_OUT_CHANNELS: i32 = 4;
/// Self-attention heads.
pub const SHARED_FIXTURE_TINY_NUM_ATTENTION_HEADS: i32 = 2;
/// Per-head width — `heads · head_dim` gives the inner width 16.
pub const SHARED_FIXTURE_TINY_ATTENTION_HEAD_DIM: i32 = 8;
/// Linear-DiT depth.
pub const SHARED_FIXTURE_TINY_NUM_LAYERS: i32 = 2;
/// Cross-attention heads.
pub const SHARED_FIXTURE_TINY_NUM_CROSS_ATTENTION_HEADS: i32 = 2;
/// Cross-attention per-head width.
pub const SHARED_FIXTURE_TINY_CROSS_ATTENTION_HEAD_DIM: i32 = 8;
/// Caption (text-encoder) width the golden's `input.caption` carries.
pub const SHARED_FIXTURE_TINY_CAPTION_CHANNELS: i32 = 24;
/// GLUMBConv Mix-FFN expansion.
pub const SHARED_FIXTURE_TINY_MLP_RATIO: f32 = 2.5;
/// Patch size — 1, so the `[1,4,4,4]` latent maps to 16 tokens.
pub const SHARED_FIXTURE_TINY_PATCH_SIZE: i32 = 1;
/// Transformer-block norm epsilon.
pub const SHARED_FIXTURE_TINY_NORM_EPS: f32 = 1e-6;
/// Caption-norm epsilon.
pub const SHARED_FIXTURE_TINY_CAPTION_NORM_EPS: f32 = 1e-5;
/// Attention qk-norm epsilon (read only when `qk_norm` is on — the Sprint superset).
pub const SHARED_FIXTURE_TINY_ATTN_QK_NORM_EPS: f32 = 1e-5;
/// ReLU-linear-attention `1/(Σ + eps)` normalizer epsilon.
pub const SHARED_FIXTURE_TINY_ATTN_EPS: f32 = 1e-15;
/// Base SANA: no guidance embedder (that is a Sprint delta).
pub const SHARED_FIXTURE_TINY_GUIDANCE_EMBEDS: bool = false;
/// Guidance-embedder input scale, declared on both configs so the Sprint delta is only the two flags.
pub const SHARED_FIXTURE_TINY_GUIDANCE_EMBEDS_SCALE: f32 = 0.1;
/// Base SANA: no qk-norm (the other Sprint delta).
pub const SHARED_FIXTURE_TINY_QK_NORM: bool = false;

/// SANA-Sprint delta: the guidance embedder is ON.
pub const SHARED_FIXTURE_SPRINT_GUIDANCE_EMBEDS: bool = true;
/// SANA-Sprint delta: `qk_norm="rms_norm_across_heads"` is ON.
pub const SHARED_FIXTURE_SPRINT_QK_NORM: bool = true;

/// Tiny config matching `dump_sana_transformer_golden.py`'s tiny instance.
///
/// Built from the `SHARED_FIXTURE_TINY_*` constants, which `check_cross_backend_geometry` holds equal
/// to the candle lane's — the two lanes load byte-identical golden bytes and must agree about their
/// geometry before either compares a tensor.
pub fn tiny_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        in_channels: SHARED_FIXTURE_TINY_IN_CHANNELS,
        out_channels: SHARED_FIXTURE_TINY_OUT_CHANNELS,
        num_attention_heads: SHARED_FIXTURE_TINY_NUM_ATTENTION_HEADS,
        attention_head_dim: SHARED_FIXTURE_TINY_ATTENTION_HEAD_DIM,
        num_layers: SHARED_FIXTURE_TINY_NUM_LAYERS,
        num_cross_attention_heads: SHARED_FIXTURE_TINY_NUM_CROSS_ATTENTION_HEADS,
        cross_attention_head_dim: SHARED_FIXTURE_TINY_CROSS_ATTENTION_HEAD_DIM,
        caption_channels: SHARED_FIXTURE_TINY_CAPTION_CHANNELS,
        mlp_ratio: SHARED_FIXTURE_TINY_MLP_RATIO,
        patch_size: SHARED_FIXTURE_TINY_PATCH_SIZE,
        norm_eps: SHARED_FIXTURE_TINY_NORM_EPS,
        caption_norm_eps: SHARED_FIXTURE_TINY_CAPTION_NORM_EPS,
        attn_qk_norm_eps: SHARED_FIXTURE_TINY_ATTN_QK_NORM_EPS,
        attn_eps: SHARED_FIXTURE_TINY_ATTN_EPS,
        guidance_embeds: SHARED_FIXTURE_TINY_GUIDANCE_EMBEDS,
        guidance_embeds_scale: SHARED_FIXTURE_TINY_GUIDANCE_EMBEDS_SCALE,
        qk_norm: SHARED_FIXTURE_TINY_QK_NORM,
    }
}

/// Tiny SANA-**Sprint** config (guidance embedder + qk-norm ON), matching `dump_sana_sprint_golden.py`.
pub fn tiny_sprint_config() -> SanaTransformerConfig {
    SanaTransformerConfig {
        guidance_embeds: SHARED_FIXTURE_SPRINT_GUIDANCE_EMBEDS,
        qk_norm: SHARED_FIXTURE_SPRINT_QK_NORM,
        ..tiny_config()
    }
}
