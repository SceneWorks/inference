//! Z-Image text encoder — a Qwen3-style decoder-only LM that turns the prompt into `cap_feats`
//! (the DiT's `Conditioning`). Port of the fork's `z_image_text_encoder`.
//!
//! Qwen3-style (not Qwen2): per-head `q_norm`/`k_norm`, **no biases**, HF half-split RoPE,
//! GQA (32 query / 8 kv heads), pre-norm residual blocks. The encoder returns the **second-to-
//! last** layer's hidden states (no final norm). The sub-modules live here; the full
//! [`encoder::TextEncoder`] assembly + prompt encoding are in [`encoder`].

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;
pub(crate) mod stream;

pub use attention::TextAttention;
pub use encoder::{TextEncoder, ZTextEncoderConfig};
pub use layer::EncoderLayer;
pub use mlp::TextMlp;
// The HF half-split text RoPE is identical across families and now lives in core (F-006).
pub use mlx_gen::nn::TextRope;

/// Epsilon for the per-head `q_norm`/`k_norm`. Z-Image's text encoder is exactly Qwen3-4B, whose
/// released `config.json` carries `rms_norm_eps: 1e-06` and declares no separate qk-norm epsilon —
/// HF's `Qwen3Attention` builds both per-head norms with `config.rms_norm_eps`, the same value the
/// block-level norms take. This read 1e-5, the `mlx_rs::fast::rms_norm` library default left in
/// place where the fork passes no explicit eps, until the sc-17137 sync review settled it against
/// the checkpoint. `ENCODER_CONTRACT.qk_norm_eps` in `lib.rs` publishes the same number.
pub(crate) const QK_NORM_EPS: f32 = 1e-6;

/// Join a module prefix with a leaf name, tolerating an empty prefix (so flat fixtures and
/// real `layers.{i}` trees both resolve without a stray leading dot).
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
