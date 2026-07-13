//! Ideogram 4's **Qwen3-VL-8B-Instruct** text encoder (text path only — the vision tower is
//! unused for text-to-image). A 36-layer decoder-only LM whose hidden states at the 13 indices
//! in [`crate::config::EXTRACTED_LAYERS`] (`0,3,…,33,35`) are concatenated into the
//! `13·4096 = 53248`-wide features the DiT projects (`llm_cond_proj`).
//!
//! Mirrors the `mlx-gen-flux2` Qwen3 assembly over the shared `mlx-gen` core primitives
//! (`TextRope`, `TokenEmbedding`, `AdaptableLinear`, `rms_norm`, SDPA), differing only in:
//!   * **θ = 5e6** (klein's Qwen3 is 1e6),
//!   * **13** captured layers (klein concatenates 3), and
//!   * the `language_model.*` key prefix.
//!
//! GQA (32 query / 8 kv heads), bias-less q/k/v/o, **per-head q/k RMSNorm**, HF half-split RoPE,
//! SwiGLU MLP, pre-norm residual blocks. The text-only path uses plain 1-D RoPE: Qwen3-VL's MRoPE
//! sections all index the same sequential text position when there are no image tokens, so it
//! reduces to standard RoPE. Dense-only for now; quantization is E6 (sc-5989).

pub mod attention;
pub mod encoder;
pub mod layer;
pub mod mlp;

pub use attention::Qwen3Attention;
pub use encoder::Ideogram4TextEncoder;
pub use layer::Qwen3DecoderLayer;
pub use mlp::Qwen3Mlp;

// The HF half-split text RoPE is identical across families and lives in core.
pub use mlx_gen::nn::TextRope;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Load a bias-less Qwen3 projection from its `{base}.weight` `key`, auto-detecting a pre-quantized
/// packed snapshot (see [`crate::quant::lin`]). Every Qwen3 projection is a bias-less Linear
/// (`attention_bias = false`).
pub(crate) fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    let base = key.strip_suffix(".weight").unwrap_or(key);
    crate::quant::lin(w, base, false)
}

/// Join a module prefix with a leaf name, tolerating an empty prefix.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
