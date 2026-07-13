//! Packed (pre-quantized) weight loading helpers — auto-detect a Q4/Q8 snapshot by the presence of
//! `{base}.scales` and build the quantized module directly (no dense bf16 transient), else load
//! dense. The same loaders serve a dense bf16 snapshot and a pre-quantized one (E8). Mirrors the
//! ideogram crate's `quant` helpers.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::TokenEmbedding;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Group size for every Boogu group-wise-affine quantization (pack + load). **32, not the codebase
/// default 64** — the DiT hidden size is **3360 = 32·105**, which is divisible by 32 but NOT 64, so
/// MLX `quantize` (which requires the last dim be a multiple of the group size) rejects the bulk of
/// the DiT at group 64. 32 is the largest power-of-two group that divides 3360, and it also divides
/// every Qwen3-VL TE dimension (4096 / 12288), so one group size serves both stacks.
pub(crate) const GROUP_SIZE: i32 = 32;

/// Load `{base}` as an [`AdaptableLinear`] at Boogu's [`GROUP_SIZE`] — packed when `{base}.scales` is
/// present, else dense. The shared [`mlx_gen::quant::lin`] with `bias` loading the dense `{base}.bias`
/// (distinct from the quant's `{base}.biases`).
pub(crate) fn lin(w: &Weights, base: &str, bias: bool) -> Result<AdaptableLinear> {
    mlx_gen::quant::lin(w, base, bias, GROUP_SIZE)
}

/// Load `{base}` as a [`TokenEmbedding`] at Boogu's [`GROUP_SIZE`] — packed when `{base}.scales` is
/// present, else dense ([`mlx_gen::quant::embedding`]).
pub(crate) fn embedding(w: &Weights, base: &str) -> Result<TokenEmbedding> {
    mlx_gen::quant::embedding(w, base, GROUP_SIZE)
}
