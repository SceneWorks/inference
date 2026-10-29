//! Qwen3 decoder block (pre-norm residual): `h += attn(input_ln(h))`, then `h += mlp(post_ln(h))`.
//! The shared [`Qwen3TextDecoderLayer`] (sc-27326) over FLUX.2's `lin` loads (packed when the
//! snapshot carries `{base}.scales` under a [`Flux2Quant`]), plain (non-upcast) projections, and the
//! q/k RMSNorm gated by `qk_norm` (klein Qwen3: on; dev Mistral: off).

use mlx_rs::Array;

use mlx_gen::text_decoder::{Qwen3TextConfig, Qwen3TextDecoderLayer};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::lin;
use crate::config::Flux2Quant;

pub struct Qwen3DecoderLayer(Qwen3TextDecoderLayer);

impl Qwen3DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
        qk_norm: bool,
        quant: Option<Flux2Quant>,
    ) -> Result<Self> {
        let cfg = Qwen3TextConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            qk_norm,
            upcast: false,
        };
        Ok(Self(Qwen3TextDecoderLayer::from_weights(
            w,
            prefix,
            cfg,
            |key| lin(w, key, quant),
        )?))
    }

    /// Quantize every projection (group_size 64). Norms stay full precision.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        self.0.forward(x, cos, sin, mask)
    }

    pub(crate) fn forward_step(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: &mut mlx_llm::primitives::ContiguousKvCache,
        layer_idx: usize,
    ) -> Result<Array> {
        self.0.forward_with(x, |attn, normed| {
            super::attention::forward_step(attn, normed, cos, sin, cache, layer_idx)
        })
    }
}
