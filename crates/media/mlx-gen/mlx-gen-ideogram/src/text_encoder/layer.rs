//! Qwen3 decoder block (pre-norm residual): `h += attn(input_ln(h))`, then `h += mlp(post_ln(h))`.
//! The shared [`Qwen3TextDecoderLayer`] (sc-27326) over Ideogram's packed-detect `lin` loads, upcast projections.

use mlx_rs::Array;

use mlx_gen::text_decoder::{Qwen3TextConfig, Qwen3TextDecoderLayer};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct Qwen3DecoderLayer(Qwen3TextDecoderLayer);

impl Qwen3DecoderLayer {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        let cfg = Qwen3TextConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            qk_norm: true,
            upcast: true,
        };
        Ok(Self(Qwen3TextDecoderLayer::from_weights(
            w,
            prefix,
            cfg,
            |key| super::lin(w, key),
        )?))
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        self.0.forward(x, cos, sin, mask)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }
}
