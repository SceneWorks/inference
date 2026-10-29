//! Qwen3 decoder block (pre-norm residual): `h += attn(input_ln(h))`, then `h += mlp(post_ln(h))`.
//! The shared [`Qwen3TextDecoderLayer`] (sc-27326) over MiniMax-H3's packed-detect `lin` loads (group 64), upcast projections.

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
        self.0.quantize(bits, Some(super::GROUP_SIZE))
    }

    /// The packed width shared by all seven projections, or `None` if any loaded dense or they
    /// disagree.
    pub fn packed_bits(&self) -> Option<i32> {
        super::uniform_packed_bits(self.0.projections())
    }

    /// Device bytes the block holds (norms + projections).
    pub fn nbytes(&self) -> usize {
        self.0.norm_weights().map(Array::nbytes).sum::<usize>()
            + self
                .0
                .projections()
                .into_iter()
                .map(crate::quant::nbytes)
                .sum::<usize>()
    }
}
