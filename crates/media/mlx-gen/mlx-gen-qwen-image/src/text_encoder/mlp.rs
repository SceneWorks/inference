//! SwiGLU MLP: `down(silu(gate(x)) · up(x))`. No biases. Port of the fork's `QwenMLP`.

use mlx_rs::Array;

use mlx_gen::nn::SwiGluMlp;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;

/// The shared [`SwiGluMlp`] (sc-27326) over dense, bias-less projections. A dense bias-less
/// [`mlx_gen::adapters::AdaptableLinear`] with no adapters forwards as `matmul(x, wᵀ)` — the op this crate's
/// `matmul_t` issued — so the graph is unchanged.
pub struct QwenMlp(SwiGluMlp);

impl QwenMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let linear = |leaf: &str| {
            mlx_gen::quant::lin(
                w,
                &join(prefix, leaf),
                false,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )
        };
        Ok(Self(SwiGluMlp::new(
            linear("gate_proj")?,
            linear("up_proj")?,
            linear("down_proj")?,
        )))
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward(x)
    }
}
