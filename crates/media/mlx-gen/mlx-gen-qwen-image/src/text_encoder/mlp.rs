//! SwiGLU MLP: `down(silu(gate(x)) · up(x))`. No biases. Port of the fork's `QwenMLP`.

use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::SwiGluMlp;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;

/// The shared [`SwiGluMlp`] (sc-27326) over dense, bias-less projections. A dense bias-less
/// [`AdaptableLinear`] with no adapters forwards as `matmul(x, wᵀ)` — the op this crate's
/// `matmul_t` issued — so the graph is unchanged.
pub struct QwenMlp(SwiGluMlp);

impl QwenMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let dense = |leaf: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&join(prefix, leaf))?.clone(),
                None,
            ))
        };
        Ok(Self(SwiGluMlp::new(
            dense("gate_proj.weight")?,
            dense("up_proj.weight")?,
            dense("down_proj.weight")?,
        )))
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward(x)
    }
}
