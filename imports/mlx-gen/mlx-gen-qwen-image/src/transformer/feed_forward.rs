//! Per-stream feed-forward: `mlp_out(gelu_approx(mlp_in(x)))` (both biased, 4× expansion).
//! Port of the fork's `QwenFeedForward`.

use mlx_rs::nn::gelu_approximate;
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;

pub struct FeedForward {
    in_w: Array,
    in_b: Array,
    out_w: Array,
    out_b: Array,
}

impl FeedForward {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            in_w: w.require(&join(prefix, "mlp_in.weight"))?.clone(),
            in_b: w.require(&join(prefix, "mlp_in.bias"))?.clone(),
            out_w: w.require(&join(prefix, "mlp_out.weight"))?.clone(),
            out_b: w.require(&join(prefix, "mlp_out.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = gelu_approximate(linear(x, &self.in_w, &self.in_b)?)?;
        linear(&h, &self.out_w, &self.out_b)
    }
}
