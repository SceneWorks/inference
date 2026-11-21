//! `VisionMlp`: a biased SwiGLU MLP (`down(silu(gate(x)) · up(x))`). Port of the fork's
//! `qwen_vision_mlp.py`.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::text_encoder::join;

pub struct VisionMlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl VisionMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: mlx_gen::quant::lin(
                w,
                &join(prefix, "gate_proj"),
                true,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            up: mlx_gen::quant::lin(
                w,
                &join(prefix, "up_proj"),
                true,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            down: mlx_gen::quant::lin(
                w,
                &join(prefix, "down_proj"),
                true,
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&self.gate.forward(x)?)?;
        let up = self.up.forward(x)?;
        self.down.forward(&multiply(&gate, &up)?)
    }
}
