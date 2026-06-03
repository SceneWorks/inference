//! Qwen3 SwiGLU MLP: `down(silu(gate(x)) · up(x))`. No biases. Port of `Qwen3VLMLP`.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, matmul_t};

pub struct Qwen3Mlp {
    gate: Array,
    up: Array,
    down: Array,
}

impl Qwen3Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: w.require(&join(prefix, "gate_proj.weight"))?.clone(),
            up: w.require(&join(prefix, "up_proj.weight"))?.clone(),
            down: w.require(&join(prefix, "down_proj.weight"))?.clone(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&matmul_t(x, &self.gate)?)?, &matmul_t(x, &self.up)?)?;
        matmul_t(&gated, &self.down)
    }
}
