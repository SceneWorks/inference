//! Qwen3 SwiGLU MLP: `down(silu(gate(x)) · up(x))`. No biases.

use mlx_rs::Array;

use mlx_gen::nn::SwiGluMlp;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

/// The shared [`SwiGluMlp`] (sc-27326) over this crate's packed-detect `lin` loads, run through the
/// upcast projection forward.
pub struct Qwen3Mlp(SwiGluMlp);

impl Qwen3Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self(SwiGluMlp::new(
            lin(w, &join(prefix, "gate_proj.weight"))?,
            lin(w, &join(prefix, "up_proj.weight"))?,
            lin(w, &join(prefix, "down_proj.weight"))?,
        )))
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward_upcast(x)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }

    pub(crate) fn materialize_weights(&self) -> Result<()> {
        self.0.materialize_weights()
    }
}
