//! Qwen3-VL LM feed-forward: **SwiGLU** — `down(silu(gate(x)) · up(x))`, no biases.
//!
//! `hidden_act: "silu"` ([`TE_HIDDEN_ACT`](crate::config::TE_HIDDEN_ACT)), `intermediate_size`
//! 9728. This is the *text encoder*; the **DiT** uses `gelu-approximate` instead
//! ([`FFN_ACTIVATION`](crate::config::FFN_ACTIVATION)) — the two must not be conflated, which is
//! why both live as named constants in [`crate::config`].

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

/// One decoder layer's SwiGLU MLP.
pub struct Qwen3VlMlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Qwen3VlMlp {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.gate.quantize(bits, None)?;
        self.up.quantize(bits, None)?;
        self.down.quantize(bits, None)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        [&self.gate, &self.up, &self.down]
            .into_iter()
            .filter(|linear| linear.is_quantized())
            .count()
    }

    /// Load from `{prefix}.{gate,up,down}_proj.weight`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: lin(w, &join(prefix, "gate_proj"))?,
            up: lin(w, &join(prefix, "up_proj"))?,
            down: lin(w, &join(prefix, "down_proj"))?,
        })
    }

    /// `x`: `[b, s, hidden]` → `[b, s, hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(
            &silu(&self.gate.forward_upcast(x)?)?,
            &self.up.forward_upcast(x)?,
        )?;
        self.down.forward_upcast(&gated)
    }
}
