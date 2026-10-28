//! Qwen3-VL LM feed-forward: **SwiGLU** — `down(silu(gate(x)) · up(x))`, no biases.
//!
//! `hidden_act: "silu"` ([`TE_HIDDEN_ACT`](crate::config::TE_HIDDEN_ACT)), `intermediate_size`
//! 9728. This is the *text encoder*; the **DiT** uses `gelu-approximate` instead
//! ([`FFN_ACTIVATION`](crate::config::FFN_ACTIVATION)) — the two must not be conflated, which is
//! why both live as named constants in [`crate::config`].

use mlx_rs::Array;

use mlx_gen::nn::SwiGluMlp;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

/// One decoder layer's SwiGLU MLP: the shared [`SwiGluMlp`] (sc-27326) over mage's packed-detect
/// `lin` loads, run through the upcast projection forward.
pub struct Qwen3VlMlp(SwiGluMlp);

impl Qwen3VlMlp {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        self.0
            .projections()
            .into_iter()
            .filter(|linear| linear.is_quantized())
            .count()
    }

    /// Load from `{prefix}.{gate,up,down}_proj.weight`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self(SwiGluMlp::new(
            lin(w, &join(prefix, "gate_proj"))?,
            lin(w, &join(prefix, "up_proj"))?,
            lin(w, &join(prefix, "down_proj"))?,
        )))
    }

    /// `x`: `[b, s, hidden]` → `[b, s, hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward_upcast(x)
    }
}
