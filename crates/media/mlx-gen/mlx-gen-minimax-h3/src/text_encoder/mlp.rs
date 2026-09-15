//! Qwen3 SwiGLU feed-forward: `down(silu(gate(x)) * up(x))`, bias-less. FFN width 25600 over a
//! 5120 hidden — the widest single tensor in the encoder.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join_key, lin};

/// One Qwen3 SwiGLU MLP.
pub struct Qwen3Mlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl Qwen3Mlp {
    /// Load `{prefix}.{gate,up,down}_proj.weight`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gate: lin(w, &join_key(prefix, "gate_proj.weight"))?,
            up: lin(w, &join_key(prefix, "up_proj.weight"))?,
            down: lin(w, &join_key(prefix, "down_proj.weight"))?,
        })
    }

    /// Quantize the three projections in place.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.gate.quantize(bits, Some(super::GROUP_SIZE))?;
        self.up.quantize(bits, Some(super::GROUP_SIZE))?;
        self.down.quantize(bits, Some(super::GROUP_SIZE))?;
        Ok(())
    }

    /// The packed width shared by all three projections, or `None` if any loaded dense or they
    /// disagree.
    pub fn packed_bits(&self) -> Option<i32> {
        super::uniform_packed_bits([&self.gate, &self.up, &self.down])
    }

    /// Device bytes the three projections hold.
    pub fn nbytes(&self) -> usize {
        [&self.gate, &self.up, &self.down]
            .into_iter()
            .map(crate::quant::nbytes)
            .sum()
    }

    /// `x`: `[b, s, hidden]` → `[b, s, hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let g = silu(&self.gate.forward_upcast(x)?)?;
        let u = self.up.forward_upcast(x)?;
        self.down.forward_upcast(&multiply(&g, &u)?)
    }
}
