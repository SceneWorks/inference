//! Text-encoder SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Port of the fork's `MLP`
//! (no biases). `gate`/`up`/`down` are `nn.Linear` — Q4/Q8 targets when the encoder is quantized
//! (the fork's `nn.quantize` predicate hits every Linear); they are not LoRA targets (Z-Image LoRAs
//! hit the DiT).

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;

pub struct TextMlp {
    gate: AdaptableLinear,
    up: AdaptableLinear,
    down: AdaptableLinear,
}

impl TextMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let dense = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&join(prefix, name))?.clone(),
                None,
            ))
        };
        Ok(Self {
            gate: dense("gate_proj.weight")?,
            up: dense("up_proj.weight")?,
            down: dense("down_proj.weight")?,
        })
    }

    /// Quantize the three projections to Q4/Q8 (group_size 64) — the fork quantizes every Linear
    /// in the text encoder. Activations run f32 here, so the quantized matmuls need no dtype guard.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.gate, &mut self.up, &mut self.down] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.gate.forward(x)?)?, &self.up.forward(x)?)?;
        self.down.forward(&gated)
    }
}
