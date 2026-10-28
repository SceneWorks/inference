//! Text-encoder SwiGLU MLP: `down(silu(gate(x)) * up(x))`. Port of the fork's `MLP`
//! (no biases). `gate`/`up`/`down` are `nn.Linear` — Q4/Q8 targets when the encoder is quantized
//! (the fork's `nn.quantize` predicate hits every Linear); they are not LoRA targets (Z-Image LoRAs
//! hit the DiT).

use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::SwiGluMlp;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::join;

pub(crate) const QUANTIZATION_LINEAR_PATHS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// The shared [`SwiGluMlp`] (sc-27326) loaded through Z-Image's packed-detect text seam; runs the
/// plain (activation-dtype) projection forward.
pub struct TextMlp(SwiGluMlp);

impl TextMlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        // Packed-detect (sc-8670): the three SwiGLU projections load packed from a pre-quantized
        // snapshot or dense otherwise. No biases.
        let lin = |name: &str| {
            crate::quant::lin(w, &join(prefix, name), crate::quant::TEXT_LINEAR_HAS_BIAS)
        };
        Ok(Self(SwiGluMlp::new(
            lin(QUANTIZATION_LINEAR_PATHS[0])?,
            lin(QUANTIZATION_LINEAR_PATHS[1])?,
            lin(QUANTIZATION_LINEAR_PATHS[2])?,
        )))
    }

    /// Quantize the three projections to Q4/Q8 (group_size 64) — the fork quantizes every Linear
    /// in the text encoder. Activations run f32 here, so the quantized matmuls need no dtype guard.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.0.quantize(bits, None)
    }

    /// The three projections, in `gate, up, down` order (iOS quantum accounting reads their shapes).
    pub(crate) fn projections(&self) -> [&AdaptableLinear; 3] {
        self.0.projections()
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        self.0.forward(x)
    }
}
