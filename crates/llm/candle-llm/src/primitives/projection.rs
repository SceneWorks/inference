//! A linear projection that is either dense or group-wise quantized.
//!
//! The decoders hold their attention/MLP projections behind this so quantize-on-load is a load-time
//! choice with no decoder changes: a dense `[out, in]` weight either stays dense (a
//! [`candle_nn::Linear`]) or is quantized to Q4/Q8 ([`QuantizedLinear`]) via Candle's quant.

use candle_core::quantized::GgmlDType;
use candle_core::Tensor;
use candle_nn::{Linear, Module};

use crate::error::Result;
use crate::primitives::quant::QuantizedLinear;

/// Group-wise quantization spec, mapped to a Candle GGML dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantSpec {
    /// The target GGML block-quant dtype.
    pub dtype: GgmlDType,
}

impl QuantSpec {
    /// 4-bit (GGML Q4_K).
    pub fn q4() -> Self {
        Self {
            dtype: GgmlDType::Q4K,
        }
    }

    /// 8-bit (GGML Q8_0).
    pub fn q8() -> Self {
        Self {
            dtype: GgmlDType::Q8_0,
        }
    }

    /// Map a persisted `quantization.bits` value to a spec: `4 → Q4_K`, `8 → Q8_0`. Any other width
    /// is unrecognized (`None`) — the snapshot writer only ever emits 4 or 8.
    pub fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            4 => Some(Self::q4()),
            8 => Some(Self::q8()),
            _ => None,
        }
    }

    /// The bit width to persist in a snapshot's `quantization` block (`Q4_K → 4`, `Q8_0 → 8`).
    pub fn bits(&self) -> u32 {
        match self.dtype {
            GgmlDType::Q8_0 => 8,
            _ => 4,
        }
    }
}

/// A linear projection weight, dense or quantized.
pub enum Projection {
    /// A dense `[out, in]` weight wrapped in a Candle linear.
    Dense(Linear),
    /// A group-wise quantized weight.
    Quantized(QuantizedLinear),
}

impl Projection {
    /// Load from a dense `[out, in]` weight, quantizing it if `quant` is set.
    pub fn load(weight: Tensor, quant: Option<QuantSpec>) -> Result<Self> {
        Self::load_with_bias(weight, None, quant)
    }

    /// Load from a dense `[out, in]` weight plus an optional `[out]` bias (Qwen2 attention carries
    /// q/k/v bias), quantizing the weight if `quant` is set. The bias is always applied dense.
    pub fn load_with_bias(
        weight: Tensor,
        bias: Option<Tensor>,
        quant: Option<QuantSpec>,
    ) -> Result<Self> {
        match quant {
            None => Ok(Projection::Dense(Linear::new(weight, bias))),
            Some(q) => Ok(Projection::Quantized(QuantizedLinear::quantize(
                &weight, q.dtype, bias,
            )?)),
        }
    }

    /// `x @ weightᵀ`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Projection::Dense(l) => Ok(l.forward(x)?),
            Projection::Quantized(q) => q.forward(x),
        }
    }

    /// Whether this projection is quantized.
    pub fn is_quantized(&self) -> bool {
        matches!(self, Projection::Quantized(_))
    }
}
