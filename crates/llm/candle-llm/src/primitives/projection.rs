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

/// A layer's key **and** value projections, which may be one shared weight.
///
/// Gemma 4's `attention_k_eq_v` makes the `full_attention` layers reuse the key projection's output
/// as the value projection's — there is no `v_proj` weight in the checkpoint at all. That is a
/// projection-layer fact, not a decoder one: the value path still gets its own (scale-free) per-head
/// norm afterwards, so K and V remain different tensors; only the matmul and the weight are shared.
///
/// Holding it here keeps the saving real. A decoder that "supported" `k_eq_v` by running the same
/// weight through two projections would produce identical numbers while paying twice the matmul and
/// twice the (quantized) weight footprint — the whole point of the flag.
pub struct KvProjection {
    k: Projection,
    /// `None` means `attention_k_eq_v`: the value heads come from `k`'s output.
    v: Option<Projection>,
}

impl KvProjection {
    /// Independent key and value projections (every architecture before Gemma 4, and Gemma 4's
    /// `sliding_attention` layers).
    pub fn separate(k: Projection, v: Projection) -> Self {
        Self { k, v: Some(v) }
    }

    /// One shared projection feeding both key and value heads (`attention_k_eq_v: true`).
    pub fn shared(k: Projection) -> Self {
        Self { k, v: None }
    }

    /// Whether K and V share a projection.
    pub fn k_eq_v(&self) -> bool {
        self.v.is_none()
    }

    /// The key projection.
    pub fn key(&self) -> &Projection {
        &self.k
    }

    /// The value projection, or `None` when it is shared with the key's.
    pub fn value(&self) -> Option<&Projection> {
        self.v.as_ref()
    }

    /// Project `x` into the **raw** key and value tensors, before any per-head norm or RoPE.
    ///
    /// When shared, the key projection runs **once** and both returned handles reference that one
    /// result (Candle tensors are refcounted, so this is a handle clone, not a copy).
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = self.k.forward(x)?;
        match &self.v {
            Some(v) => {
                let v = v.forward(x)?;
                Ok((k, v))
            }
            None => Ok((k.clone(), k)),
        }
    }

    /// Whether either half is quantized.
    pub fn is_quantized(&self) -> bool {
        self.k.is_quantized() || self.v.as_ref().is_some_and(Projection::is_quantized)
    }
}
